#! /usr/bin/env python
"""Train a Transformer language model from a TOML configuration file.

Usage:
    python -m cs336_basics.train [-c CONFIG] [-v] [--dry-run [TOKENS]] [--no-wandb]

The config file (``train_config.toml`` in the current directory by default)
reads the tokenizer and data locations plus the model/optimization
hyperparameters, e.g.::

    # data / tokenizer
    dataset        = "data/tinystories/tinystories_10.txt"
    valid_dataset  = "data/TinyStoriesV2-GPT4-valid.txt"   # optional
    tokenizer      = "experiments/tokenizer-v1"            # experiment dir
    output_dir     = "training"                            # artifact dir

    # model
    # vocab_size is resolved from the tokenizer's training/meta.json.
    d_model        = 256
    num_layers     = 4
    context_length = 256
    theta          = 10000.0       # RoPE theta
    num_heads      = 8
    d_ff           = 704           # optional, defaults to 8/3 * d_model (rounded)

    # optimization
    batch_size     = 32
    num_epochs     = 5             # or set `num_steps` for a fixed total
    num_steps      = 5000          # optional; total steps spread over num_epochs
    max_lr         = 3e-4
    min_lr         = 3e-5
    warmup_fraction = 0.02         # fraction of total steps spent warming up
    cosine_fraction = 1.0          # fraction of total steps in the cosine decay
    weight_decay   = 0.01
    beta1          = 0.9
    beta2          = 0.999
    eps            = 1e-8
    grad_clip      = 1.0
    seed           = 0
    device         = "auto"        # auto | cpu | cuda | mps
    dtype          = "float32"

    # logging / wandb
    log_every          = 1
    eval_every         = 100
    eval_batches       = 20
    checkpoint_every   = 500
    use_wandb          = true
    wandb_project      = "cs336-assignment1"
    wandb_entity       = ""        # optional
    run_name           = ""        # optional, defaults to output dir name

Relative paths are resolved against the directory containing the config file.
The output directory defaults to ``training/`` next to the config file and
receives ``config.json``, ``meta.json``, ``logs.txt`` and ``checkpoints/``.

The ``--dry-run`` flag estimates parameters, FLOPs, peak memory and expected
wall-clock time for the configured model on the current hardware without
loading the tokenizer or running any training. Pass a dataset token count
directly to it (``--dry-run 2000000``) to also derive batches/epoch, total
steps, total time and throughput.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import sys
import time
import tomllib
from itertools import islice
from pathlib import Path

import humanize
import numpy as np
import torch
import torch.nn.functional as F
from quantiphy import Quantity

from cs336_basics.tokenizer.bpe import BPETokenizer
from cs336_basics.training_utils import get_batch, load_checkpoint, save_checkpoint, stream_batch
from cs336_basics.transformer import AdamW, TransformerLM, clip_grad, cosine_lr_schedule

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("train_config.toml")
DEFAULT_OUTPUT_DIR = "training"
DEFAULT_WANDB_PROJECT = "cs336n"

# Number of hex characters kept from the dataset's SHA-256 digest. 20 hex chars
# = 80 bits, so the chance of a collision across even a billion distinct
# datasets is < 1e-12.
HASH_CHARS = 20


# --------------------------------------------------------------------------- #
# Config handling
# --------------------------------------------------------------------------- #
def _resolve(base_dir: Path, path: str) -> Path:
    """Resolve ``path`` against ``base_dir`` unless it is already absolute."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else (base_dir / resolved).resolve()


def load_config(config_path: Path) -> dict:
    """Read and validate the TOML config, resolving paths against its directory."""
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    required = (
        "dataset",
        "tokenizer",
        "d_model",
        "num_layers",
        "context_length",
        "theta",
        "num_heads",
        "batch_size",
        "max_lr",
        "min_lr",
    )
    for key in required:
        if key not in raw:
            raise KeyError(f"missing required config key: {key}")

    if "num_epochs" not in raw and "num_steps" not in raw:
        raise KeyError("config must specify either `num_epochs` or `num_steps`")

    if not isinstance(raw["dataset"], str):
        raise TypeError("`dataset` must be a string path")
    if not isinstance(raw["tokenizer"], str):
        raise TypeError("`tokenizer` must be a string path")
    if not isinstance(raw["context_length"], int) or raw["context_length"] <= 0:
        raise ValueError("`context_length` must be a positive integer")
    if not isinstance(raw["num_layers"], int) or raw["num_layers"] <= 0:
        raise ValueError("`num_layers` must be a positive integer")
    if not isinstance(raw["num_heads"], int) or raw["num_heads"] <= 0:
        raise ValueError("`num_heads` must be a positive integer")

    base_dir = config_path.parent
    dataset = _resolve(base_dir, raw["dataset"])
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset file not found: {dataset}")

    valid_dataset = None
    if "valid_dataset" in raw:
        if not isinstance(raw["valid_dataset"], str):
            raise TypeError("`valid_dataset` must be a string path")
        valid_dataset = _resolve(base_dir, raw["valid_dataset"])
        if not valid_dataset.is_file():
            raise FileNotFoundError(f"valid dataset file not found: {valid_dataset}")

    tokenizer_path = _resolve(base_dir, raw["tokenizer"])
    if not tokenizer_path.is_dir():
        raise NotADirectoryError(f"tokenizer path is not a directory: {tokenizer_path}")

    with (tokenizer_path / "training" / "meta.json").open(encoding="utf-8") as f:
        vocab_size = json.load(f)["result"]["vocab_size"]

    d_model = raw["d_model"]
    # FFN hidden size defaults to 8/3 * d_model, rounded up to the nearest multiple of 64.
    d_ff = raw.get("d_ff", -(-8 * d_model // (3 * 64)) * 64)
    output_dir = _resolve(base_dir, raw.get("output_dir", DEFAULT_OUTPUT_DIR))

    num_epochs = raw.get("num_epochs")
    num_steps = raw.get("num_steps")
    for name, value in (("num_epochs", num_epochs), ("num_steps", num_steps)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"`{name}` must be a positive integer")

    warmup_fraction = raw.get("warmup_fraction", 0.0)
    cosine_fraction = raw.get("cosine_fraction", 1.0)
    for name, fraction in (("warmup_fraction", warmup_fraction), ("cosine_fraction", cosine_fraction)):
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"`{name}` must be a fraction in [0, 1]")
    if cosine_fraction < warmup_fraction:
        raise ValueError("`cosine_fraction` must be >= `warmup_fraction`")

    return {
        "config_path": str(config_path),
        "dataset": dataset,
        "valid_dataset": valid_dataset,
        "tokenizer_path": tokenizer_path,
        "output_dir": output_dir,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": raw["num_layers"],
        "context_length": raw["context_length"],
        "theta": raw["theta"],
        "num_heads": raw["num_heads"],
        "batch_size": raw["batch_size"],
        "num_epochs": num_epochs,
        "num_steps": num_steps,
        "max_lr": raw["max_lr"],
        "min_lr": raw["min_lr"],
        "warmup_fraction": warmup_fraction,
        "cosine_fraction": cosine_fraction,
        "weight_decay": raw.get("weight_decay", 0.01),
        "beta1": raw.get("beta1", 0.9),
        "beta2": raw.get("beta2", 0.999),
        "eps": raw.get("eps", 1e-8),
        "grad_clip": raw.get("grad_clip", 1.0),
        "seed": raw.get("seed", 0),
        "device": raw.get("device", "auto"),
        "dtype": raw.get("dtype", "float32"),
        "log_every": raw.get("log_every", 1),
        "eval_every": raw.get("eval_every", 100),
        "eval_batches": raw.get("eval_batches", 20),
        "checkpoint_every": raw.get("checkpoint_every", 500),
        "resume_from": raw.get("resume_from"),
        "use_wandb": raw.get("use_wandb", True),
        "wandb_project": raw.get("wandb_project", DEFAULT_WANDB_PROJECT),
        "wandb_entity": raw.get("wandb_entity"),
        "run_name": raw.get("run_name"),
        "hardware_flops": raw.get("hardware_flops"),
    }


def _serializable(value):
    """Recursively convert ``Path`` values into strings for JSON serialization."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Tokenizer + dataset encoding
# --------------------------------------------------------------------------- #
def load_tokenizer(tokenizer_path: Path) -> tuple[BPETokenizer, str, dict]:
    """Instantiate a :class:`BPETokenizer` from an experiment directory.

    Tokenizer artifacts are always read from ``<tokenizer>/training/`` and the
    special tokens are inferred from the published ``meta.json``.
    """
    training_dir = tokenizer_path / "training"
    vocab_path = training_dir / "vocab.pkl"
    merges_path = training_dir / "merges.pkl"
    logger.info("loading tokenizer: vocab=%s, merges=%s", vocab_path, merges_path)

    with vocab_path.open("rb") as f_vocab, merges_path.open("rb") as f_merges:
        vocab = pickle.load(f_vocab)
        merges = pickle.load(f_merges)

    with (training_dir / "meta.json").open(encoding="utf-8") as f:
        special_tokens = json.load(f)["config"]["special_tokens"]

    tokenizer = BPETokenizer(vocab, merges, special_tokens=special_tokens)
    meta = {
        "tokenizer_path": str(tokenizer_path),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": special_tokens,
    }
    logger.info("tokenizer ready: vocab_size=%d, special_tokens=%s", len(vocab), special_tokens)
    return tokenizer, tokenizer_path.name, meta


def encoded_paths(dataset_path: Path, tokenizer_id: str) -> tuple[Path, Path]:
    """Return ``(<hash>__<tokenizer id>.bin, <hash>__<tokenizer id>.json)``.

    The hash is the truncated SHA-256 hex digest of the dataset contents, and
    the encoded files are placed adjacent to the dataset.
    """
    digest = hashlib.sha256()
    with dataset_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    stem = f"{digest.hexdigest()[:HASH_CHARS]}__{tokenizer_id}"
    return dataset_path.parent / f"{stem}.bin", dataset_path.parent / f"{stem}.json"


def ensure_encoded(
    tokenizer: BPETokenizer,
    tokenizer_id: str,
    tokenizer_meta: dict,
    dataset_path: Path,
) -> tuple[np.memmap, dict]:
    """Return the encoded dataset, encoding it first if not already cached.

    The encoded file lives next to the dataset as ``<dataset hash>__<tokenizer
    id>.bin`` with a ``.json`` sidecar; if both already exist they are reused.
    """
    max_token_id = max(tokenizer_meta["vocab_size"], 1)
    dtype = "<u2" if max_token_id <= np.iinfo(np.uint16).max else "<u4"
    bin_path, meta_path = encoded_paths(dataset_path, tokenizer_id)

    if bin_path.is_file() and meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        logger.info("using cached encoding: %s (%s tokens)", bin_path, f"{meta.get('num_tokens', 0):,}")
        return np.memmap(bin_path, mode="r", dtype=np.dtype(meta["dtype"])), meta

    storage_dtype = np.dtype(dtype)
    n_input_bytes = dataset_path.stat().st_size
    # Worst case is one token per input byte; truncate once the true count is known.
    tokens_np = np.memmap(bin_path, mode="w+", dtype=storage_dtype, shape=n_input_bytes)

    max_workers = os.cpu_count() or 4
    worker_batch_size = 100_000

    logger.info(
        "encoding %s (%s) -> %s (dtype=%s, max_workers=%d)",
        dataset_path,
        humanize.naturalsize(n_input_bytes),
        bin_path,
        storage_dtype.str,
        max_workers,
    )

    start_time = time.perf_counter()
    w_off = 0
    with dataset_path.open() as f:
        stream = tokenizer.encode_iterable(f, batch_size=worker_batch_size, max_workers=max_workers)
        # buffer a batch of tokens into memory
        while batch := list(islice(stream, 1_000_000)):
            n = len(batch)
            tokens_np[w_off : w_off + n] = batch
            w_off += n

    tokens_np.flush()
    with bin_path.open("r+b") as f_out:
        f_out.truncate(w_off * storage_dtype.itemsize)
    elapsed = time.perf_counter() - start_time

    meta = {
        "dataset": str(dataset_path),
        "dataset_bytes": n_input_bytes,
        "dataset_hash": bin_path.name.split("__", 1)[0],
        "encoded_path": str(bin_path),
        "dtype": storage_dtype.str,
        "num_tokens": w_off,
        "compression_ratio": n_input_bytes / w_off if w_off else float("inf"),
        "encoding_time_s": elapsed,
        "throughput_bytes_per_s": n_input_bytes / elapsed if elapsed else 0.0,
        "max_workers": max_workers,
        "worker_batch_size": worker_batch_size,
        "tokenizer": tokenizer_meta,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "encoded %d tokens in %.3fs (%s bytes/s, compression ratio %.3f) -> %s",
        w_off,
        elapsed,
        f"{meta['throughput_bytes_per_s']:,.0f}",
        meta["compression_ratio"],
        bin_path,
    )
    return np.memmap(bin_path, mode="r", dtype=np.dtype(meta["dtype"])), meta


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(config: dict) -> dict:
    """Run training described by a validated ``config`` and write artifacts."""
    output_dir: Path = config["output_dir"]
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "logs.txt"

    device = pick_device(config["device"])
    torch_dtype = getattr(torch, config["dtype"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    logger.info("device: %s, dtype: %s", device, config["dtype"])
    logger.info("output directory: %s", output_dir)
    logger.debug("resolved config: %s", _serializable(config))

    # --- tokenizer + dataset ------------------------------------------------ #
    tokenizer, tokenizer_id, tokenizer_meta = load_tokenizer(config["tokenizer_path"])
    train_data, train_encoding_meta = ensure_encoded(tokenizer, tokenizer_id, tokenizer_meta, config["dataset"])
    valid_data, valid_encoding_meta = (None, None)
    if config["valid_dataset"] is not None:
        valid_data, valid_encoding_meta = ensure_encoded(
            tokenizer, tokenizer_id, tokenizer_meta, config["valid_dataset"]
        )
    logger.debug(
        "dataset ready: %d train tokens, %d valid tokens",
        len(train_data),
        len(valid_data) if valid_data is not None else 0,
    )

    per_epoch = steps_per_epoch(config, len(train_data))
    num_steps = config.get("num_steps", None)
    if num_steps is None:
        # Normalise `num_epochs` into a single `num_steps` currency. `num_steps`, when
        # set, wins; otherwise it is `num_epochs` passes over the dataset.
        num_steps = int(config["num_epochs"] * per_epoch)
        logger.info(
            "`num_steps` not set; resolved from num_epochs=%s at %d batches/epoch -> %d steps",
            config["num_epochs"],
            per_epoch,
            num_steps,
        )
    else:
        logger.info("`num_steps`=%d set in config (%d batches/epoch)", num_steps, per_epoch)

    Tw = round(config["warmup_fraction"] * num_steps)
    Tc = max(round(config["cosine_fraction"] * num_steps), Tw + 1)
    tokens_per_step = config["batch_size"] * config["context_length"]
    logger.info(
        "training for %d steps (%d batches/epoch, warmup %d, cosine %d, %s tokens/step, %s tokens)",
        num_steps,
        per_epoch,
        Tw,
        Tc,
        f"{tokens_per_step:,}",
        f"{len(train_data):,}",
    )

    # --- model / optimizer -------------------------------------------------- #
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["theta"],
        device=device,
        dtype=torch_dtype,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("model initialized: %s parameters", humanize.intword(num_params, format="%.2f"))

    optimizer = AdamW(
        model.parameters(),
        lr=config["max_lr"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )

    start_step = 0
    resume_from = config["resume_from"]
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = (Path(config["config_path"]).parent / resume_path).resolve()
        start_step = load_checkpoint(resume_path, model, optimizer)
        logger.info("resumed from %s at step %d", resume_path, start_step)

    # --- wandb -------------------------------------------------------------- #
    run = None
    if config["use_wandb"]:
        import wandb

        run_name = config["run_name"] or output_dir.name
        run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"] or None,
            name=run_name,
            dir=str(output_dir),
            config=_serializable(
                {
                    **{k: v for k, v in config.items() if k != "config_path"},
                    "resolved_num_steps": num_steps,
                    "num_params": num_params,
                    "tokenizer_id": tokenizer_id,
                }
            ),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("eval/*", step_metric="train/step")

    # --- loop --------------------------------------------------------------- #
    model.train()
    history: list[dict] = []
    run_start = time.perf_counter()
    last_log_time = run_start
    last_log_step = start_step

    # stream_batch is an infinite generator that reshuffles shard order each pass.
    # With a fixed seed the order is deterministic, so skipping the first start_step
    # batches resumes the exact same data stream. The loop is indexed purely by steps.
    batches = stream_batch(
        train_data,
        len(train_data),
        config["batch_size"],
        config["context_length"],
        device,
        seed=config["seed"],
    )
    for _ in range(start_step):
        next(batches)

    for step in range(start_step, num_steps):
        x, y = next(batches)

        lr = cosine_lr_schedule(step, config["max_lr"], config["min_lr"], Tw, Tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        grad_norm = clip_grad(model.parameters(), config["grad_clip"] or math.inf)
        optimizer.step()

        if step % config["log_every"] == 0:
            now = time.perf_counter()
            interval = now - last_log_time
            steps_done = step - last_log_step + 1
            tokens_per_s = tokens_per_step * steps_done / interval if interval else 0.0
            last_log_time, last_log_step = now, step

            logger.info(
                "step %d/%d | loss %.4f | lr %.3e | grad_norm %.3f | %s tok/s",
                step,
                num_steps,
                loss.item(),
                lr,
                grad_norm,
                f"{tokens_per_s:,.0f}",
            )
            metrics = {
                "train/step": step,
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/grad_norm": grad_norm,
                "train/tokens_per_s": tokens_per_s,
                "train/elapsed_s": now - run_start,
            }
            history.append(metrics)
            if run is not None:
                wandb.log(metrics, step=step)

        if valid_data is not None and config["eval_every"] and step % config["eval_every"] == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(config["eval_batches"]):
                    x, y = get_batch(valid_data, config["batch_size"], config["context_length"], device)
                    logits = model(x)
                    val_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            val_loss /= config["eval_batches"]
            model.train()
            logger.info("step %d | val loss %.4f | perplexity %.2f", step, val_loss, math.exp(val_loss))
            if run is not None:
                wandb.log(
                    {"train/step": step, "eval/loss": val_loss, "eval/perplexity": math.exp(val_loss)},
                    step=step,
                )

        if config["checkpoint_every"] and (step + 1) % config["checkpoint_every"] == 0:
            ckpt_path = ckpt_dir / f"step_{step + 1}.pt"
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            save_checkpoint(model, optimizer, step + 1, ckpt_dir / "latest.pt")
            logger.info("saved checkpoint %s", ckpt_path)

    total_time = time.perf_counter() - run_start
    final_ckpt = ckpt_dir / "final.pt"
    save_checkpoint(model, optimizer, num_steps, final_ckpt)
    logger.info("saved final checkpoint %s", final_ckpt)

    # --- artifacts ---------------------------------------------------------- #
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(_serializable(config), f, ensure_ascii=False, indent=2)

    meta = {
        "config": _serializable(config),
        "tokenizer": tokenizer_meta,
        "dataset": {
            "train": train_encoding_meta,
            "valid": valid_encoding_meta,
        },
        "result": {
            "num_steps": num_steps,
            "num_params": num_params,
            "final_loss": history[-1]["train/loss"] if history else None,
            "final_val_loss": None,
        },
        "timing_s": {
            "total": total_time,
            "per_step": total_time / max(1, num_steps - start_step),
        },
        "artifacts": {
            "config": str(config_path),
            "logs": str(logs_path),
            "report": str(output_dir / "meta.json"),
            "final_checkpoint": str(final_ckpt),
            "checkpoints_dir": str(ckpt_dir),
        },
    }
    meta_path = output_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "training complete in %s (%d steps, %.3f s/step)",
        humanize.precisedelta(total_time, minimum_unit="seconds"),
        num_steps - start_step,
        total_time / max(1, num_steps - start_step),
    )
    logger.info("wrote report to %s", meta_path)

    if run is not None:
        import wandb

        wandb.summary.update(
            {
                "num_params": num_params,
                "num_steps": num_steps,
                "final_loss": meta["result"]["final_loss"],
                "total_time_s": total_time,
            }
        )
        wandb.finish()

    return meta


# --------------------------------------------------------------------------- #
# Hardware / dry-run estimation
# --------------------------------------------------------------------------- #
# The model-statistics formulae below are inlined (the assignment's driver.py is
# a transient analysis script, so train.py has no dependency on it).
DTYPE = "float32"
DTYPE_BYTES = 4


def param_counter(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, **kwargs
) -> dict[str, int]:
    mha = 4 * (d_model**2)
    ffn = 3 * d_ff * d_model
    rms_per_layer = 2 * d_model

    tot_mha = num_layers * mha
    tot_ffn = num_layers * ffn
    tot_rms = num_layers * rms_per_layer + d_model
    embeddings = vocab_size * d_model
    lm_head = d_model * vocab_size

    return {
        "mha": tot_mha,
        "ffn": tot_ffn,
        "rmsnorm": tot_rms,
        "embeddings": embeddings,
        "lm_head": lm_head,
        "total": tot_mha + tot_ffn + tot_rms + embeddings + lm_head,
    }


def flop_counter(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    batch_size: int = 1,
    **kwargs,
) -> dict[str, int]:
    mha = batch_size * (8 * context_length * (d_model**2) + 4 * (context_length**2) * d_model)
    ffn = batch_size * 6 * context_length * d_model * d_ff
    lm_head = batch_size * 2 * context_length * d_model * vocab_size

    tot_mha = mha * num_layers
    tot_ffn = ffn * num_layers

    return {"mha": tot_mha, "ffn": tot_ffn, "lm_head": lm_head, "total": tot_mha + tot_ffn + lm_head}


def activation_counter(
    batch_size: int, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, **kwargs
) -> dict[str, int]:
    bcd = batch_size * context_length * d_model

    tot_mha = (5 * bcd + batch_size * num_heads * context_length**2) * num_layers
    tot_ffn = 35 * bcd * num_layers // 3
    tot_rms = 2 * bcd * num_layers + bcd
    output_embedding = bcd
    cross_entropy = batch_size * context_length * vocab_size

    return {
        "mha": tot_mha,
        "ffn": tot_ffn,
        "rmsnorm": tot_rms,
        "output_embedding": output_embedding,
        "cross_entropy": cross_entropy,
        "total": tot_mha + tot_ffn + tot_rms + output_embedding + cross_entropy,
    }


def peak_memory_bytes(num_params: int, num_activations: int) -> int:
    """Peak training memory: params + grads + activations + optimizer state (2 * params)."""
    return DTYPE_BYTES * (4 * num_params + num_activations)


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_flops(device: str, dtype: torch.dtype = torch.float32, size: int = 4096, iters: int = 5) -> float | None:
    """Estimate achievable FLOP/s on the current hardware via a square matmul.

    A ``size x size @ size x size`` matmul costs ``2 * size**3`` FLOPs.
    """
    try:
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)
        for _ in range(1):  # warmup
            a @ b
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            a @ b
        _synchronize(device)
        elapsed = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - any backend failure must not abort the run
        logger.warning("FLOP benchmark failed on %s: %s", device, exc)
        return None
    return 2 * size**3 * iters / elapsed if elapsed else None


def steps_per_epoch(config: dict, dataset_tokens: int) -> int:
    """Number of batches ``stream_batch`` yields in one pass over the dataset.

    ``stream_batch`` aligns batch starts to the ``batch_size * context_length``
    grid and drops the incomplete tail, so the count is ``(n - 1) // tokens_per_step``.
    """
    tokens_per_step = config["batch_size"] * config["context_length"]
    return max(1, (dataset_tokens - 1) // tokens_per_step)


def build_stats_table(
    name: str,
    config: dict[str, int],
    batch_size: int,
    params: dict[str, int],
    flops: dict[str, int],
    activations: dict[str, int],
):
    """Per-component params/FLOPs/activations table for the dry-run report."""
    from rich.table import Table

    total_flops = flops["total"]
    config_str = "  ".join(f"{key}={value}" for key, value in config.items())
    title = f"{name}\n{config_str}\nbatch_size={batch_size}"
    table = Table(title=title, title_style="bold", header_style="bold", caption_style="bold")
    table.add_column("component")
    table.add_column("params", justify="right")
    table.add_column("flops", justify="right")
    table.add_column("flops %", justify="right")
    table.add_column("activations", justify="right")

    rows = [
        ("attention", "mha", "mha", "mha"),
        ("ffn", "ffn", "ffn", "ffn"),
        ("rmsnorm", "rmsnorm", None, "rmsnorm"),
        ("embedding", "embeddings", None, "output_embedding"),
        ("lm_head", "lm_head", "lm_head", "cross_entropy"),
    ]
    for label, p_key, f_key, a_key in rows:
        p = params.get(p_key, 0) if p_key else 0
        f = flops.get(f_key, 0) if f_key else 0
        a = activations.get(a_key, 0) if a_key else 0
        table.add_row(
            label,
            humanize.intword(p, format="%.2f") if p else "—",
            str(Quantity(f, "FLOPs")) if f else "—",
            f"{100 * f / total_flops:.1f}%" if f else "—",
            humanize.intword(a, format="%.2f") if a else "—",
        )

    table.add_row(
        "total",
        humanize.intword(params["total"], format="%.2f"),
        str(Quantity(total_flops, "FLOPs")),
        "100.0%",
        humanize.intword(activations["total"], format="%.2f"),
        style="bold",
    )
    peak_bytes = peak_memory_bytes(params["total"], activations["total"])
    table.caption = (
        f"peak memory ({DTYPE}, params + grads + activations + optimizer state): {humanize.naturalsize(peak_bytes)}"
    )
    return table


def print_dry_run(config: dict, tokens: int | None = None) -> None:
    """Estimate params, FLOPs, memory and runtime, then return.

    ``tokens`` is the dataset size in tokens. When provided, the report also
    derives batches/epoch, total steps, total time and throughput from
    ``num_steps`` (or ``num_epochs``); otherwise those totals cannot be known.
    """
    if tokens is not None and tokens <= 0:
        raise ValueError("`--dry-run` token count must be a positive integer")
    from rich.console import Console
    from rich.table import Table

    device = pick_device(config["device"])
    torch_dtype = getattr(torch, config["dtype"])
    dims = {
        "vocab_size": config["vocab_size"],
        "context_length": config["context_length"],
        "num_layers": config["num_layers"],
        "d_model": config["d_model"],
        "num_heads": config["num_heads"],
        "d_ff": config["d_ff"],
    }
    batch_size = config["batch_size"]
    logger.debug("dry-run model dims: %s, batch_size=%d", dims, batch_size)

    params = param_counter(**dims)
    flops = flop_counter(batch_size=batch_size, **dims)
    activations = activation_counter(batch_size=batch_size, **dims)
    peak_bytes = peak_memory_bytes(params["total"], activations["total"])

    console = Console()
    console.print(
        build_stats_table(
            config.get("run_name") or "dry-run",
            dims,
            batch_size,
            params,
            flops,
            activations,
        )
    )

    throughput = config.get("hardware_flops") or benchmark_flops(device, dtype=torch_dtype)
    table = Table(title="Runtime estimate", title_style="bold", header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("device", device)
    table.add_row("dtype", config["dtype"])
    table.add_row("params", humanize.intword(params["total"], format="%.2f"))
    table.add_row("FLOPs / step", str(Quantity(flops["total"], "FLOPs")))
    table.add_row("activations / step", humanize.naturalsize(activations["total"] * 4))
    table.add_row("peak memory", humanize.naturalsize(peak_bytes))

    if throughput:
        step_time = flops["total"] / throughput
        table.add_row("measured throughput", f"{throughput / 1e12:.2f} TFLOP/s")
        table.add_row("est. time / step", f"{step_time * 1e3:.1f} ms")
        if tokens is not None or config["num_steps"] is not None:
            tokens_per_step = config["batch_size"] * config["context_length"]
            if config["num_steps"] is not None:
                num_steps = config["num_steps"]
            else:
                per_epoch = steps_per_epoch(config, tokens)
                num_steps = int(config["num_epochs"] * per_epoch)
            total_time = step_time * num_steps
            tokens_processed = num_steps * tokens_per_step
            if tokens is not None:
                table.add_row("dataset tokens", humanize.intword(tokens))
                table.add_row("batches / epoch", humanize.intword(steps_per_epoch(config, tokens)))
            table.add_row("total steps", humanize.intword(num_steps))
            table.add_row("est. total time", humanize.precisedelta(total_time, minimum_unit="seconds"))
            table.add_row("tokens processed", humanize.intword(tokens_processed, format="%.2f"))
            table.add_row("est. tokens / s", humanize.intword(tokens_processed / total_time, format="%.2f"))
        else:
            table.add_row("est. total time", "pass a token count, e.g. --dry-run 2000000")
    else:
        table.add_row("throughput", "unavailable (benchmark failed)")

    console.print(table)
    console.print(
        "[dim]Estimates use float32 FLOP/activation formulae and ignore data loading, "
        "eval and optimizer step overhead.[/dim]"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM from a TOML config.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"path to the config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument(
        "--dry-run",
        nargs="?",
        type=int,
        const=0,
        default=None,
        metavar="TOKENS",
        help=(
            "estimate memory, FLOPs and runtime without training; pass the dataset "
            "token count to derive batches/epoch and, when `num_steps` is not set in "
            "the config, total steps and total time, e.g. `--dry-run 2000000`"
        ),
    )
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging for this run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except (FileNotFoundError, NotADirectoryError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.no_wandb:
        config["use_wandb"] = False

    if args.dry_run is not None:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        try:
            print_dry_run(config, tokens=args.dry_run or None)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    file_handler = logging.FileHandler(output_dir / "logs.txt", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[stream_handler, file_handler],
        force=True,
    )

    logger.info("config: %s", config["config_path"])
    logger.info("dataset: %s", config["dataset"])
    logger.info("tokenizer: %s", config["tokenizer_path"])

    try:
        train(config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
