#! /usr/bin/env python
"""Generate text from a trained Transformer LM checkpoint.

Usage:
    python -m cs336_basics.infer <model_dir> "<prompt>" [--checkpoint final.pt]

``model_dir`` is the experiment directory (e.g. ``experiments/lm-v1``) whose
``training/`` subdirectory holds ``config.json`` and ``checkpoints/``. The
checkpoint file name is relative to that ``checkpoints/`` directory and defaults
to ``final.pt``. Decoding is greedy (no temperature / nucleus sampling) and stops
at ``<|endoftext|>`` or after ``--max-new-tokens`` tokens.
"""

import argparse
import json
import pickle
from codecs import getincrementaldecoder
from collections.abc import Iterator
from pathlib import Path

import torch

from cs336_basics.tokenizer.bpe import BPETokenizer
from cs336_basics.transformer import TransformerLM


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(tokenizer_path: Path) -> tuple[BPETokenizer, list[str]]:
    training_dir = tokenizer_path / "training"
    with (training_dir / "vocab.pkl").open("rb") as f:
        vocab = pickle.load(f)
    with (training_dir / "merges.pkl").open("rb") as f:
        merges = pickle.load(f)
    with (training_dir / "meta.json").open(encoding="utf-8") as f:
        special_tokens = json.load(f)["config"]["special_tokens"]
    return BPETokenizer(vocab, merges, special_tokens=special_tokens), special_tokens


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    device: str,
) -> Iterator[int]:
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(x[:, -context_length:])[:, -1, :]
        next_id = int(logits.argmax(dim=-1))
        yield next_id
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy text generation from a trained Transformer LM.")
    parser.add_argument("model_dir", type=Path, help="experiment directory, e.g. experiments/lm-v1")
    parser.add_argument("prompt", help="text to condition on")
    parser.add_argument(
        "--checkpoint", default="final.pt", help="checkpoint file name under training/checkpoints (default: final.pt)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=200, help="maximum tokens to generate (default: 200)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps (default: auto)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = pick_device(args.device)

    training_dir = args.model_dir / "training"
    checkpoint_path = training_dir / "checkpoints" / args.checkpoint

    with (training_dir / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)

    dtype = getattr(torch, config.get("dtype", "float32"))
    tokenizer, special_tokens = load_tokenizer(Path(config["tokenizer_path"]))

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["theta"],
        device=device,
        dtype=dtype,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    stop_ids = {tokenizer.vocab_index[st.encode("utf-8")] for st in special_tokens}

    print(f"checkpoint: {checkpoint_path} (step {state.get('iter', '?')})")
    print(f"device: {device}")
    print(args.prompt, end="", flush=True)

    token_iter = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=config["context_length"],
        device=device,
    )

    # Byte-level BPE tokens are raw bytes, so a character can span tokens. The
    # incremental decoder buffers partial UTF-8 sequences and yields only
    # complete characters as they become available.
    utf8_decoder = getincrementaldecoder("utf-8")(errors="replace")
    for next_id in token_iter:
        if next_id in stop_ids:
            # print("Received <|endoftext|>")
            break
        text = utf8_decoder.decode(tokenizer.vocab[next_id])
        if text:
            print(text, end="", flush=True)
    print(utf8_decoder.decode(b"", final=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
