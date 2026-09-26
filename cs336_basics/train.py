#! /usr/bin/env python
"""Train a BPE tokenizer from a TOML configuration file.

Usage:
    python -m cs336_basics.train [-c CONFIG] [-v]

The config file (``config.toml`` in the current directory by default) supports:

    corpus          = "data/tinystories.txt"   # required, path to training corpus
    vocab_size      = 10000                     # required, target vocab size
    special_tokens  = ["<|endoftext|>"]         # required, list of special tokens
    output_dir      = "training"                # optional, artifact directory

Relative paths are resolved against the directory containing the config file.
The output directory defaults to ``training/`` next to the config file. It will
contain ``vocab.pkl``, ``merges.pkl``, ``logs.txt`` and ``report.json``.
"""

import argparse
import json
import logging
import pickle
import sys
import time
import tomllib
from pathlib import Path

import humanize

from cs336_basics.tokenizer.bpe import train_bpe_fast

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.toml")
DEFAULT_OUTPUT_DIR = "training"


def save_tokenizer_artifacts(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Pickle the trained ``vocab`` and ``merges`` into ``output_dir``.

    Returns a mapping of artifact name to written path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"
    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)
    with merges_path.open("wb") as f:
        pickle.dump(merges, f)

    return {"vocab": str(vocab_path), "merges": str(merges_path)}


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

    for key in ("corpus", "vocab_size", "special_tokens"):
        if key not in raw:
            raise KeyError(f"missing required config key: {key}")

    if not isinstance(raw["corpus"], str):
        raise TypeError("`corpus` must be a string path")
    if not isinstance(raw["vocab_size"], int):
        raise TypeError("`vocab_size` must be an integer")
    if raw["vocab_size"] <= 0:
        raise ValueError("`vocab_size` must be positive")
    if not isinstance(raw["special_tokens"], list) or not all(isinstance(s, str) for s in raw["special_tokens"]):
        raise TypeError("`special_tokens` must be a list of strings")
    if "output_dir" in raw and not isinstance(raw["output_dir"], str):
        raise TypeError("`output_dir` must be a string path")

    base_dir = config_path.parent
    corpus = _resolve(base_dir, raw["corpus"])
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus file not found: {corpus}")

    return {
        "config_path": str(config_path),
        "corpus": corpus,
        "vocab_size": raw["vocab_size"],
        "special_tokens": raw["special_tokens"],
        "output_dir": _resolve(base_dir, raw.get("output_dir", DEFAULT_OUTPUT_DIR)),
    }


def train_tokenizer(config: dict) -> dict:
    """Train the tokenizer from a validated ``config`` and write its artifacts.

    Returns the contents of the generated ``report.json``.
    """
    corpus: Path = config["corpus"]
    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "logs.txt"

    logger.info("config: %s", config["config_path"])
    logger.info("corpus: %s", corpus)
    logger.info("output directory: %s", output_dir)
    logger.info(
        "target vocab_size=%d, special_tokens=%s",
        config["vocab_size"],
        config["special_tokens"],
    )

    corpus_bytes = corpus.stat().st_size
    logger.debug("corpus size: %s", humanize.naturalsize(corpus_bytes))

    stats: dict = {}
    start_time = time.perf_counter()
    vocab, merges = train_bpe_fast(
        input_path=corpus,
        vocab_size=config["vocab_size"],
        special_tokens=config["special_tokens"],
        stats=stats,
    )
    total_time = time.perf_counter() - start_time

    logger.info("writing tokenizer artifacts to %s", output_dir)
    artifact_paths = save_tokenizer_artifacts(vocab, merges, output_dir)
    for name, path in artifact_paths.items():
        logger.debug("wrote %s: %s (%s)", name, path, humanize.naturalsize(Path(path).stat().st_size))

    report_path = output_dir / "report.json"
    report = {
        "config": {
            "config_path": config["config_path"],
            "corpus": str(corpus),
            "vocab_size_requested": config["vocab_size"],
            "special_tokens": config["special_tokens"],
            "output_dir": str(output_dir),
        },
        "corpus": {
            "path": str(corpus),
            "bytes": corpus_bytes,
        },
        "result": {
            "vocab_size": len(vocab),
            "num_merges": len(merges),
            "num_special_tokens": len(config["special_tokens"]),
            "num_unique_pretokens": stats.get("num_unique_pretokens"),
        },
        "timing_s": {
            "pre_tokenize": stats.get("pre_tokenize_time_s"),
            "merges": stats.get("merge_time_s"),
            "per_merge": stats.get("merge_per_iter_s"),
            "total": total_time,
        },
        "artifacts": {
            **artifact_paths,
            "logs": str(logs_path),
            "report": str(report_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "training complete in %.3fs: vocab_size=%d, num_merges=%d",
        total_time,
        len(vocab),
        len(merges),
    )
    logger.info("wrote report to %s", report_path)
    logger.debug("artifacts: %s", artifact_paths)

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from a TOML config.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"path to the config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except (FileNotFoundError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logs go to stdout and alongside the training artifacts. The file always
    # captures debug logs; the stream verbosity is controlled by -v.
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

    train_tokenizer(config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
