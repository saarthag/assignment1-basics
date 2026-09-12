#! /usr/bin/env python
import argparse
import json
import logging
import os
import pickle
import time
from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer.bpe import Tokenizer

logger = logging.getLogger(__name__)


def run_tokenizer_encode(
    input_path: str | os.PathLike,
    vocab_path: str | os.PathLike,
    merge_pairs_path: str | os.PathLike,
    output_path: str | os.PathLike | None = None,
    special_tokens: list[str] | None = None,
    max_workers: int | None = None,
    worker_batch_size: int = 100_000,
    dtype: str = "<u2",
) -> dict:
    """Encode a text file into BPE token ids using a pretrained tokenizer.

    Args:
        input_path: text file to encode
        vocab_path: pickle containing the vocab (dict[int, bytes])
        merge_pairs_path: pickle containing the merge pairs
        output_path: where to write the encoded ids; defaults to
            `<input stem>_encoded.bin` in the current directory
        special_tokens: special tokens passed to the tokenizer
        max_workers: workers for the parallel encoder (default: CPU count)
        dtype: numpy dtype for the stored token ids
        worker_batch_size: how many raw chars to accumulate before handing a
            batch to an encode() worker

    Returns:
        dict with the encoding data (read-only np.memmap of token ids) and
        stats about the run.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else Path.cwd() / f"{input_path.stem}_encoded.bin"
    special_tokens = special_tokens if special_tokens is not None else ["<|endoftext|>"]

    logger.debug(
        "run_tokenizer_encode(input_path=%s, vocab_path=%s, merge_pairs_path=%s, "
        "output_path=%s, special_tokens=%s, max_workers=%s, worker_batch_size=%d, dtype=%s)",
        input_path,
        vocab_path,
        merge_pairs_path,
        output_path,
        special_tokens,
        max_workers,
        worker_batch_size,
        dtype,
    )

    max_workers = max_workers or os.cpu_count() or 4
    logger.debug("resolved max_workers=%d", max_workers)

    logger.info("loading tokenizer: vocab=%s, merge_pairs=%s", vocab_path, merge_pairs_path)
    with open(vocab_path, "rb") as f_vocab, open(merge_pairs_path, "rb") as f_mp:
        vocab = pickle.load(f_vocab)
        merge_pairs = pickle.load(f_mp)
    tokenizer = Tokenizer(vocab, merge_pairs, special_tokens=special_tokens)
    logger.info("tokenizer ready: vocab_size=%d, special_tokens=%s", len(vocab), special_tokens)

    storage_dtype = np.dtype(dtype)
    n_input_bytes = input_path.stat().st_size

    # worst case is one token per byte, so allocate that big and truncate at the end
    tokens_np = np.memmap(filename=output_path, mode="w+", dtype=storage_dtype, shape=n_input_bytes)
    logger.debug(
        "allocated memmap %s: dtype=%s, %d entries (%.1f MiB)",
        output_path,
        storage_dtype.str,
        n_input_bytes,
        n_input_bytes * storage_dtype.itemsize / (1 << 20),
    )

    meta_path = output_path.with_name(f"{output_path.stem}_meta.json")
    with meta_path.open("w") as f_meta:
        json.dump({"dtype": storage_dtype.str, "max_workers": max_workers}, f_meta)
    logger.debug("wrote metadata to %s", meta_path)

    logger.info(
        "encoding %s (%d bytes) -> %s (max_workers=%d, worker_batch_size=%d)",
        input_path,
        n_input_bytes,
        output_path,
        max_workers,
        worker_batch_size,
    )

    start_time = time.perf_counter()
    w_off = 0  # write offset
    # how many encoded tokens to buffer in memory before flushing to disk
    buf_size = 1_000_000

    with input_path.open() as f:
        token_stream = tokenizer.encode_iterable(f, batch_size=worker_batch_size, max_workers=max_workers)

        while buf := list(islice(token_stream, buf_size)):
            n_tokens = len(buf)
            tokens_np[w_off : w_off + n_tokens] = buf
            w_off += n_tokens
            logger.debug("flushed %d tokens (total %d so far)", n_tokens, w_off)

    tot_tokens = w_off
    tot_time = time.perf_counter() - start_time

    # flush encodings to disk and truncate the output file to the real token count
    tokens_np.flush()
    with output_path.open("r+b") as f_out:
        f_out.truncate(tot_tokens * storage_dtype.itemsize)
    logger.debug(
        "truncated %s to %d entries (%.1f MiB)",
        output_path,
        tot_tokens,
        tot_tokens * storage_dtype.itemsize / (1 << 20),
    )

    compression_ratio = n_input_bytes / tot_tokens if tot_tokens else float("inf")
    throughput_bytes_per_s = n_input_bytes / tot_time if tot_time else 0.0

    logger.info(
        "done: %s -> %s (%d tokens, %.3fs, %s bytes/s, compression ratio %.3f)",
        input_path,
        output_path,
        tot_tokens,
        tot_time,
        f"{throughput_bytes_per_s:,.0f}",
        compression_ratio,
    )
    logger.debug("encoding data written to %s, metadata to %s", output_path, meta_path)

    return {
        "tokens": np.memmap(filename=output_path, mode="r", dtype=storage_dtype),
        "output_path": str(output_path),
        "meta_path": str(meta_path),
        "dtype": storage_dtype.str,
        "n_tokens": tot_tokens,
        "n_input_bytes": n_input_bytes,
        "compression_ratio": compression_ratio,
        "encoding_time_s": tot_time,
        "throughput_bytes_per_s": throughput_bytes_per_s,
        "max_workers": max_workers,
        "worker_batch_size": worker_batch_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a text file into BPE token ids.")
    parser.add_argument("input", help="text file to encode")
    parser.add_argument("vocab", help="pickle with the vocab (dict[int, bytes])")
    parser.add_argument("merge_pairs", help="pickle with the merge pairs")
    parser.add_argument(
        "--output",
        default=None,
        help="output .bin path (default: `<input stem>_encoded.bin` in the current dir)",
    )
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--max-workers", type=int, default=None, help="default: CPU count")
    parser.add_argument("--worker-batch-size", type=int, default=100_000)
    parser.add_argument("--dtype", default="<u2")
    parser.add_argument("-v", "--verbose", action="store_true", help="print debug logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_tokenizer_encode(
        args.input,
        args.vocab,
        args.merge_pairs,
        output_path=args.output,
        special_tokens=args.special_tokens,
        max_workers=args.max_workers,
        dtype=args.dtype,
        worker_batch_size=args.worker_batch_size,
    )
