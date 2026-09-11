#! /usr/bin/env python
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer.bpe import Tokenizer, train_bpe_fast


def train_bpe_tinystories():
    sample_data_dir = Path("/Users/saarthak/Projects/stanford-cs336/assignment1-basics/data")
    sample_file_name = Path("TinyStoriesV2-GPT4-train.txt")
    # sample_file_name = Path("tinystories_100mb.txt")
    vocab, merge_pairs = train_bpe_fast(
        input_path=sample_data_dir / sample_file_name, vocab_size=10000, special_tokens=["<|endoftext|>"]
    )

    with (
        open(f"vocab_{sample_file_name.stem}.pickle", "wb") as f_vocab,
        open(f"mergepairs_{sample_file_name.stem}.pickle", "wb") as f_mp,
    ):
        pickle.dump(vocab, f_vocab)
        pickle.dump(merge_pairs, f_mp)

    # vocab_decoded = {k: v.decode("latin-1") for k, v in vocab.items()}
    # merge_pairs_decoded = [(p[0].decode("latin-1"), p[1].decode("latin-1")) for p in merge_pairs]
    #
    # with open(f"bpe_{sample_file_name.stem}.json", "w") as f:
    #     d = {"vocab": vocab_decoded, "merge_pairs": merge_pairs_decoded}
    #     json.dump(d, f, indent=2, sort_keys=True, ensure_ascii=False)


def run_tokenizer_encode():
    train_data = "TinyStoriesV2-GPT4-train"
    with (
        open(f"vocab_{train_data}.pickle", "rb") as f_vocab,
        open(f"mergepairs_{train_data}.pickle", "rb") as f_mp,
    ):
        vocab = pickle.load(f_vocab)
        merge_pairs = pickle.load(f_mp)

    sample_data_dir = Path("/Users/saarthak/Projects/stanford-cs336/assignment1-basics/data")
    sample_file_name = Path(sys.argv[1])

    tokenizer = Tokenizer(vocab, merge_pairs, special_tokens=["<|endoftext|>"])

    with open(sample_data_dir / sample_file_name) as f:
        # print("encode")
        # start_time = time.perf_counter()
        # n_tokens = len(tokenizer.encode(f.read()))
        # print(f"{sample_file_name}: {n_tokens:,} tokens, {time.perf_counter() - start_time}s")
        #
        # for mw in (2**i for i in range(6)):
        f.seek(0, os.SEEK_SET)
        mw = 4

        start_time = time.perf_counter()
        n_tokens = sum(1 for _ in tokenizer.encode_iterable(f, max_workers=mw))
        tot_time = time.perf_counter() - start_time
        tot_bytes = f.tell()

        print(f"{sample_file_name}: {n_tokens:,} tokens, {tot_time:.3f}s, max_workers={mw}")
        print(f"compression ratio={tot_bytes / n_tokens:.3f}")
        print(f"throughput={tot_bytes / tot_time:,.3f} bytes/s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("filename required!")
        sys.exit(1)

    run_tokenizer_encode()
    # train_bpe_tinystories()
