#! /usr/bin/env python

from cs336_basics.tokenizer.bpe import train_bpe_fast
from pathlib import Path


if __name__ == "__main__":
    sample_file = "/Users/saarthak/Projects/stanford-cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab, merge_pairs = train_bpe_fast(
        input_path=Path(sample_file), vocab_size=10000, special_tokens=["<|endoftext|>"]
    )

    # sample_file_basedir = "/Users/saarthak/Projects/stanford-cs336/assignment1-basics/data"
    # for i in range(100, 400, 100):
    #     file_name = f"tinystories_{i}mb.txt"
    #     print(file_name)
    #     train_bpe_fast(
    #         input_path=Path(sample_file_basedir) / file_name, vocab_size=1000, special_tokens=["<|endoftext|>"]
    #     )
