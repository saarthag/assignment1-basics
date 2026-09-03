#! /usr/bin/env python

from cs336_basics.tokenizer.bpe import train_bpe_fast
from pathlib import Path


if __name__ == "__main__":
    sample_file = "/Users/saarthak/Projects/stanford-cs336/assignment1-basics/tests/fixtures/corpus.en"
    _, _ = train_bpe_fast(input_path=Path(sample_file), vocab_size=500, special_tokens=["<|endoftext|>"])
    # sample_file = "/Users/saarthak/Projects/stanford-cs336/assignment1-basics/cs336_basics/tokenizer/sample_doc.txt"
    # v, mp = train_bpe_fast(sample_file, 257, [])
    # print("vocab=", v)
    # print("merge pairs=", mp)
