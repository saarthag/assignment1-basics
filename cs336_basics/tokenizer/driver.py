from cs336_basics.tokenizer.bpe import train_bpe_naive
from pathlib import Path

if __name__ == "__main__":
    sample_file = "/Users/saarthak/Projects/stanford-cs336/assignment1-basics/tests/fixtures/corpus.en"
    _, _ = train_bpe_naive(input_path=Path(sample_file), vocab_size=500, special_tokens=["<|endoftext|>"])
