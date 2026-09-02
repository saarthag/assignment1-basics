# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import regex as re2
from pathlib import Path
from collections import Counter
from itertools import pairwise, chain
from rich.pretty import pprint
import time

# %%
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re_PAT = re2.compile(PAT)


# %%
def pre_tokenize(input_path: os.PathLike, special_tokens: list[str] = []) -> Counter[tuple[bytes]]:
    with open(input_path, "r", encoding="utf-8") as f:
        pre_tokens = Counter()

        chunks = re2.split("|".join(re2.escape(s) for s in special_tokens), f.read())
        for c in chunks:
            pre_tokens.update(tuple(bytes([b]) for b in m[0].encode("utf-8")) for m in re_PAT.finditer(c))

        return pre_tokens


# %%
def train_bpe_naive(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: i.to_bytes(1) for i in range(256)} | {
        256 + i: st.encode("utf-8") for i, st in enumerate(special_tokens)
    }
    vocab_size_cur = len(vocab)
    merge_pairs: list[tuple[bytes, bytes]] = []

    pre_tokens = pre_tokenize(Path(input_path), special_tokens=special_tokens)

    while vocab_size_cur < vocab_size:
        bp_cnt = Counter()

        start_time = time.perf_counter()
        for k, cnt in pre_tokens.items():
            for p in pairwise(k):
                bp_cnt[p] += cnt
        print(time.perf_counter() - start_time)

        best_bp = max((cnt, bp) for bp, cnt in bp_cnt.items())[1]

        merge_pairs.append(best_bp)
        vocab[vocab_size_cur] = best_bp[0] + best_bp[1]
        vocab_size_cur += 1

        pre_tokens_upd = {}
        for k in pre_tokens:
            k_upd = []
            num_tokens = len(k)

            i = 0
            while i < num_tokens:
                if i < num_tokens - 1 and k[i] == best_bp[0] and k[i + 1] == best_bp[1]:
                    k_upd.append(best_bp[0] + best_bp[1])
                    i += 1
                else:
                    k_upd.append(k[i])
                i += 1

            pre_tokens_upd[tuple(k_upd)] = pre_tokens[k]

        pre_tokens = pre_tokens_upd

    return vocab, merge_pairs
