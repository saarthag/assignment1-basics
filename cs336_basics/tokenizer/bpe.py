import os
import regex as re2
from pathlib import Path
from collections import Counter, defaultdict
from itertools import pairwise, compress, repeat
from rich.pretty import pprint
from concurrent.futures import ProcessPoolExecutor
import time
import heapq
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(
    input_path: os.PathLike, start: int, end: int, special_tokens: list[str] = []
) -> Counter[tuple[bytes, ...]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    chunk_counter = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        content = f.read(end - start).decode("utf-8", errors="ignore")
        # split across special tokens
        sub_chunks = [content]
        if len(special_tokens) > 0:
            sub_chunks = re2.split("|".join(re2.escape(s) for s in special_tokens), content)

        for c in sub_chunks:
            chunk_counter.update(tuple(bytes([b]) for b in m[0].encode("utf-8")) for m in re2.finditer(PAT, c))

        return chunk_counter


def pre_tokenize(input_path: os.PathLike, special_tokens: list[str] = []) -> Counter[tuple[bytes]]:
    with open(input_path, "rb") as f:
        num_processes = os.cpu_count() or 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        pre_tokens = Counter()

        with ProcessPoolExecutor() as executor:
            for chunk_counter in executor.map(
                pre_tokenize_chunk, repeat(input_path), boundaries[:-1], boundaries[1:], repeat(special_tokens)
            ):
                pre_tokens.update(chunk_counter)

        return pre_tokens


class BPWrapper:
    def __init__(self, bp: tuple[bytes, bytes]):
        self.bp = bp

    def __lt__(self, other: "BPWrapper"):
        # return self.bp[0] + self.bp[1] > other.bp[0] + other.bp[1]
        return self.bp > other.bp

    def __repr__(self):
        return repr(self.bp)


def train_bpe_fast(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: i.to_bytes(1) for i in range(256)} | {
        256 + i: st.encode("utf-8") for i, st in enumerate(special_tokens)
    }
    vocab_size_cur = len(vocab)
    merge_pairs: list[tuple[bytes, bytes]] = []

    start_time = time.perf_counter()
    pre_tokens_raw = pre_tokenize(Path(input_path), special_tokens=special_tokens)
    print("pre_tokenize", time.perf_counter() - start_time)

    # pre-allocate since size is known
    pre_tokens = [None] * len(pre_tokens_raw)
    # index of all positions (pre_tokens index) of a byte pair
    bp_pos_index: dict[tuple[bytes, bytes], list[int]] = defaultdict(list)
    # index of all counts of a byte pair
    bp_cnt_index: dict[tuple[bytes, bytes], int] = Counter()

    # process and fill raw pre-tokens into pre_tokens
    i = 0
    for tok, cnt in pre_tokens_raw.items():
        bp_list = [None] * (len(tok) - 1)

        for j, bp in enumerate(pairwise(tok)):
            bp_list[j] = bp
            bp_pos_index[bp].append(i)
            bp_cnt_index[bp] += cnt

        pre_tokens[i] = (bp_list, cnt)
        i += 1

    # max-heap to store byte pair counts
    # count is inverted and the byte pair is stored in a wrapper class to emulate a max-heap
    # python support only min-heap for versions <3.14
    bp_heap = [(-v, BPWrapper(k)) for k, v in bp_cnt_index.items()]
    heapq.heapify(bp_heap)

    start_time = time.perf_counter()
    while vocab_size_cur < vocab_size:
        top = heapq.heappop(bp_heap)
        while -top[0] != (v := bp_cnt_index[top[1].bp]):
            if v > 0:
                heapq.heappush(bp_heap, (-v, top[1]))
            top = heapq.heappop(bp_heap)

        best_bp = top[1].bp
        # update vocabulary
        merge_pairs.append(best_bp)
        new_token = best_bp[0] + best_bp[1]
        vocab[vocab_size_cur] = new_token
        vocab_size_cur += 1

        heap_candidates = []
        positions = bp_pos_index[best_bp]

        for i in range(len(positions)):
            bp_list, pretok_cnt = pre_tokens[positions[i]]
            select_bp = [1] * len(bp_list)
            for j in range(len(bp_list)):
                if bp_list[j] == best_bp:
                    select_bp[j] = 0

                    if j > 0:
                        left_bp = bp_list[j - 1]
                        bp_cnt_index[left_bp] -= pretok_cnt

                        left_bp_upd = (left_bp[0], new_token)
                        bp_cnt_index[left_bp_upd] += pretok_cnt
                        bp_pos_index[left_bp_upd].append(positions[i])

                        bp_list[j - 1] = left_bp_upd
                        heap_candidates.append(left_bp_upd)

                    if j < len(bp_list) - 1:
                        right_bp = bp_list[j + 1]
                        bp_cnt_index[right_bp] -= pretok_cnt

                        right_bp_upd = (new_token, right_bp[1])
                        bp_cnt_index[right_bp_upd] += pretok_cnt
                        bp_pos_index[right_bp_upd].append(positions[i])

                        bp_list[j + 1] = right_bp_upd
                        heap_candidates.append(right_bp_upd)

            pre_tokens[positions[i]] = (list(compress(bp_list, select_bp)), pretok_cnt)

        del bp_pos_index[best_bp]
        bp_cnt_index[best_bp] = 0

        for h in heap_candidates:
            if (v := bp_cnt_index[h]) > 0:
                heapq.heappush(bp_heap, (-v, BPWrapper(h)))

    # profiling
    merge_tottime = time.perf_counter() - start_time
    merge_niters = vocab_size - 256 - len(special_tokens)
    print("merge_tottime", merge_tottime)
    print("merge_niters", merge_niters)
    print("merge_periter", merge_tottime / merge_niters)

    return vocab, merge_pairs


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

        for k, cnt in pre_tokens.items():
            for p in pairwise(k):
                bp_cnt[p] += cnt

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
