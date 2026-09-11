import heapq
import os
import tempfile
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, compress, cycle, islice, pairwise, repeat, zip_longest
from pathlib import Path
from typing import BinaryIO

import regex as re2
from rich.pretty import pprint

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re2_PAT = re2.compile(PAT)


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
    input_path: os.PathLike, start: int, end: int, special_tokens: list[str] | None = None
) -> Counter[tuple[int, ...]]:
    special_tokens = special_tokens or []

    with open(input_path, "rb") as f:
        f.seek(start)
        content = f.read(end - start).decode("utf-8", errors="ignore")
        # split across special tokens
        sub_chunks = [content]
        if len(special_tokens) > 0:
            sub_chunks = re2.split("|".join(re2.escape(s) for s in special_tokens), content)

        pre_tokens = Counter(tuple(m[0].encode("utf-8")) for c in sub_chunks for m in re2_PAT.finditer(c))

        return pre_tokens


def pre_tokenize(input_path: os.PathLike, special_tokens: list[str] | None = None) -> Counter[tuple[int, ...]]:
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


def train_bpe_fast(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: i.to_bytes(1) for i in range(256)} | {
        256 + i: st.encode("utf-8") for i, st in enumerate(special_tokens)
    }
    vocab_size_cur = len(vocab)
    merge_pairs: list[tuple[int, int]] = []

    start_time = time.perf_counter()
    pre_tokens_tmp = Counter(pre_tokenize(Path(input_path), special_tokens=special_tokens))
    print("pre_tokenize", time.perf_counter() - start_time)

    # pre-allocate since size is known
    pre_tokens = [None] * len(pre_tokens_tmp)
    # index of all positions (pre_tokens index) of a byte pair
    bp_pos_index: dict[tuple[int, int], list[int]] = defaultdict(list)
    # index of all counts of a byte pair
    bp_cnt_index: dict[tuple[int, int], int] = Counter()

    # process and fill raw pre-tokens into pre_tokens
    i = 0
    for tok, cnt in pre_tokens_tmp.items():
        bp_list = [None] * (len(tok) - 1)

        for j, bp in enumerate(pairwise(tok)):
            bp_list[j] = bp
            bp_pos_index[bp].append(i)
            bp_cnt_index[bp] += cnt

        pre_tokens[i] = (bp_list, cnt)
        i += 1  # noqa: SIM113

    class BPWrapper:
        def __init__(self, bp: tuple[int, int]):
            self.bp = bp

        def __lt__(self, other: "BPWrapper"):
            # return self.bp[0] + self.bp[1] > other.bp[0] + other.bp[1]
            return tuple(vocab[b] for b in self.bp) > tuple(vocab[b] for b in other.bp)

        def __repr__(self):
            return repr(self.bp)

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
        vocab[vocab_size_cur] = vocab[best_bp[0]] + vocab[best_bp[1]]
        new_token = vocab_size_cur
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

    return vocab, [(vocab[mp[0]], vocab[mp[1]]) for mp in merge_pairs]


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_index = {v: i for i, v in vocab.items()}

        self.merges = merges
        self.merges_rank_index = {mp: i for i, mp in enumerate(merges)}

        self.special_tokens = sorted(special_tokens or [], key=lambda s: -len(s))
        self.special_tokens_prefix = sorted(
            (st[: i + 1] for st in self.special_tokens for i in range(len(st) - 1)), key=lambda s: -len(s)
        )

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        cache: dict[tuple[bytes, ...], list[int]] = {}

        re2_split_special = re2.compile("(" + "|".join(re2.escape(s) for s in self.special_tokens) + ")")

        def text_iter():
            chunks = [text]
            if len(self.special_tokens) > 0:
                chunks = re2_split_special.split(text)

            for c in chunks:
                if c in self.special_tokens:
                    yield (c.encode("utf-8"),)
                else:
                    yield from (tuple(bytes([b]) for b in m[0].encode("utf-8")) for m in re2_PAT.finditer(c))

        MAX_RANK = len(self.merges)

        for tok in text_iter():
            # print("tok", tok, sep="=")
            if len(tok) == 1:
                encoded.append(self.vocab_index[tok[0]])
                continue

            if tok in cache:
                encoded.extend(cache[tok])
                continue

            bp_list = [bp for bp in pairwise(tok)]
            n_bp = len(bp_list)

            for _ in range(n_bp):
                # print("bp_list", bp_list, sep="=")
                best_bp_tup = min((self.merges_rank_index.get(bp, MAX_RANK), bp) for bp in bp_list)

                if best_bp_tup[0] == MAX_RANK:
                    break

                best_bp = best_bp_tup[1]
                new_token = best_bp[0] + best_bp[1]
                # if only a single byte pair is remaining
                if len(bp_list) == 1:
                    bp_list[0] = (new_token,)
                    break

                keep = [1] * len(bp_list)

                for i, bp in enumerate(bp_list):
                    if bp == best_bp:
                        keep[i] = 0

                        if i > 0:
                            left_bp_upd = (bp_list[i - 1][0], new_token)
                            bp_list[i - 1] = left_bp_upd

                        if i < len(bp_list) - 1:
                            right_bp_upd = (new_token, bp_list[i + 1][1])
                            bp_list[i + 1] = right_bp_upd

                bp_list_upd = list(compress(bp_list, keep))
                bp_list = bp_list_upd

            tok_encoded = [self.vocab_index[bp[0]] for bp in bp_list]
            # append the last element of the bp list
            if len(bp_list[-1]) > 1:
                tok_encoded.append(self.vocab_index[bp_list[-1][1]])

            cache[tok] = tok_encoded
            encoded.extend(tok_encoded)

        # print("encoded",encoded,sep="=")
        return encoded

    def encode_iterable(
        self, iterable: Iterable[str], batch_size: int = 500, max_workers: int | None = None
    ) -> Iterator[int]:
        def batched_token_aware():
            pt_iter = (m[0] for elem in iterable for m in re2_PAT.finditer(elem))
            pt_pending = deque()

            while batch := list(islice(pt_iter, batch_size)):
                pt_pending.extend(batch)
                batch_str = "".join(pt_pending)
                pt_pending.clear()

                n_retain = None

                for pre in self.special_tokens_prefix:
                    if batch_str.endswith(pre):
                        n_retain = len(pre)
                        break

                if n_retain is not None:
                    yield batch_str[:-n_retain]
                    pt_pending.extend(batch_str[-n_retain:])
                else:
                    yield batch_str

        with ProcessPoolExecutor() as executor:
            yield from chain.from_iterable(executor.map(self.encode, batched_token_aware()))

    def decode(self, ids: list[int]) -> str:
        encoded_str = b"".join(self.vocab[i] for i in ids)
        return encoded_str.decode("utf-8", errors="replace")
