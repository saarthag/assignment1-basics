import os
import typing
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange
from torch import Tensor


def stream_batch(
    dataset: npt.NDArray,
    dataset_tokens: int,
    batch_size: int,
    context_length: int,
    device: str,
    seed: int | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    batch_tokens = batch_size * context_length
    # each shard is ~10% of the dataset, aligned to whole batches
    shard_size = max((dataset_tokens // 10 // batch_tokens) * batch_tokens, batch_tokens)
    num_shards = -(-dataset_tokens // shard_size)

    shard_idx = np.arange(num_shards)

    epoch = 0
    while True:
        # A fixed `seed` makes every pass deterministic (one independent order per
        # pass), so consuming batches reproduces the exact same stream and a
        # skipped prefix lines up with a previous run.
        rng = np.random.default_rng(None if seed is None else seed + epoch)
        epoch += 1
        rng.shuffle(shard_idx)

        for i in shard_idx:
            shard_start = i * shard_size
            shard_end = min(shard_start + shard_size, dataset_tokens - 1)
            for j in range(shard_start, shard_end, batch_tokens):
                if j + batch_tokens >= dataset_tokens:
                    continue
                batch = dataset[j : j + batch_tokens + 1]
                # torch.tensor copies (and casts) so the read-only memmap slice is
                # never shared as a writable-tensor backing store.
                yield (
                    rearrange(
                        torch.tensor(batch[:batch_tokens], dtype=torch.long, device=device),
                        "(b t) -> b t",
                        b=batch_size,
                    ),
                    rearrange(
                        torch.tensor(batch[1:], dtype=torch.long, device=device),
                        "(b t) -> b t",
                        b=batch_size,
                    ),
                )


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    # Number of valid start indices: a window of length context_length + 1
    # (input + its shifted label) must fit inside the dataset.
    num_valid_starts = len(dataset) - context_length

    # Sample batch_size random start indices uniformly from [0, num_valid_starts).
    starts = np.random.randint(0, num_valid_starts, (batch_size,))

    # Build input/label windows. Use numpy advanced indexing for efficiency.
    offsets = np.arange(context_length)
    x = dataset[starts[:, None] + offsets]
    y = dataset[starts[:, None] + offsets + 1]

    return (
        torch.from_numpy(x).long().to(device),
        torch.from_numpy(y).long().to(device),
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    state_obj = {"model": model.state_dict(), "optim": optimizer.state_dict(), "iter": iteration}
    torch.save(state_obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    state_obj = torch.load(src)
    model.load_state_dict(state_obj["model"])
    optimizer.load_state_dict(state_obj["optim"])

    return state_obj["iter"]
