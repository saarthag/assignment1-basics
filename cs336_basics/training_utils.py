import os
import typing

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


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
