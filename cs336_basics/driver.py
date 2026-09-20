#! /usr/bin/env python
import sys

import humanize
import torch
from quantiphy import Quantity
from rich.console import Console
from rich.table import Table

from cs336_basics import transformer


def flop_counter(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
) -> dict[str, int]:
    mha = 8 * context_length * (d_model**2) + 4 * (context_length**2) * d_model
    ffn = 6 * context_length * d_model * d_ff
    lm_head = 2 * context_length * d_model * vocab_size

    tot_mha = mha * num_layers
    tot_ffn = ffn * num_layers

    return {
        "mha": tot_mha,
        "ffn": tot_ffn,
        "lm_head": lm_head,
        "total": tot_mha + tot_ffn + lm_head,
    }


def param_counter(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
) -> dict[str, int]:
    mha = 4 * (d_model**2)
    ffn = 3 * d_ff * d_model
    rms_per_layer = 2 * d_model

    tot_mha = num_layers * mha
    tot_ffn = num_layers * ffn
    tot_rms = num_layers * rms_per_layer + d_model
    embeddings = vocab_size * d_model
    lm_head = d_model * vocab_size

    return {
        "mha": tot_mha,
        "ffn": tot_ffn,
        "rmsnorm": tot_rms,
        "embeddings": embeddings,
        "lm_head": lm_head,
        "total": tot_mha + tot_ffn + tot_rms + embeddings + lm_head,
    }


def d_ff_from_d_model(d_model: int) -> int:
    """FFN hidden size: 8/3 * d_model, rounded up to the nearest multiple of 64."""
    return -(-8 * d_model // (3 * 64)) * 64


def build_table(name: str, params: dict[str, int], flops: dict[str, int]) -> Table:
    total_flops = flops["total"]
    table = Table(title=name, title_style="bold", header_style="bold")
    table.add_column("component")
    table.add_column("params", justify="right")
    table.add_column("flops", justify="right")
    table.add_column("flops %", justify="right")

    rows = [
        ("attention", params["mha"], flops["mha"]),
        ("ffn", params["ffn"], flops["ffn"]),
        ("rmsnorm", params["rmsnorm"], 0),
        ("embeddings", params["embeddings"], 0),
        ("lm_head", params["lm_head"], flops["lm_head"]),
    ]
    for label, p, f in rows:
        table.add_row(
            label,
            humanize.intword(p, format="%.2f"),
            str(Quantity(f, "FLOPs")) if f else "—",
            f"{100 * f / total_flops:.1f}%" if f else "—",
        )

    table.add_row(
        "total",
        humanize.intword(params["total"], format="%.2f"),
        str(Quantity(total_flops, "FLOPs")),
        "100.0%",
        style="bold",
    )
    return table


if __name__ == "__main__":
    models = {
        # "gpt2-small": {
        #     "vocab_size": 50257,
        #     "context_length": 1024,
        #     "num_layers": 12,
        #     "d_model": 768,
        #     "num_heads": 12,
        #     "d_ff": d_ff_from_d_model(768),
        # },
        # "gpt2-medium": {
        #     "vocab_size": 50257,
        #     "context_length": 1024,
        #     "num_layers": 24,
        #     "d_model": 1024,
        #     "num_heads": 16,
        #     "d_ff": d_ff_from_d_model(1024),
        # },
        # "gpt2-large": {
        #     "vocab_size": 50257,
        #     "context_length": 1024,
        #     "num_layers": 36,
        #     "d_model": 1280,
        #     "num_heads": 20,
        #     "d_ff": d_ff_from_d_model(1280),
        # },
        "gpt2-xl": {
            "vocab_size": 50257,
            "context_length": 1024,
            "num_layers": 48,
            "d_model": 1600,
            "num_heads": 25,
            "d_ff": d_ff_from_d_model(1600),
        },
        "gpt2-xl_ctx16384": {
            "vocab_size": 50257,
            "context_length": 16384,
            "num_layers": 48,
            "d_model": 1600,
            "num_heads": 25,
            "d_ff": d_ff_from_d_model(1600),
        },
    }

    console = Console()
    for name, config in models.items():
        console.print(build_table(name, param_counter(**config), flop_counter(**config)))
        console.print()
