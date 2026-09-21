#! /usr/bin/env python
import sys

import humanize
import plotly.graph_objects as go
import torch
from quantiphy import Quantity
from rich.console import Console
from rich.table import Table

from cs336_basics import transformer

DTYPE = "float32"
DTYPE_BYTES = 4


def flop_counter(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    batch_size: int = 1,
    **kwargs,
) -> dict[str, int]:
    mha = batch_size * (8 * context_length * (d_model**2) + 4 * (context_length**2) * d_model)
    ffn = batch_size * 6 * context_length * d_model * d_ff
    lm_head = batch_size * 2 * context_length * d_model * vocab_size

    tot_mha = mha * num_layers
    tot_ffn = ffn * num_layers

    return {
        "mha": tot_mha,
        "ffn": tot_ffn,
        "lm_head": lm_head,
        "total": tot_mha + tot_ffn + lm_head,
    }


def param_counter(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, **kwargs
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


def activation_counter(
    batch_size: int, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, **kwargs
) -> dict[str, int]:
    bcd = batch_size * context_length * d_model

    tot_mha = (5 * bcd + batch_size * num_heads * context_length**2) * num_layers
    tot_ffn = 35 * bcd * num_layers // 3
    tot_rms = 2 * bcd * num_layers + bcd
    output_embedding = bcd
    cross_entropy = batch_size * context_length * vocab_size

    return {
        "mha": tot_mha,
        "ffn": tot_ffn,
        "rmsnorm": tot_rms,
        "output_embedding": output_embedding,
        "cross_entropy": cross_entropy,
        "total": tot_mha + tot_ffn + tot_rms + output_embedding + cross_entropy,
    }


def d_ff_from_d_model(d_model: int) -> int:
    """FFN hidden size: 8/3 * d_model, rounded up to the nearest multiple of 64."""
    return -(-8 * d_model // (3 * 64)) * 64


def peak_memory_bytes(num_params: int, num_activations: int) -> int:
    """Peak training memory: params + grads + activations + optimizer state (2 * params)."""
    return DTYPE_BYTES * (4 * num_params + num_activations)


def build_table(
    name: str,
    config: dict[str, int],
    batch_size: int,
    params: dict[str, int],
    flops: dict[str, int],
    activations: dict[str, int],
) -> Table:
    total_flops = flops["total"]
    config_str = "  ".join(f"{key}={value}" for key, value in config.items())
    title = f"{name}\n{config_str}\nbatch_size={batch_size}"
    table = Table(title=title, title_style="bold", header_style="bold", caption_style="bold")
    table.add_column("component")
    table.add_column("params", justify="right")
    table.add_column("flops", justify="right")
    table.add_column("flops %", justify="right")
    table.add_column("activations", justify="right")

    # (label, param key, flop key, activation key); None means the component has no such entry.
    rows = [
        ("attention", "mha", "mha", "mha"),
        ("ffn", "ffn", "ffn", "ffn"),
        ("rmsnorm", "rmsnorm", None, "rmsnorm"),
        ("embedding", "embeddings", None, "output_embedding"),
        ("lm_head", "lm_head", "lm_head", "cross_entropy"),
    ]
    for label, p_key, f_key, a_key in rows:
        p = params.get(p_key, 0) if p_key else 0
        f = flops.get(f_key, 0) if f_key else 0
        a = activations.get(a_key, 0) if a_key else 0
        table.add_row(
            label,
            humanize.intword(p, format="%.2f") if p else "—",
            str(Quantity(f, "FLOPs")) if f else "—",
            f"{100 * f / total_flops:.1f}%" if f else "—",
            humanize.intword(a, format="%.2f") if a else "—",
        )

    table.add_row(
        "total",
        humanize.intword(params["total"], format="%.2f"),
        str(Quantity(total_flops, "FLOPs")),
        "100.0%",
        humanize.intword(activations["total"], format="%.2f"),
        style="bold",
    )

    # Peak training memory: params + grads + activations + optimizer state (2 * params).
    peak_bytes = peak_memory_bytes(params["total"], activations["total"])
    table.caption = (
        f"peak memory ({DTYPE}, params + grads + activations + optimizer state): {humanize.naturalsize(peak_bytes)}"
    )
    return table


def plot_peak_memory(models: dict[str, dict[str, int]], batch_sizes: list[int]) -> go.Figure:
    """Plot peak training memory as a function of batch size, one line per model."""
    fig = go.Figure()
    for name, config in models.items():
        num_params = param_counter(**config)["total"]
        peak_gb = [
            peak_memory_bytes(num_params, activation_counter(batch_size=b, **config)["total"]) / 1e9
            for b in batch_sizes
        ]
        fig.add_trace(go.Scatter(x=batch_sizes, y=peak_gb, mode="lines+markers", name=name))

    fig.update_layout(
        title=f"Peak training memory ({DTYPE}) vs. batch size",
        xaxis_title="batch size",
        yaxis_title="peak memory (GB)",
        legend_title="model",
    )
    return fig


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
        # "gpt2-xl_ctx16384": {
        #     "vocab_size": 50257,
        #     "context_length": 16384,
        #     "num_layers": 48,
        #     "d_model": 1600,
        #     "num_heads": 25,
        #     "d_ff": d_ff_from_d_model(1600),
        # },
    }

    console = Console()
    batch_size = 1024

    for name, config in models.items():
        console.print(
            build_table(
                name,
                config,
                batch_size,
                param_counter(**config),
                flop_counter(batch_size=batch_size, **config),
                activation_counter(batch_size=batch_size, **config),
            )
        )
        console.print()

    batch_sizes = list(range(1, 65))
    fig = plot_peak_memory(models, batch_sizes)
    output_path = "peak_memory.html"
    fig.write_html(output_path)
    console.print(f"wrote peak memory plot to [bold]{output_path}[/bold]")
