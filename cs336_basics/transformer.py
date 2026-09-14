import math

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        std_ = math.sqrt(2 / (in_features + out_features))
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros([out_features, in_features], dtype=dtype, device=device),
                mean=0.0,
                std=std_,
                a=-3 * std_,
                b=3 * std_,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = embedding_dim
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros([num_embeddings, embedding_dim], dtype=dtype, device=device),
                mean=0.0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids,]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)

        var_ = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_norm = x * torch.rsqrt(var_)
        rmsnorm = x_norm * self.weights

        return rmsnorm.to(x_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        std_ = math.sqrt(2 / (d_ff + d_model))

        self.weights_1: Float[Tensor, " d_ff d_model"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros([d_ff, d_model], dtype=dtype, device=device),
                mean=0.0,
                std=std_,
                a=-3 * std_,
                b=3 * std_,
            )
        )

        self.weights_2: Float[Tensor, " d_model d_ff"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros([d_model, d_ff], dtype=dtype, device=device),
                mean=0.0,
                std=std_,
                a=-3 * std_,
                b=3 * std_,
            )
        )

        self.weights_3: Float[Tensor, " d_ff d_model"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros([d_ff, d_model], dtype=dtype, device=device),
                mean=0.0,
                std=std_,
                a=-3 * std_,
                b=3 * std_,
            )
        )

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        w1_x = einsum(x, self.weights_1, "... d_model, d_ff d_model -> ... d_ff")
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        w3_x = einsum(x, self.weights_3, "... d_model, d_ff d_model -> ... d_ff")

        return einsum(silu_w1_x * w3_x, self.weights_2, "... d_ff, d_model d_ff -> ... d_model")
