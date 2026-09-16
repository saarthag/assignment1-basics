import math

import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int
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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.d_k = d_k
        exp = torch.arange(d_k >> 1, device=device) * 2 / d_k
        angles = einsum(torch.arange(max_seq_len, device=device), theta**-exp, "max_seq_len, dk2 -> max_seq_len dk2")

        self.register_buffer("sin_precomp", torch.sin(angles), persistent=False)
        self.register_buffer("cos_precomp", torch.cos(angles), persistent=False)

    def forward(self, x: Float[Tensor, " ... seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"]) -> Tensor:
        sin_ord = self.get_buffer("sin_precomp")[token_positions]
        cos_ord = self.get_buffer("cos_precomp")[token_positions]

        x_pair = rearrange(x, "... seq_len (dk2 p) -> ... seq_len dk2 p", dk2=(self.d_k >> 1), p=2)
        x_pair_flip = x_pair[..., [1, 0]] * torch.tensor([-1, 1])
        x_pair_roped = x_pair * cos_ord.unsqueeze(dim=-1) + x_pair_flip * sin_ord.unsqueeze(dim=-1)

        return rearrange(x_pair_roped, "... dk2 p -> ... (dk2 p)")


def my_softmax(x: Tensor, dim: int):
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return exp_x * exp_x.sum(dim=dim, keepdim=True).reciprocal()
