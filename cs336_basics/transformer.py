import math

import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        std_ = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros([out_features, in_features], dtype=dtype, device=device),
                mean=0.0,
                std=std_,
                a=-3 * std_,
                b=3 * std_,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


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
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)

        var_ = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_norm = x * torch.rsqrt(var_)
        rmsnorm = x_norm * self.weight

        return rmsnorm.to(x_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        w1_x = self.w1(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3(x)

        return self.w2(silu_w1_x * w3_x)


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
        x_pair_flip = x_pair[..., [1, 0]] * torch.tensor([-1, 1], device=x_pair.device)
        x_pair_roped = x_pair * cos_ord.unsqueeze(dim=-1) + x_pair_flip * sin_ord.unsqueeze(dim=-1)

        return rearrange(x_pair_roped, "... dk2 p -> ... (dk2 p)")


def my_softmax(x: Tensor, dim: int):
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return exp_x * exp_x.sum(dim=dim, keepdim=True).reciprocal()


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    sdp = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    pre_softmax = sdp.masked_fill(~mask, -torch.inf) if mask is not None else sdp

    return einsum(my_softmax(pre_softmax, dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.q_proj = Linear(in_features=d_model, out_features=d_model)
        self.k_proj = Linear(in_features=d_model, out_features=d_model)
        self.v_proj = Linear(in_features=d_model, out_features=d_model)
        self.output_proj = Linear(in_features=d_model, out_features=d_model)

        self.rope = None
        if self.theta is not None and self.max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=self.theta, d_k=self.d_model // self.num_heads, max_seq_len=self.max_seq_len
            )

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        # project x into query/key/value subspaces
        # here d_out = h.d_k = h.d_model/h = d_model
        x_query = self.q_proj(x)
        x_key = self.k_proj(x)
        x_value = self.v_proj(x)

        # split into attention heads
        x_query_heads = rearrange(x_query, "... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.num_heads)
        x_key_heads = rearrange(x_key, "... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.num_heads)
        x_value_heads = rearrange(x_value, "... sequence_length (h d_v) -> ... h sequence_length d_v", h=self.num_heads)

        if self.rope is not None:
            assert token_positions is not None, "token_positions must be provided for RoPE"
            x_query_heads = self.rope(x_query_heads, token_positions)
            x_key_heads = self.rope(x_key_heads, token_positions)

        seq_len = x.shape[-2]
        # mask will automatically broadcast over other dims
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        sdpa = rearrange(
            scaled_dot_product_attention(Q=x_query_heads, K=x_key_heads, V=x_value_heads, mask=mask),
            "... h sequence_length d_v -> ... sequence_length (h d_v)",
        )

        return self.output_proj(sdpa)


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)

        y1 = x + self.attn(self.ln1(x), token_positions)
        y2 = y1 + self.ffn(self.ln2(y1))

        return y2
