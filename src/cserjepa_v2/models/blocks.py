from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden: int | None = None,
                 depth: int = 2, dropout: float = 0.0):
        super().__init__()
        h = hidden if hidden is not None else max(dim_in, dim_out) * 2
        layers: list[nn.Module] = [nn.Linear(dim_in, h), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.GELU()]
        layers.append(nn.Linear(h, dim_out))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block. Causal self-attn when ``causal=True``."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        mask = None
        if self.causal:
            T = x.size(1)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x
