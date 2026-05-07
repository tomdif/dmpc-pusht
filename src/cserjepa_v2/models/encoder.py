"""Per-frame ViT encoder. Produces a single mean-pooled embedding per frame.

For the toy: img=32, patch=4 → 64 patches → mean-pool → (B, T, embed_dim)
or (B, embed_dim) for a single frame. No spatial patch sequence retained
— the v2 design predicts in mean-pooled space (matches the v5.1-validated
regime, sidesteps the per-patch-loss problem).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import TransformerBlock


class VideoEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C, H, W) or (B, C, H, W).

        Returns:
            (B, T, embed_dim) for video input — mean-pooled per frame.
            (B, embed_dim) for single-frame input.
        """
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
            tokens = self._encode(x).mean(dim=1)             # (B*T, D)
            return tokens.view(B, T, self.embed_dim)
        return self._encode(x).mean(dim=1)                   # (B, D)

    def _encode(self, x: Tensor) -> Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)
