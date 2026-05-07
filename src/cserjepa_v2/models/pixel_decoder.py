"""Pixel decoder for grounded planning.

Tiny CNN decoder that maps a latent z (D-dim) back to a 96x96 RGB image.
Used at plan time to ground CEM's score in pixel space: imagined latents
must decode to images that look like progress toward the goal frame, not
just have favorable value/goal-distance scores in latent space.

Trained on a frozen encoder via pixel-MSE reconstruction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PixelDecoder(nn.Module):
    def __init__(self, d_z: int = 768, img_size: int = 96, init_hw: int = 12) -> None:
        super().__init__()
        # init_hw=12 for img_size=96 → three 2x upsamples reach 96.
        assert img_size == init_hw * 8, f"need img_size={init_hw*8}, got {img_size}"
        self.init_hw = init_hw
        self.proj = nn.Sequential(
            nn.Linear(d_z, 256),
            nn.GELU(),
            nn.Linear(256, init_hw * init_hw * 64),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 12→24
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 24→48
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # 48→96
            nn.GELU(),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, D). Returns (B, 3, 96, 96) in [0, 1]."""
        b = z.size(0)
        h = self.proj(z).view(b, 64, self.init_hw, self.init_hw)
        return self.deconv(h)
