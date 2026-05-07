"""Behavior-cloning policy: π(a_chunk | frame_embeds).

Used as a CEM proposal prior — the planner searches *from* an imitative
starting point rather than from zero. Helps the agent engage the
environment in initial configurations the world model otherwise has no
signal on.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import MLP


class BCPolicy(nn.Module):
    def __init__(self, d: int, d_a: int, chunk_size: int, context_len: int,
                 hidden: int = 256, depth: int = 3):
        super().__init__()
        self.d_a = d_a
        self.chunk_size = chunk_size
        self.context_len = context_len
        self.trunk = MLP(d * context_len, hidden, hidden=hidden, depth=depth)
        self.head = nn.Linear(hidden, chunk_size * d_a)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, frame_embeds: Tensor) -> Tensor:
        """frame_embeds: (B, T_ctx, D). Returns (B, chunk_size, d_a)."""
        x = frame_embeds.flatten(1)
        h = self.trunk(x)
        return self.head(h).view(-1, self.chunk_size, self.d_a)
