"""Auxiliary heads (general meta-methods, not domain priors).

EncoderActionDecoder (IDM-z): predicts a_t from (z_t, z_target). A general
supervised auxiliary loss that grounds the encoder against action-invariant
collapse. Bitter-lesson-defensible: encodes no assumption about *how*
actions act on state — just a generic "if you have action labels, use
them" signal, structurally analogous to SIGReg.

On data with rich appearance variation (real video), this head's loss
naturally goes to zero as the encoder becomes action-aware on its own.
On sparse-state toys it provides the gradient channel that keeps the
encoder honest.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import MLP


class StateDecoder(nn.Module):
    """Decodes (agent_xy, block_xytheta) from z. Used to compute a dense
    proximity reward at planning time: predicted distance from agent to
    block. Trained on self-play episodes that saved gym-pusht state info.
    """

    def __init__(self, d_z: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        # Output: 2 (agent xy) + 3 (block x, y, theta) = 5 scalars
        self.trunk = MLP(d_z, hidden, hidden=hidden, depth=depth)
        self.head = nn.Linear(hidden, 5)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: Tensor) -> Tensor:
        h = self.trunk(z)
        out = self.head(h)
        return out  # (B, 5) — agent_xy + block_xytheta


class EncoderActionDecoder(nn.Module):
    def __init__(self, d_z: int, d_a: int, chunk_size: int = 1, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.d_a = d_a
        self.chunk_size = chunk_size
        self.trunk = MLP(2 * d_z, hidden, hidden=hidden, depth=depth)
        self.head = nn.Linear(hidden, chunk_size * d_a)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_t: Tensor, z_target: Tensor) -> Tensor:
        x = torch.cat([z_t, z_target], dim=-1)
        h = self.trunk(x)
        return self.head(h).view(-1, self.chunk_size, self.d_a)
