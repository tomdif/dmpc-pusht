"""Core data types — chunked actions are first-class.

For Stage 0 toy at K=1, ``a_chunk`` has shape (B, 1, d_a). The architecture
treats K=1 and K>1 uniformly — bump K via config without changing types.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class LabeledBatch:
    x_context: Tensor                       # (B, T_ctx, C, H, W)
    x_target: Tensor                        # (B, C, H, W) at t+K
    a_chunk: Tensor                         # (B, K, d_a)
    r_chunk: Tensor | None = None           # (B, K) — for reward training (Stage 1 B onward)
    rtg_target: Tensor | None = None        # (B,) — return-to-go at target frame (Stage Z)
    rtg_mask: Tensor | None = None          # (B,) — 1 if rtg is real, 0 if padded
    state_target: Tensor | None = None      # (B, 5) — agent_xy + block_xytheta (Stage AF)
    state_mask: Tensor | None = None        # (B,) — 1 if state labels present, 0 if padded
    x_target_multi: Tensor | None = None    # (B, H-1, C, H, W) — Stage AH multi-step targets
    a_chunk_multi: Tensor | None = None     # (B, H-1, K, d_a) — actions per multi-step
    proprio: Tensor | None = None
    embodiment: Tensor | None = None


@dataclass
class UnlabeledBatch:
    x_context: Tensor
    x_target: Tensor
    proprio: Tensor | None = None
    embodiment: Tensor | None = None
