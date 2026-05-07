"""Calibrated 2D toy: v_{t+1} = ρ·v + α·a + ε; s_{t+1} = clip(s + v).

Renders the state as a 32×32 RGB image with a Gaussian blob at (x, y).
For Stage 0 we run α=1, ρ=0 (action-essential, no inertia) — translation
in (x, y) per step. The natural simple algorithm is "action ≈ phase shift
in x, y" — discoverable via the Action-Spectral Path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..types import LabeledBatch


@dataclass
class ToyConfig:
    image_size: int = 32
    in_chans: int = 3
    state_dim: int = 2
    action_dim: int = 2
    context_len: int = 4
    chunk_size: int = 1
    blob_sigma: float = 1.5
    action_scale: float = 0.25
    alpha: float = 1.0
    rho: float = 0.0
    noise_std: float = 0.0


def _render(states: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    leading = states.shape[:-1]
    flat = states.reshape(-1, 2)
    cx = (flat[:, 0] + 1.0) / 2.0 * (size - 1)
    cy = (flat[:, 1] + 1.0) / 2.0 * (size - 1)
    yy = torch.arange(size, dtype=torch.float32)
    xx = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    img = torch.exp(
        -((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2)
        / (2.0 * sigma * sigma)
    )
    img = img.unsqueeze(1).expand(-1, 3, -1, -1)
    return img.reshape(*leading, 3, size, size).clone()


def _gen_episodes(n: int, T_total: int, cfg: ToyConfig, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    D = cfg.state_dim
    s = (torch.rand(n, D, generator=g) * 2.0 - 1.0)
    v = torch.zeros(n, D)
    actions = (torch.rand(n, T_total - 1, D, generator=g) * 2.0 - 1.0) * cfg.action_scale
    states = [s]
    for t in range(T_total - 1):
        eps = torch.randn(n, D, generator=g) * cfg.noise_std
        v = cfg.rho * v + cfg.alpha * actions[:, t] + eps
        s = (s + v).clamp(-1.0, 1.0)
        states.append(s)
    return torch.stack(states, dim=1), actions


def build_toy_batches(
    cfg: ToyConfig, n: int, device: torch.device | str = "cpu", seed: int | None = None
) -> LabeledBatch:
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))

    T_ctx, K = cfg.context_len, cfg.chunk_size
    T_total = T_ctx + K
    states, actions = _gen_episodes(n, T_total, cfg, g)
    imgs = _render(states, cfg.image_size, cfg.blob_sigma)

    x_context = imgs[:, :T_ctx]
    x_target = imgs[:, -1]                              # frame at t+K
    a_chunk = actions[:, T_ctx - 1: T_ctx - 1 + K]

    return LabeledBatch(
        x_context=x_context.to(device),
        x_target=x_target.to(device),
        a_chunk=a_chunk.to(device),
    )
