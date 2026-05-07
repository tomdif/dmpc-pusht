"""Model-Predictive Path Integral (MPPI) planner.

Like CEM but uses importance-weighted Gaussian updates rather than top-K
elites. Better suited to over-confident world models because every
sample contributes (weighted by exp(reward/τ)) instead of binary
elite/non-elite — softer optimization that doesn't over-fit to model
errors.

Mathematical form (per iter):
    weights ∝ exp((rewards - max(rewards)) / τ)
    μ_new   = Σ_i weights_i · samples_i
    σ_new   = Σ_i weights_i · (samples_i - μ_new)²
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from ..models.world_model import CSERJEPAv2


@dataclass
class MPPIConfig:
    horizon: int = 12
    n_samples: int = 256
    n_iters: int = 4
    init_std: float = 0.3
    min_std: float = 0.1
    temperature: float = 1.0
    action_clip: float | None = None


class MPPIPlanner:
    def __init__(self, model: CSERJEPAv2, cfg: MPPIConfig, action_mean: Tensor, action_std: Tensor):
        self.model = model.eval()
        self.cfg = cfg
        self.action_mean = action_mean
        self.action_std = action_std
        self.d_a = int(action_mean.numel())

    @torch.no_grad()
    def plan(self, frame_embeds: Tensor,
             prior_fn: Callable[[Tensor], Tensor] | None = None) -> Tensor:
        cfg = self.cfg
        device = frame_embeds.device
        K = self.model.chunk_size

        mu = torch.zeros(cfg.horizon, K, self.d_a, device=device)
        if prior_fn is not None:
            a0 = prior_fn(frame_embeds).squeeze(0).to(device)
            mu[0] = a0
        sigma = torch.full((cfg.horizon, K, self.d_a), cfg.init_std, device=device)

        for _ in range(cfg.n_iters):
            samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
                cfg.n_samples, cfg.horizon, K, self.d_a, device=device
            )
            if cfg.action_clip is not None:
                samples = samples.clamp(-cfg.action_clip, cfg.action_clip)

            scores = torch.zeros(cfg.n_samples, device=device)
            ctx = frame_embeds.expand(cfg.n_samples, -1, -1).contiguous()
            max_ctx = self.model.predictor.frame_pos_embed.size(1)
            for h in range(cfg.horizon):
                a_h = samples[:, h]
                z_h, r_h = self.model.predict(ctx, a_h)
                if r_h.dim() > 1 and r_h.size(-1) > 1:
                    scores = scores + r_h.sum(dim=-1)
                else:
                    scores = scores + r_h.squeeze(-1)
                ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
                if ctx.size(1) > max_ctx:
                    ctx = ctx[:, -max_ctx:]

            # Importance-weighted update.
            scores_centered = scores - scores.max()
            weights = torch.softmax(scores_centered / max(cfg.temperature, 1e-8), dim=0)
            # weights: (N,). samples: (N, horizon, K, d_a).
            w = weights.view(-1, 1, 1, 1)
            mu = (w * samples).sum(dim=0)
            var = (w * (samples - mu.unsqueeze(0)).pow(2)).sum(dim=0)
            sigma = var.sqrt().clamp_min(cfg.min_std)

        a0_norm = mu[0]
        a0 = a0_norm * self.action_std.to(device) + self.action_mean.to(device)
        return a0.unsqueeze(0)
