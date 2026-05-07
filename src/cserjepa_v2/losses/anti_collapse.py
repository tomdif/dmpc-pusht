"""Epps-Pulley SIGReg.

Ported from sai_product/sai_jepa_v3.py — the LeWorldModel/LeJEPA recipe.
Cramér-Wold-correct anti-collapse: catches per-dim collapse, direction
collapse, and non-Gaussian shape via random unit-norm projections + the
Epps-Pulley univariate normality statistic.

Critically, this regularizer is strong enough that LeWorldModel drops the
EMA target encoder and stop-gradient asymmetric tricks entirely.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def sigreg_epps_pulley(z: Tensor, n_directions: int = 16, eps: float = 1e-5) -> Tensor:
    if z.dim() == 3:
        z = z.flatten(0, 1)
    if z.dim() != 2:
        return z.new_zeros(())
    N, D = z.shape
    if N < 4:
        return z.new_zeros(())

    zf = z.float()
    u = torch.randn(D, n_directions, device=zf.device, dtype=zf.dtype)
    u = F.normalize(u, p=2, dim=0)
    h = zf @ u
    h = h - h.mean(dim=0, keepdim=True)

    h_sq = h * h
    gram = torch.einsum("nm,Nm->nNm", h, h)
    d2 = (h_sq.unsqueeze(0) + h_sq.unsqueeze(1) - 2.0 * gram).clamp(min=0.0)
    K = torch.exp(-(d2.clamp(max=60.0)) / 2.0)

    sum_K = K.sum(dim=(0, 1)) / float(N)
    sum_L = (math.sqrt(2.0) * torch.exp(-h_sq.clamp(max=120.0) / 4.0)).sum(dim=0)
    const = float(N) / math.sqrt(3.0)

    T_dir = (sum_K - sum_L + const).clamp(min=0.0)
    return T_dir.mean().to(z.dtype)


def sigreg_lewm(z: Tensor, num_proj: int = 1024, knots: int = 17) -> Tensor:
    """LeWM SIGReg — exact port of lucas-maes/le-wm `module.py:SIGReg`.

    Different formulation from sigreg_epps_pulley above: integrates moment-
    fit error against a Gaussian window using a quadrature rule on a fixed
    knot grid. Uses 64× more random projections (1024 vs 16) and is the
    statistic that drove LeWM's 98% PushT result with only this + pred MSE.
    """
    if z.dim() == 3:
        z = z.transpose(0, 1)  # (T, B, D) — matches LeWM's input convention
    elif z.dim() == 2:
        z = z.unsqueeze(0)  # (1, B, D)

    T_dim, B, D = z.shape
    if B < 4:
        return z.new_zeros(())

    zf = z.float()
    t = torch.linspace(0, 3, knots, device=zf.device, dtype=zf.dtype)
    dt = 3.0 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=zf.device, dtype=zf.dtype)
    weights[0] = dt
    weights[-1] = dt
    window = torch.exp(-t.square() / 2.0)
    weights = weights * window

    A = torch.randn(D, num_proj, device=zf.device, dtype=zf.dtype)
    A = A.div_(A.norm(p=2, dim=0))                                   # (D, P)
    proj = zf @ A                                                     # (T, B, P)

    x_t = proj.unsqueeze(-1) * t                                      # (T, B, P, K)
    err = (x_t.cos().mean(dim=-3) - window).square() + x_t.sin().mean(dim=-3).square()
    statistic = (err @ weights) * float(B)                            # (T, P)
    return statistic.mean().to(z.dtype)
