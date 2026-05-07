from __future__ import annotations

import math

import torch
from torch import Tensor


class RunningScalar:
    def __init__(self, init: float = 1.0, momentum: float = 0.99) -> None:
        self.value = float(init)
        self.momentum = float(momentum)
        self._initialized = False

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        v = float(x.detach().float().var().clamp_min(1e-8).item())
        if not self._initialized:
            self.value = v
            self._initialized = True
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * v

    def get(self) -> float:
        return max(self.value, 1e-8)


def normalized_mse(pred: Tensor, target: Tensor, sigma2: float, dim: int) -> Tensor:
    diff = pred - target
    sq = diff.pow(2).sum(dim=-1)
    return sq.mean() / (dim * max(sigma2, 1e-8))


def gaussian_nll(target: Tensor, mu: Tensor, log_sigma: Tensor) -> Tensor:
    """Diagonal Gaussian NLL summed over feature dim, meaned over leading dims."""
    var = (2.0 * log_sigma).exp()
    nll = 0.5 * (
        math.log(2.0 * math.pi)
        + (2.0 * log_sigma)
        + (target - mu).pow(2) / var.clamp_min(1e-8)
    )
    if nll.dim() > 2:
        return nll.flatten(start_dim=-2).sum(dim=-1).mean()
    return nll.sum(dim=-1).mean()
