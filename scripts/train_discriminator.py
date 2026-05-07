"""Train a small CNN+MLP success/fail classifier on the discriminator buffer.

Input: pixels (B, n_obs, 3, H, W), states (B, n_obs, 2), actions (B, n_action, 2)
Output: P(success | obs, action_chunk) — sigmoid logit

Goal at eval time: add log P(success) as an additional D-MPC rerank term, biasing
candidate selection toward action chunks the classifier predicts will lead to
success. Trained on ~270K windows, 50/50 positive/negative class balance.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class DiscriminatorBuffer(Dataset):
    def __init__(self, path: str):
        b = torch.load(path, map_location="cpu", weights_only=False)
        self.pixels = b["pixels"]   # (N, n_obs, H, W, 3) uint8
        self.states = b["states"]
        self.actions = b["actions"]
        self.labels = b["labels"]
        self.n_pos = b["n_pos"]
        self.n_neg = b["n_neg"]
        self.n_obs = b["n_obs_steps"]
        self.n_action = b["n_action_steps"]
        print(f"[buffer] {len(self.pixels)} windows: {self.n_pos} pos / {self.n_neg} neg")

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx: int):
        pix = torch.from_numpy(self.pixels[idx]).float() / 255.0       # (n_obs, H, W, 3)
        pix = pix.permute(0, 3, 1, 2).contiguous()                      # (n_obs, 3, H, W)
        # Stack obs frames along channel: (n_obs * 3, H, W).
        pix = pix.flatten(0, 1)
        state = torch.from_numpy(self.states[idx]).float().flatten()    # (n_obs * 2,)
        # Normalize state to roughly [-1, 1] using known PushT range.
        state = (state - 256.0) / 256.0
        action = torch.from_numpy(self.actions[idx]).float().flatten()  # (n_action * 2,)
        action = (action - 256.0) / 256.0
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return pix, state, action, label


class SuccessDiscriminator(nn.Module):
    """Small CNN + MLP classifier."""

    def __init__(self, in_channels: int, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # 96→48
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),            # 48→24
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),           # 24→12
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),          # 12→6
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128 + state_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pix, state, action):
        v = self.cnn(pix)
        h = torch.cat([v, state, action], dim=-1)
        return self.mlp(h).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"=== device: {device} ===")
    ds = DiscriminatorBuffer(args.buffer)

    # Class-balanced sampler.
    weights = np.where(ds.labels.astype(np.float32) > 0.5, 1.0 / ds.n_pos, 1.0 / ds.n_neg)
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), num_samples=10**8, replacement=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False)

    in_ch = ds.n_obs * 3
    state_dim = ds.n_obs * 2
    action_dim = ds.n_action * 2
    model = SuccessDiscriminator(in_ch, state_dim, action_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    step = 0
    t0 = time.time()
    iters = iter(loader)
    while step < args.steps:
        try:
            pix, state, action, label = next(iters)
        except StopIteration:
            iters = iter(loader)
            pix, state, action, label = next(iters)

        pix = pix.to(device, non_blocking=True)
        state = state.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(pix, state, action)
        loss = F.binary_cross_entropy_with_logits(logits, label)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                pred = (logits > 0).float()
                acc = (pred == label).float().mean().item()
            dt = time.time() - t0
            print(f"step={step:>5}  loss={loss.item():.4f}  acc={acc:.3f}  dt={dt:.1f}s", flush=True)
        step += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {
            "in_channels": in_ch, "state_dim": state_dim, "action_dim": action_dim,
            "n_obs_steps": ds.n_obs, "n_action_steps": ds.n_action,
        },
    }, args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
