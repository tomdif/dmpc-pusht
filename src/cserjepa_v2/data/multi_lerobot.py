"""Multi-dataset LeRobot adapter — pads heterogeneous action dims and
resizes heterogeneous image sizes so a single model trains across them.

Bitter-lesson rationale: more diverse data → better representations, even
when datasets cover different tasks. The predictor learns dynamics across
embodiments; the encoder doesn't get to overfit to one visual style.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from ..types import LabeledBatch
from .lerobot import LeRobotConfig, LeRobotWindows


@dataclass
class MultiLeRobotConfig:
    repos: list[dict] = field(default_factory=list)
    """List of per-dataset configs:
        [{"repo_id": "lerobot/pusht_image", "image_key": "observation.image",
          "action_key": "action", "reward_key": "next.reward"}, ...]
    """
    context_len: int = 4
    chunk_size: int = 8
    target_image_size: int = 96
    target_action_dim: int = 3       # pad shorter-action datasets to this
    normalize_actions: bool = True
    normalize_rewards: bool = False
    stats_n: int = 4096


class _NormalizingResizingWrapper(Dataset):
    """Wraps a LeRobotWindows so its outputs match (target_image_size,
    target_action_dim, embodiment_id). Action padded with zeros, image
    resized via bilinear interpolation, embodiment_id added.
    """

    def __init__(self, windows: LeRobotWindows, embodiment_id: int,
                 target_image_size: int, target_action_dim: int):
        self.w = windows
        self.embodiment_id = embodiment_id
        self.target_image_size = target_image_size
        self.target_action_dim = target_action_dim

    def __len__(self) -> int:
        return len(self.w)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        d = self.w[idx]
        x_ctx = d["x_context"]
        x_tgt = d["x_target"]
        a = d["a_chunk"]
        r = d["r_chunk"]
        if x_ctx.shape[-1] != self.target_image_size:
            T, C, H, W = x_ctx.shape
            x_ctx = F.interpolate(
                x_ctx, size=(self.target_image_size, self.target_image_size),
                mode="bilinear", align_corners=False,
            )
            x_tgt = F.interpolate(
                x_tgt.unsqueeze(0),
                size=(self.target_image_size, self.target_image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        if a.shape[-1] < self.target_action_dim:
            pad = self.target_action_dim - a.shape[-1]
            a = F.pad(a, (0, pad), value=0.0)
        return {
            "x_context": x_ctx,
            "x_target": x_tgt,
            "a_chunk": a,
            "r_chunk": r,
            "embodiment": torch.tensor(self.embodiment_id, dtype=torch.long),
        }


class MultiLeRobotWindows(Dataset):
    """Concatenated dataset of multiple LeRobot datasets with action/image
    homogenized via _NormalizingResizingWrapper."""

    def __init__(self, cfg: MultiLeRobotConfig):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.cfg = cfg
        self.embodiments = [r["repo_id"] for r in cfg.repos]
        wrapped: list[Dataset] = []
        action_means, action_stds = [], []
        reward_means, reward_stds = [], []
        for i, r in enumerate(cfg.repos):
            sub_cfg = LeRobotConfig(
                repo_id=r["repo_id"],
                image_key=r.get("image_key", "observation.image"),
                action_key=r.get("action_key", "action"),
                reward_key=r.get("reward_key", "next.reward"),
                context_len=cfg.context_len,
                chunk_size=cfg.chunk_size,
                normalize_actions=cfg.normalize_actions,
                normalize_rewards=cfg.normalize_rewards,
                stats_n=cfg.stats_n,
            )
            base = LeRobotDataset(r["repo_id"])
            windows = LeRobotWindows(base, sub_cfg)
            # Pad per-dataset action stats to target_action_dim with mean=0, std=1.
            am = windows.action_mean.flatten()
            asd = windows.action_std.flatten()
            if am.numel() < cfg.target_action_dim:
                pad = cfg.target_action_dim - am.numel()
                am = torch.cat([am, torch.zeros(pad)])
                asd = torch.cat([asd, torch.ones(pad)])
            action_means.append(am)
            action_stds.append(asd)
            reward_means.append(windows.reward_mean)
            reward_stds.append(windows.reward_std)
            wrapped.append(_NormalizingResizingWrapper(
                windows, embodiment_id=i,
                target_image_size=cfg.target_image_size,
                target_action_dim=cfg.target_action_dim,
            ))
            print(f"[multi-data] {r['repo_id']}: {len(windows)} windows")
        self.concat = ConcatDataset(wrapped)
        # Per-dataset stats (for online inference, eval picks one)
        self.action_means = torch.stack(action_means, dim=0)        # (n_emb, target_action_dim)
        self.action_stds = torch.stack(action_stds, dim=0)
        self.reward_means = torch.tensor(reward_means)
        self.reward_stds = torch.tensor(reward_stds)

    def __len__(self) -> int:
        return len(self.concat)

    def __getitem__(self, idx: int):
        return self.concat[idx]


def collate_multi_lerobot(batch: list[dict[str, Tensor]]) -> LabeledBatch:
    x_context = torch.stack([b["x_context"] for b in batch], dim=0)
    x_target = torch.stack([b["x_target"] for b in batch], dim=0)
    a_chunk = torch.stack([b["a_chunk"] for b in batch], dim=0)
    r_chunk = torch.stack([b["r_chunk"] for b in batch], dim=0)
    embodiment = torch.stack([b["embodiment"] for b in batch], dim=0)
    return LabeledBatch(
        x_context=x_context, x_target=x_target,
        a_chunk=a_chunk, r_chunk=r_chunk, embodiment=embodiment,
    )


def build_multi_lerobot_loader(
    cfg: MultiLeRobotConfig, *, batch_size: int, num_workers: int = 4, shuffle: bool = True,
):
    ds = MultiLeRobotWindows(cfg)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_multi_lerobot,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, cfg.target_action_dim, cfg.target_image_size, ds
