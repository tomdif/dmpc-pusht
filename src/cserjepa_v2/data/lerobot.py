"""LeRobot dataset adapter -> LabeledBatch with (x_context, x_target, a_chunk).

Wraps a LeRobotDataset (image variant) and yields windows aligned with our
existing toy interface so the trainer is dataset-agnostic. Builds an
episode-aware sampler so we never cross episode boundaries when forming
context/target windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..types import LabeledBatch


@dataclass
class LeRobotConfig:
    repo_id: str = "lerobot/pusht_image"
    image_key: str = "observation.image"
    action_key: str = "action"
    reward_key: str = "next.reward"
    context_len: int = 4
    chunk_size: int = 1
    normalize_actions: bool = True
    normalize_rewards: bool = False
    stats_n: int = 4096    # frames sampled for mean/std estimation


class LeRobotWindows(Dataset):
    """Yields (x_context [T,C,H,W], x_target [C,H,W], a_chunk [K,d_a], r [K])
    over all valid windows of a LeRobot dataset, never crossing episodes.
    """

    def __init__(self, lerobot_dataset, cfg: LeRobotConfig):
        self.ds = lerobot_dataset
        self.cfg = cfg

        # Episode boundaries — the dataset stores 1D `episode_index` per frame.
        ep_idx = torch.as_tensor(self.ds.hf_dataset["episode_index"], dtype=torch.long)
        self.episode_starts: dict[int, list[int]] = {}
        T_min = cfg.context_len + cfg.chunk_size
        valid: list[int] = []
        boundaries = (ep_idx[1:] != ep_idx[:-1]).nonzero(as_tuple=True)[0].tolist() + [len(ep_idx) - 1]
        prev = 0
        for end in boundaries:
            length = end - prev + 1
            if length >= T_min:
                for i in range(prev, prev + length - T_min + 1):
                    valid.append(i)
            prev = end + 1
        self.valid_starts = valid

        # Action stats — pixel-coord actions in pusht have range O(100s);
        # without normalization, IDM-z MSE swamps everything by 10^4.
        self.action_mean = torch.zeros(1)
        self.action_std = torch.ones(1)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        if cfg.normalize_actions or cfg.normalize_rewards:
            n_total = len(self.ds)
            n = min(int(cfg.stats_n), n_total)
            idx = torch.linspace(0, n_total - 1, n).long().tolist()
            a_samples, r_samples = [], []
            for i in idx:
                f = self.ds[i]
                a_samples.append(f[cfg.action_key])
                r_samples.append(f[cfg.reward_key])
            a_stack = torch.stack(a_samples, dim=0).float()
            self.action_mean = a_stack.mean(dim=0)
            self.action_std = a_stack.std(dim=0).clamp_min(1e-6)
            r_stack = torch.stack(r_samples, dim=0).float()
            self.reward_mean = float(r_stack.mean().item())
            self.reward_std = float(r_stack.std().clamp_min(1e-6).item())

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        cfg = self.cfg
        T_ctx, K = cfg.context_len, cfg.chunk_size
        start = self.valid_starts[idx]
        imgs, actions, rewards = [], [], []
        for t in range(T_ctx + K):
            f = self.ds[start + t]
            imgs.append(f[cfg.image_key])
            actions.append(f[cfg.action_key])
            rewards.append(f[cfg.reward_key])
        imgs = torch.stack(imgs, dim=0)        # (T_ctx+K, C, H, W)
        actions = torch.stack(actions, dim=0).float()  # (T_ctx+K, d_a)
        rewards = torch.stack(rewards, dim=0).float()  # (T_ctx+K,)
        if cfg.normalize_actions:
            actions = (actions - self.action_mean) / self.action_std
        if cfg.normalize_rewards:
            rewards = (rewards - self.reward_mean) / self.reward_std
        return {
            "x_context": imgs[:T_ctx],
            "x_target": imgs[-1],
            "a_chunk": actions[T_ctx - 1:T_ctx - 1 + K],
            "r_chunk": rewards[T_ctx - 1:T_ctx - 1 + K],
        }


def collate_lerobot(batch: list[dict[str, Tensor]]) -> LabeledBatch:
    x_context = torch.stack([b["x_context"] for b in batch], dim=0)
    x_target = torch.stack([b["x_target"] for b in batch], dim=0)
    a_chunk = torch.stack([b["a_chunk"] for b in batch], dim=0)
    r_chunk = torch.stack([b["r_chunk"] for b in batch], dim=0)
    return LabeledBatch(x_context=x_context, x_target=x_target, a_chunk=a_chunk, r_chunk=r_chunk)


def build_lerobot_loader(cfg: LeRobotConfig, *, batch_size: int, num_workers: int = 4,
                         shuffle: bool = True, image_size: int | None = None):
    """Returns (loader, action_dim, image_size)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    base = LeRobotDataset(cfg.repo_id)
    sample = base[0]
    a_dim = sample[cfg.action_key].shape[-1]
    if image_size is None:
        image_size = sample[cfg.image_key].shape[-1]
    windows = LeRobotWindows(base, cfg)
    loader = torch.utils.data.DataLoader(
        windows,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_lerobot,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, a_dim, image_size
