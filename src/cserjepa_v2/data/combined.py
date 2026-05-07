"""Combined dataset: original LeRobot pusht + self-play synthetic rollouts.

Both datasets share the same action dim (2D) and image format, so no
padding pathology — unlike the multi-dataset (R) attempt with xarm.
The synthetic data inherits action normalization stats from the canonical
LeRobot dataset so both halves normalize identically.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from ..types import LabeledBatch
from .lerobot import LeRobotConfig, LeRobotWindows
from .synthetic import SyntheticConfig, SyntheticWindows


def collate_combined(batch: list[dict[str, Tensor]]) -> LabeledBatch:
    x_context = torch.stack([b["x_context"] for b in batch], dim=0)
    x_target = torch.stack([b["x_target"] for b in batch], dim=0)
    a_chunk = torch.stack([b["a_chunk"] for b in batch], dim=0)
    r_chunk = torch.stack([b["r_chunk"] for b in batch], dim=0)
    # rtg_target may be present (synthetic) or absent (LeRobot). Pad missing
    # with 0 and mask them so value loss only fires on real returns.
    rtgs, masks = [], []
    for b in batch:
        if "rtg_target" in b:
            rtgs.append(b["rtg_target"].float())
            masks.append(torch.tensor(1.0))
        else:
            rtgs.append(torch.tensor(0.0))
            masks.append(torch.tensor(0.0))
    rtg_target = torch.stack(rtgs, dim=0)
    rtg_mask = torch.stack(masks, dim=0)
    # State labels (Stage AF onward).
    states, smasks = [], []
    for b in batch:
        if "state_target" in b:
            states.append(b["state_target"].float())
            smasks.append(b.get("state_mask", torch.tensor(1.0)))
        else:
            states.append(torch.zeros(5, dtype=torch.float32))
            smasks.append(torch.tensor(0.0))
    state_target = torch.stack(states, dim=0)
    state_mask = torch.stack(smasks, dim=0)
    # Multi-step targets (Stage AH). All-or-none across batch — synthetic
    # samples have them, lerobot doesn't, so when mixed we fall back.
    if all("x_target_multi" in b for b in batch):
        x_target_multi = torch.stack([b["x_target_multi"] for b in batch], dim=0)
        a_chunk_multi = torch.stack([b["a_chunk_multi"] for b in batch], dim=0)
    else:
        x_target_multi = None
        a_chunk_multi = None
    return LabeledBatch(
        x_context=x_context, x_target=x_target, a_chunk=a_chunk, r_chunk=r_chunk,
        rtg_target=rtg_target, rtg_mask=rtg_mask,
        state_target=state_target, state_mask=state_mask,
        x_target_multi=x_target_multi, a_chunk_multi=a_chunk_multi,
    )


def build_combined_loader(
    lerobot_cfg: LeRobotConfig,
    synthetic_cfg: SyntheticConfig,
    *,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    base = LeRobotDataset(lerobot_cfg.repo_id)
    sample = base[0]
    a_dim = sample[lerobot_cfg.action_key].shape[-1]
    image_size = sample[lerobot_cfg.image_key].shape[-1]
    lerobot_w = LeRobotWindows(base, lerobot_cfg)
    synthetic_w = SyntheticWindows(synthetic_cfg)
    # Force synthetic to share LeRobot's action normalization.
    synthetic_w.set_action_stats(lerobot_w.action_mean, lerobot_w.action_std)

    print(f"[combined] lerobot windows : {len(lerobot_w)}")
    print(f"[combined] synthetic windows: {len(synthetic_w)}")

    concat = ConcatDataset([lerobot_w, synthetic_w])
    loader = torch.utils.data.DataLoader(
        concat,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_combined,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, a_dim, image_size, lerobot_w
