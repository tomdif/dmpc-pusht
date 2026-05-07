"""Synthetic-rollout dataset — wraps self-play episodes saved by
`collect_self_play.py` so they slot into the existing training pipeline
exactly like a LeRobotWindows dataset.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class SyntheticConfig:
    rollout_dir: str = ""
    rollout_dirs: list[str] | None = None  # if set, overrides rollout_dir
    context_len: int = 4
    chunk_size: int = 8
    normalize_actions: bool = True
    normalize_rewards: bool = False
    return_to_go: bool = True              # compute MC return-to-go per frame
    discount: float = 0.99
    # Object-aware pixel masking (light C-JEPA, Stage AG onward).
    object_mask_prob: float = 0.0
    env_coord_scale: float = 0.1875        # 96/512 — env→image coord
    # Multi-step prediction targets (Stage AH onward). When >1, dataset
    # returns extra (x_target_2, a_chunk_2) for autoregressive 2-step
    # supervision. Trains predictor to be robust over the 12-step CEM
    # rollouts used at inference.
    multistep: int = 1


class SyntheticWindows(Dataset):
    """Loads all episode_*.pt files from rollout_dir and exposes
    (x_context, x_target, a_chunk, r_chunk) windows.

    Each episode file contains:
        frames  : (T, C, H, W) float32 in [0, 1]
        actions : (T, d_a) raw (unnormalized)
        rewards : (T,)
    """

    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        dirs = cfg.rollout_dirs if cfg.rollout_dirs else [cfg.rollout_dir]
        all_paths: list[str] = []
        for d in dirs:
            all_paths.extend(sorted(glob.glob(os.path.join(d, "episode_*.pt"))))
        if not all_paths:
            raise ValueError(f"no episodes found in {dirs}")
        self.episode_paths = all_paths

        # Lazy index of (episode_path, start_frame) for every valid window.
        # Keep the per-episode tensors loaded only when needed via a small cache.
        self._cache: dict[str, dict] = {}
        T_min = cfg.context_len + cfg.chunk_size * cfg.multistep
        valid: list[tuple[int, int]] = []
        all_actions = []
        all_rewards = []
        for ei, path in enumerate(self.episode_paths):
            ep = torch.load(path, map_location="cpu", weights_only=False)
            T = ep["frames"].size(0)
            if T >= T_min:
                for j in range(T - T_min + 1):
                    valid.append((ei, j))
            all_actions.append(ep["actions"])
            all_rewards.append(ep["rewards"])
            # Compute return-to-go per frame (MC, discounted).
            if cfg.return_to_go:
                rewards = ep["rewards"].float()
                rtg = torch.zeros_like(rewards)
                running = 0.0
                for t in range(T - 1, -1, -1):
                    running = float(rewards[t]) + cfg.discount * running
                    rtg[t] = running
                ep["return_to_go"] = rtg
            self._cache[path] = ep
        self.valid = valid

        a_stack = torch.cat(all_actions, dim=0).float()
        self.action_mean = a_stack.mean(dim=0)
        self.action_std = a_stack.std(dim=0).clamp_min(1e-6)
        r_stack = torch.cat(all_rewards, dim=0).float()
        self.reward_mean = float(r_stack.mean().item())
        self.reward_std = float(r_stack.std().clamp_min(1e-6).item())

    def set_action_stats(self, mean: Tensor, std: Tensor) -> None:
        """Use external action stats (e.g. from the canonical LeRobot dataset)
        so synthetic data normalizes the same way."""
        self.action_mean = mean.clone()
        self.action_std = std.clone()

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        cfg = self.cfg
        T_ctx, K = cfg.context_len, cfg.chunk_size
        ei, j = self.valid[idx]
        ep = self._cache[self.episode_paths[ei]]
        frames = ep["frames"]
        actions = ep["actions"].float()
        rewards = ep["rewards"].float()
        a = actions[j + T_ctx - 1: j + T_ctx - 1 + K]
        r = rewards[j + T_ctx - 1: j + T_ctx - 1 + K]
        if cfg.normalize_actions:
            a = (a - self.action_mean) / self.action_std
        if cfg.normalize_rewards:
            r = (r - self.reward_mean) / self.reward_std
        out = {
            "x_context": frames[j:j + T_ctx],
            "x_target": frames[j + T_ctx + K - 1],
            "a_chunk": a,
            "r_chunk": r,
        }
        # Multi-step targets: stack of (x_target_h, a_chunk_h) for h=2..multistep.
        if cfg.multistep > 1:
            extra_targets = []
            extra_actions = []
            for h in range(2, cfg.multistep + 1):
                t_offset = j + T_ctx + h * K - 1
                a_offset_lo = j + T_ctx + (h - 1) * K - 1
                a_offset_hi = a_offset_lo + K
                extra_targets.append(frames[t_offset])
                a_h = actions[a_offset_lo:a_offset_hi]
                if cfg.normalize_actions:
                    a_h = (a_h - self.action_mean) / self.action_std
                extra_actions.append(a_h)
            out["x_target_multi"] = torch.stack(extra_targets, dim=0)        # (multistep-1, C, H, W)
            out["a_chunk_multi"] = torch.stack(extra_actions, dim=0)         # (multistep-1, K, d_a)
        if cfg.return_to_go and "return_to_go" in ep:
            # Return-to-go at the TARGET frame (so V(z_target) predicts it).
            out["rtg_target"] = ep["return_to_go"][j + T_ctx + K - 1].clone()
        else:
            out["rtg_target"] = torch.tensor(0.0, dtype=torch.float32)
        # State labels at TARGET frame (for state decoder training).
        if "agent_positions" in ep and "block_poses" in ep:
            tgt_idx = j + T_ctx + K - 1
            out["state_target"] = torch.cat([
                ep["agent_positions"][tgt_idx].float(),
                ep["block_poses"][tgt_idx].float(),
            ], dim=0)  # (5,)
            out["state_mask"] = torch.tensor(1.0)

            # Object-aware pixel masking — mask one randomly chosen object's
            # pixel region in x_context. Forces encoder to embed
            # "missing-object" representations the predictor must fill in.
            if cfg.object_mask_prob > 0 and torch.rand(1).item() < cfg.object_mask_prob:
                obj = "agent" if torch.rand(1).item() < 0.5 else "block"
                radius = 12 if obj == "agent" else 28
                positions = ep["agent_positions"] if obj == "agent" else ep["block_poses"]
                xc = out["x_context"].clone()  # (T_ctx, C, H, W)
                H = xc.size(-1)
                for t in range(T_ctx):
                    cx = float(positions[j + t][0]) * cfg.env_coord_scale
                    cy = float(positions[j + t][1]) * cfg.env_coord_scale
                    x0, x1 = max(0, int(cx - radius)), min(H, int(cx + radius))
                    y0, y1 = max(0, int(cy - radius)), min(H, int(cy + radius))
                    xc[t, :, y0:y1, x0:x1] = 0.0
                out["x_context"] = xc
        else:
            out["state_target"] = torch.zeros(5, dtype=torch.float32)
            out["state_mask"] = torch.tensor(0.0)
        return out
