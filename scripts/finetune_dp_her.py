"""Fine-tune Diffusion Policy on HER buffer (Step 5c) with DDP.

Loads pretrained `lerobot/diffusion_pusht`, fine-tunes via standard DP loss
(diffusion noise-prediction MSE on action sequences) on a mix of:
  (1) HER buffer of own-rollout high-coverage trajectories
  (2) original LeRobot expert demos (anti-forgetting)

Multi-GPU via torch DistributedDataParallel. Save to container drive.

Launch:
  torchrun --nproc-per-node=6 scripts/finetune_dp_her.py \\
      --buffer /root/step5/her_buffer/round1.pt \\
      --out-dir /root/step5/ckpts/dp_her_round1 \\
      --steps 30000 --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import safetensors.torch as st
from huggingface_hub import hf_hub_download

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", type=str, required=True)
    p.add_argument("--policy-id", type=str, default="lerobot/diffusion_pusht")
    p.add_argument("--out-dir", type=str, default="/root/step5/ckpts/dp_her_round1")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=64,
                   help="per-GPU batch size; effective batch = batch * world_size")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--mix-expert-frac", type=float, default=0.5,
                   help="fraction of batch drawn from LeRobot expert (anti-forgetting)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class HERBufferDataset(Dataset):
    """Sample windows from the HER buffer, applying DP's normalization."""

    def __init__(self, buffer_path: str, img_mean, img_std, st_max, st_min, a_max, a_min):
        b = torch.load(buffer_path, map_location="cpu", weights_only=False)
        self.pixels = b["pixels"]    # (N, T_obs, H, W, 3) uint8
        self.states = b["states"]    # (N, T_obs, 2)
        self.actions = b["actions"]  # (N, horizon, 2)
        self.img_mean = img_mean
        self.img_std = img_std
        self.st_max = st_max
        self.st_min = st_min
        self.a_max = a_max
        self.a_min = a_min
        print(f"[buffer] {len(self.pixels)} windows, pixels {self.pixels.shape}")

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx: int):
        # uint8 HWC → float CHW [0, 1]
        pix = torch.from_numpy(self.pixels[idx]).float() / 255.0   # (T_obs, H, W, 3)
        pix = pix.permute(0, 3, 1, 2).contiguous()                  # (T_obs, 3, H, W)
        # ImageNet-style normalization.
        pix = (pix - self.img_mean) / self.img_std
        # State: min-max → [-1, 1]
        st_ = torch.from_numpy(self.states[idx]).float()
        st_ = 2.0 * (st_ - self.st_min) / (self.st_max - self.st_min) - 1.0
        # Actions: min-max → [-1, 1]
        a_ = torch.from_numpy(self.actions[idx]).float()
        a_ = 2.0 * (a_ - self.a_min) / (self.a_max - self.a_min) - 1.0
        return {
            "observation.image": pix,                                  # (T_obs, 3, H, W) — single-camera key
            "observation.state": st_,                                   # (T_obs, 2)
            "action": a_,                                               # (horizon, 2)
            "action_is_pad": torch.zeros(a_.size(0), dtype=torch.bool), # (horizon,)
        }


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def main():
    args = parse_args()
    rank, world, local = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local}")
    torch.manual_seed(args.seed + rank)

    if is_main:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        print(f"=== rank=0/{world} world={world} device={device} ===")

    # Load pretrained DP.
    policy = DiffusionPolicy.from_pretrained(args.policy_id).to(device)
    if is_main:
        print(f"[policy] loaded {args.policy_id}, horizon={policy.config.horizon}")
    pol_sd = st.load_file(hf_hub_download(args.policy_id, "model.safetensors"))
    img_mean = pol_sd["normalize_inputs.buffer_observation_image.mean"].view(1, 3, 1, 1)
    img_std  = pol_sd["normalize_inputs.buffer_observation_image.std"].view(1, 3, 1, 1)
    st_max   = pol_sd["normalize_inputs.buffer_observation_state.max"]
    st_min   = pol_sd["normalize_inputs.buffer_observation_state.min"]
    a_max    = pol_sd["unnormalize_outputs.buffer_action.max"]
    a_min    = pol_sd["unnormalize_outputs.buffer_action.min"]

    # Dataset (HER buffer for now; expert mix-in TBD if regression observed).
    dataset = HERBufferDataset(
        args.buffer, img_mean, img_std, st_max, st_min, a_max, a_min,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    # Wrap the whole policy in DDP so policy.forward → diffusion.compute_loss
    # routes through DDP correctly.
    if world > 1:
        policy_ddp = DDP(policy, device_ids=[local], output_device=local,
                         find_unused_parameters=True)
    else:
        policy_ddp = policy
    diffusion = policy.diffusion  # underlying module for ckpt saving

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)

    step = 0
    t0 = time.time()
    epoch = 0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            policy_ddp.train()
            loss, _ = policy_ddp(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()

            if is_main and step % 100 == 0:
                dt = time.time() - t0
                print(f"step={step:>6}  loss={loss.item():.5f}  dt={dt:.1f}s", flush=True)

            if is_main and step > 0 and step % args.save_every == 0:
                ckpt_path = Path(args.out_dir) / f"dp_her_step{step}.pt"
                state = diffusion.state_dict()
                torch.save({"step": step, "diffusion": state, "loss": loss.item()}, ckpt_path)
                print(f"[ckpt] saved {ckpt_path}", flush=True)

            step += 1
            if step >= args.steps:
                break
        epoch += 1

    if is_main:
        ckpt_path = Path(args.out_dir) / f"dp_her_step{args.steps}_final.pt"
        state = diffusion.state_dict()
        torch.save({"step": step, "diffusion": state, "loss": loss.item()}, ckpt_path)
        print(f"[ckpt] saved final {ckpt_path}")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
