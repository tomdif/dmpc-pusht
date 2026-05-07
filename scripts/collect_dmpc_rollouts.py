"""Collect D-MPC rollouts in parallel across multiple GPUs for Step 5 (HER + Dyna).

Distributes episodes across N GPUs via multiprocessing — one process per GPU,
each runs a contiguous slice of episode seeds. Saves per-episode trajectories
to disk for downstream HER relabeling.

Each saved episode contains:
  pixels:  (T+1, 3, 96, 96) uint8 frames including final
  states:  (T+1, 2)         float32 agent_pos
  actions: (T,   2)         float32 raw action commands executed
  rewards: (T,)             float32 per-step env reward (= coverage)
  max_cov: float            episode-level max coverage
  success: bool
  seed:    int

Output: <out-dir>/episode_<seed:05d>.pt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import yaml

from huggingface_hub import hf_hub_download
import safetensors.torch as st

from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader
from cserjepa_v2.models import CSERJEPAv2
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--world-config", type=str, required=True)
    p.add_argument("--world-ckpt", type=str, required=True)
    p.add_argument("--policy-id", type=str, default="lerobot/diffusion_pusht")
    p.add_argument("--out-dir", type=str, default="/root/step5/rollouts/round1")
    p.add_argument("--n-episodes", type=int, default=1500)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--n-gpus", type=int, default=6)
    # D-MPC reranker config — match our 74% setup.
    p.add_argument("--use-value", action="store_true", default=True)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--use-goal", action="store_true", default=True)
    p.add_argument("--goal-weight", type=float, default=0.3)
    p.add_argument("--goal-multi", action="store_true", default=True)
    p.add_argument("--goal-file", type=str, required=True)
    return p.parse_args()


def _img_to_tensor(pixels: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pixels).float().permute(2, 0, 1).contiguous() / 255.0


def worker(rank: int, args, ep_seeds: list[int]):
    """Per-GPU worker: runs a slice of episodes on rank-th GPU."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.out_dir) / f"worker_{rank}.log"
    log = open(log_path, "w")

    def _log(msg):
        line = f"[gpu{rank}] {msg}"
        log.write(line + "\n"); log.flush()
        print(line, flush=True)

    _log(f"start; episodes {ep_seeds[0]}..{ep_seeds[-1]} ({len(ep_seeds)} total)")

    # World model.
    cfg = yaml.safe_load(open(args.world_config))
    data_section = cfg["data"]
    if "lerobot" in data_section and isinstance(data_section["lerobot"], dict):
        data_section = data_section["lerobot"]
    data_cfg = LeRobotConfig(**data_section)
    loader, a_dim, image_size = build_lerobot_loader(
        data_cfg, batch_size=1, num_workers=0, shuffle=False,
    )
    windows = loader.dataset
    action_mean = windows.action_mean.to(device).flatten()
    action_std = windows.action_std.to(device).flatten()
    cfg["model"]["encoder"]["img_size"] = image_size
    cfg["model"]["action"]["d_a"] = a_dim
    T_ctx = int(data_cfg.context_len)
    world = CSERJEPAv2(cfg["model"]).to(device)
    ck = torch.load(args.world_ckpt, map_location=device, weights_only=False)
    sd = ck["model"]; cur = world.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    world.load_state_dict(filtered, strict=False)
    world.eval()
    _log("world model loaded")

    # Diffusion policy.
    policy = DiffusionPolicy.from_pretrained(args.policy_id).to(device)
    policy.eval()
    pol_sd = st.load_file(hf_hub_download(args.policy_id, "model.safetensors"))
    img_mean = pol_sd["normalize_inputs.buffer_observation_image.mean"].to(device)
    img_std  = pol_sd["normalize_inputs.buffer_observation_image.std"].to(device)
    st_max   = pol_sd["normalize_inputs.buffer_observation_state.max"].to(device)
    st_min   = pol_sd["normalize_inputs.buffer_observation_state.min"].to(device)
    a_max    = pol_sd["unnormalize_outputs.buffer_action.max"].to(device)
    a_min    = pol_sd["unnormalize_outputs.buffer_action.min"].to(device)
    n_obs_steps = policy.config.n_obs_steps
    horizon_p = policy.config.horizon
    _log("policy loaded")

    # Goal embeddings.
    gd = torch.load(args.goal_file, map_location=device, weights_only=False)
    cf = gd["candidate_frames"].to(device)
    z_cands = world.encode(cf)
    z_goal = z_cands if args.goal_multi else z_cands.mean(dim=0)

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    K_world = world.chunk_size
    max_ctx = world.predictor.frame_pos_embed.size(1)

    @torch.no_grad()
    def plan(dp_pixels, dp_states, world_buf):
        img_stack = torch.stack(dp_pixels, dim=0).to(device)
        img_stack = (img_stack - img_mean) / img_std
        state_stack = torch.stack(dp_states, dim=0).to(device)
        state_stack = 2.0 * (state_stack - st_min) / (st_max - st_min) - 1.0
        batch = {
            "observation.images": img_stack.unsqueeze(0).unsqueeze(2),
            "observation.state": state_stack.unsqueeze(0),
        }
        global_cond = policy.diffusion._prepare_global_conditioning(batch)
        global_cond = global_cond.expand(args.n_samples, -1).contiguous()
        actions_norm = policy.diffusion.conditional_sample(
            batch_size=args.n_samples, global_cond=global_cond,
        )
        start = n_obs_steps - 1
        actions_norm_chunk = actions_norm[:, start : start + args.n_action_steps]
        actions_raw = ((actions_norm_chunk + 1.0) / 2.0 * (a_max - a_min) + a_min).clamp(0.0, 512.0)
        actions_world = (actions_raw - action_mean) / action_std

        # Rerank.
        world_ctx_imgs = torch.stack(world_buf, dim=0).unsqueeze(0).to(device)
        fe = world.encode(world_ctx_imgs)
        ctx = fe.expand(args.n_samples, -1, -1).contiguous()
        scores = torch.zeros(args.n_samples, device=device)
        z_final = None
        goal_dists = []
        n_chunks = args.n_action_steps // K_world
        for h in range(n_chunks):
            a_h = actions_world[:, h * K_world : (h + 1) * K_world]
            z_h, r_h = world.predict(ctx, a_h)
            scores = scores + (r_h.sum(dim=-1) if r_h.dim() > 1 and r_h.size(-1) > 1 else r_h.squeeze(-1))
            if z_goal.dim() == 1:
                d2 = (z_h - z_goal.unsqueeze(0)).pow(2).mean(dim=-1)
            else:
                diff = z_h.unsqueeze(1) - z_goal.unsqueeze(0)
                d2 = diff.pow(2).mean(dim=-1).min(dim=-1).values
            goal_dists.append(d2)
            ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx: ctx = ctx[:, -max_ctx:]
            z_final = z_h
        if args.use_value and z_final is not None:
            scores = scores + args.value_weight * world.predictor.value(z_final)
        agg = torch.stack(goal_dists, dim=0).min(dim=0).values
        scores = scores - args.goal_weight * agg
        best = scores.argmax().item()
        return actions_raw[best]

    n_done = 0
    t0 = time.time()
    for ep_seed in ep_seeds:
        ep_path = out_dir / f"episode_{ep_seed:05d}.pt"
        if ep_path.exists():
            n_done += 1; continue

        obs, info = env.reset(seed=ep_seed)
        x0 = _img_to_tensor(obs["pixels"])
        s0 = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
        dp_pixels = [x0.clone() for _ in range(n_obs_steps)]
        dp_states = [s0.clone() for _ in range(n_obs_steps)]
        world_buf = [x0.clone() for _ in range(T_ctx)]

        # Trajectory buffers.
        # Store pixels as uint8 (factor 4× space saving) — convert on load.
        traj_pixels = [(obs["pixels"]).astype(np.uint8)]            # (H, W, 3) HWC uint8
        traj_states = [s0.numpy()]                                   # (2,)
        traj_actions = []                                            # (2,) per step
        traj_rewards = []                                            # scalar per step
        ep_max_cov = 0.0; ep_success = False; steps = 0

        while steps < args.max_steps:
            a_chunk = plan(dp_pixels, dp_states, world_buf).cpu().numpy().astype(np.float32)
            for k in range(args.n_action_steps):
                if steps >= args.max_steps: break
                obs, r, terminated, truncated, info = env.step(a_chunk[k])
                ep_max_cov = max(ep_max_cov, float(info.get("coverage", 0.0)))
                if info.get("is_success", False): ep_success = True
                steps += 1

                traj_pixels.append((obs["pixels"]).astype(np.uint8))
                traj_states.append(np.asarray(obs["agent_pos"], dtype=np.float32))
                traj_actions.append(a_chunk[k].copy())
                traj_rewards.append(float(r))

                xt = _img_to_tensor(obs["pixels"])
                st_ = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
                dp_pixels = dp_pixels[1:] + [xt]
                dp_states = dp_states[1:] + [st_]
                world_buf = world_buf[1:] + [xt]
                if terminated or truncated: break
            if terminated or truncated: break

        # Stack & save (uint8 frames in HWC).
        torch.save({
            "pixels": np.stack(traj_pixels, axis=0),                # (T+1, H, W, 3) uint8
            "states": np.stack(traj_states, axis=0),                # (T+1, 2) float32
            "actions": np.stack(traj_actions, axis=0) if traj_actions else np.zeros((0, 2), np.float32),
            "rewards": np.asarray(traj_rewards, dtype=np.float32),
            "max_cov": ep_max_cov,
            "success": ep_success,
            "seed": ep_seed,
            "steps": steps,
        }, ep_path)
        n_done += 1
        if n_done % 25 == 0 or n_done == len(ep_seeds):
            dt = time.time() - t0
            _log(f"  done {n_done}/{len(ep_seeds)} (last seed={ep_seed}, max_cov={ep_max_cov:.3f}, succ={ep_success}, dt={dt:.0f}s)")

    log.close()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    # Round-robin shard across GPUs.
    shards = [seeds[i :: args.n_gpus] for i in range(args.n_gpus)]
    print(f"=== launching {args.n_gpus} workers, {args.n_episodes} total eps ===", flush=True)
    for i, s in enumerate(shards):
        print(f"  gpu{i}: {len(s)} eps (seeds {s[0]}..{s[-1]})", flush=True)

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(i, args, shards[i])) for i in range(args.n_gpus)]
    for p in procs: p.start()
    for p in procs: p.join()
    print("=== all workers done ===")


if __name__ == "__main__":
    main()
