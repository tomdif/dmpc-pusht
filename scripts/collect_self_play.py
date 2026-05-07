"""Self-play data collection in gym-pusht.

Roll out episodes using a mix of strategies (BC+noise, pure random) and
save (frames, actions, rewards) to disk in a format the SyntheticWindows
class can consume. Quality-filters by max coverage so we keep only
informative episodes.
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import yaml

from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader
from cserjepa_v2.models import BCPolicy, CSERJEPAv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--bc-ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--noise-std", type=float, default=0.4,
                   help="exploration noise added to BC actions in normalized space")
    p.add_argument("--random-frac", type=float, default=0.25,
                   help="fraction of episodes that use pure random actions")
    p.add_argument("--min-max-coverage", type=float, default=0.02,
                   help="discard episodes whose max coverage is below this — pure noise filter")
    p.add_argument("--seed-base", type=int, default=10000)
    return p.parse_args()


def _obs_to_tensor(obs_pixels: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(obs_pixels).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = torch.device(args.device)
    print(f"=== device: {device} ===", flush=True)

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

    model = CSERJEPAv2(cfg["model"]).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"]
    cur = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"[ckpt] loaded {args.ckpt}", flush=True)

    bc_ck = torch.load(args.bc_ckpt, map_location=device, weights_only=False)
    bc = BCPolicy(**bc_ck["cfg"]).to(device)
    bc.load_state_dict(bc_ck["model"])
    bc.eval()
    print(f"[bc] loaded {args.bc_ckpt}", flush=True)

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    os.makedirs(args.out, exist_ok=True)
    n_kept = 0
    n_random = int(args.random_frac * args.n_episodes)
    saved_max_cov = []
    for ep in range(args.n_episodes):
        use_random = ep < n_random
        seed = args.seed_base + ep
        obs, info = env.reset(seed=seed)
        buf = [_obs_to_tensor(obs["pixels"]).to(device) for _ in range(T_ctx)]

        frames = []
        actions = []
        rewards = []
        agent_positions = []  # (2,)
        block_poses = []       # (3,) — x, y, theta
        max_cov = 0.0
        for step_i in range(args.max_steps):
            x_t = buf[-1]
            frames.append(x_t.cpu())
            # Capture state labels for proximity reward training.
            agent_positions.append(torch.tensor(info.get("pos_agent", [0.0, 0.0]), dtype=torch.float32))
            block_pose_raw = info.get("block_pose", [0.0, 0.0, 0.0])
            block_poses.append(torch.tensor(block_pose_raw, dtype=torch.float32))

            if use_random:
                a_unnorm_np = env.action_space.sample()
            else:
                x_ctx = torch.stack(buf, dim=0).unsqueeze(0)
                fe = model.encode(x_ctx)
                a_norm = bc(fe).squeeze(0)                      # (1, d_a)? actually (chunk, d_a) — chunk=8
                # Use just the first action of the chunk; add Gaussian noise.
                a0_norm = a_norm[0] + args.noise_std * torch.randn_like(a_norm[0])
                a0_unnorm = a0_norm * action_std + action_mean
                a0_unnorm = a0_unnorm.clamp(0.0, 512.0)
                a_unnorm_np = a0_unnorm.detach().cpu().numpy().astype(np.float32)

            actions.append(torch.from_numpy(a_unnorm_np.copy()))
            obs, r, term, trunc, info = env.step(a_unnorm_np)
            rewards.append(float(r))
            max_cov = max(max_cov, float(info.get("coverage", 0.0)))
            xt = _obs_to_tensor(obs["pixels"]).to(device)
            buf = buf[1:] + [xt]
            if term or trunc:
                break

        if max_cov < args.min_max_coverage and not use_random:
            continue

        # We always keep random episodes (they form the exploration backbone).
        ep_data = {
            "frames": torch.stack(frames, dim=0),                    # (T, C, H, W) float32 in [0, 1]
            "actions": torch.stack(actions, dim=0).float(),           # (T, d_a) — RAW unnormalized
            "rewards": torch.tensor(rewards, dtype=torch.float32),    # (T,)
            "agent_positions": torch.stack(agent_positions, dim=0),   # (T, 2)
            "block_poses": torch.stack(block_poses, dim=0),           # (T, 3)
            "policy": "random" if use_random else "bc_noise",
            "max_coverage": max_cov,
            "seed": seed,
        }
        torch.save(ep_data, os.path.join(args.out, f"episode_{n_kept:05d}.pt"))
        saved_max_cov.append(max_cov)
        n_kept += 1
        if (ep + 1) % 50 == 0:
            print(f"[collect] {ep+1}/{args.n_episodes}  kept={n_kept}  "
                  f"recent_max_cov={max_cov:.3f}  policy={'random' if use_random else 'bc_noise'}",
                  flush=True)

    saved_arr = np.array(saved_max_cov) if saved_max_cov else np.array([0.0])
    print(f"\n=== self-play done ===", flush=True)
    print(f"  collected {n_kept} episodes (of {args.n_episodes} attempts)", flush=True)
    print(f"  max_coverage distribution: mean={saved_arr.mean():.3f}  "
          f"median={np.median(saved_arr):.3f}  max={saved_arr.max():.3f}  "
          f"frac>0.1={(saved_arr > 0.1).mean():.2f}  frac>0.3={(saved_arr > 0.3).mean():.2f}", flush=True)
    print(f"  output: {args.out}", flush=True)


if __name__ == "__main__":
    main()
