"""BC-only eval on seed 102 (the one D-MPC + K=20 cannot crack).

Uses our trained BCPolicy directly as the proposer — bypassing DP entirely.
BC was trained on the same expert demos but has a totally different inductive
bias (MLP regression to action chunks). If DP's mode collapse on seed 102
is architecture-specific, BC might find a path DP can't.
"""
from __future__ import annotations

import argparse
import sys, pathlib

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader  # noqa: E402
from cserjepa_v2.models import BCPolicy, CSERJEPAv2  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-config", default="configs/stage1ab_reward.yaml")
    ap.add_argument("--world-ckpt", default="ckpts/stage_AB/reward/ckpt_step5000.pt")
    ap.add_argument("--bc-ckpt", default="ckpts/stage_AB/bc_policy_AB.pt")
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--n-attempts", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()
    device = torch.device("cuda")

    cfg = yaml.safe_load(open(args.world_config))
    data_section = cfg["data"]
    if "lerobot" in data_section: data_section = data_section["lerobot"]
    data_cfg = LeRobotConfig(**data_section)
    loader, a_dim, image_size = build_lerobot_loader(data_cfg, batch_size=1, num_workers=0, shuffle=False)
    windows = loader.dataset
    action_mean = windows.action_mean.to(device).flatten()
    action_std = windows.action_std.to(device).flatten()
    cfg["model"]["encoder"]["img_size"] = image_size; cfg["model"]["action"]["d_a"] = a_dim
    T_ctx = int(data_cfg.context_len)

    world = CSERJEPAv2(cfg["model"]).to(device).eval()
    ck = torch.load(args.world_ckpt, map_location=device, weights_only=False)
    sd = ck["model"]; cur = world.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    world.load_state_dict(filtered, strict=False)

    bc_ck = torch.load(args.bc_ckpt, map_location=device, weights_only=False)
    bc = BCPolicy(**bc_ck["cfg"]).to(device).eval()
    bc.load_state_dict(bc_ck["model"])
    K_bc = bc.chunk_size

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    print(f"=== seed {args.seed}, BC-only proposer, K={args.n_attempts} attempts ===")
    best_max_cov = 0.0; best_success = False
    for attempt in range(args.n_attempts):
        torch.manual_seed(args.seed * 31 + 13 * attempt)
        obs, info = env.reset(seed=args.seed)
        x0 = torch.from_numpy(obs["pixels"]).float().permute(2,0,1) / 255.0
        world_buf = [x0.clone() for _ in range(T_ctx)]
        ep_max_cov = 0.0; ep_success = False; steps = 0
        while steps < args.max_steps:
            x_ctx = torch.stack(world_buf, dim=0).unsqueeze(0).to(device)
            fe = world.encode(x_ctx)
            a_norm = bc(fe).squeeze(0)                                    # (K_bc, d_a) normalized
            a_raw = (a_norm * action_std + action_mean).clamp(0.0, 512.0)
            for k in range(K_bc):
                if steps >= args.max_steps: break
                obs, r, term, trunc, info = env.step(a_raw[k].cpu().numpy().astype(np.float32))
                ep_max_cov = max(ep_max_cov, float(info.get("coverage", 0.0)))
                if info.get("is_success", False): ep_success = True
                steps += 1
                xt = torch.from_numpy(obs["pixels"]).float().permute(2,0,1)/255.0
                world_buf = world_buf[1:] + [xt]
                if term or trunc: break
            if term or trunc: break
        print(f"  attempt {attempt}: max_cov={ep_max_cov:.3f}  success={ep_success}  steps={steps}")
        if ep_max_cov > best_max_cov:
            best_max_cov = ep_max_cov; best_success = ep_success
        if best_success: break
    print(f"\nbest: max_cov={best_max_cov:.3f}  success={best_success}")


if __name__ == "__main__":
    main()
