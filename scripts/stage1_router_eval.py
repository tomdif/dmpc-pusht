"""AE router eval: AB-primary + Z-fallback based on runtime engagement.

Logic:
  - Run AB (planner + BC) for first --probe-steps actions
  - If cumulative return < --engage-threshold, switch to Z
  - Else commit to AB for the remainder
  - No env reset — agent's actions through probe phase are real

Tests whether runtime engagement signal can recover the seed-103 +89 spike
that AB misses while AB still handles its broader strengths.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import yaml

from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader
from cserjepa_v2.models import BCPolicy, CSERJEPAv2
from cserjepa_v2.planning import CEMConfig, CEMPlanner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ab-config", type=str, required=True)
    p.add_argument("--ab-ckpt", type=str, required=True)
    p.add_argument("--ab-bc-ckpt", type=str, required=True)
    p.add_argument("--z-config", type=str, required=True)
    p.add_argument("--z-ckpt", type=str, required=True)
    p.add_argument("--z-bc-ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--probe-steps", type=int, default=30)
    p.add_argument("--engage-threshold", type=float, default=0.5)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--cem-samples", type=int, default=256)
    p.add_argument("--cem-elite", type=int, default=32)
    p.add_argument("--cem-iters", type=int, default=4)
    p.add_argument("--cem-init-std", type=float, default=0.3)
    p.add_argument("--replan-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _obs_to_tensor(obs_pixels: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(obs_pixels).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def _load_member(cfg_path, ckpt_path, bc_ckpt_path, args, device):
    cfg = yaml.safe_load(open(cfg_path))
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
    T_ctx = int(data_section["context_len"])
    K = int(cfg["model"]["action"]["chunk_size"])

    model = CSERJEPAv2(cfg["model"]).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model"]
    cur = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    planner = CEMPlanner(
        model,
        CEMConfig(
            horizon=args.horizon, n_samples=args.cem_samples, n_elite=args.cem_elite,
            n_iters=args.cem_iters, init_std=args.cem_init_std,
            use_value=True, value_weight=1.0,
        ),
        action_mean=action_mean, action_std=action_std,
    )

    bc_ck = torch.load(bc_ckpt_path, map_location=device, weights_only=False)
    bc = BCPolicy(**bc_ck["cfg"]).to(device)
    bc.load_state_dict(bc_ck["model"])
    bc.eval()

    return {"model": model, "planner": planner, "bc": bc, "T_ctx": T_ctx, "K": K}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"=== device: {device} ===", flush=True)

    print("[router] loading AB (primary)", flush=True)
    AB = _load_member(args.ab_config, args.ab_ckpt, args.ab_bc_ckpt, args, device)
    print("[router] loading Z (fallback)", flush=True)
    Z = _load_member(args.z_config, args.z_ckpt, args.z_bc_ckpt, args, device)
    T_ctx = AB["T_ctx"]
    K = AB["K"]
    print(f"[router] probe={args.probe_steps} threshold={args.engage_threshold}", flush=True)

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    returns, max_covs, switches = [], [], []
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        x0 = _obs_to_tensor(obs["pixels"]).to(device)
        buf = [x0.clone() for _ in range(T_ctx)]
        ep_return = 0.0
        ep_max = 0.0
        steps = 0
        active = "AB"  # start with AB

        while steps < args.max_steps:
            # Check if we should switch from AB to Z based on engagement.
            if active == "AB" and steps >= args.probe_steps and ep_return < args.engage_threshold:
                active = "Z"

            member = AB if active == "AB" else Z
            x_ctx = torch.stack(buf, dim=0).unsqueeze(0)
            fe = member["model"].encode(x_ctx)
            a_chunk = member["planner"].plan(
                fe, prior_fn=(lambda fe_: member["bc"](fe_)),
            ).squeeze(0)
            a_chunk = a_chunk.clamp(0.0, 512.0)

            n_apply = K if args.replan_every is None else min(args.replan_every, K)
            for k in range(n_apply):
                if steps >= args.max_steps:
                    break
                a_np = a_chunk[k].detach().cpu().numpy().astype(np.float32)
                obs, r, term, trunc, info = env.step(a_np)
                ep_return += float(r)
                ep_max = max(ep_max, float(info.get("coverage", 0.0)))
                steps += 1
                xt = _obs_to_tensor(obs["pixels"]).to(device)
                buf = buf[1:] + [xt]
                if term or trunc:
                    break
            if term or trunc:
                break

        switches.append(active)
        print(f"  ep {ep}: return={ep_return:+.3f}  max_cov={ep_max:.3f}  active={active}  steps={steps}",
              flush=True)
        returns.append(ep_return)
        max_covs.append(ep_max)

    print(f"\n=== router summary over {args.n_episodes} episodes ===", flush=True)
    print(f"  mean return  : {np.mean(returns):+.3f}", flush=True)
    print(f"  mean max_cov : {np.mean(max_covs):.3f}", flush=True)
    n_ab = sum(1 for s in switches if s == "AB")
    n_z = sum(1 for s in switches if s == "Z")
    print(f"  active        : AB={n_ab}/10, Z={n_z}/10", flush=True)


if __name__ == "__main__":
    main()
