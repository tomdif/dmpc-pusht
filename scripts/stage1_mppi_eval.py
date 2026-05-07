"""Online eval with MPPI planner instead of CEM. Same reward / BC / env loop.

MPPI substitution: importance-weighted Gaussian update instead of CEM's
top-K elite refit. Less greedy, better suited to over-confident world
models that the bigger Y/X checkpoints surface.
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
from cserjepa_v2.planning import MPPIConfig, MPPIPlanner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--bc-ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--n-iters", type=int, default=4)
    p.add_argument("--init-std", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--replan-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
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
    T_ctx = int(data_section["context_len"])
    K = int(cfg["model"]["action"]["chunk_size"])

    model = CSERJEPAv2(cfg["model"]).to(device)
    print(f"[ckpt] loading {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    planner = MPPIPlanner(
        model,
        MPPIConfig(
            horizon=args.horizon, n_samples=args.n_samples, n_iters=args.n_iters,
            init_std=args.init_std, temperature=args.temperature,
        ),
        action_mean=action_mean,
        action_std=action_std,
    )
    print(f"[plan] MPPI H={args.horizon} N={args.n_samples} iters={args.n_iters} τ={args.temperature}", flush=True)

    bc = None
    if args.bc_ckpt:
        bc_ck = torch.load(args.bc_ckpt, map_location=device, weights_only=False)
        bc = BCPolicy(**bc_ck["cfg"]).to(device)
        bc.load_state_dict(bc_ck["model"])
        bc.eval()
        print(f"[bc] loaded {args.bc_ckpt}", flush=True)

    def prior_fn(frame_embeds): return bc(frame_embeds)

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    returns, max_covs, successes = [], [], []
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        x0 = _obs_to_tensor(obs["pixels"]).to(device)
        buf = [x0.clone() for _ in range(T_ctx)]
        ep_return = 0.0
        ep_max = 0.0
        ep_success = False
        steps = 0
        while steps < args.max_steps:
            x_ctx = torch.stack(buf, dim=0).unsqueeze(0)
            fe = model.encode(x_ctx)
            a_chunk = planner.plan(fe, prior_fn=prior_fn if bc is not None else None).squeeze(0)
            a_chunk = a_chunk.clamp(0.0, 512.0)
            n_apply = K if args.replan_every is None else min(args.replan_every, K)
            for k in range(n_apply):
                if steps >= args.max_steps:
                    break
                a_np = a_chunk[k].detach().cpu().numpy().astype(np.float32)
                obs, r, term, trunc, info = env.step(a_np)
                ep_return += float(r)
                ep_max = max(ep_max, float(info.get("coverage", 0.0)))
                if info.get("is_success", False):
                    ep_success = True
                steps += 1
                xt = _obs_to_tensor(obs["pixels"]).to(device)
                buf = buf[1:] + [xt]
                if term or trunc:
                    break
            if term or trunc:
                break
        print(f"  ep {ep}: return={ep_return:+.3f}  max_cov={ep_max:.3f}  success={ep_success}  steps={steps}", flush=True)
        returns.append(ep_return)
        max_covs.append(ep_max)
        successes.append(int(ep_success))

    print(f"\n=== summary over {args.n_episodes} episodes ===", flush=True)
    print(f"  mean return  : {np.mean(returns):+.3f}", flush=True)
    print(f"  mean max_cov : {np.mean(max_covs):.3f}", flush=True)
    print(f"  success rate : {np.mean(successes):.2%}", flush=True)


if __name__ == "__main__":
    main()
