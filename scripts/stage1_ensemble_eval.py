"""Action-averaging ensemble eval across multiple checkpoints.

At each replan step, every member's CEM produces an action chunk; we average
the first actions across members to get the executed action. Members with
different specialties (T2 broad, U/W deep, Z/AB on hard configs) combine —
the averaging smooths over disagreement and amplifies consensus.
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
    p.add_argument("--ensemble", type=str, required=True,
                   help="comma-separated list of MEMBER specs as "
                        "config_path:reward_ckpt:bc_ckpt:use_value:init_std (use_value=0/1)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--cem-samples", type=int, default=256)
    p.add_argument("--cem-elite", type=int, default=32)
    p.add_argument("--cem-iters", type=int, default=4)
    p.add_argument("--replan-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _obs_to_tensor(obs_pixels: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(obs_pixels).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"=== device: {device} ===", flush=True)

    members = []
    for spec in args.ensemble.split(","):
        parts = spec.strip().split(":")
        cfg_path, reward_ckpt, bc_ckpt, use_value_str, init_std_str = parts
        members.append({
            "cfg_path": cfg_path,
            "reward_ckpt": reward_ckpt,
            "bc_ckpt": bc_ckpt if bc_ckpt and bc_ckpt != "_" else None,
            "use_value": bool(int(use_value_str)),
            "init_std": float(init_std_str),
        })
    print(f"[ensemble] {len(members)} members", flush=True)

    # Load each member: cfg, model, BC, planner, action stats.
    member_runtime = []
    image_size_canon = None
    T_ctx_canon = None
    K_canon = None
    action_mean_canon = None
    action_std_canon = None

    for i, m in enumerate(members):
        cfg = yaml.safe_load(open(m["cfg_path"]))
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
        ck = torch.load(m["reward_ckpt"], map_location=device, weights_only=False)
        # Filter mismatched-shape keys (e.g. value_head present in some, not others).
        sd = ck["model"]
        cur = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        model.eval()
        print(f"[ensemble] member {i} reward ckpt: {m['reward_ckpt']}", flush=True)

        planner = CEMPlanner(
            model,
            CEMConfig(
                horizon=args.horizon, n_samples=args.cem_samples, n_elite=args.cem_elite,
                n_iters=args.cem_iters, init_std=m["init_std"],
                use_value=m["use_value"],
            ),
            action_mean=action_mean, action_std=action_std,
        )

        bc = None
        if m["bc_ckpt"]:
            bc_ck = torch.load(m["bc_ckpt"], map_location=device, weights_only=False)
            bc = BCPolicy(**bc_ck["cfg"]).to(device)
            bc.load_state_dict(bc_ck["model"])
            bc.eval()
            print(f"[ensemble] member {i} bc ckpt: {m['bc_ckpt']}", flush=True)

        member_runtime.append({
            "model": model, "planner": planner, "bc": bc,
            "action_mean": action_mean, "action_std": action_std,
        })
        if image_size_canon is None:
            image_size_canon = image_size
            T_ctx_canon = T_ctx
            K_canon = K
            action_mean_canon = action_mean
            action_std_canon = action_std

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    returns, max_covs, successes = [], [], []
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        x0 = _obs_to_tensor(obs["pixels"]).to(device)
        buf = [x0.clone() for _ in range(T_ctx_canon)]
        ep_return = 0.0
        ep_max = 0.0
        ep_success = False
        steps = 0
        while steps < args.max_steps:
            x_ctx = torch.stack(buf, dim=0).unsqueeze(0)

            # Each member plans independently.
            actions_per_member = []
            for mr in member_runtime:
                fe = mr["model"].encode(x_ctx)
                bc = mr["bc"]
                prior = (lambda fe_: bc(fe_)) if bc is not None else None
                a_chunk = mr["planner"].plan(fe, prior_fn=prior).squeeze(0)  # (K, d_a) — RAW
                actions_per_member.append(a_chunk)

            # Average across members. Stack -> (M, K, d_a) -> mean -> (K, d_a).
            a_avg = torch.stack(actions_per_member, dim=0).mean(dim=0)
            a_avg = a_avg.clamp(0.0, 512.0)

            n_apply = K_canon if args.replan_every is None else min(args.replan_every, K_canon)
            for k in range(n_apply):
                if steps >= args.max_steps:
                    break
                a_np = a_avg[k].detach().cpu().numpy().astype(np.float32)
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
        print(f"  ep {ep}: return={ep_return:+.3f}  max_cov={ep_max:.3f}  success={ep_success}  steps={steps}",
              flush=True)
        returns.append(ep_return)
        max_covs.append(ep_max)
        successes.append(int(ep_success))

    print(f"\n=== ensemble summary over {args.n_episodes} episodes ===", flush=True)
    print(f"  mean return  : {np.mean(returns):+.3f}", flush=True)
    print(f"  mean max_cov : {np.mean(max_covs):.3f}", flush=True)
    print(f"  success rate : {np.mean(successes):.2%}", flush=True)


if __name__ == "__main__":
    main()
