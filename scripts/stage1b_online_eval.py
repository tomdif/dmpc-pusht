"""Stage 1B online evaluation in gym-pusht.

Runs episodes in the actual PushT environment using the world model + CEM
planner. Each episode:
  1. reset env, get initial observation (pixels)
  2. seed context buffer by replicating frame 0 (T_ctx times)
  3. loop:
     a. encode buffer to (1, T_ctx, D)
     b. CEM plan -> first action chunk (K actions, in raw action space)
     c. execute the K actions one-by-one, accumulating env reward
     d. push the new frame into the buffer (sliding T_ctx)
     e. re-plan
  4. record episode return + success

This is the canonical era-of-experience smoke test: agent acts in
environment using ONLY its world model + planner, and earns real return.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import gym_pusht  # noqa: F401 — registers the env
import numpy as np
import torch
import yaml

from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader
from cserjepa_v2.models import BCPolicy, CSERJEPAv2, PixelDecoder
from cserjepa_v2.planning import CEMConfig, CEMPlanner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--cem-iters", type=int, default=4)
    p.add_argument("--cem-samples", type=int, default=256)
    p.add_argument("--cem-elite", type=int, default=32)
    p.add_argument("--cem-init-std", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bc-ckpt", type=str, default=None,
                   help="optional BC policy ckpt to use as CEM proposal prior")
    p.add_argument("--replan-every", type=int, default=None,
                   help="execute this many actions per plan before replanning. "
                        "Default = chunk_size (1 plan per chunk). "
                        "Set to 1 for per-step replanning (more reactive, slower).")
    p.add_argument("--use-value", action="store_true",
                   help="bootstrap CEM score with V(z_final) (Stage Z onward)")
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--use-proximity", action="store_true",
                   help="add per-step proximity bonus (Stage AF onward)")
    p.add_argument("--proximity-weight", type=float, default=0.01)
    p.add_argument("--use-goal", action="store_true",
                   help="goal-conditioned planning: subtract -||z_h - z_goal||² (Stage AI)")
    p.add_argument("--goal-weight", type=float, default=1.0)
    p.add_argument("--goal-aggregate", type=str, default="min", choices=["min", "final", "mean"])
    p.add_argument("--goal-drift-weight", type=float, default=0.0,
                   help="extra penalty on final-step goal distance (drift fix)")
    p.add_argument("--goal-multi", action="store_true",
                   help="use all K candidate goal embeddings (best-of-K min) instead of averaging")
    p.add_argument("--goal-file", type=str, default=None,
                   help="path to .pt file with 'candidate_frames' tensor")
    p.add_argument("--stuck-window", type=int, default=0,
                   help="if max_cov hasn't increased by stuck-thresh in this many env steps, "
                        "bump CEM init_std for the next plans (0 = disabled)")
    p.add_argument("--stuck-thresh", type=float, default=0.01,
                   help="min increase in max_cov over the stuck-window")
    p.add_argument("--stuck-init-std", type=float, default=1.5,
                   help="boosted init_std when stuck")
    p.add_argument("--use-bc-anchor", action="store_true",
                   help="penalize CEM elite drift from BC prior (anti-exploitation, Phase 1.5)")
    p.add_argument("--bc-anchor-weight", type=float, default=0.0)
    p.add_argument("--use-pixel-ground", action="store_true",
                   help="pixel-grounded planning: decode imagined latents and add pixel-MSE-to-goal cost")
    p.add_argument("--pixel-ground-weight", type=float, default=10.0)
    p.add_argument("--pixel-ground-aggregate", type=str, default="min", choices=["min", "final", "mean"])
    p.add_argument("--decoder-ckpt", type=str, default=None,
                   help="path to trained PixelDecoder ckpt")
    return p.parse_args()


def _obs_to_tensor(obs_pixels: np.ndarray) -> torch.Tensor:
    """gym-pusht returns (96, 96, 3) uint8. LeRobot dataset produces (3, 96, 96) float32 in [0, 1]."""
    x = torch.from_numpy(obs_pixels).float() / 255.0
    return x.permute(2, 0, 1).contiguous()  # (C, H, W)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = torch.device(args.device)
    print(f"=== device: {device} ===", flush=True)

    # Recover normalization stats from the dataset.
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
    K = int(cfg["model"]["action"]["chunk_size"])

    model = CSERJEPAv2(cfg["model"]).to(device)
    print(f"[ckpt] loading {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Filter shape-mismatched keys (e.g. M ckpt's reward_head Linear(d,1) vs
    # current Linear(d,K), and value_head present only on Z/AB).
    sd = ck["model"]
    cur = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    planner = CEMPlanner(
        model,
        CEMConfig(
            horizon=args.horizon,
            n_samples=args.cem_samples,
            n_elite=args.cem_elite,
            n_iters=args.cem_iters,
            init_std=args.cem_init_std,
            use_value=args.use_value,
            value_weight=args.value_weight,
            use_proximity=args.use_proximity,
            proximity_weight=args.proximity_weight,
            use_goal=args.use_goal,
            goal_weight=args.goal_weight,
            goal_aggregate=args.goal_aggregate,
            goal_drift_weight=args.goal_drift_weight,
            use_bc_anchor=args.use_bc_anchor,
            bc_anchor_weight=args.bc_anchor_weight,
            use_pixel_ground=args.use_pixel_ground,
            pixel_ground_weight=args.pixel_ground_weight,
            pixel_ground_aggregate=args.pixel_ground_aggregate,
        ),
        action_mean=action_mean,
        action_std=action_std,
    )

    z_goal = None
    x_goal = None
    if args.use_goal and args.goal_file:
        gd = torch.load(args.goal_file, map_location=device, weights_only=False)
        cf = gd["candidate_frames"].to(device)            # (K, C, H, W)
        with torch.no_grad():
            z_cands = model.encode(cf)                    # (K, D)
        if args.goal_multi:
            z_goal = z_cands                              # (K, D) — best-of-K
        else:
            z_goal = z_cands.mean(dim=0)                  # (D,) — averaged
        x_goal = gd["goal_frame"].to(device)              # (C, H, W)
        mode = "multi-goal best-of-K" if args.goal_multi else "averaged"
        print(f"[goal] loaded {args.goal_file}, K={cf.size(0)} frames, z_goal {tuple(z_goal.shape)} ({mode})", flush=True)
    print(f"[plan] H={args.horizon} N={args.cem_samples} elite={args.cem_elite} iters={args.cem_iters}", flush=True)

    decoder = None
    if args.use_pixel_ground:
        if not args.decoder_ckpt:
            raise SystemExit("--use-pixel-ground requires --decoder-ckpt")
        dck = torch.load(args.decoder_ckpt, map_location=device, weights_only=False)
        decoder = PixelDecoder(d_z=cfg["model"]["encoder"]["embed_dim"], img_size=image_size).to(device)
        decoder.load_state_dict(dck["decoder"])
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad_(False)
        print(f"[decoder] loaded {args.decoder_ckpt}", flush=True)

    bc = None
    if args.bc_ckpt:
        bc_ck = torch.load(args.bc_ckpt, map_location=device, weights_only=False)
        bc_cfg = bc_ck["cfg"]
        bc = BCPolicy(**bc_cfg).to(device)
        bc.load_state_dict(bc_ck["model"])
        bc.eval()
        print(f"[bc] loaded {args.bc_ckpt}", flush=True)

    def prior_fn(frame_embeds: torch.Tensor) -> torch.Tensor:
        return bc(frame_embeds)

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    returns, successes, max_coverages = [], [], []
    base_init_std = float(planner.cfg.init_std)
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        x0 = _obs_to_tensor(obs["pixels"]).to(device)        # (C, H, W)
        # Seed context: replicate frame 0 T_ctx times.
        buf = [x0.clone() for _ in range(T_ctx)]

        ep_return = 0.0
        ep_max_coverage = 0.0
        ep_success = False
        steps = 0
        random_action_steps = 0
        last_action = None  # for "no replan" debug
        # Stuck-detector state.
        cov_at_step: list[tuple[int, float]] = []
        stuck_active = False
        planner.cfg.init_std = base_init_std
        while steps < args.max_steps:
            # Stuck-detector: if max_cov hasn't risen by stuck_thresh in the
            # last stuck_window env steps, bump init_std to escape local minimum.
            if args.stuck_window > 0:
                cov_at_step.append((steps, ep_max_coverage))
                if cov_at_step[-1][0] - cov_at_step[0][0] >= args.stuck_window:
                    while cov_at_step and steps - cov_at_step[0][0] > args.stuck_window:
                        cov_at_step.pop(0)
                    delta = cov_at_step[-1][1] - cov_at_step[0][1]
                    if delta < args.stuck_thresh and not stuck_active:
                        planner.cfg.init_std = args.stuck_init_std
                        stuck_active = True
                    elif delta >= args.stuck_thresh and stuck_active:
                        planner.cfg.init_std = base_init_std
                        stuck_active = False
            x_ctx = torch.stack(buf, dim=0).unsqueeze(0)      # (1, T_ctx, C, H, W)
            fe = model.encode(x_ctx)                           # (1, T_ctx, D)
            a_chunk = planner.plan(
                fe,
                prior_fn=prior_fn if bc is not None else None,
                z_goal=z_goal,
                decoder=decoder,
                x_goal=x_goal,
            ).squeeze(0)
            # (K, d_a) — RAW action space
            # Clip to env action box.
            a_chunk = a_chunk.clamp(0.0, 512.0)
            n_apply = K if args.replan_every is None else min(args.replan_every, K)
            for k in range(n_apply):
                if steps >= args.max_steps:
                    break
                a_np = a_chunk[k].detach().cpu().numpy().astype(np.float32)
                obs, r, terminated, truncated, info = env.step(a_np)
                ep_return += float(r)
                ep_max_coverage = max(ep_max_coverage, float(info.get("coverage", 0.0)))
                if info.get("is_success", False):
                    ep_success = True
                steps += 1
                # Update buffer.
                xt = _obs_to_tensor(obs["pixels"]).to(device)
                buf = buf[1:] + [xt]
                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        print(f"  ep {ep}: return={ep_return:+.3f}  max_cov={ep_max_coverage:.3f}  "
              f"success={ep_success}  steps={steps}", flush=True)
        returns.append(ep_return)
        successes.append(int(ep_success))
        max_coverages.append(ep_max_coverage)

    print(f"\n=== summary over {args.n_episodes} episodes ===", flush=True)
    print(f"  mean return     : {np.mean(returns):+.3f}", flush=True)
    print(f"  mean max_cov    : {np.mean(max_coverages):.3f}", flush=True)
    print(f"  success rate    : {np.mean(successes):.2%}", flush=True)


if __name__ == "__main__":
    main()
