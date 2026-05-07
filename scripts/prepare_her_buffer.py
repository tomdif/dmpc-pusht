"""Prepare HER+Dyna training buffer from D-MPC rollouts.

For non-goal-conditional Diffusion Policy, "HER" reduces to "filtered behavior
cloning of self-generated successful trajectories" — the policy learns from
its OWN best rollouts, with the original LeRobot expert demos mixed in for
anti-forgetting.

Strategy: keep trajectories with max_cov ≥ keep_threshold. Each trajectory
becomes a sequence of (obs, action_chunk) windows matching DP's expected
input format (n_obs_steps observations, n_action_steps actions).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts-dir", type=str, default="/root/step5/rollouts/round1")
    p.add_argument("--out-path", type=str, default="/root/step5/her_buffer/round1.pt")
    p.add_argument("--keep-threshold", type=float, default=0.85,
                   help="keep trajectories with max_cov >= this")
    p.add_argument("--n-obs-steps", type=int, default=2)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--horizon", type=int, default=16,
                   help="DP's horizon — we extract (obs[t-n_obs+1:t+1], action[t:t+horizon]) windows")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(f"{args.rollouts_dir}/episode_*.pt"))
    print(f"scanning {len(paths)} rollouts")

    kept_eps = 0
    skipped_eps = 0
    successes = 0
    windows = {
        "pixels": [],         # list of (n_obs, H, W, 3) uint8
        "states": [],         # list of (n_obs, 2) float32
        "actions": [],        # list of (horizon, 2) float32
        "max_cov": [],        # per-window source max_cov (for weighting)
    }

    for path in paths:
        ep = torch.load(path, map_location="cpu", weights_only=False)
        if ep["max_cov"] < args.keep_threshold:
            skipped_eps += 1; continue
        if ep["success"]:
            successes += 1
        kept_eps += 1
        T_steps = ep["actions"].shape[0]
        # We need at least n_obs frames AND horizon actions remaining.
        # pixels has T_steps+1 frames (obs at each step + final).
        for t in range(args.n_obs_steps - 1, T_steps - args.horizon + 1):
            obs_pix = ep["pixels"][t - args.n_obs_steps + 1 : t + 1]   # (n_obs, H, W, 3)
            obs_st  = ep["states"][t - args.n_obs_steps + 1 : t + 1]   # (n_obs, 2)
            act_window = ep["actions"][t : t + args.horizon]            # (horizon, 2)
            windows["pixels"].append(obs_pix)
            windows["states"].append(obs_st)
            windows["actions"].append(act_window)
            windows["max_cov"].append(ep["max_cov"])

    print(f"kept {kept_eps} / {len(paths)} eps ({successes} formal successes)")
    print(f"  → {len(windows['pixels'])} training windows")

    if not windows["pixels"]:
        raise SystemExit("No windows extracted — nothing to save.")

    out = {
        "pixels":  np.stack(windows["pixels"], axis=0),    # (N, n_obs, H, W, 3) uint8
        "states":  np.stack(windows["states"], axis=0),    # (N, n_obs, 2)
        "actions": np.stack(windows["actions"], axis=0),   # (N, horizon, 2)
        "max_cov": np.asarray(windows["max_cov"], dtype=np.float32),
        "n_obs_steps": args.n_obs_steps,
        "horizon": args.horizon,
        "kept_eps": kept_eps,
        "successes": successes,
    }
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_path, pickle_protocol=4)
    print(f"[saved] {args.out_path}")
    print(f"  pixels: {out['pixels'].shape} {out['pixels'].dtype}")
    print(f"  total bytes: {out['pixels'].nbytes / 1e9:.2f} GB (uint8)")


if __name__ == "__main__":
    main()
