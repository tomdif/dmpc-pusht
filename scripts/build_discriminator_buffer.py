"""Build a per-window success/fail buffer for discriminator training.

Each window: (obs[t-1:t+1], action[t:t+8]) → label

Label strategy: per-WINDOW outcome — did the agent's coverage at end-of-window
reach the success threshold within the next 16 steps? This is finer-grained
than episode-level labeling and gives the classifier intra-episode signal.

Output: pixels (uint8), states (float32), actions (float32), labels (int8).
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
    p.add_argument("--out-path", type=str, default="/root/step5/her_buffer/discriminator.pt")
    p.add_argument("--n-obs-steps", type=int, default=2)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--lookahead-steps", type=int, default=24,
                   help="window is positive if max coverage in next L steps reaches threshold")
    p.add_argument("--success-threshold", type=float, default=0.95)
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(f"{args.rollouts_dir}/episode_*.pt"))
    print(f"scanning {len(paths)} rollouts")

    pos_wins = {"pixels": [], "states": [], "actions": []}
    neg_wins = {"pixels": [], "states": [], "actions": []}

    for path in paths:
        ep = torch.load(path, map_location="cpu", weights_only=False)
        T_steps = ep["actions"].shape[0]
        rewards = ep["rewards"]   # per-step coverage as reward
        # Per-window labels: window starting at obs index t covers t-(n_obs-1)..t.
        # We label as positive if coverage in [t, t+lookahead] reaches threshold.
        for t in range(args.n_obs_steps - 1, T_steps - args.n_action_steps + 1):
            future_end = min(t + args.lookahead_steps, T_steps)
            future_max = float(rewards[t : future_end].max()) if future_end > t else 0.0
            label = 1 if future_max >= args.success_threshold else 0
            obs_pix = ep["pixels"][t - args.n_obs_steps + 1 : t + 1]   # (n_obs, H, W, 3) uint8
            obs_st  = ep["states"][t - args.n_obs_steps + 1 : t + 1]
            act     = ep["actions"][t : t + args.n_action_steps]
            (pos_wins if label else neg_wins)["pixels"].append(obs_pix)
            (pos_wins if label else neg_wins)["states"].append(obs_st)
            (pos_wins if label else neg_wins)["actions"].append(act)

    n_pos = len(pos_wins["pixels"])
    n_neg = len(neg_wins["pixels"])
    print(f"  positives: {n_pos}, negatives: {n_neg}, ratio: {n_pos/(n_pos+n_neg):.2%}")

    # Stack and save with both classes.
    out = {
        "pixels":  np.concatenate([np.stack(pos_wins["pixels"]), np.stack(neg_wins["pixels"])], axis=0),
        "states":  np.concatenate([np.stack(pos_wins["states"]), np.stack(neg_wins["states"])], axis=0),
        "actions": np.concatenate([np.stack(pos_wins["actions"]), np.stack(neg_wins["actions"])], axis=0),
        "labels":  np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)], axis=0),
        "n_pos": n_pos, "n_neg": n_neg,
        "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps,
        "lookahead_steps": args.lookahead_steps,
        "success_threshold": args.success_threshold,
    }
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_path, pickle_protocol=4)
    print(f"[saved] {args.out_path}, pixels {out['pixels'].shape} ({out['pixels'].nbytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
