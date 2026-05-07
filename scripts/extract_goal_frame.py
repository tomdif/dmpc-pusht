"""Extract a single 'goal' frame from self-play episodes.

Heuristic: the highest-coverage frame across all saved self-play rollouts
is the closest visual approximation of the solved state. Save as a tensor.
"""

from __future__ import annotations

import argparse
import glob
import os

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dirs", type=str, nargs="+",
                   default=["/workspace/cser-jepa-v2/rollouts/self_play_af",
                            "/workspace/cser-jepa-v2/rollouts/self_play_z",
                            "/workspace/cser-jepa-v2/rollouts/self_play_u"])
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--top-k", type=int, default=8,
                   help="average top-K-coverage frames as a smoother goal embedding")
    return p.parse_args()


def main():
    args = parse_args()
    paths = []
    for d in args.rollout_dirs:
        paths.extend(sorted(glob.glob(os.path.join(d, "episode_*.pt"))))
    print(f"Scanning {len(paths)} episodes...")
    candidates = []  # (max_cov, path, peak_frame_idx)
    for path in paths:
        ep = torch.load(path, map_location="cpu", weights_only=False)
        # Find the frame with highest reward (proxy for highest coverage at that step).
        rewards = ep["rewards"]
        peak_idx = int(rewards.argmax().item())
        candidates.append((float(rewards.max().item()), path, peak_idx, ep["max_coverage"]))
    candidates.sort(key=lambda x: x[3], reverse=True)  # by max_coverage
    print(f"Top-{args.top_k} episodes by max_coverage:")
    top_frames = []
    for max_r, path, peak_idx, mc in candidates[:args.top_k]:
        ep = torch.load(path, map_location="cpu", weights_only=False)
        top_frames.append(ep["frames"][peak_idx])
        print(f"  cov={mc:.3f}  {os.path.basename(path)}  peak_t={peak_idx}")
    # Goal frame: pick the single highest-coverage frame.
    # (Averaging frames doesn't make sense in pixel space; we'll average
    # the embeddings later inside the eval script.)
    goal = top_frames[0]
    out = {
        "goal_frame": goal,                                  # (C, H, W)
        "candidate_frames": torch.stack(top_frames, dim=0),  # (K, C, H, W) for embedding-averaging
        "max_coverage": candidates[0][3],
    }
    torch.save(out, args.out)
    print(f"\n[saved] {args.out}")
    print(f"  best max_coverage: {candidates[0][3]:.3f}")
    print(f"  candidate_frames shape: {out['candidate_frames'].shape}")


if __name__ == "__main__":
    main()
