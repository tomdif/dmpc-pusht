"""Extract a high-coverage goal frame from LeRobot expert pusht episodes.

The existing goal frame (extracted from self-play) has max_coverage=0.875,
below the 95% success threshold. LeRobot expert episodes ALL reach the
goal, so their final frames are the natural "goal" reference.
"""

from __future__ import annotations

import argparse

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", type=str, default="lerobot/pusht_image")
    p.add_argument("--image-key", type=str, default="observation.image")
    p.add_argument("--reward-key", type=str, default="next.reward")
    p.add_argument("--top-k", type=int, default=8,
                   help="average top-K-reward final frames")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"loading {args.repo_id}...")
    ds = LeRobotDataset(args.repo_id)
    print(f"  total frames: {len(ds)}")
    print(f"  episodes: {ds.num_episodes}")

    # Rank ALL frames by reward (= coverage). Many high-coverage frames
    # don't have next.success=True if the env didn't terminate at that exact
    # step — but their pixels are still high-coverage goal references.
    hf = ds.hf_dataset
    rewards = hf[args.reward_key]
    cands = [(float(r), i) for i, r in enumerate(rewards)]
    cands.sort(key=lambda x: -x[0])
    print(f"\nTop-{args.top_k} frames by reward (coverage):")
    top_frames = []
    for r, frame_idx in cands[: args.top_k]:
        sample = ds[frame_idx]
        img = sample[args.image_key]  # (C, H, W) float32 in [0, 1]
        top_frames.append(img)
        ep_idx = int(sample["episode_index"])
        print(f"  reward={r:.4f}  ep={ep_idx}  frame_idx={frame_idx}  img={tuple(img.shape)}")
    candidates = cands

    out = {
        "goal_frame": top_frames[0],
        "candidate_frames": torch.stack(top_frames, dim=0),
        "max_coverage": float(candidates[0][0]),
        "source": "lerobot_expert_final_frames",
    }
    torch.save(out, args.out)
    print(f"\n[saved] {args.out}")
    print(f"  best final reward: {out['max_coverage']:.4f}")
    print(f"  candidate_frames shape: {out['candidate_frames'].shape}")


if __name__ == "__main__":
    main()
