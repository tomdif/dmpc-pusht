"""Stage 1B planner evaluation (offline smoke).

Loads the Stage-1B reward-fine-tuned checkpoint, runs CEM planning on
held-out pusht trajectories using ONLY the world model (no env), and
reports:

  - planner-vs-expert action distance (in normalized action space)
  - predicted total reward (from world model rollout) vs ground-truth
    total reward over the same horizon

These numbers don't say "the agent solves pusht" — they say "the world
model + reward head + planner pipeline is internally consistent and the
planner's choices correlate with the expert's." That is the right
smoke-test signal before plugging into a real environment loop.
"""

from __future__ import annotations

import argparse

import torch
import yaml

from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader
from cserjepa_v2.models import CSERJEPAv2
from cserjepa_v2.planning import CEMPlanner, CEMConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-trajectories", type=int, default=8)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--cem-iters", type=int, default=4)
    p.add_argument("--cem-samples", type=int, default=256)
    p.add_argument("--cem-elite", type=int, default=32)
    p.add_argument("--cem-init-std", type=float, default=1.0)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = torch.device(args.device)
    print(f"=== device: {device} ===", flush=True)

    data_cfg = LeRobotConfig(**cfg["data"])
    loader, a_dim, image_size = build_lerobot_loader(
        data_cfg, batch_size=1, num_workers=0, shuffle=True,
    )
    cfg["model"]["encoder"]["img_size"] = image_size
    cfg["model"]["action"]["d_a"] = a_dim

    # Recover normalization stats from the dataset.
    windows = loader.dataset
    action_mean = windows.action_mean.to(device).flatten()
    action_std = windows.action_std.to(device).flatten()

    model = CSERJEPAv2(cfg["model"]).to(device)
    print(f"[ckpt] loading {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    planner = CEMPlanner(
        model,
        CEMConfig(
            horizon=args.horizon,
            n_samples=args.cem_samples,
            n_elite=args.cem_elite,
            n_iters=args.cem_iters,
            init_std=args.cem_init_std,
        ),
        action_mean=action_mean,
        action_std=action_std,
    )

    print(f"[plan] horizon={args.horizon} samples={args.cem_samples} elite={args.cem_elite} iters={args.cem_iters}", flush=True)

    # Compare planner choices vs expert on held-out windows.
    # Note: r_chunk in the loader is in raw scale (not normalized).
    K = int(cfg["model"]["action"]["chunk_size"])
    a_dists, pred_returns, true_returns = [], [], []
    n = 0
    for batch in loader:
        if n >= args.n_trajectories:
            break
        x_ctx = batch.x_context.to(device)
        x_tgt = batch.x_target.to(device)
        a_expert = batch.a_chunk.to(device)            # already normalized by loader
        r_expert = batch.r_chunk.to(device)            # raw rewards

        # Encode context
        fe = model.encode(x_ctx)                       # (1, T_ctx, D)

        # CEM plan
        a_planned = planner.plan(fe)                    # (1, K, d_a) — UNNORMALIZED
        a_planned_norm = (a_planned - action_mean.view(1, 1, -1)) / action_std.view(1, 1, -1)

        # Distance in normalized action space (apples-to-apples with expert).
        a_dist = (a_planned_norm - a_expert).pow(2).mean().item() ** 0.5

        # Predicted return from world model under the planned action chunk.
        z_pred, r_pred = model.predict(fe, a_planned_norm)  # (1, D), (1, 1)
        pred_return = r_pred.squeeze().item()
        true_return = r_expert.sum().item()

        a_dists.append(a_dist)
        pred_returns.append(pred_return)
        true_returns.append(true_return)
        print(f"  traj {n:>2}: ‖a_plan - a_expert‖={a_dist:.3f}   "
              f"r̂(plan)={pred_return:+.3f}   r(expert chunk)={true_return:+.3f}", flush=True)
        n += 1

    print(f"\n=== summary over {n} trajectories ===", flush=True)
    a_arr = torch.tensor(a_dists)
    pr = torch.tensor(pred_returns)
    tr = torch.tensor(true_returns)
    print(f"  mean ‖a_plan - a_expert‖     : {a_arr.mean().item():.3f}", flush=True)
    print(f"  mean predicted-plan return    : {pr.mean().item():+.3f}", flush=True)
    print(f"  mean true-expert chunk return : {tr.mean().item():+.3f}", flush=True)
    if pr.std() > 1e-6 and tr.std() > 1e-6:
        cov = ((pr - pr.mean()) * (tr - tr.mean())).mean()
        corr = cov / (pr.std() * tr.std())
        print(f"  corr(predicted, true)         : {corr.item():+.3f}", flush=True)


if __name__ == "__main__":
    main()
