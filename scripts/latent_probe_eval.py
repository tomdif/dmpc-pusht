"""Latent probing eval — encoder-quality benchmark vs JEPA-family papers.

Trains a linear probe and an MLP probe to predict ground-truth state
(agent_xy, block_xytheta) from the frozen encoder output. Comparable to
the LeWM/DINO-WM/PLDM 'Latent probing' eval in the CS25 deck.

Reports per-target MSE on held-out frames (normalized so MSE ≈ 1 - R²).
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
import yaml

from cserjepa_v2.models import CSERJEPAv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--rollout-dir", type=str, default="/workspace/cser-jepa-v2/rollouts/self_play_af",
                   help="must contain episodes with agent_positions + block_poses")
    p.add_argument("--n-frames", type=int, default=4000, help="frames sampled for probe data")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--label", type=str, default="ckpt")
    return p.parse_args()


@torch.no_grad()
def collect_probe_data(model, rollout_dir, n_frames, device):
    """Returns (z [N, D], targets [N, 5]) where targets = [agent_x, agent_y, bx, by, btheta]."""
    paths = sorted(glob.glob(os.path.join(rollout_dir, "episode_*.pt")))
    if not paths:
        raise ValueError(f"no episodes in {rollout_dir}")
    zs, ys = [], []
    n_collected = 0
    for path in paths:
        if n_collected >= n_frames:
            break
        ep = torch.load(path, map_location="cpu", weights_only=False)
        if "agent_positions" not in ep:
            continue
        T = ep["frames"].size(0)
        # Sample 8 frames per episode for diversity.
        idx = torch.linspace(0, T - 1, 8).long().tolist()
        for t in idx:
            if n_collected >= n_frames:
                break
            x = ep["frames"][t].to(device).unsqueeze(0)         # (1, C, H, W)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z = model.encode(x).float().squeeze(0).cpu()      # (D,)
            target = torch.cat([
                ep["agent_positions"][t].float(),
                ep["block_poses"][t].float(),
            ], dim=0)                                              # (5,)
            zs.append(z)
            ys.append(target)
            n_collected += 1
    Z = torch.stack(zs, dim=0)
    Y = torch.stack(ys, dim=0)
    return Z, Y


def train_probe(z_train, y_train, z_val, y_val, *, kind: str, epochs: int = 80, device: str = "cpu"):
    """kind: 'linear' or 'mlp'. Returns dict of per-dim MSE on val."""
    D_in = z_train.size(-1)
    D_out = y_train.size(-1)
    if kind == "linear":
        net = torch.nn.Linear(D_in, D_out).to(device)
    else:
        net = torch.nn.Sequential(
            torch.nn.Linear(D_in, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 256), torch.nn.GELU(),
            torch.nn.Linear(256, D_out),
        ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    z_train, y_train = z_train.to(device), y_train.to(device)
    z_val, y_val = z_val.to(device), y_val.to(device)
    bs = min(256, z_train.size(0))
    for _ in range(epochs):
        order = torch.randperm(z_train.size(0), device=device)
        for i in range(0, z_train.size(0), bs):
            sl = order[i:i + bs]
            pred = net(z_train[sl])
            loss = ((pred - y_train[sl]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pv = net(z_val)
    mse_per_dim = ((pv - y_val) ** 2).mean(dim=0).cpu().tolist()
    overall = float(((pv - y_val) ** 2).mean().item())
    var_per_dim = y_val.var(dim=0, unbiased=False).cpu().tolist()
    r2_per_dim = [1.0 - m / max(v, 1e-12) for m, v in zip(mse_per_dim, var_per_dim)]
    return {"mse_per_dim": mse_per_dim, "r2_per_dim": r2_per_dim, "overall_mse": overall,
            "overall_r2": 1.0 - overall / max(float(y_val.var(dim=0, unbiased=False).mean().item()), 1e-12)}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device)
    print(f"=== latent probe: {args.label} ===", flush=True)
    print(f"  ckpt: {args.ckpt}", flush=True)

    cfg["model"]["encoder"]["img_size"] = 96
    cfg["model"]["action"]["d_a"] = 2
    model = CSERJEPAv2(cfg["model"]).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"]
    cur = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"  d={model.d}", flush=True)

    Z, Y = collect_probe_data(model, args.rollout_dir, args.n_frames, device)
    print(f"  collected {Z.size(0)} (z, target) pairs, target dim={Y.size(-1)}", flush=True)

    # Standardize targets per-dim.
    Y_mean = Y.mean(dim=0, keepdim=True)
    Y_std = Y.std(dim=0, keepdim=True).clamp_min(1e-6)
    Y_norm = (Y - Y_mean) / Y_std

    n = Z.size(0)
    n_val = int(0.2 * n)
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    z_train, y_train = Z[tr_idx], Y_norm[tr_idx]
    z_val, y_val = Z[val_idx], Y_norm[val_idx]

    # Standardize z too (linear probe sensitive to scale).
    z_mean = z_train.mean(dim=0, keepdim=True)
    z_std = z_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    z_train = (z_train - z_mean) / z_std
    z_val = (z_val - z_mean) / z_std

    for kind in ["linear", "mlp"]:
        res = train_probe(z_train, y_train, z_val, y_val, kind=kind, epochs=args.epochs, device=device.type)
        print(f"\n  {kind.upper()} probe:", flush=True)
        labels = ["agent_x", "agent_y", "block_x", "block_y", "block_theta"]
        for lbl, mse, r2 in zip(labels, res["mse_per_dim"], res["r2_per_dim"]):
            print(f"    {lbl:<14} MSE={mse:.4f}  R²={r2:+.4f}", flush=True)
        print(f"    {'overall':<14} MSE={res['overall_mse']:.4f}  R²={res['overall_r2']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
