"""Train a small behavior-cloning policy on top of a frozen world-model encoder.

The policy takes (frame_embeds) and predicts the action chunk in normalized
action space. It serves as a CEM proposal prior — search refines around BC,
agent doesn't have to discover engagement from scratch.
"""

from __future__ import annotations

import argparse
import time

import torch
import yaml

from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader
from cserjepa_v2.models import BCPolicy, CSERJEPAv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True, help="Stage-1B world model ckpt (encoder source)")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device)
    print(f"=== device: {device} ===", flush=True)

    # Support both flat (data: {repo_id: ...}) and nested (data: {lerobot: {...}, synthetic: {...}})
    # config layouts. BC trains on canonical expert actions only, so always pull the lerobot half.
    data_section = cfg["data"]
    if "lerobot" in data_section and isinstance(data_section["lerobot"], dict):
        data_section = data_section["lerobot"]
    data_cfg = LeRobotConfig(**data_section)
    loader, a_dim, image_size = build_lerobot_loader(
        data_cfg,
        batch_size=cfg["batch"]["size"],
        num_workers=args.num_workers,
        shuffle=True,
    )
    cfg["model"]["encoder"]["img_size"] = image_size
    cfg["model"]["action"]["d_a"] = a_dim

    # Load world model and pin its encoder.
    wm = CSERJEPAv2(cfg["model"]).to(device)
    print(f"[ckpt] loading {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    wm.load_state_dict(ck["model"])
    wm.eval()
    for p in wm.encoder.parameters():
        p.requires_grad_(False)

    d = wm.d
    K = wm.chunk_size
    T_ctx = int(data_section["context_len"])
    bc = BCPolicy(
        d=d, d_a=a_dim, chunk_size=K, context_len=T_ctx,
        hidden=args.hidden, depth=args.depth,
    ).to(device)
    n_params = sum(p.numel() for p in bc.parameters())
    print(f"[bc] {n_params/1e6:.2f}M params", flush=True)

    optimizer = torch.optim.AdamW(bc.parameters(), lr=args.lr, weight_decay=1e-4)

    step = 0
    last_log_t = time.time()
    print(f"[train] starting, total_steps={args.steps}", flush=True)
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            x_ctx = batch.x_context.to(device, non_blocking=True)
            a_chunk = batch.a_chunk.to(device, non_blocking=True)  # (B, K, d_a)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    fe = wm.encode(x_ctx)                          # (B, T_ctx, D)
            optimizer.zero_grad(set_to_none=True)
            pred = bc(fe.float())                                  # (B, K, d_a)
            loss = (pred - a_chunk).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc.parameters(), 1.0)
            optimizer.step()

            if step % args.log_every == 0:
                dt = time.time() - last_log_t
                last_log_t = time.time()
                # Variance baseline: action_chunk has unit variance per dim
                # after normalization. R² ≈ 1 - loss.
                print(f"step={step:>5}  bc_mse={loss.item():.4f}  R²≈{1.0 - loss.item():.3f}  dt={dt:.2f}s",
                      flush=True)
            step += 1

    torch.save({
        "step": step,
        "model": bc.state_dict(),
        "cfg": {"d": d, "d_a": a_dim, "chunk_size": K, "context_len": T_ctx,
                "hidden": args.hidden, "depth": args.depth},
    }, args.out)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
