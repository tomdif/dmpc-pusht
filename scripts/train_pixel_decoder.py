"""Train pixel decoder on top of frozen JEPA encoder.

Loss: |decoder(encoder(x)) - x|^2 over real PushT frames.

The encoder is FROZEN — we only learn how to invert it back to pixels.
This gives us a pixel-space reconstruction we can use at plan time to
catch phantom-optimum latents that don't correspond to coherent images.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW

from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader
from cserjepa_v2.models import CSERJEPAv2
from cserjepa_v2.models.pixel_decoder import PixelDecoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--encoder-ckpt", type=str, required=True,
                   help="path to AB world-model ckpt; encoder will be frozen")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"=== device: {device} ===", flush=True)

    cfg = yaml.safe_load(open(args.config))
    data_section = cfg["data"]
    if "lerobot" in data_section and isinstance(data_section["lerobot"], dict):
        data_section = data_section["lerobot"]
    data_cfg = LeRobotConfig(**data_section)
    loader, a_dim, image_size = build_lerobot_loader(
        data_cfg, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True,
    )
    print(f"[data] image_size={image_size}, total_windows={len(loader.dataset)}", flush=True)

    cfg["model"]["encoder"]["img_size"] = image_size
    cfg["model"]["action"]["d_a"] = a_dim
    model = CSERJEPAv2(cfg["model"]).to(device)

    print(f"[ckpt] loading frozen encoder from {args.encoder_ckpt}", flush=True)
    ck = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    sd = ck["model"]
    cur = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_z = cfg["model"]["encoder"]["embed_dim"]
    decoder = PixelDecoder(d_z=d_z, img_size=image_size).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"[decoder] {n_params/1e6:.2f}M params", flush=True)

    opt = AdamW(decoder.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    iters = iter(loader)
    step = 0
    t0 = time.time()
    decoder.train()
    while step < args.steps:
        try:
            batch = next(iters)
        except StopIteration:
            iters = iter(loader)
            batch = next(iters)

        x_ctx = batch.x_context.to(device)              # (B, T, C, H, W)
        x_tgt = batch.x_target.to(device)               # (B, C, H, W)

        with torch.no_grad():
            z_ctx = model.encode(x_ctx)                  # (B, T, D)
            z_tgt = model.encode(x_tgt)                  # (B, D)
        z_all = torch.cat([z_ctx.flatten(0, 1), z_tgt], dim=0)  # (B*T + B, D)
        x_all = torch.cat([x_ctx.flatten(0, 1), x_tgt], dim=0)  # (B*T + B, C, H, W)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            x_hat = decoder(z_all)
            loss = F.mse_loss(x_hat, x_all)

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()

        if step % 100 == 0:
            dt = time.time() - t0
            print(f"step={step:>5}  loss={loss.item():.6f}  dt={dt:.1f}s", flush=True)
        step += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "step": step,
        "decoder": decoder.state_dict(),
        "cfg": cfg,
    }, args.out)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
