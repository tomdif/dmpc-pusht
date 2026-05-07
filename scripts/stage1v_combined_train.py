"""Stage 1 (V) self-play combined-data training driver.

Same architecture, same losses as stage1_pretrain.py — only the data
source changes. Trains on original LeRobot pusht + self-play rollouts
collected by collect_self_play.py. Both halves share action stats so
normalization is consistent.
"""

from __future__ import annotations

import argparse
import time

import torch
import yaml

from cserjepa_v2.data import LeRobotConfig, SyntheticConfig, build_combined_loader
from cserjepa_v2.models import CSERJEPAv2
from cserjepa_v2.training import Stage0Config, Stage0Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--diag-every", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ckpt-every", type=int, default=2500)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def _shuffle_and_pred_loss(model, batch, device):
    model.eval()
    fe = model.encode(batch.x_context.to(device))
    z = model.encode(batch.x_target.to(device))
    a = batch.a_chunk.to(device)
    z_pred, _ = model.predict(fe, a)
    clean = ((z_pred - z) ** 2).mean().item()
    perm = torch.randperm(a.size(0), device=device)
    z_pred_s, _ = model.predict(fe, a[perm])
    shuffled = ((z_pred_s - z) ** 2).mean().item()
    model.train()
    return shuffled / max(clean, 1e-8), clean


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.steps is not None:
        cfg["trainer"]["total_steps"] = args.steps

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"=== device: {device} ===", flush=True)

    le_cfg = LeRobotConfig(**cfg["data"]["lerobot"])
    syn_cfg = SyntheticConfig(**cfg["data"]["synthetic"])

    loader, a_dim, image_size, lerobot_w = build_combined_loader(
        le_cfg, syn_cfg,
        batch_size=cfg["batch"]["size"],
        num_workers=args.num_workers,
        shuffle=True,
    )
    print(f"[data] action_dim={a_dim}, image_size={image_size}, total_windows={len(loader.dataset)}", flush=True)

    cfg["model"]["encoder"]["img_size"] = image_size
    cfg["model"]["action"]["d_a"] = a_dim

    model = CSERJEPAv2(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params", flush=True)

    trainer = Stage0Trainer(model, Stage0Config(**cfg["trainer"]))
    base_lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, betas=tuple(cfg["optimizer"]["betas"]),
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    warmup_steps = int(cfg["optimizer"].get("warmup_steps", 0))
    min_lr_ratio = float(cfg["optimizer"].get("min_lr_ratio", 1.0))
    total_steps_for_sched = int(cfg["trainer"]["total_steps"])

    def lr_at(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        if min_lr_ratio < 1.0:
            import math
            t = min(1.0, (step - warmup_steps) / max(1, total_steps_for_sched - warmup_steps))
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)
        return base_lr

    use_amp = bool(cfg["optimizer"].get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16

    if args.resume:
        print(f"[ckpt] loading {args.resume}", flush=True)
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ck["model"]
        # Drop keys whose shape doesn't match the current model — handles
        # head reshapes (e.g. reward_head Linear(d,1) → Linear(d,K)).
        cur = model.state_dict()
        dropped = []
        filtered = {}
        for k, v in sd.items():
            if k in cur and cur[k].shape != v.shape:
                dropped.append((k, tuple(v.shape), tuple(cur[k].shape)))
            else:
                filtered[k] = v
        info = model.load_state_dict(filtered, strict=False)
        if dropped:
            print(f"[ckpt] dropped {len(dropped)} mismatched-shape keys:", flush=True)
            for k, sh_old, sh_new in dropped:
                print(f"  {k}: {sh_old} → {sh_new}", flush=True)
        if info.missing_keys or info.unexpected_keys:
            print(f"[ckpt] missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}",
                  flush=True)
        try:
            optimizer.load_state_dict(ck["optimizer"])
        except Exception as e:
            print(f"[ckpt] optimizer state mismatch: {e}", flush=True)

    total_steps = cfg["trainer"]["total_steps"]
    step = 0
    last_log_t = time.time()
    print(f"[train] starting, total_steps={total_steps}, amp={use_amp}", flush=True)

    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            batch = type(batch)(
                x_context=batch.x_context.to(device, non_blocking=True),
                x_target=batch.x_target.to(device, non_blocking=True),
                a_chunk=batch.a_chunk.to(device, non_blocking=True),
                r_chunk=batch.r_chunk.to(device, non_blocking=True) if batch.r_chunk is not None else None,
                rtg_target=batch.rtg_target.to(device, non_blocking=True) if batch.rtg_target is not None else None,
                rtg_mask=batch.rtg_mask.to(device, non_blocking=True) if batch.rtg_mask is not None else None,
                state_target=batch.state_target.to(device, non_blocking=True) if batch.state_target is not None else None,
                state_mask=batch.state_mask.to(device, non_blocking=True) if batch.state_mask is not None else None,
                x_target_multi=batch.x_target_multi.to(device, non_blocking=True) if batch.x_target_multi is not None else None,
                a_chunk_multi=batch.a_chunk_multi.to(device, non_blocking=True) if batch.a_chunk_multi is not None else None,
            )
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss, diags = trainer.step(batch, global_step=step)
            else:
                loss, diags = trainer.step(batch, global_step=step)
            if not torch.isfinite(loss):
                print(f"[fatal] non-finite loss at step {step}", flush=True)
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["clip_grad"])
            optimizer.step()

            if step % args.log_every == 0:
                dt = time.time() - last_log_t
                last_log_t = time.time()
                line = (
                    f"step={step:>6}  L={loss.item():+.4f}  "
                    f"pred={diags['loss/pred'].item():.4f}  "
                    f"idmz={diags['loss/idm_z'].item():.4f}  "
                    f"reg={diags['loss/reg'].item():.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                    f"dt={dt:.2f}s"
                )
                print(line, flush=True)

            if (step + 1) % args.diag_every == 0:
                sr, clean = _shuffle_and_pred_loss(model, batch, device)
                print(f"\n--- diag @ step {step} ---", flush=True)
                print(f"  shuffle ratio (live batch): {sr:.3f}x   pred MSE={clean:.5f}", flush=True)
                print(flush=True)

            if args.ckpt_dir and (step + 1) % args.ckpt_every == 0:
                import os
                os.makedirs(args.ckpt_dir, exist_ok=True)
                path = os.path.join(args.ckpt_dir, f"ckpt_step{step + 1}.pt")
                torch.save({"step": step + 1, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "cfg": cfg}, path)
                print(f"[ckpt] saved {path}", flush=True)

            step += 1

    print("\n=== final ===", flush=True)
    sr_list, clean_list = [], []
    for j, batch in enumerate(loader):
        if j >= 16:
            break
        batch = type(batch)(
            x_context=batch.x_context.to(device), x_target=batch.x_target.to(device),
            a_chunk=batch.a_chunk.to(device),
        )
        sr, c = _shuffle_and_pred_loss(model, batch, device)
        sr_list.append(sr); clean_list.append(c)
    print(f"  shuffle ratio (mean over 16 batches): {sum(sr_list)/len(sr_list):.3f}x", flush=True)
    print(f"  clean pred MSE                      : {sum(clean_list)/len(clean_list):.5f}", flush=True)


if __name__ == "__main__":
    main()
