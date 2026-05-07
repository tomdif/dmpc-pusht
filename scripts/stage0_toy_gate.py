"""Stage 0 toy-gate driver for CSER-JEPA-v3.

Three gates, all general-purpose (no spec-path-specific signals):

  shuffle ratio   : pred MSE under shuffled actions / pred MSE under true.
                    >> 1 means model uses the action.
  time-shift      : same but with action sequence rolled by 1.
                    >> 1 means model is time-coherent in action use.
  Probe Z R²      : R² of an MLP fit (z_t, z_target) -> a_t. Tests whether
                    the encoder is action-grounded — i.e. whether action
                    information is decodable from the pre/post-step
                    embedding pair. This is the only gate that probes the
                    encoder; it caught the mean-pool-on-sparse-signal
                    pathology in v2 attempt D.
"""

from __future__ import annotations

import argparse

import torch
import yaml

from cserjepa_v2.data import ToyConfig, build_toy_batches
from cserjepa_v2.models import CSERJEPAv2
from cserjepa_v2.training import Stage0Config, Stage0Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/toy_v3.yaml")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--diag-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def _pred_obs_loss(model, labeled):
    fe = model.encode(labeled.x_context)
    z = model.encode(labeled.x_target)
    z_pred, _ = model.predict(fe, labeled.a_chunk)
    return ((z_pred - z) ** 2).mean(dim=-1).mean().item()


def _shuffle_canary(model, labeled):
    clean = _pred_obs_loss(model, labeled)
    perm = torch.randperm(labeled.a_chunk.size(0), device=labeled.a_chunk.device)
    shuffled_batch = type(labeled)(
        x_context=labeled.x_context, x_target=labeled.x_target, a_chunk=labeled.a_chunk[perm]
    )
    shuffled = _pred_obs_loss(model, shuffled_batch)
    return shuffled / max(clean, 1e-8)


def _time_shift_canary(model, labeled, delta=1):
    clean = _pred_obs_loss(model, labeled)
    shifted = torch.roll(labeled.a_chunk, shifts=delta, dims=0)
    shifted_batch = type(labeled)(
        x_context=labeled.x_context, x_target=labeled.x_target, a_chunk=shifted
    )
    s = _pred_obs_loss(model, shifted_batch)
    return s / max(clean, 1e-8)


def _train_probe_mlp(inputs, targets, *, hidden=128, depth=2, epochs=60, val_split=0.2, device="cpu", seed=0):
    torch.manual_seed(seed)
    n = inputs.size(0)
    n_val = max(8, int(n * val_split))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_tr, y_tr = inputs[tr_idx].to(device), targets[tr_idx].to(device)
    x_va, y_va = inputs[val_idx].to(device), targets[val_idx].to(device)
    mu = x_tr.mean(0, keepdim=True)
    std = x_tr.std(0, keepdim=True).clamp_min(1e-6)
    layers = [torch.nn.Linear(inputs.size(-1), hidden), torch.nn.GELU()]
    for _ in range(depth - 1):
        layers += [torch.nn.Linear(hidden, hidden), torch.nn.GELU()]
    layers.append(torch.nn.Linear(hidden, targets.size(-1)))
    net = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = min(256, x_tr.size(0))
    for _ in range(epochs):
        order = torch.randperm(x_tr.size(0), device=device)
        for i in range(0, x_tr.size(0), bs):
            sl = order[i:i + bs]
            xb = (x_tr[sl] - mu) / std
            loss = ((net(xb) - y_tr[sl]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pv = net((x_va - mu) / std)
    ss_res = ((pv - y_va) ** 2).sum(0)
    ss_tot = (y_va.var(0, unbiased=False).clamp_min(1e-12) * y_va.size(0))
    r2 = (1.0 - ss_res / ss_tot.clamp_min(1e-12)).mean().item()
    return r2


@torch.no_grad()
def _collect_probe_z_data(model, toy_cfg, n_batches, batch_size, eval_seed, device):
    a_list, zt_list, znext_list = [], [], []
    for i in range(n_batches):
        labeled = build_toy_batches(toy_cfg, n=batch_size, device=device, seed=eval_seed + i)
        fe = model.encode(labeled.x_context)
        z = model.encode(labeled.x_target)
        z_t = fe[:, -1, :] if fe.dim() == 3 else fe
        a_list.append(labeled.a_chunk.flatten(1).cpu())
        zt_list.append(z_t.cpu())
        znext_list.append(z.cpu())
    return {
        "a": torch.cat(a_list, dim=0),
        "z_t": torch.cat(zt_list, dim=0),
        "z_next": torch.cat(znext_list, dim=0),
    }


def _run_diagnostics(model, toy_cfg, batch_size, device, eval_seed, n_eval=8, n_probe=16):
    model.eval()
    shuffles, tshifts = [], []
    for i in range(n_eval):
        labeled = build_toy_batches(toy_cfg, n=batch_size, device=device, seed=eval_seed + i)
        shuffles.append(_shuffle_canary(model, labeled))
        tshifts.append(_time_shift_canary(model, labeled))
    sr = sum(shuffles) / len(shuffles)
    ts = sum(tshifts) / len(tshifts)

    pdat = _collect_probe_z_data(model, toy_cfg, n_probe, batch_size, eval_seed + 1000, device)
    z_pair = torch.cat([pdat["z_t"], pdat["z_next"]], dim=-1)
    pz_r2 = _train_probe_mlp(z_pair, pdat["a"], epochs=80, device=device, seed=eval_seed)

    g_shuffle = sr > 1.5
    g_tshift = ts > 1.3
    g_probe_z = pz_r2 > 0.7

    print(f"  shuffle ratio        : {sr:.3f}x   {'PASS' if g_shuffle else 'FAIL'}")
    print(f"  time-shift ratio     : {ts:.3f}x   {'PASS' if g_tshift else 'FAIL'}")
    print(f"  Probe Z R² (z_t,z→a) : {pz_r2:+.3f}     {'PASS' if g_probe_z else 'FAIL'}  [encoder-grounding]")
    if g_shuffle and g_tshift and g_probe_z:
        print("  [STAGE 0 PASS] all gates clear; safe to advance.")
    else:
        print("  [STAGE 0 FAIL] do not advance.")
    model.train()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.steps is not None:
        cfg["trainer"]["total_steps"] = args.steps

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"=== device: {device} ===")
    print(f"=== α={cfg['data']['alpha']}, ρ={cfg['data']['rho']}, blob_sigma={cfg['data']['blob_sigma']} ===")

    model = CSERJEPAv2(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.2f}M params")

    trainer = Stage0Trainer(model, Stage0Config(**cfg["trainer"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    toy_cfg = ToyConfig(**cfg["data"])
    bs = cfg["batch"]["size"]
    total_steps = cfg["trainer"]["total_steps"]

    for step in range(total_steps):
        labeled = build_toy_batches(toy_cfg, n=bs, device=device, seed=args.seed * 1_000_000 + step)
        optimizer.zero_grad(set_to_none=True)
        loss, diags = trainer.step(labeled, global_step=step)
        if not torch.isfinite(loss):
            print(f"[fatal] non-finite loss at step {step}: {loss.item()}")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["clip_grad"])
        optimizer.step()

        if step % args.log_every == 0:
            line = (
                f"step={step:>5}  L={loss.item():+.4f}  "
                f"pred={diags['loss/pred'].item():.4f}  "
                f"rew={diags['loss/reward'].item():.4f}  "
                f"idmz={diags['loss/idm_z'].item():.4f}  "
                f"reg={diags['loss/reg'].item():.4f}"
            )
            print(line, flush=True)

        if (step + 1) % args.diag_every == 0 or step == total_steps - 1:
            print(f"\n--- diag @ step {step} ---")
            _run_diagnostics(model, toy_cfg, bs, device, eval_seed=10_000 + step)
            print()

    print("\n=== final ===")
    _run_diagnostics(model, toy_cfg, bs, device, eval_seed=99_000, n_eval=16, n_probe=32)


if __name__ == "__main__":
    main()
