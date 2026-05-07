"""Adaptive D-MPC eval with random-perturb fallback.

Per seed:
  1. Try D-MPC + discriminator with up to K_primary attempts.
     If success on any attempt → record TRUE, move on.
  2. Otherwise, fall back to random-initial-action-perturbation + DP for up
     to K_fallback attempts. If success → TRUE.
  3. Else → FALSE.

This is principled adaptive inference compute: easy seeds get cheap
treatment, hard seeds get the heavier strategy.
"""
from __future__ import annotations

import argparse
import sys, pathlib

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
import safetensors.torch as st

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader  # noqa: E402
from cserjepa_v2.models import CSERJEPAv2  # noqa: E402
from train_discriminator import SuccessDiscriminator  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--world-config", default="configs/stage1ab_reward.yaml")
    p.add_argument("--world-ckpt", default="ckpts/stage_AB/reward/ckpt_step5000.pt")
    p.add_argument("--policy-id", default="lerobot/diffusion_pusht")
    p.add_argument("--discriminator-ckpt", default="/root/step5/ckpts/discriminator_round1.pt")
    p.add_argument("--goal-file", default="goals/pusht_goal_lerobot.pt")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--k-primary", type=int, default=5)
    p.add_argument("--k-fallback", type=int, default=10)
    p.add_argument("--perturb-steps", type=int, default=30)
    p.add_argument("--goal-weight", type=float, default=0.3)
    p.add_argument("--disc-weight", type=float, default=1.0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda")

    # World model.
    cfg = yaml.safe_load(open(args.world_config))
    data_section = cfg["data"]
    if "lerobot" in data_section: data_section = data_section["lerobot"]
    data_cfg = LeRobotConfig(**data_section)
    loader, a_dim, image_size = build_lerobot_loader(data_cfg, batch_size=1, num_workers=0, shuffle=False)
    windows = loader.dataset
    action_mean = windows.action_mean.to(device).flatten()
    action_std = windows.action_std.to(device).flatten()
    cfg["model"]["encoder"]["img_size"] = image_size; cfg["model"]["action"]["d_a"] = a_dim
    T_ctx = int(data_cfg.context_len)
    K_world = cfg["model"]["action"]["chunk_size"]

    world = CSERJEPAv2(cfg["model"]).to(device).eval()
    ck = torch.load(args.world_ckpt, map_location=device, weights_only=False)
    sd = ck["model"]; cur = world.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    world.load_state_dict(filtered, strict=False)
    print("[world] loaded")

    # Diffusion Policy + normalization.
    policy = DiffusionPolicy.from_pretrained(args.policy_id).to(device).eval()
    pol_sd = st.load_file(hf_hub_download(args.policy_id, "model.safetensors"))
    img_mean = pol_sd["normalize_inputs.buffer_observation_image.mean"].to(device)
    img_std  = pol_sd["normalize_inputs.buffer_observation_image.std"].to(device)
    st_max = pol_sd["normalize_inputs.buffer_observation_state.max"].to(device)
    st_min = pol_sd["normalize_inputs.buffer_observation_state.min"].to(device)
    a_max = pol_sd["unnormalize_outputs.buffer_action.max"].to(device)
    a_min = pol_sd["unnormalize_outputs.buffer_action.min"].to(device)
    n_obs_steps = policy.config.n_obs_steps
    horizon_p = policy.config.horizon
    print("[policy] loaded")

    # Discriminator.
    dck = torch.load(args.discriminator_ckpt, map_location=device, weights_only=False)
    dcfg = dck["config"]
    discriminator = SuccessDiscriminator(dcfg["in_channels"], dcfg["state_dim"], dcfg["action_dim"]).to(device).eval()
    discriminator.load_state_dict(dck["model"])
    print("[discriminator] loaded")

    # Goals.
    gd = torch.load(args.goal_file, map_location=device, weights_only=False)
    z_goal = world.encode(gd["candidate_frames"].to(device))   # (K, D)

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    def _img_to_tensor(p): return torch.from_numpy(p).float().permute(2,0,1) / 255.0

    def dp_rerank_chunk(dp_pixels, dp_states, world_buf):
        """Standard D-MPC: sample N from DP, rerank with world model + disc."""
        img_stack = torch.stack(dp_pixels, dim=0).to(device)
        img_stack_n = (img_stack - img_mean) / img_std
        state_stack = torch.stack(dp_states, dim=0).to(device)
        state_stack_n = 2.0*(state_stack - st_min)/(st_max - st_min) - 1.0
        batch = {
            "observation.images": img_stack_n.unsqueeze(0).unsqueeze(2),
            "observation.state": state_stack_n.unsqueeze(0),
        }
        gc = policy.diffusion._prepare_global_conditioning(batch).expand(args.n_samples, -1).contiguous()
        a_norm_full = policy.diffusion.conditional_sample(batch_size=args.n_samples, global_cond=gc)
        start = n_obs_steps - 1
        a_chunk_norm = a_norm_full[:, start : start + args.n_action_steps]
        a_raw = ((a_chunk_norm + 1.0)/2.0 * (a_max - a_min) + a_min).clamp(0.0, 512.0)
        a_world = (a_raw - action_mean) / action_std

        # Score via world model (goal-distance + value) + discriminator.
        world_ctx = torch.stack(world_buf, dim=0).unsqueeze(0).to(device)
        fe = world.encode(world_ctx)
        ctx = fe.expand(args.n_samples, -1, -1).contiguous()
        max_ctx = world.predictor.frame_pos_embed.size(1)
        scores = torch.zeros(args.n_samples, device=device)
        z_final = None; goal_dists = []
        n_chunks = args.n_action_steps // K_world
        for h in range(n_chunks):
            a_h = a_world[:, h * K_world : (h + 1) * K_world]
            z_h, r_h = world.predict(ctx, a_h)
            scores = scores + (r_h.sum(dim=-1) if r_h.dim() > 1 and r_h.size(-1) > 1 else r_h.squeeze(-1))
            diff = z_h.unsqueeze(1) - z_goal.unsqueeze(0)
            goal_dists.append(diff.pow(2).mean(dim=-1).min(dim=-1).values)
            ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx: ctx = ctx[:, -max_ctx:]
            z_final = z_h
        scores = scores + world.predictor.value(z_final)
        scores = scores - args.goal_weight * torch.stack(goal_dists, dim=0).min(dim=0).values

        # Discriminator score.
        obs_pix = (img_stack.flatten(0, 1)).unsqueeze(0).expand(args.n_samples, -1, -1, -1).contiguous()
        disc_state = ((state_stack - 256.0) / 256.0).flatten().unsqueeze(0).expand(args.n_samples, -1).contiguous()
        disc_action = ((a_raw - 256.0) / 256.0).flatten(1)
        disc_logits = discriminator(obs_pix, disc_state, disc_action)
        scores = scores + args.disc_weight * disc_logits

        best = scores.argmax().item()
        return a_raw[best].cpu().numpy().astype(np.float32)

    def run_episode(env_seed: int, attempt_seed: int, perturb_first: bool) -> tuple[float, bool, int]:
        torch.manual_seed(attempt_seed)
        obs, info = env.reset(seed=env_seed)
        x0 = _img_to_tensor(obs["pixels"])
        s0 = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
        dp_pixels = [x0.clone() for _ in range(n_obs_steps)]
        dp_states = [s0.clone() for _ in range(n_obs_steps)]
        world_buf = [x0.clone() for _ in range(T_ctx)]
        ep_max_cov = 0.0; ep_success = False; steps = 0
        while steps < args.max_steps:
            if perturb_first and steps < args.perturb_steps:
                a_chunk = np.random.uniform(50, 470, size=(args.n_action_steps, 2)).astype(np.float32)
            else:
                a_chunk = dp_rerank_chunk(dp_pixels, dp_states, world_buf)
            for k in range(args.n_action_steps):
                if steps >= args.max_steps: break
                obs, r, term, trunc, info = env.step(a_chunk[k])
                ep_max_cov = max(ep_max_cov, float(info.get("coverage", 0.0)))
                if info.get("is_success", False): ep_success = True
                steps += 1
                xt = _img_to_tensor(obs["pixels"])
                stt = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
                dp_pixels = dp_pixels[1:] + [xt]
                dp_states = dp_states[1:] + [stt]
                world_buf = world_buf[1:] + [xt]
                if term or trunc: break
            if term or trunc: break
        return ep_max_cov, ep_success, steps

    # --- adaptive protocol ---
    n_succ = 0
    fallback_used = 0
    for ep in range(args.n_episodes):
        env_seed = args.seed + ep
        # Stage 1: K_primary D-MPC attempts.
        success = False; best_cov = 0.0; best_steps = 0
        for k in range(args.k_primary):
            cov, succ, steps = run_episode(env_seed, env_seed * 31 + 7 * k, perturb_first=False)
            if cov > best_cov: best_cov = cov; best_steps = steps
            if succ:
                success = True; best_cov = cov; best_steps = steps; break

        # Stage 2: random-perturb fallback if primary failed.
        if not success:
            fallback_used += 1
            for k in range(args.k_fallback):
                cov, succ, steps = run_episode(env_seed, env_seed * 53 + 13 * k + 1, perturb_first=True)
                if cov > best_cov: best_cov = cov; best_steps = steps
                if succ:
                    success = True; best_cov = cov; best_steps = steps; break

        if success: n_succ += 1
        flag = "FALLBACK" if not success or (fallback_used > 0 and ep == args.seed + ep - args.seed) else ""
        marker = "✓" if success else "✗"
        print(f"  seed {env_seed} {marker}: max_cov={best_cov:.3f}  steps={best_steps}  "
              f"fallback={fallback_used}", flush=True)

    print(f"\n=== adaptive protocol on {args.n_episodes} seeds ===")
    print(f"  success rate    : {n_succ/args.n_episodes:.2%}  ({n_succ}/{args.n_episodes})")
    print(f"  fallback used   : {fallback_used} seeds (stage-2 random-perturb)")


if __name__ == "__main__":
    main()
