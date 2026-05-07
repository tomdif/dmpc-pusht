"""Try multiple strategies on seed 102 to break the mode collapse.

Strategy A: High-temperature DP — scale noise by τ during sampling.
Strategy B: BC+DP alternating mix (uses BC's chunk every other attempt).
Strategy C: DP with random initial-action perturbation (replace first 4 actions).

Reports max_cov per attempt across strategies.
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
from cserjepa_v2.models import BCPolicy, CSERJEPAv2  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-config", default="configs/stage1ab_reward.yaml")
    ap.add_argument("--world-ckpt", default="ckpts/stage_AB/reward/ckpt_step5000.pt")
    ap.add_argument("--bc-ckpt", default="ckpts/stage_AB/bc_policy_AB.pt")
    ap.add_argument("--policy-id", default="lerobot/diffusion_pusht")
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--seeds-list", type=str, default=None)
    ap.add_argument("--n-episodes", type=int, default=1, help="if seeds-list not given, eval seed..seed+n-1")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--n-attempts", type=int, default=8)
    ap.add_argument("--strategy", type=str, default="high_temp",
                    choices=["high_temp", "bc_dp_mix", "random_perturb"])
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--n-action-steps", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda")

    # World model + BC.
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

    world = CSERJEPAv2(cfg["model"]).to(device).eval()
    ck = torch.load(args.world_ckpt, map_location=device, weights_only=False)
    sd = ck["model"]; cur = world.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    world.load_state_dict(filtered, strict=False)
    K_world = world.chunk_size

    bc_ck = torch.load(args.bc_ckpt, map_location=device, weights_only=False)
    bc = BCPolicy(**bc_ck["cfg"]).to(device).eval()
    bc.load_state_dict(bc_ck["model"])
    K_bc = bc.chunk_size

    # Diffusion Policy.
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

    def dp_sample(dp_pixels, dp_states, batch_size: int, temperature: float = 1.0):
        img_stack = torch.stack(dp_pixels, dim=0).to(device)
        img_stack = (img_stack - img_mean) / img_std
        state_stack = torch.stack(dp_states, dim=0).to(device)
        state_stack = 2.0*(state_stack - st_min)/(st_max - st_min) - 1.0
        batch = {
            "observation.images": img_stack.unsqueeze(0).unsqueeze(2),
            "observation.state": state_stack.unsqueeze(0),
        }
        global_cond = policy.diffusion._prepare_global_conditioning(batch).expand(batch_size, -1).contiguous()
        # High-temperature sampling: scale the initial noise.
        noise = temperature * torch.randn(
            (batch_size, horizon_p, policy.diffusion.config.action_feature.shape[0]),
            device=device, dtype=next(policy.diffusion.parameters()).dtype,
        )
        a_norm = policy.diffusion.conditional_sample(batch_size=batch_size, global_cond=global_cond, noise=noise)
        start = n_obs_steps - 1
        a_chunk_norm = a_norm[:, start : start + args.n_action_steps]
        a_raw = ((a_chunk_norm + 1.0)/2.0 * (a_max - a_min) + a_min).clamp(0.0, 512.0)
        return a_raw

    def bc_chunk(world_buf):
        x_ctx = torch.stack(world_buf, dim=0).unsqueeze(0).to(device)
        fe = world.encode(x_ctx)
        a_norm = bc(fe).squeeze(0)
        a_raw = (a_norm * action_std + action_mean).clamp(0.0, 512.0)
        return a_raw

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    if args.seeds_list:
        seeds = [int(s) for s in args.seeds_list.split(",")]
    else:
        seeds = [args.seed + i for i in range(args.n_episodes)]
    print(f"=== seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}), strategy={args.strategy}, K={args.n_attempts} ===")

    all_best_cov = []; all_best_succ = []
    for env_seed in seeds:
     best_max_cov = 0.0; best_success = False
     for attempt in range(args.n_attempts):
        torch.manual_seed(env_seed * 31 + 53 * attempt + 7)
        obs, info = env.reset(seed=env_seed)
        x0 = torch.from_numpy(obs["pixels"]).float().permute(2,0,1) / 255.0
        s0 = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
        dp_pixels = [x0.clone() for _ in range(n_obs_steps)]
        dp_states = [s0.clone() for _ in range(n_obs_steps)]
        world_buf = [x0.clone() for _ in range(T_ctx)]

        ep_max_cov = 0.0; ep_success = False; steps = 0
        while steps < args.max_steps:
            if args.strategy == "high_temp":
                a_chunk = dp_sample(dp_pixels, dp_states, 1, temperature=args.temperature)[0].cpu().numpy().astype(np.float32)
            elif args.strategy == "bc_dp_mix":
                if attempt % 2 == 0:
                    a_chunk = dp_sample(dp_pixels, dp_states, 1, 1.0)[0].cpu().numpy().astype(np.float32)
                else:
                    a_chunk = bc_chunk(world_buf).cpu().numpy().astype(np.float32)
                    a_chunk = a_chunk[: args.n_action_steps]
            elif args.strategy == "random_perturb":
                # First 30 env steps: random pushes; then DP takes over.
                if steps < 30:
                    a_chunk = np.random.uniform(50, 470, size=(args.n_action_steps, 2)).astype(np.float32)
                else:
                    a_chunk = dp_sample(dp_pixels, dp_states, 1, 1.0)[0].cpu().numpy().astype(np.float32)

            for k in range(args.n_action_steps):
                if steps >= args.max_steps: break
                obs, r, term, trunc, info = env.step(a_chunk[k])
                ep_max_cov = max(ep_max_cov, float(info.get("coverage", 0.0)))
                if info.get("is_success", False): ep_success = True
                steps += 1
                xt = torch.from_numpy(obs["pixels"]).float().permute(2,0,1)/255.0
                stt = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
                dp_pixels = dp_pixels[1:] + [xt]
                dp_states = dp_states[1:] + [stt]
                world_buf = world_buf[1:] + [xt]
                if term or trunc: break
            if term or trunc: break

        if ep_max_cov > best_max_cov:
            best_max_cov = ep_max_cov; best_success = ep_success
        if best_success: break
     all_best_cov.append(best_max_cov); all_best_succ.append(best_success)
     print(f"  seed {env_seed}: best max_cov={best_max_cov:.3f}  success={best_success}", flush=True)
    n_succ = sum(all_best_succ); n = len(all_best_succ)
    print(f"\n=== summary {n} seeds ({args.strategy}, K={args.n_attempts}) ===")
    print(f"  success rate : {n_succ/n:.2%}  ({n_succ}/{n})")
    print(f"  mean max_cov : {sum(all_best_cov)/n:.3f}")


if __name__ == "__main__":
    main()
