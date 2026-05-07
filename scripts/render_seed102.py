"""Render frames from a single seed=102 D-MPC rollout. Saves a strip
of every-Nth frame as a single side-by-side PNG so we can see what
the agent is actually doing on this stuck seed.
"""
from __future__ import annotations

import argparse
import sys, pathlib
from pathlib import Path

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
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-config", default="configs/stage1ab_reward.yaml")
    ap.add_argument("--world-ckpt", default="ckpts/stage_AB/reward/ckpt_step5000.pt")
    ap.add_argument("--policy-id", default="lerobot/diffusion_pusht")
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--n-action-steps", type=int, default=8)
    ap.add_argument("--out", type=str, default="/root/step5/seed102_strip.png")
    ap.add_argument("--every", type=int, default=15, help="save every-Nth env step")
    args = ap.parse_args()
    device = torch.device("cuda")

    cfg = yaml.safe_load(open(args.world_config))
    data_section = cfg["data"]
    if "lerobot" in data_section: data_section = data_section["lerobot"]
    data_cfg = LeRobotConfig(**data_section)
    loader, a_dim, image_size = build_lerobot_loader(data_cfg, batch_size=1, num_workers=0, shuffle=False)
    cfg["model"]["encoder"]["img_size"] = image_size; cfg["model"]["action"]["d_a"] = a_dim
    T_ctx = int(data_cfg.context_len)

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

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)
    x0 = torch.from_numpy(obs["pixels"]).float().permute(2,0,1) / 255.0
    s0 = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
    dp_pixels = [x0.clone() for _ in range(n_obs_steps)]
    dp_states = [s0.clone() for _ in range(n_obs_steps)]
    frames = [obs["pixels"].copy()]
    coverages = [float(info.get("coverage", 0.0))]

    torch.manual_seed(args.seed * 31 + 7)
    while len(frames) < args.max_steps + 1:
        img_stack = torch.stack(dp_pixels, dim=0).to(device)
        img_stack = (img_stack - img_mean) / img_std
        state_stack = torch.stack(dp_states, dim=0).to(device)
        state_stack = 2.0*(state_stack - st_min)/(st_max - st_min) - 1.0
        batch = {
            "observation.images": img_stack.unsqueeze(0).unsqueeze(2),
            "observation.state": state_stack.unsqueeze(0),
        }
        global_cond = policy.diffusion._prepare_global_conditioning(batch)
        a_norm = policy.diffusion.conditional_sample(batch_size=1, global_cond=global_cond)[0]
        start = n_obs_steps - 1
        a_chunk_norm = a_norm[start : start + args.n_action_steps]
        a_chunk = ((a_chunk_norm + 1.0)/2.0 * (a_max - a_min) + a_min).clamp(0.0, 512.0).cpu().numpy()
        for k in range(args.n_action_steps):
            if len(frames) >= args.max_steps + 1: break
            obs, r, terminated, truncated, info = env.step(a_chunk[k])
            frames.append(obs["pixels"].copy())
            coverages.append(float(info.get("coverage", 0.0)))
            xt = torch.from_numpy(obs["pixels"]).float().permute(2,0,1)/255.0
            stt = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
            dp_pixels = dp_pixels[1:] + [xt]
            dp_states = dp_states[1:] + [stt]
            if terminated or truncated: break
        if terminated or truncated: break

    print(f"steps={len(frames)-1}  max_cov={max(coverages):.3f}")
    print(f"first/mid/last cov: {coverages[0]:.3f} / {coverages[len(coverages)//2]:.3f} / {coverages[-1]:.3f}")

    import imageio
    sample = frames[::args.every]
    H, W, _ = sample[0].shape
    strip = np.concatenate(sample, axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, strip)
    print(f"[saved] {args.out}  ({len(sample)} frames, every {args.every})")


if __name__ == "__main__":
    main()
