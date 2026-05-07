"""Run pretrained LeRobot Diffusion Policy on gym-pusht.

This is the calibration anchor: the LeRobot card claims 65.4% success on
500 episodes at canonical 95%-coverage threshold. Without this number on
our eval harness, every other result is uncalibrated.

Reference: lerobot/diffusion_pusht (HuggingFace), Chi et al. 2023 (arXiv:2303.04137).
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-id", type=str, default="lerobot/diffusion_pusht")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=100)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"=== device: {device} ===")
    print(f"[policy] loading {args.policy_id} ...")
    policy = DiffusionPolicy.from_pretrained(args.policy_id).to(device)
    policy.eval()
    print(f"  horizon={policy.config.horizon}  n_obs={policy.config.n_obs_steps}  n_action={policy.config.n_action_steps}")

    # New lerobot policy classes drop the on-model normalize_inputs/targets
    # buffers, but pretrained checkpoints still contain them. Pull stats
    # directly from the safetensors file and apply manually.
    from huggingface_hub import hf_hub_download
    import safetensors.torch as st
    sd = st.load_file(hf_hub_download(args.policy_id, "model.safetensors"))
    img_mean = sd["normalize_inputs.buffer_observation_image.mean"].to(device)            # (3, 1, 1)
    img_std  = sd["normalize_inputs.buffer_observation_image.std"].to(device)             # (3, 1, 1)
    st_max   = sd["normalize_inputs.buffer_observation_state.max"].to(device)             # (2,)
    st_min   = sd["normalize_inputs.buffer_observation_state.min"].to(device)             # (2,)
    a_max    = sd["unnormalize_outputs.buffer_action.max"].to(device)                     # (2,)
    a_min    = sd["unnormalize_outputs.buffer_action.min"].to(device)                     # (2,)
    print(f"  norm: image mean/std loaded; state min={st_min.tolist()}, max={st_max.tolist()}")
    print(f"  norm: action min={a_min.tolist()}, max={a_max.tolist()}")

    def norm_state(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - st_min) / (st_max - st_min) - 1.0

    def unnorm_action(a: torch.Tensor) -> torch.Tensor:
        return (a + 1.0) / 2.0 * (a_max - a_min) + a_min

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    returns, successes, max_coverages = [], [], []
    for ep in range(args.n_episodes):
        policy.reset()
        obs, info = env.reset(seed=args.seed + ep)

        ep_return = 0.0
        ep_max_coverage = 0.0
        ep_success = False
        for step in range(args.max_steps):
            # Normalize observations using the stats from the pretrained ckpt.
            img = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1).contiguous() / 255.0  # (3, H, W)
            img = img.to(device)
            img = (img - img_mean) / img_std                                                      # ImageNet-style
            state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).to(device)   # (2,)
            state = norm_state(state)
            obs_in = {
                "observation.image": img.unsqueeze(0),                # (1, 3, H, W)
                "observation.state": state.unsqueeze(0),              # (1, 2)
            }
            a_norm = policy.select_action(obs_in)                      # (1, 2) in [-1, 1]
            a = unnorm_action(a_norm.squeeze(0))                       # (2,) in raw action space
            a_np = a.cpu().numpy().astype(np.float32)
            a_np = np.clip(a_np, 0.0, 512.0)

            obs, r, terminated, truncated, info = env.step(a_np)
            ep_return += float(r)
            ep_max_coverage = max(ep_max_coverage, float(info.get("coverage", 0.0)))
            if info.get("is_success", False):
                ep_success = True
            if terminated or truncated:
                break

        print(f"  ep {ep}: return={ep_return:+.3f}  max_cov={ep_max_coverage:.3f}  "
              f"success={ep_success}  steps={step + 1}", flush=True)
        returns.append(ep_return)
        successes.append(int(ep_success))
        max_coverages.append(ep_max_coverage)

    print(f"\n=== summary over {args.n_episodes} episodes ===")
    print(f"  mean return     : {np.mean(returns):+.3f}")
    print(f"  mean max_cov    : {np.mean(max_coverages):.3f}")
    print(f"  success rate    : {np.mean(successes):.2%}")


if __name__ == "__main__":
    main()
