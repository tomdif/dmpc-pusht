"""D-MPC: Diffusion Policy proposal + JEPA world-model reranker.

Draws N candidate action sequences from the pretrained Diffusion Policy
using the underlying conditional_sample (the on-manifold proposal that
fixes our 30-iter CEM phantom-optimum problem), then ranks each via
rollout through our predictor + value head + best-of-K goal distance.
Picks the highest-scoring chunk, executes T_a=8 actions, replans.

Reference: Zhao et al. "Diffusion Model Predictive Control" arXiv:2410.05364
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import yaml

from huggingface_hub import hf_hub_download
import safetensors.torch as st

from cserjepa_v2.data import LeRobotConfig, build_lerobot_loader
from cserjepa_v2.models import CSERJEPAv2, PixelDecoder
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--world-config", type=str, required=True)
    p.add_argument("--world-ckpt", type=str, required=True)
    p.add_argument("--policy-id", type=str, default="lerobot/diffusion_pusht")
    p.add_argument("--diffusion-ckpt", type=str, default=None,
                   help="optional path to fine-tuned diffusion state_dict (overrides pretrained weights)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--n-samples", type=int, default=64,
                   help="number of candidate action sequences per plan")
    p.add_argument("--n-action-steps", type=int, default=8,
                   help="execute this many actions before replanning")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--seeds-list", type=str, default=None,
                   help="comma-separated explicit env seeds to eval (overrides --seed and --n-episodes)")
    p.add_argument("--n-attempts", type=int, default=1,
                   help="K-shot best-of-rollouts: run K rollouts per env seed, take best max_cov")
    # Variance-based adaptive K (option 2 from PAPER §5):
    # After K_probe standard attempts, compute max_cov variance. If all attempts
    # fail with low variance (mode collapse), skip remaining standard attempts
    # and go directly to perturbation fallback.
    p.add_argument("--adaptive-k", action="store_true",
                   help="enable variance-based adaptive K with mode-collapse detection")
    p.add_argument("--probe-k", type=int, default=2,
                   help="number of probe attempts before variance check")
    p.add_argument("--collapse-threshold", type=float, default=0.02,
                   help="if max(probe_covs) - min(probe_covs) < threshold AND all fail, declare mode collapse")
    p.add_argument("--fallback-k", type=int, default=10,
                   help="number of perturbation-fallback attempts after mode collapse OR after standard K exhausted")
    p.add_argument("--perturb-steps", type=int, default=30,
                   help="env steps of random actions before DP resumes, in fallback rollouts")
    p.add_argument("--use-value", action="store_true")
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--use-goal", action="store_true")
    p.add_argument("--goal-weight", type=float, default=0.3)
    p.add_argument("--goal-aggregate", type=str, default="min")
    p.add_argument("--goal-multi", action="store_true")
    p.add_argument("--goal-file", type=str, default=None)
    p.add_argument("--rerank-mode", type=str, default="dmpc",
                   choices=["dmpc", "diffusion_only"],
                   help="dmpc=score+pick best; diffusion_only=just take first sample")
    # Self-GAD temporal coherence: bias denoising toward previous chunk's tail.
    # Self-Guided Action Diffusion (Malhotra 2025, arXiv:2508.12189).
    p.add_argument("--self-gad", action="store_true",
                   help="apply Self-GAD temporal coherence guidance in denoising")
    p.add_argument("--gad-scale", type=float, default=0.5,
                   help="guidance scale; ~0.3-1.0 typical")
    p.add_argument("--gad-anchor-len", type=int, default=4,
                   help="number of leading horizon positions to anchor to prev tail")
    # Pixel grounding for D-MPC reranker (Lever #3): decode imagined latents
    # and add pixel-MSE-to-goal as a tie-breaker that catches phantom-z trajs.
    p.add_argument("--use-pixel-ground", action="store_true")
    p.add_argument("--pixel-ground-weight", type=float, default=10.0)
    p.add_argument("--decoder-ckpt", type=str, default=None)
    # Drift penalty for D-MPC reranker (Lever #4): also penalize final-step
    # goal distance to discourage touch-and-drift trajectories.
    p.add_argument("--goal-drift-weight", type=float, default=0.0)
    # BID — Bidirectional Decoding (Liu+ ICLR 2025, arXiv:2408.17355).
    # Backward-coherence: score candidates by consistency with the previous
    # plan's predictions at overlapping absolute-time positions. Direct PushT
    # precedent +7pp (78% → 85%).
    p.add_argument("--use-bid", action="store_true")
    p.add_argument("--bid-weight", type=float, default=1.0)
    p.add_argument("--bid-rho", type=float, default=0.7,
                   help="exponential decay of coherence weight along horizon")
    # Discriminator reranker — adds log P(success | obs, action_chunk) to score.
    p.add_argument("--discriminator-ckpt", type=str, default=None)
    p.add_argument("--discriminator-weight", type=float, default=1.0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"=== device: {device} ===  rerank-mode: {args.rerank_mode} ===")

    # --- world model ---
    cfg = yaml.safe_load(open(args.world_config))
    data_section = cfg["data"]
    if "lerobot" in data_section and isinstance(data_section["lerobot"], dict):
        data_section = data_section["lerobot"]
    data_cfg = LeRobotConfig(**data_section)
    loader, a_dim, image_size = build_lerobot_loader(data_cfg, batch_size=1, num_workers=0, shuffle=False)
    windows = loader.dataset
    action_mean = windows.action_mean.to(device).flatten()
    action_std = windows.action_std.to(device).flatten()
    cfg["model"]["encoder"]["img_size"] = image_size
    cfg["model"]["action"]["d_a"] = a_dim
    T_ctx = int(data_cfg.context_len)

    world = CSERJEPAv2(cfg["model"]).to(device)
    print(f"[world] loading {args.world_ckpt}")
    ck = torch.load(args.world_ckpt, map_location=device, weights_only=False)
    sd = ck["model"]
    cur = world.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    world.load_state_dict(filtered, strict=False)
    world.eval()

    # --- diffusion policy + its normalization stats ---
    print(f"[policy] loading {args.policy_id}")
    policy = DiffusionPolicy.from_pretrained(args.policy_id).to(device)
    if args.diffusion_ckpt:
        print(f"  overriding diffusion weights from {args.diffusion_ckpt}")
        dck = torch.load(args.diffusion_ckpt, map_location=device, weights_only=False)
        sd = dck["diffusion"] if "diffusion" in dck else dck
        missing, unexpected = policy.diffusion.load_state_dict(sd, strict=False)
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    policy.eval()
    pol_sd = st.load_file(hf_hub_download(args.policy_id, "model.safetensors"))
    img_mean = pol_sd["normalize_inputs.buffer_observation_image.mean"].to(device)
    img_std  = pol_sd["normalize_inputs.buffer_observation_image.std"].to(device)
    st_max   = pol_sd["normalize_inputs.buffer_observation_state.max"].to(device)
    st_min   = pol_sd["normalize_inputs.buffer_observation_state.min"].to(device)
    a_max    = pol_sd["unnormalize_outputs.buffer_action.max"].to(device)
    a_min    = pol_sd["unnormalize_outputs.buffer_action.min"].to(device)

    n_obs_steps = policy.config.n_obs_steps
    horizon_p = policy.config.horizon

    # --- goals ---
    z_goal = None
    x_goal = None
    if args.use_goal and args.goal_file:
        gd = torch.load(args.goal_file, map_location=device, weights_only=False)
        cf = gd["candidate_frames"].to(device)
        z_cands = world.encode(cf)
        z_goal = z_cands if args.goal_multi else z_cands.mean(dim=0)
        x_goal = gd["goal_frame"].to(device)
        print(f"[goal] z_goal {tuple(z_goal.shape)}, mode={'multi' if args.goal_multi else 'avg'}")

    # --- discriminator reranker ---
    discriminator = None
    if args.discriminator_ckpt:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from train_discriminator import SuccessDiscriminator
        dck = torch.load(args.discriminator_ckpt, map_location=device, weights_only=False)
        dcfg = dck["config"]
        discriminator = SuccessDiscriminator(dcfg["in_channels"], dcfg["state_dim"], dcfg["action_dim"]).to(device)
        discriminator.load_state_dict(dck["model"])
        discriminator.eval()
        for p in discriminator.parameters():
            p.requires_grad_(False)
        print(f"[discriminator] loaded {args.discriminator_ckpt}")

    # --- pixel decoder for grounded reranking (Lever #3) ---
    decoder = None
    if args.use_pixel_ground:
        if not args.decoder_ckpt:
            raise SystemExit("--use-pixel-ground requires --decoder-ckpt")
        dck = torch.load(args.decoder_ckpt, map_location=device, weights_only=False)
        decoder = PixelDecoder(d_z=cfg["model"]["encoder"]["embed_dim"], img_size=image_size).to(device)
        decoder.load_state_dict(dck["decoder"])
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad_(False)
        print(f"[decoder] loaded {args.decoder_ckpt}")

    # --- env ---
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    def _img_to_tensor(pixels: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(pixels).float().permute(2, 0, 1).contiguous() / 255.0

    def _norm_state(x):
        return 2.0 * (x - st_min) / (st_max - st_min) - 1.0

    def _unnorm_action(a_norm):
        return (a_norm + 1.0) / 2.0 * (a_max - a_min) + a_min

    # Self-GAD state: previous chunk's full normalized action tensor (for anchoring).
    prev_chunk_norm: list = [None]  # use list as nonlocal mutable

    def _guided_sample(global_cond: torch.Tensor) -> torch.Tensor:
        """Custom denoising loop with optional Self-GAD temporal coherence guidance.
        Returns (N, horizon_p, action_dim) actions in [-1, 1] normalized space.
        """
        diffusion = policy.diffusion
        action_dim = diffusion.config.action_feature.shape[0]
        sample = torch.randn(
            (args.n_samples, horizon_p, action_dim),
            device=device, dtype=next(diffusion.parameters()).dtype,
        )
        diffusion.noise_scheduler.set_timesteps(diffusion.num_inference_steps)
        anchor = prev_chunk_norm[0]  # (horizon_p, action_dim) or None
        for t in diffusion.noise_scheduler.timesteps:
            t_batch = torch.full(sample.shape[:1], t, dtype=torch.long, device=device)
            model_output = diffusion.unet(sample, t_batch, global_cond=global_cond)
            # Self-GAD: nudge sample toward anchor at the leading horizon positions.
            if args.self_gad and anchor is not None:
                L = min(args.gad_anchor_len, horizon_p)
                # ANCHOR is the previous chunk's actions for the SAME first-L positions.
                # Penalize ||sample[:, :L] - anchor[:L]||² → grad = 2*(sample[:, :L] - anchor[:L]).
                # Add this gradient (scaled, sign-correct) to model_output to bias denoising.
                grad = sample[:, :L] - anchor[:L].to(sample).unsqueeze(0)         # (N, L, A)
                # Exponential weights wᵢ = 0.5^i (Self-GAD form); biases earlier
                # positions more strongly.
                w = (0.5 ** torch.arange(L, device=device, dtype=sample.dtype))    # (L,)
                grad_weighted = grad * w.view(1, L, 1)
                # Add into the noise-prediction's leading positions only.
                model_output_mod = model_output.clone()
                model_output_mod[:, :L] = model_output_mod[:, :L] + args.gad_scale * grad_weighted
                model_output = model_output_mod
            sample = diffusion.noise_scheduler.step(model_output, t, sample).prev_sample
        return sample

    @torch.no_grad()
    def plan(obs_buf_pixels, obs_buf_state, ctx_buf_world):
        """Sample N candidates from DP, score with world model, return best chunk.

        obs_buf_pixels: list of (3, H, W) tensors (length n_obs_steps)
        obs_buf_state:  list of (2,) tensors          (length n_obs_steps)
        ctx_buf_world:  list of (3, H, W) tensors  (length T_ctx) — for world encoder
        """
        # --- prepare DP global conditioning ---
        img_stack = torch.stack(obs_buf_pixels, dim=0).to(device)             # (T_o, 3, H, W) [0,1]
        img_stack = (img_stack - img_mean) / img_std
        state_stack = torch.stack(obs_buf_state, dim=0).to(device)            # (T_o, 2)
        state_stack = _norm_state(state_stack)
        # New lerobot expects "observation.images" with shape (B, S, N_cam, C, H, W).
        batch = {
            "observation.images": img_stack.unsqueeze(0).unsqueeze(2),  # (1, T_o, 1, 3, H, W)
            "observation.state": state_stack.unsqueeze(0),               # (1, T_o, 2)
        }
        global_cond = policy.diffusion._prepare_global_conditioning(batch)    # (1, D_cond)
        global_cond = global_cond.expand(args.n_samples, -1).contiguous()

        # --- sample N candidate full-horizon action sequences (with Self-GAD if on) ---
        actions_norm = _guided_sample(global_cond)                # (N, horizon_p, 2) in [-1, 1]
        # Slice to "from the current observation" window: same as generate_actions.
        start = n_obs_steps - 1
        actions_norm_chunk = actions_norm[:, start : start + args.n_action_steps]  # (N, T_a, 2)
        # Un-normalize to raw action space.
        actions_raw = _unnorm_action(actions_norm_chunk)                            # (N, T_a, 2)
        # Clip to env bounds.
        actions_raw = actions_raw.clamp(0.0, 512.0)
        # Normalize for world-model (its own action stats).
        actions_world = (actions_raw - action_mean) / action_std                    # (N, T_a, 2)

        if args.rerank_mode == "diffusion_only":
            return actions_raw[0]  # just first sample, no reranking

        # --- world-model rollout to score each candidate ---
        world_ctx_imgs = torch.stack(ctx_buf_world, dim=0).unsqueeze(0).to(device)  # (1, T_ctx, 3, H, W)
        fe = world.encode(world_ctx_imgs)                                           # (1, T_ctx, D)
        ctx = fe.expand(args.n_samples, -1, -1).contiguous()
        max_ctx = world.predictor.frame_pos_embed.size(1)
        K_world = world.chunk_size

        scores = torch.zeros(args.n_samples, device=device)
        z_final = None
        goal_dists = []
        pixel_dists = []
        # We feed world model in chunks of K_world actions per predictor step.
        n_chunks = args.n_action_steps // K_world
        for h in range(n_chunks):
            a_h = actions_world[:, h * K_world : (h + 1) * K_world]                  # (N, K_world, 2)
            z_h, r_h = world.predict(ctx, a_h)
            if r_h.dim() > 1 and r_h.size(-1) > 1:
                scores = scores + r_h.sum(dim=-1)
            else:
                scores = scores + r_h.squeeze(-1)
            if args.use_goal and z_goal is not None:
                if z_goal.dim() == 1:
                    d2 = (z_h - z_goal.unsqueeze(0)).pow(2).mean(dim=-1)
                else:
                    diff = z_h.unsqueeze(1) - z_goal.unsqueeze(0)
                    d2 = diff.pow(2).mean(dim=-1).min(dim=-1).values
                goal_dists.append(d2)
            # Pixel grounding (Lever #3): decode imagined latent, pixel-MSE to goal.
            if args.use_pixel_ground and decoder is not None and x_goal is not None:
                x_h = decoder(z_h)                                                    # (N, 3, H, W)
                pix_d = (x_h - x_goal.unsqueeze(0)).pow(2).mean(dim=(1, 2, 3))         # (N,)
                pixel_dists.append(pix_d)
            ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
            z_final = z_h
        if args.use_value and z_final is not None:
            scores = scores + args.value_weight * world.predictor.value(z_final)
        if args.use_goal and goal_dists:
            stacked = torch.stack(goal_dists, dim=0)
            agg = stacked.min(dim=0).values if args.goal_aggregate == "min" else stacked[-1]
            scores = scores - args.goal_weight * agg
            # Drift penalty (Lever #4): also penalize final-step distance.
            if args.goal_drift_weight > 0:
                scores = scores - args.goal_drift_weight * stacked[-1]
        if args.use_pixel_ground and pixel_dists:
            pix_stacked = torch.stack(pixel_dists, dim=0)
            pix_agg = pix_stacked.min(dim=0).values
            scores = scores - args.pixel_ground_weight * pix_agg

        # Discriminator reranker: log P(success | obs, action_chunk).
        # The classifier scores each candidate's first n_action_steps actions
        # against the current observation history. Adds a learned signal that
        # captures patterns the JEPA reranker misses (success/fail dynamics).
        if discriminator is not None:
            # Build batch input: pixels stacked across n_obs frames.
            obs_pix_stack = torch.stack(obs_buf_pixels, dim=0)         # (n_obs, 3, H, W) [0,1]
            obs_pix_stack = obs_pix_stack.flatten(0, 1).to(device)      # (n_obs*3, H, W)
            obs_state_stack = torch.stack(obs_buf_state, dim=0).to(device)
            # Normalize for discriminator (centered around PushT board midpoint).
            disc_state = ((obs_state_stack - 256.0) / 256.0).flatten()  # (n_obs*2,)
            disc_pix = obs_pix_stack.unsqueeze(0).expand(args.n_samples, -1, -1, -1).contiguous()  # (N, n_obs*3, H, W)
            disc_state_b = disc_state.unsqueeze(0).expand(args.n_samples, -1).contiguous()           # (N, n_obs*2)
            disc_action = ((actions_raw - 256.0) / 256.0).flatten(1)                                  # (N, n_action*2)
            disc_logits = discriminator(disc_pix, disc_state_b, disc_action)                          # (N,)
            scores = scores + args.discriminator_weight * disc_logits

        # BID backward coherence (Lever S1): penalize candidates whose first
        # (horizon_p - n_action_steps) actions deviate from previous plan's
        # predictions at the same absolute-time positions. Exponential decay
        # along the overlap window with rate ρ.
        if args.use_bid and prev_chunk_norm[0] is not None:
            L_overlap = horizon_p - args.n_action_steps
            if L_overlap > 0:
                # actions_norm: (N, horizon_p, A); prev: (horizon_p, A).
                cand_first = actions_norm[:, :L_overlap]                  # (N, L, A)
                prev_match = prev_chunk_norm[0][args.n_action_steps : args.n_action_steps + L_overlap].to(actions_norm)
                diff_b = cand_first - prev_match.unsqueeze(0)            # (N, L, A)
                rho_w = (args.bid_rho ** torch.arange(L_overlap, device=device, dtype=actions_norm.dtype))  # (L,)
                bid_loss = (diff_b.pow(2).sum(dim=-1) * rho_w.unsqueeze(0)).sum(dim=-1)  # (N,)
                scores = scores - args.bid_weight * bid_loss

        # Pick best.
        best = scores.argmax().item()
        # Store the FULL chosen sequence's normalized form for next plan's Self-GAD anchor.
        prev_chunk_norm[0] = actions_norm[best].detach()                             # (horizon_p, 2)
        return actions_raw[best]                                                    # (T_a, 2)

    def run_one_rollout(env_seed: int, attempt_seed: int, perturb_first_steps: int = 0):
        """Run a single full episode rollout. If perturb_first_steps > 0, replace
        first N env steps with uniform-random actions before D-MPC takes over."""
        torch.manual_seed(attempt_seed)
        obs, info = env.reset(seed=env_seed)
        x0 = _img_to_tensor(obs["pixels"])
        s0 = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
        dp_pixels = [x0.clone() for _ in range(n_obs_steps)]
        dp_states = [s0.clone() for _ in range(n_obs_steps)]
        world_buf = [x0.clone() for _ in range(T_ctx)]

        ep_return = 0.0; ep_max_coverage = 0.0; ep_success = False; steps = 0
        terminated = False; truncated = False
        prev_chunk_norm[0] = None
        while steps < args.max_steps:
            if perturb_first_steps > 0 and steps < perturb_first_steps:
                a_chunk = np.random.uniform(50, 470, size=(args.n_action_steps, 2)).astype(np.float32)
            else:
                a_chunk = plan(dp_pixels, dp_states, world_buf).cpu().numpy().astype(np.float32)
            for k in range(args.n_action_steps):
                if steps >= args.max_steps: break
                obs, r, terminated, truncated, info = env.step(a_chunk[k])
                ep_return += float(r)
                ep_max_coverage = max(ep_max_coverage, float(info.get("coverage", 0.0)))
                if info.get("is_success", False):
                    ep_success = True
                steps += 1
                xt = _img_to_tensor(obs["pixels"])
                st_ = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
                dp_pixels = dp_pixels[1:] + [xt]
                dp_states = dp_states[1:] + [st_]
                world_buf = world_buf[1:] + [xt]
                if terminated or truncated: break
            if terminated or truncated: break
        return ep_return, ep_max_coverage, ep_success, steps

    # --- eval loop ---
    if args.seeds_list:
        seeds_to_eval = [int(s) for s in args.seeds_list.split(",")]
    else:
        seeds_to_eval = [args.seed + i for i in range(args.n_episodes)]
    returns, successes, max_coverages = [], [], []
    total_attempts_used = 0
    fallback_triggers = 0   # via mode-collapse detection
    fallback_post_K = 0     # via standard K exhausted
    for ep, env_seed in enumerate(seeds_to_eval):
        best_return = 0.0; best_max_cov = 0.0; best_success = False; best_steps = 0
        probe_covs = []
        attempts_used = 0
        in_fallback = False

        # ----- adaptive-K mode -----
        if args.adaptive_k:
            # 1) Probe phase: K=probe_k standard attempts.
            for attempt in range(args.probe_k):
                ep_return, ep_max_coverage, ep_success, steps = run_one_rollout(
                    env_seed, 1000 * env_seed + 31 * ep + 7 * attempt, perturb_first_steps=0,
                )
                attempts_used += 1; probe_covs.append(ep_max_coverage)
                if ep_max_coverage > best_max_cov or ep_success:
                    best_return = ep_return; best_max_cov = ep_max_coverage
                    best_success = ep_success; best_steps = steps
                print(f"    seed {env_seed} probe {attempt}: max_cov={ep_max_coverage:.3f}  success={ep_success}", flush=True)
                if best_success: break

            # 2) Decide: if probe didn't succeed AND variance is low → mode collapse → fallback.
            if not best_success:
                cov_range = max(probe_covs) - min(probe_covs)
                # Two collapse signatures, EITHER triggers fallback:
                # 1. Tight variance (deterministic mode collapse): range < threshold AND max < 0.85
                #    (excludes near-miss cluster which has tight variance but is on the cusp).
                # 2. Very low max (clearly stuck — barely engaged with block): max < 0.55.
                low_var_collapse = (cov_range < args.collapse_threshold) and (max(probe_covs) < 0.85)
                low_max_collapse = max(probe_covs) < 0.55
                detected_collapse = low_var_collapse or low_max_collapse
                if detected_collapse:
                    fallback_triggers += 1
                    print(f"    seed {env_seed} ▶ mode-collapse detected "
                          f"(cov_range={cov_range:.3f}, max={max(probe_covs):.3f}); "
                          f"skipping standard K, going to perturb fallback K={args.fallback_k}", flush=True)
                else:
                    # Standard K=remaining attempts before fallback.
                    n_more_standard = max(0, args.n_attempts - args.probe_k)
                    for attempt in range(args.probe_k, args.probe_k + n_more_standard):
                        ep_return, ep_max_coverage, ep_success, steps = run_one_rollout(
                            env_seed, 1000 * env_seed + 31 * ep + 7 * attempt, perturb_first_steps=0,
                        )
                        attempts_used += 1
                        if ep_max_coverage > best_max_cov or ep_success:
                            best_return = ep_return; best_max_cov = ep_max_coverage
                            best_success = ep_success; best_steps = steps
                        print(f"    seed {env_seed} std {attempt}: max_cov={ep_max_coverage:.3f}  success={ep_success}", flush=True)
                        if best_success: break
                    if not best_success:
                        fallback_post_K += 1

            # 3) Perturbation fallback if still not solved.
            if not best_success:
                in_fallback = True
                for attempt in range(args.fallback_k):
                    ep_return, ep_max_coverage, ep_success, steps = run_one_rollout(
                        env_seed, 1000 * env_seed + 53 * ep + 13 * attempt + 1,
                        perturb_first_steps=args.perturb_steps,
                    )
                    attempts_used += 1
                    if ep_max_coverage > best_max_cov or ep_success:
                        best_return = ep_return; best_max_cov = ep_max_coverage
                        best_success = ep_success; best_steps = steps
                    print(f"    seed {env_seed} perturb {attempt}: max_cov={ep_max_coverage:.3f}  success={ep_success}", flush=True)
                    if best_success: break

        # ----- non-adaptive (legacy K-shot) mode -----
        else:
            for attempt in range(args.n_attempts):
                ep_return, ep_max_coverage, ep_success, steps = run_one_rollout(
                    env_seed, 1000 * env_seed + 31 * ep + 7 * attempt, perturb_first_steps=0,
                )
                attempts_used += 1
                attempt_better = (
                    (ep_success and not best_success)
                    or (ep_success == best_success and ep_max_coverage > best_max_cov)
                    or (ep_success == best_success and ep_max_coverage == best_max_cov and ep_return > best_return)
                )
                if attempt == 0 or attempt_better:
                    best_return = ep_return; best_max_cov = ep_max_coverage
                    best_success = ep_success; best_steps = steps
                if args.n_attempts > 1:
                    print(f"    seed {env_seed} attempt {attempt}: return={ep_return:+.3f}  "
                          f"max_cov={ep_max_coverage:.3f}  success={ep_success}", flush=True)
                if best_success: break
        total_attempts_used += attempts_used

        print(f"  seed {env_seed}: return={best_return:+.3f}  max_cov={best_max_cov:.3f}  "
              f"success={best_success}  steps={best_steps}", flush=True)
        returns.append(best_return)
        successes.append(int(best_success))
        max_coverages.append(best_max_cov)

    n = len(seeds_to_eval)
    print(f"\n=== summary over {n} episodes ({args.rerank_mode}) ===")
    print(f"  mean return     : {np.mean(returns):+.3f}")
    print(f"  mean max_cov    : {np.mean(max_coverages):.3f}")
    print(f"  success rate    : {np.mean(successes):.2%}  ({sum(successes)}/{n})")
    if args.adaptive_k:
        print(f"  total attempts  : {total_attempts_used}  (avg {total_attempts_used/n:.2f}/seed)")
        print(f"  fallback triggers (collapse-detected): {fallback_triggers}")
        print(f"  fallback triggers (post-K-exhausted): {fallback_post_K}")


if __name__ == "__main__":
    main()
