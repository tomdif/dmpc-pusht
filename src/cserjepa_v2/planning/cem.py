"""Cross-Entropy-Method planner for action selection over a learned world model.

Given a world model with a `rollout(frame_embeds, action_seq) -> (z_traj, r_traj)`
API, CEM iteratively refines a Gaussian proposal distribution over action
sequences by selecting the top-K-return rollouts and refitting.

Era-of-Experience role: this is the "plan" loop in pillar 4. The agent
imagines futures using its world model, scores them with the learned
reward head, and picks the action that leads to the best imagined
trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from ..models.world_model import CSERJEPAv2


def _colored_noise(
    shape: tuple[int, ...], beta: float, time_dim: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Generate temporally-correlated noise with 1/f^beta power spectrum.
    iCEM (Pinneri 2020) shows beta=2.5 dominates white noise on manipulation."""
    n = shape[time_dim]
    f = torch.fft.rfftfreq(n, device=device).clamp(min=1e-8)
    spectrum = (1.0 / f) ** (beta / 2.0)
    spectrum[0] = 0.0  # zero-mean
    white = torch.randn(*shape, device=device, dtype=dtype)
    white_fft = torch.fft.rfft(white, dim=time_dim)
    spec_shape = [1] * len(shape)
    spec_shape[time_dim] = -1
    colored_fft = white_fft * spectrum.view(*spec_shape)
    colored = torch.fft.irfft(colored_fft, n=n, dim=time_dim)
    std = colored.std(dim=time_dim, keepdim=True).clamp(min=1e-6)
    return (colored / std).to(dtype)


@dataclass
class CEMConfig:
    horizon: int = 16          # plan H steps ahead (in action-chunks of chunk_size)
    n_samples: int = 256       # candidate action sequences per iteration
    n_elite: int = 32          # top-K kept for refitting
    n_iters: int = 4           # CEM iterations
    init_std: float = 1.0      # initial Gaussian std (assumes normalized action space)
    min_std: float = 0.1
    action_clip: float | None = None  # |a| <= clip; None = no clip
    use_value: bool = False    # if True, add V(z_final) to score (Stage Z bootstrap)
    value_weight: float = 1.0  # multiplier on V(z_final) when use_value=True
    use_proximity: bool = False  # if True, add -dist(agent, block) per-step (Stage AF)
    proximity_weight: float = 0.01  # scale on proximity bonus (raw px-coord scale)
    # Goal-conditioned planning (Stage AI). Score = -||z_h - z_goal||² aggregated
    # over horizon. The single biggest LeWM/DINO-WM differentiator from
    # reward-based CEM — gives the planner a smooth gradient everywhere.
    use_goal: bool = False
    goal_weight: float = 1.0
    goal_aggregate: str = "min"  # "min" (closest-approach) or "final" or "mean"
    # Drift penalty: also penalize the FINAL-step goal distance, on top of min.
    # This rewards trajectories that REACH the goal AND STAY there, fixing
    # "touch-and-drift" trajectories that score well on min alone.
    goal_drift_weight: float = 0.0
    # BC-anchor penalty: -weight * ||first_horizon_action - bc_action||² per sample.
    # Anchors CEM elite distribution to BC prior to dampen world-model exploitation
    # under high-iter CEM (a Goodhart's-law fix from offline MBRL literature).
    use_bc_anchor: bool = False
    bc_anchor_weight: float = 0.0
    # Pixel-grounded planning: decode imagined latents back to images and add
    # -||x_decoded - x_goal||² as cost. Forces CEM-explored latents to map to
    # coherent goal-aligned pixel configurations, breaking phantom-optimum
    # exploitation that pure latent-space cost is vulnerable to.
    use_pixel_ground: bool = False
    pixel_ground_weight: float = 0.0
    pixel_ground_aggregate: str = "min"  # "min" | "final" | "mean"
    # iCEM upgrades (Pinneri et al. 2020, arXiv:2008.06389):
    # colored-noise sampling with low-frequency power for manipulation tasks +
    # elite memory across plan calls (carry top elites forward).
    colored_noise_beta: float = 0.0   # 0 = white (current); 2.5 typical for low-freq push tasks
    elite_carry_frac: float = 0.0     # fraction of last plan's elites to seed next plan


class CEMPlanner:
    def __init__(self, model: CSERJEPAv2, cfg: CEMConfig, action_mean: Tensor, action_std: Tensor):
        self.model = model.eval()
        self.cfg = cfg
        # In normalized action space: a_norm = (a - mean) / std. Planner
        # operates in normalized coords; we unnormalize when returning.
        self.action_mean = action_mean
        self.action_std = action_std
        self.d_a = int(action_mean.numel())
        # iCEM: keep last plan's elite samples for warm-start.
        self._elite_carry: Tensor | None = None

    @torch.no_grad()
    def plan(self, frame_embeds: Tensor,
             prior_fn: Callable[[Tensor], Tensor] | None = None,
             z_goal: Tensor | None = None,
             decoder: nn.Module | None = None,
             x_goal: Tensor | None = None) -> Tensor:
        """frame_embeds: (1, T_ctx, D). Returns the chosen FIRST action chunk
        as a (1, chunk_size, d_a) tensor in original (unnormalized) action space.

        prior_fn: optional callable (frame_embeds) -> (1, K, d_a) in NORMALIZED
        action space. If provided, sets the initial CEM mean for step 0 from
        the prior; remaining horizon steps start at zero (CEM iterations refit
        the rest from elites).
        """
        cfg = self.cfg
        device = frame_embeds.device
        K = self.model.chunk_size

        mu = torch.zeros(cfg.horizon, K, self.d_a, device=device)
        bc_anchor = None
        if prior_fn is not None:
            a0 = prior_fn(frame_embeds).squeeze(0).to(device)   # (K, d_a)
            mu[0] = a0
            bc_anchor = a0.detach()                              # frozen BC ref for KL-style penalty
        sigma = torch.full((cfg.horizon, K, self.d_a), cfg.init_std, device=device)

        for _ in range(cfg.n_iters):
            # Sample N candidate sequences in normalized space.
            # iCEM: colored noise (β>0) for temporally-correlated samples.
            if cfg.colored_noise_beta > 0:
                eps = _colored_noise(
                    (cfg.n_samples, cfg.horizon, K, self.d_a),
                    cfg.colored_noise_beta, time_dim=1, device=device,
                )
            else:
                eps = torch.randn(cfg.n_samples, cfg.horizon, K, self.d_a, device=device)
            samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
            # iCEM: inject carried elites (warm-start) into the sample pool.
            if cfg.elite_carry_frac > 0 and self._elite_carry is not None:
                n_carry = min(int(cfg.elite_carry_frac * cfg.n_samples), self._elite_carry.size(0))
                if n_carry > 0 and self._elite_carry.shape[1:] == samples.shape[1:]:
                    samples[:n_carry] = self._elite_carry[:n_carry]
            if cfg.action_clip is not None:
                samples = samples.clamp(-cfg.action_clip, cfg.action_clip)

            # Score each via rollout. Predictor.rollout takes (B, H, d_a) per-step
            # actions where each H step is itself a chunk of K actions; we feed it
            # the per-step planning sequence directly with chunk_size=K.
            # Actually our rollout takes action_seq of shape (B, K_total, d_a) and
            # uses each as one chunk_size=1 chunk if predictor.chunk_size=1, or
            # one full chunk if predictor.chunk_size matches. We flatten horizon
            # × K into K_total = horizon * K steps when chunk_size=1; for
            # chunk_size > 1 we need to feed full chunks. Treat each horizon step
            # as a chunk: shape becomes (B, horizon, K * d_a) folded.
            samples_chunked = samples.reshape(cfg.n_samples, cfg.horizon, K * self.d_a)
            # rollout expects (B, K_total, d_a) — we're effectively setting d_a' = K*d_a.
            # That doesn't match the predictor's d_a, so instead loop horizon manually.

            scores = torch.zeros(cfg.n_samples, device=device)
            ctx = frame_embeds.expand(cfg.n_samples, -1, -1).contiguous()
            max_ctx = self.model.predictor.frame_pos_embed.size(1)
            z_final = None
            goal_dists = []  # per-step distance to goal (when use_goal)
            pixel_dists = []  # per-step pixel-distance to goal image (when use_pixel_ground)
            for h in range(cfg.horizon):
                a_h = samples[:, h]                                # (N, K, d_a)
                z_h, r_h = self.model.predict(ctx, a_h)            # (N, D), (N, K) or (N, 1)
                if r_h.dim() > 1 and r_h.size(-1) > 1:
                    scores = scores + r_h.sum(dim=-1)
                else:
                    scores = scores + r_h.squeeze(-1)
                # Per-step proximity bonus from state decoder.
                if cfg.use_proximity:
                    s_pred = self.model.state_decoder(z_h)         # (N, 5)
                    agent_xy = s_pred[:, :2]
                    block_xy = s_pred[:, 2:4]
                    dist = (agent_xy - block_xy).pow(2).sum(-1).sqrt()
                    scores = scores - cfg.proximity_weight * dist
                # Goal-distance per step — gives CEM a smooth gradient
                # everywhere (the LeWM/DINO-WM differentiator).
                if cfg.use_goal and z_goal is not None:
                    # Multi-goal best-of-K: score against K candidate goals,
                    # take MIN distance per sample. Lets the agent target ANY
                    # solved-state cluster, not the average (which may be in
                    # an unreachable basin).
                    if z_goal.dim() == 1:
                        zg = z_goal.unsqueeze(0)                   # (1, D)
                        d2 = (z_h - zg).pow(2).mean(dim=-1)         # (N,)
                    else:
                        # z_h: (N, D); z_goal: (K, D) → pairwise (N, K) → min over K
                        diff = z_h.unsqueeze(1) - z_goal.unsqueeze(0)   # (N, K, D)
                        d2_all = diff.pow(2).mean(dim=-1)              # (N, K)
                        d2 = d2_all.min(dim=-1).values                  # (N,)
                    goal_dists.append(d2)
                # Pixel-grounded cost: decode z_h, measure pixel-MSE to goal image.
                # This catches phantom-optimum latents — if z_h is OOD, decoder
                # produces noise → high pixel distance → low score → CEM stays
                # in physically-plausible regions.
                if cfg.use_pixel_ground and decoder is not None and x_goal is not None:
                    x_h = decoder(z_h)                              # (N, 3, H, W) in [0, 1]
                    if x_goal.dim() == 3:
                        xg = x_goal.unsqueeze(0)                    # (1, 3, H, W)
                    else:
                        xg = x_goal
                    pix_d = (x_h - xg).pow(2).mean(dim=(1, 2, 3))   # (N,)
                    pixel_dists.append(pix_d)
                ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
                if ctx.size(1) > max_ctx:
                    ctx = ctx[:, -max_ctx:]
                z_final = z_h
            if cfg.use_value and z_final is not None:
                v_final = self.model.predictor.value(z_final)      # (N,)
                scores = scores + cfg.value_weight * v_final
            if cfg.use_goal and z_goal is not None and goal_dists:
                stacked = torch.stack(goal_dists, dim=0)            # (H, N)
                if cfg.goal_aggregate == "min":
                    agg = stacked.min(dim=0).values
                elif cfg.goal_aggregate == "final":
                    agg = stacked[-1]
                else:  # "mean"
                    agg = stacked.mean(dim=0)
                scores = scores - cfg.goal_weight * agg
                # Drift penalty: also penalize the final-step distance.
                if cfg.goal_drift_weight > 0:
                    scores = scores - cfg.goal_drift_weight * stacked[-1]

            if cfg.use_pixel_ground and pixel_dists:
                stacked_p = torch.stack(pixel_dists, dim=0)          # (H, N)
                if cfg.pixel_ground_aggregate == "min":
                    agg_p = stacked_p.min(dim=0).values
                elif cfg.pixel_ground_aggregate == "final":
                    agg_p = stacked_p[-1]
                else:  # "mean"
                    agg_p = stacked_p.mean(dim=0)
                scores = scores - cfg.pixel_ground_weight * agg_p

            # BC anchor: penalize divergence of first-horizon action chunk from BC's
            # prior. Shrinks the elite distribution toward BC under more CEM iters,
            # preventing world-model exploitation. Only applies if prior_fn was given.
            if cfg.use_bc_anchor and bc_anchor is not None:
                a_first = samples[:, 0]                              # (N, K, d_a)
                drift = (a_first - bc_anchor.unsqueeze(0)).pow(2).sum(dim=(-1, -2))  # (N,)
                scores = scores - cfg.bc_anchor_weight * drift

            # Top-K-elite refit.
            elite_idx = scores.topk(cfg.n_elite).indices
            elite = samples[elite_idx]                              # (n_elite, horizon, K, d_a)
            mu = elite.mean(dim=0)
            sigma = elite.std(dim=0).clamp_min(cfg.min_std)

        # iCEM: stash final elites (shifted by 1 in horizon, last step random)
        # for next plan's warm-start. Shifting accounts for the fact that we
        # execute K_action steps before replanning.
        if cfg.elite_carry_frac > 0 and elite is not None:
            shifted = torch.zeros_like(elite)
            shifted[:, :-1] = elite[:, 1:].detach()
            shifted[:, -1] = elite[:, -1].detach()  # repeat last (or could random-init)
            self._elite_carry = shifted

        # Return first chunk in original action space.
        a0_norm = mu[0]                                             # (K, d_a)
        a0 = a0_norm * self.action_std.to(device) + self.action_mean.to(device)
        return a0.unsqueeze(0)
