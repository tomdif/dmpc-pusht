"""Stage 0 trainer for CSER-JEPA-v3 (bitter-lesson + era-of-experience).

Two losses, no curriculum:

    L = L_pred                                   normalized MSE on z_target
      + lambda_reward * L_reward                  MSE on r_target (0 in toy)
      + lambda_reg * L_sigreg_epps_pulley         anti-collapse on encoder features

That is the entire training objective. No spectral path, no residual
decomposition, no inverse model, no contrastive head, no cleanup ramp,
no generator regularizer. The transformer learns whatever structure
relates (history, action) to next-state from data.

Reward head is wired but lambda_reward defaults to 0 in toy — keeps the
architecture rollout-ready for downstream agentic training where reward
becomes the grounding signal per Era of Experience §Rewards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..losses import sigreg_epps_pulley, sigreg_lewm
from ..models.world_model import CSERJEPAv2
from ..types import LabeledBatch
from ..utils.normalization import RunningScalar, normalized_mse


@dataclass
class Stage0Config:
    lambda_pred: float = 1.0
    lambda_reward: float = 0.0
    lambda_value: float = 0.0     # value-head bootstrap (Stage Z onward)
    lambda_state: float = 0.0     # state decoder for proximity reward (Stage AF onward)
    lambda_pred_multi: float = 0.0  # multi-step prediction loss (Stage AH onward)
    lambda_idm_z: float = 5.0     # general auxiliary anti-collapse signal,
                                  # not a domain prior — see models/aux.py
    lambda_reg: float = 1.0
    sigreg_variant: str = "epps_pulley"  # "epps_pulley" (16-proj pairwise) | "lewm" (1024-proj moment-fit)
    total_steps: int = 2000


class Stage0Trainer:
    def __init__(self, model: CSERJEPAv2, cfg: Stage0Config) -> None:
        self.model = model
        self.cfg = cfg
        self.sigma_z = RunningScalar(init=1.0, momentum=0.99)

    def step(self, labeled: LabeledBatch, global_step: int) -> tuple[Tensor, dict[str, Tensor]]:
        diags: dict[str, Tensor] = {}
        m = self.model

        frame_embeds = m.encode(labeled.x_context)                  # (B, T, D)
        z_t = frame_embeds[:, -1, :] if frame_embeds.dim() == 3 else frame_embeds
        z_target = m.encode(labeled.x_target)                       # (B, D)
        z_pred, r_pred = m.predict(frame_embeds, labeled.a_chunk)   # (B, D), (B, 1)

        self.sigma_z.update(z_target.detach())
        sigma2 = self.sigma_z.get()
        D = z_target.size(-1)

        L_pred = normalized_mse(z_pred, z_target, sigma2, D)
        diags["loss/pred"] = L_pred.detach()

        # Multi-step prediction: roll out predictor autoregressively in z-space
        # and supervise each horizon's predicted z against the encoder output
        # of the corresponding ground-truth future frame. Trains the model to
        # be self-consistent under the same recursion CEM uses at inference.
        if (
            self.cfg.lambda_pred_multi > 0
            and getattr(labeled, "x_target_multi", None) is not None
        ):
            # x_target_multi: (B, H-1, C, H, W); a_chunk_multi: (B, H-1, K, d_a)
            B = labeled.x_target_multi.size(0)
            H_extra = labeled.x_target_multi.size(1)
            # Build rolled-context starting from frame_embeds + initial z_pred.
            ctx = torch.cat([frame_embeds, z_pred.unsqueeze(1)], dim=1)  # (B, T+1, D)
            max_ctx = m.predictor.frame_pos_embed.size(1)
            multi_terms = []
            for h in range(H_extra):
                a_h = labeled.a_chunk_multi[:, h]                        # (B, K, d_a)
                z_h, _ = m.predict(ctx, a_h)                              # (B, D)
                # Encode the corresponding target frame to get the ground-truth z.
                tgt_h = labeled.x_target_multi[:, h]                      # (B, C, H, W)
                z_tgt_h = m.encode(tgt_h)
                multi_terms.append(normalized_mse(z_h, z_tgt_h, sigma2, D))
                # Slide context.
                ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
                if ctx.size(1) > max_ctx:
                    ctx = ctx[:, -max_ctx:]
            L_pred_multi = torch.stack(multi_terms).mean()
        else:
            L_pred_multi = z_pred.new_zeros(())
        diags["loss/pred_multi"] = L_pred_multi.detach()

        if self.cfg.lambda_reward > 0:
            r_chunk = getattr(labeled, "r_chunk", None)
            if r_chunk is None:
                L_reward = z_pred.new_zeros(())
            else:
                if r_pred.shape == r_chunk.shape:
                    L_reward = (r_pred - r_chunk).pow(2).mean()
                else:
                    L_reward = (r_pred.squeeze(-1) - r_chunk.sum(dim=-1)).pow(2).mean()
        else:
            L_reward = z_pred.new_zeros(())
        diags["loss/reward"] = L_reward.detach()

        # Value bootstrap: V(z_target) → return-to-go. Trains only on samples
        # with real rtg (synthetic data); LeRobot frames pass with mask=0.
        if self.cfg.lambda_value > 0 and getattr(labeled, "rtg_target", None) is not None:
            v_pred = m.predictor.value(z_target)
            rtg = labeled.rtg_target
            mask = labeled.rtg_mask if labeled.rtg_mask is not None else torch.ones_like(rtg)
            sq = (v_pred - rtg).pow(2)
            denom = mask.sum().clamp_min(1.0)
            L_value = (sq * mask).sum() / denom
        else:
            L_value = z_pred.new_zeros(())
        diags["loss/value"] = L_value.detach()

        # State decoder: z_target → (agent_xy, block_xytheta). Trains only on
        # samples with state labels (newer self-play). Used by CEM at planning
        # for proximity reward (Stage AF).
        if self.cfg.lambda_state > 0 and getattr(labeled, "state_target", None) is not None:
            s_pred = m.state_decoder(z_target)
            s_tgt = labeled.state_target
            s_mask = labeled.state_mask if labeled.state_mask is not None else torch.ones(s_tgt.size(0), device=s_tgt.device)
            sq = (s_pred - s_tgt).pow(2).mean(dim=-1)
            denom = s_mask.sum().clamp_min(1.0)
            L_state = (sq * s_mask).sum() / denom
        else:
            L_state = z_pred.new_zeros(())
        diags["loss/state"] = L_state.detach()

        # IDM-z: general auxiliary anti-collapse on the encoder. (z_t, z_target) → a_t.
        # Encodes no assumption about how actions act on state — just uses
        # the action labels as supervised gradient channel into the encoder.
        a_pred_z = m.infer_action_z(z_t, z_target)
        L_idm_z = (a_pred_z - labeled.a_chunk).pow(2).mean()
        diags["loss/idm_z"] = L_idm_z.detach()

        # SIGReg over both context-frame and target embeddings — keeps the
        # encoder from collapsing to low-rank features regardless of how
        # predictable z_target gets.
        z_for_reg = torch.cat([frame_embeds.flatten(0, 1), z_target], dim=0)
        if self.cfg.sigreg_variant == "lewm":
            L_reg = sigreg_lewm(z_for_reg)
        else:
            L_reg = sigreg_epps_pulley(z_for_reg)
        diags["loss/reg"] = L_reg.detach()

        L_total = (
            self.cfg.lambda_pred * L_pred
            + self.cfg.lambda_pred_multi * L_pred_multi
            + self.cfg.lambda_reward * L_reward
            + self.cfg.lambda_value * L_value
            + self.cfg.lambda_state * L_state
            + self.cfg.lambda_idm_z * L_idm_z
            + self.cfg.lambda_reg * L_reg
        )
        diags["loss/total"] = L_total.detach()
        diags["scale/sigma_z2"] = torch.tensor(sigma2)
        return L_total, diags
