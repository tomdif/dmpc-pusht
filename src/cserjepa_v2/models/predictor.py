"""Bitter-lesson-pure predictor + reward head, rollout-capable.

A single transformer that ingests context-frame embeddings, an action
chunk, and emits the next-frame embedding (and reward). No residual
decomposition, no spectral path, no skew-symmetric generator prior, no
fallback head. Whatever structure relates (history, action) to next-state
is learned from data.

Designed against two principles:

  Sutton (2019, "The Bitter Lesson"): general methods that scale with
  compute beat hand-encoded structure. The predictor is a transformer
  with no domain-specific architectural priors about action dynamics.

  Silver & Sutton (2025, "Era of Experience"): a world model should
  predict action consequences AND reward, support rollouts over multiple
  steps for planning, and operate in the encoder's embedding space so it
  can be unrolled autoregressively. Reward head is included even if
  rewards are absent (λ=0) so downstream agentic training can light it up
  without re-architecting.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import TransformerBlock


class Predictor(nn.Module):
    def __init__(
        self,
        d: int,
        d_a: int,
        chunk_size: int,
        n_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_context: int = 16,
    ) -> None:
        super().__init__()
        self.d = d
        self.d_a = d_a
        self.chunk_size = chunk_size

        self.action_embed = nn.Linear(d_a * chunk_size, d)
        self.target_token = nn.Parameter(torch.zeros(1, 1, d))
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, max_context, d))
        self.role_embed = nn.Parameter(torch.zeros(1, 3, d))  # frame / action / target

        nn.init.trunc_normal_(self.target_token, std=0.02)
        nn.init.trunc_normal_(self.frame_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.role_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d, num_heads, mlp_ratio, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d)
        # Per-step reward head: predicts K scalars (one per action in the chunk)
        # rather than a single chunk-summed scalar. Lets the planner score
        # action sequences with per-step granularity, addressing the
        # over-confidence pathology that bigger world models surface.
        self.reward_head = nn.Linear(d, chunk_size)
        nn.init.zeros_(self.reward_head.bias)
        # Value head: predicts return-to-go from the predicted next-state.
        # Used by the planner as a bootstrap at the horizon — gives credit
        # for paths leading to engagement, not just paths earning immediate
        # reward. Standard MPC+V formulation.
        self.value_head = nn.Linear(d, 1)
        nn.init.zeros_(self.value_head.bias)

    def value(self, z: Tensor) -> Tensor:
        """V(z) — return-to-go bootstrap."""
        return self.value_head(z).squeeze(-1)

    def forward(self, frame_embeds: Tensor, a_chunk: Tensor) -> tuple[Tensor, Tensor]:
        """frame_embeds: (B, T, D). a_chunk: (B, K, d_a). Returns (ẑ, r̂) with shapes (B, D), (B, K)."""
        B, T, D = frame_embeds.shape
        a_flat = a_chunk.reshape(B, -1)
        a_tok = self.action_embed(a_flat).unsqueeze(1)
        target_tok = self.target_token.expand(B, -1, -1)

        frames = frame_embeds + self.frame_pos_embed[:, :T] + self.role_embed[:, 0:1]
        a_tok = a_tok + self.role_embed[:, 1:2]
        target_tok = target_tok + self.role_embed[:, 2:3]

        x = torch.cat([frames, a_tok, target_tok], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        z_pred = x[:, -1]
        r_pred = self.reward_head(z_pred)            # (B, chunk_size)
        return z_pred, r_pred

    @torch.no_grad()
    def rollout(self, frame_embeds: Tensor, action_seq: Tensor) -> tuple[Tensor, Tensor]:
        """Multi-step rollout for planning.

        action_seq: (B, K_total, d_a) — sequence of action chunks to apply.
        Returns:
            z_traj: (B, K_total, D) — predicted next-frame embeddings.
            r_traj: (B, K_total, K) — predicted per-step rewards over each chunk.

        Each step the latest predicted ẑ is appended to the context window
        (sliding) and fed back; the agent's own predicted future drives
        the next prediction.
        """
        B, K_total, _ = action_seq.shape
        ctx = frame_embeds
        max_ctx = self.frame_pos_embed.size(1)
        z_traj, r_traj = [], []
        for k in range(K_total):
            a_k = action_seq[:, k:k + 1]
            z_k, r_k = self.forward(ctx, a_k)
            z_traj.append(z_k)
            r_traj.append(r_k)
            ctx = torch.cat([ctx, z_k.unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
        return torch.stack(z_traj, dim=1), torch.stack(r_traj, dim=1)
