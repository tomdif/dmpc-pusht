"""CSER-JEPA-v3: bitter-lesson-pure, era-of-experience-shaped.

Architecture: ViT encoder + transformer predictor. That's it.

  encoder(x_context) -> frame_embeds (B, T, D)
  predictor(frame_embeds, a_chunk) -> (ẑ_target, r̂)
  predictor.rollout(frame_embeds, action_seq) -> (z_traj, r_traj)

No history transformer (predictor self-attends over context directly), no
residual decomposition, no spectral path, no fallback head, no inverse
action model, no auxiliary IDM-z, no contrastive loss, no curriculum
schedules. Whatever structure relates (history, action) to next-state is
learned from data, with anti-collapse provided by Epps-Pulley SIGReg.
"""

from __future__ import annotations

from torch import Tensor, nn

from .aux import EncoderActionDecoder, StateDecoder
from .encoder import VideoEncoder
from .predictor import Predictor


class CSERJEPAv2(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        enc = cfg["encoder"]
        pred = cfg["predictor"]
        action = cfg["action"]
        idm_cfg = cfg.get("idm_z", {})

        self.d = int(enc["embed_dim"])
        self.d_a = int(action["d_a"])
        self.chunk_size = int(action["chunk_size"])

        self.encoder = VideoEncoder(**enc)
        self.predictor = Predictor(
            d=self.d,
            d_a=self.d_a,
            chunk_size=self.chunk_size,
            **pred,
        )
        self.encoder_idm = EncoderActionDecoder(
            d_z=self.d,
            d_a=self.d_a,
            chunk_size=self.chunk_size,
            hidden=int(idm_cfg.get("hidden", 128)),
            depth=int(idm_cfg.get("depth", 3)),
        )
        # Optional state decoder: z → (agent_xy, block_xytheta). Lit by
        # lambda_state > 0 in trainer; used by CEM for proximity reward.
        self.state_decoder = StateDecoder(
            d_z=self.d,
            hidden=int(idm_cfg.get("hidden", 128)),
            depth=int(idm_cfg.get("depth", 3)),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def predict(self, frame_embeds: Tensor, a_chunk: Tensor) -> tuple[Tensor, Tensor]:
        return self.predictor(frame_embeds, a_chunk)

    def rollout(self, frame_embeds: Tensor, action_seq: Tensor) -> tuple[Tensor, Tensor]:
        return self.predictor.rollout(frame_embeds, action_seq)

    def infer_action_z(self, z_t: Tensor, z_target: Tensor) -> Tensor:
        return self.encoder_idm(z_t, z_target)
