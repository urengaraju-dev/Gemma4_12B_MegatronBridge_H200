# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 encoder-free VISION projector for Gemma-4 Unified (google/gemma-4-12B).

Reproduces HF `Gemma4UnifiedVisionEmbedder.forward` + `Gemma4UnifiedMultimodalEmbedder`
exactly, so it can be wired as the single `input_projection` of a MIMO
`VisionModalitySubmodules` (encoder-free: raw patches in, LM-dim soft tokens out).

HF pipeline (verified against transformers 5.12.1 modular_gemma4_unified.py):
    pixel_values [., num_patches, 48*48*3=6912], image_position_ids [., num_patches, 2]
      -> patch_ln1 (LayerNorm 6912)
      -> patch_dense (Linear 6912->3840)
      -> patch_ln2 (LayerNorm 3840)
      -> + factorized 2D pos_embedding[mm_posemb_size=1120, 2, 3840] indexed by (x,y)
      -> pos_norm (LayerNorm 3840)
      -> embedding_pre_projection_norm (scaleless RMSNorm)
      -> embedding_projection (Linear 3840->3840, no bias)

Checkpoint weight keys (single model.safetensors):
    model.vision_embedder.patch_ln1.{weight,bias}
    model.vision_embedder.patch_dense.{weight,bias}
    model.vision_embedder.patch_ln2.{weight,bias}
    model.vision_embedder.pos_embedding
    model.vision_embedder.pos_norm.{weight,bias}
    model.embed_vision.embedding_projection.weight     (the scaleless RMSNorm has no weight)
"""
import torch
import torch.nn as nn


class ScalelessRMSNorm(nn.Module):
    """RMSNorm with no learnable scale (fp32 compute), matching HF with_scale=False."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dt)


class Gemma4UnifiedVisionProjector(nn.Module):
    """Encoder-free vision embedder: raw pixel patches -> LM-dim soft tokens.

    forward(pixel_values [.,P,6912], image_position_ids [.,P,2]) -> [.,P,3840]
    (Caller strips padding patches — position_ids == -1 — before scatter.)
    """

    def __init__(
        self,
        patch_dim: int = 6912,      # model_patch_size(48)**2 * 3
        mm_embed_dim: int = 3840,   # text hidden_size
        mm_posemb_size: int = 1120,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.patch_ln1 = nn.LayerNorm(patch_dim, dtype=dtype)
        self.patch_dense = nn.Linear(patch_dim, mm_embed_dim, dtype=dtype)
        self.patch_ln2 = nn.LayerNorm(mm_embed_dim, dtype=dtype)
        self.pos_embedding = nn.Parameter(torch.zeros(mm_posemb_size, 2, mm_embed_dim, dtype=dtype))
        self.pos_norm = nn.LayerNorm(mm_embed_dim, dtype=dtype)
        self.embedding_pre_projection_norm = ScalelessRMSNorm(eps=rms_norm_eps)
        self.embedding_projection = nn.Linear(mm_embed_dim, mm_embed_dim, bias=False, dtype=dtype)

    def forward(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
        h = self.patch_ln1(pixel_values.to(self.patch_dense.weight.dtype))
        h = self.patch_dense(h)
        h = self.patch_ln2(h)
        # factorized 2D positional embedding (padding positions == -1 contribute 0)
        clamped = image_position_ids.clamp(min=0).long()
        valid = (image_position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
        axes = torch.arange(2, device=image_position_ids.device)
        pos = (self.pos_embedding[clamped, axes] * valid).sum(-2)
        h = h + pos
        h = self.pos_norm(h)
        # base multimodal embedder: scaleless RMSNorm -> Linear
        h = self.embedding_projection(self.embedding_pre_projection_norm(h))
        return h

    @torch.no_grad()
    def load_from_hf_state_dict(self, sd: dict, vp: str = "model.vision_embedder.", ep: str = "model.embed_vision."):
        """Copy weights from a HF Gemma4Unified state_dict (by checkpoint key names)."""
        self.patch_ln1.weight.copy_(sd[vp + "patch_ln1.weight"]);  self.patch_ln1.bias.copy_(sd[vp + "patch_ln1.bias"])
        self.patch_dense.weight.copy_(sd[vp + "patch_dense.weight"]); self.patch_dense.bias.copy_(sd[vp + "patch_dense.bias"])
        self.patch_ln2.weight.copy_(sd[vp + "patch_ln2.weight"]);  self.patch_ln2.bias.copy_(sd[vp + "patch_ln2.bias"])
        self.pos_embedding.copy_(sd[vp + "pos_embedding"])
        self.pos_norm.weight.copy_(sd[vp + "pos_norm.weight"]);    self.pos_norm.bias.copy_(sd[vp + "pos_norm.bias"])
        self.embedding_projection.weight.copy_(sd[ep + "embedding_projection.weight"])
