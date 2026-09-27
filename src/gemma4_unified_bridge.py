# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""Phase-0 text-tower bridge for Gemma-4 Unified (`google/gemma-4-12B`, omni).

`google/gemma-4-12B` is `Gemma4UnifiedForConditionalGeneration` (model_type
`gemma4_unified`) — a 12B DENSE, encoder-free omni model (text + vision + audio).
Megatron-Bridge rejects it today only because no bridge is registered for that
architecture. This module registers one **for the text tower**, reusing the
existing Gemma-4 *dense* provider + weight map wholesale.

Why this is enough for Phase 0:
  * Unified's `text_config` exposes the exact fields `_build_dense_provider`
    already reads: `num_hidden_layers`, `hidden_size`, `layer_types`,
    `sliding_window`, `rope_parameters` (dual local/global, proportional),
    `num_key_value_heads`, `num_global_key_value_heads` (→ global MQA=1),
    `head_dim`/`global_head_dim` (256/512), `attention_k_eq_v`,
    `num_kv_shared_layers`, `hidden_size_per_layer_input` (=0 → PLE disabled),
    `final_logit_softcapping`.
  * The dense weight map already maps `layer_scalar`, uses `_Gemma4DenseQKVMapping`
    that tolerates the missing `v_proj` on K=V global layers, and applies the
    logit-softcap output layer — no new attention/RoPE/norm code.

The only two things that differ for Unified:
  1. the text params live under a nested `text_config`, and
  2. the HF weights are under `model.language_model.*` (not `model.*`).

Vision (`model.embed_vision.*`) and audio (`model.embed_audio.*`) towers are
intentionally NOT mapped here — that is Phase 1/2 (MIMO projectors).

Limitation inherited from `Gemma4DenseProvider`: PP=1 only.
"""

from megatron.core.models.gpt import GPTModel

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma.gemma4_bridge import Gemma4Bridge
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider


@MegatronModelBridge.register_bridge(
    source="Gemma4UnifiedForConditionalGeneration",
    target=GPTModel,
    provider=Gemma4DenseProvider,
    model_type="gemma4_unified",
)
class Gemma4UnifiedBridge(Gemma4Bridge):
    """Text-tower bridge for the Gemma-4 Unified (encoder-free omni) model.

    Phase 0 = language model only. Reuses the Gemma-4 dense provider, weight
    map, K=V synthesis and softcap output layer; reads the nested ``text_config``
    and points the HF weight prefix at ``model.language_model.``.
    """

    def provider_bridge(self, hf_pretrained) -> Gemma4DenseProvider:
        hf_config = hf_pretrained.config
        text_config = getattr(hf_config, "text_config", hf_config)
        # Force dense path; expose the text config to mapping + K=V synthesis.
        self._is_dense = True
        self._unified_text_config = text_config
        return self._build_dense_provider(text_config)

    def _text_config(self):
        tc = getattr(self, "_unified_text_config", None)
        return tc if tc is not None else getattr(self, "hf_config", None)

    def _hf_layer_prefix(self) -> str:
        # Unified nests the language model under model.language_model.*
        return "model.language_model."
