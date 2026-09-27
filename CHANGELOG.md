# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses date-based tags while pre-1.0.

## [Unreleased]

### Planned
- **Phase 2 (audio):** wire the single `model.embed_audio` Linear (640→3840) projector onto the existing merge scaffold.
- Fold `Gemma4UnifiedVLModel` into `MegatronMIMOProvider` + `megatron_mimo_step`, and register the vision/audio weights in the bridge mapping registry (removes the single-GPU wrapper).
- Pipeline-parallel support for `Gemma4DenseProvider` (currently PP=1 only).

## [0.2.0] — 2026-09-27

### Added — Phase 1 (vision)
- `src/gemma4_vision_projector.py` — encoder-free vision embedder reproducing HF `Gemma4UnifiedVisionEmbedder` + `Gemma4UnifiedMultimodalEmbedder`; **bit-exact vs HF** (`test_vision_parity.py`, cosine 1.000000).
- `src/gemma4_vision_mask.py` — bidirectional-in-vision-span attention mask, injected via `FusedScaleMaskSoftmax` (`attn_mask_type=padding`) since MCore's local `DotProductAttention` ignores the per-call mask on its fused-causal path.
- `src/gemma4_unified_vl.py` — `Gemma4UnifiedVLModel`: merges vision soft tokens into the LM embedding stream via `masked_scatter` at `image_token_id=258880`, runs the LM via `decoder_input`.
- Parity/diagnostic harnesses: `phase1_parity.py`, `test_mask_diff.py`, `test_hidden_diff.py`, `test_layer_diff.py`.

### Fixed
- **Text embeddings scaled by √hidden** in the multimodal path (Megatron returns unscaled embeddings; HF scales text but not image features) — found via a scale-invariant hidden-state diff.
- **Bidirectional vision mask applied to all 48 layers**, not sliding-only — HF's `full_attention` per-layer mask is identical to the sliding one at short lengths.

### Result
- Image-conditioned parity vs HF: **image-position cosine 0.999 (bit-exact), text cosine 0.975 (top-1 83%)** — up from cosine 0.49 before the two fixes. Residual text gap characterized as bf16 accumulation, not a discrete bug (`test_layer_diff.py`).

## [0.1.0] — 2026-09-27

### Added — Phase 0 (text)
- `src/gemma4_unified_bridge.py` — `Gemma4UnifiedBridge`, a ~30-line subclass of `Gemma4Bridge` registered for `Gemma4UnifiedForConditionalGeneration`; reads the nested `text_config` and points the HF weight prefix at `model.language_model.`, reusing Megatron-Bridge's full Gemma-4 dense text stack.
- `tests/test_phase0.py` — build the 12B text model + forward (config only, no download).
- `tests/parity_g4.py` — HF vs Megatron logit parity.
- `tests/train_lora_g4.py` — 3-step LoRA finetune smoke.

### Result
- Phase 0 text parity vs HF: **cosine 0.9999, top-1 agreement 100%**; LoRA smoke loss 1.86 → 1.12 with no NaNs.
