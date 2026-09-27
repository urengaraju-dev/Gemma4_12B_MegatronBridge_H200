# Adding `google/gemma-4-12B` (Gemma4Unified) support to Megatron-Bridge — feasibility

**Date:** 2026-09-27 · **Dev box:** 1× NVIDIA H200 (143 GB) · **Container:** `nvcr.io/nvidia/nemo:26.08` (Megatron-Bridge 0.6.1, Megatron-Core 0.19.1)

## Executive summary (revised)

Earlier I called this "a non-trivial upstream addition, realistically a future release." **After grounding it in source + the installed container, that was too pessimistic.** The correct conclusion:

- **A text-only bridge (Phase 0) is hand-rollable now on the single H200** — it is a *fork-a-provider + register-one-bridge + populate-config + grind logit-parity* job, **not** a from-scratch Megatron-Core port.
- `AutoBridge` rejects `google/gemma-4-12B` today **only because no bridge is registered** for `Gemma4UnifiedForConditionalGeneration` — the architecture-suffix gate already passes; it's the missing registration that raises.
- Megatron-Bridge already ships **almost all** the hard machinery: a full Gemma4 **dense text stack** (dual/proportional RoPE, 5:1 sliding/full interleave, per-layer head-dim, K=V attention, softcap, MQA) **and** a **MIMO omni framework** with **vision *and* audio** modality submodules and an encoder-free (projection-only) path — exactly what this encoder-free model needs.
- The omni phases (vision, then audio/video) are more genuinely-new code but also tractable on the existing MIMO scaffold.

## Verified on the installed container (Megatron-Bridge 0.6.1)

These are empirical results from running inside `nvcr.io/nvidia/nemo:26.08`, not desk inference:

| Check | Result |
|---|---|
| `AutoBridge.from_hf_pretrained("google/gemma-4-12B")` | **`ValueError` (rejected)** — no registered bridge for the Unified arch |
| Registration mechanism | `@MegatronModelBridge.register_bridge(source="...", target=, provider=)` — string-source; e.g. `source="Gemma4ForCausalLM"→Gemma4ModelProvider`, `source="Gemma4ForConditionalGeneration"→Gemma4VLModel/Gemma4VLModelProvider` |
| MCore `TransformerConfig` | has `window_size`, `softmax_scale`, `attention_softmax_in_fp32`, `layernorm_zero_centered_gamma` (Gemma `(1+w)` RMSNorm), `kv_channels`, `num_query_groups`, `multi_latent_attention` |
| Sliding/full + dual RoPE | Gemma3 provider does it via `interleaved_attn_pattern=(5,1)`, `rotary_base=(1e4,1e6)`, `Gemma3RotaryEmbedding` (stacks local+global), `_is_local_attn_layer(layer_types)` |
| MIMO / omni framework | **present**: `megatron.core.models.mimo` → `MimoModel`, `VisionModalitySubmodules`, `AudioModalitySubmodules`; bridge `MegatronMIMOProvider`; steps `megatron_mimo_step`, `nemotron_omni_step`, `audio_lm_step` |

The installed 0.6.1 already exposes `Gemma4DenseProvider` / `Gemma4ModelProvider` and `gemma4_bridge.py` / `gemma4_vl_bridge.py`, so the fork base exists in the container (upstream `main` is a bit newer; verify a couple of items below against the *pinned* version before committing).

## Target architecture (load-bearing details)

`Gemma4UnifiedForConditionalGeneration`, `model_type="gemma4_unified"`, 12B **dense**, three towers:

- **Text** (`model.language_model.`): 48 layers, hidden 3840, ffn 15360, gelu-tanh gated MLP, vocab 262144, tied embeddings, final logit softcap 30. **Heterogeneous per-layer attention** — `layer_types=[sliding×5, full×1]×8`: sliding layers `head_dim=256`, 8 KV heads, real `v_proj`, window 1024, RoPE default θ=1e4; full/global layers `head_dim=512`, **1 KV head (MQA)**, **K=V (`attention_k_eq_v`, no `v_proj`)**, RoPE **proportional** (partial_rotary_factor 0.25) θ=1e6. Softmax scale hard-set to 1.0 (q/k pre-normed). Plain-weight RMSNorm (not Gemma3 `(1+w)`), some scaleless norms, a per-layer `layer_scalar` buffer, 4-norm sandwich, **no PLE** (`hidden_size_per_layer_input=0`).
- **Vision** (`model.embed_vision.`, encoder-free): raw 48²×3 patch → LN→Dense(→3840)→LN → +factorized 2D posemb → norm → Linear(3840→3840); ~280 soft tokens/image.
- **Audio** (`model.embed_audio.`, encoder-free): a single Linear(640→3840).
- **Splice:** placeholder token ids (image 258880, audio 258881, video 258884, + boi/eoi/boa/eoa) `masked_scatter`-ed with projected features. `use_bidirectional_attention="vision"` — bidirectional within a vision span on **sliding layers only**.

## The delta (what to build)

**Phase 0 — text bridge (makes text LoRA/SFT of gemma-4-12B work on MB)**
- `Gemma4UnifiedBridge`: `@register_bridge(source="Gemma4UnifiedForConditionalGeneration", target=<text GPTModel>, provider=Gemma4UnifiedDenseProvider, model_type="gemma4_unified")`; fork the Gemma4 dense weight map with `_hf_layer_prefix()="model.language_model."`.
- `Gemma4UnifiedDenseProvider`: fork `Gemma4DenseProvider`; populate `global_head_dim=512`, **`num_global_key_value_heads=1`** (base default 2), `head_dim=256`, softcap 30, dual-RoPE thetas, `attention_k_eq_v=true`; **disable PLE**; handle the `layer_scalar` per-layer output multiplier; ensure norm has no `+1` offset and `softmax_scale=1.0`.
- Reuse: per-layer config clone, K=V synthesis, proportional-RoPE, softcap output layer, `(1+w)`/zero-centered-gamma toggle — all already in the Gemma4 stack.

**Phase 1 — + vision** — new encoder-free vision projector module wired as a MIMO `input_projections` (no encoder); vision weight map (module-name authoritative — confirm against the real `state_dict().keys()`); the **new sliding-layer blockwise-bidirectional mask**; Gemma4 processor/collate for images. Reuse MIMO `masked_scatter` splice + `megatron_mimo_step`.

**Phase 2 — + audio/video** — audio projector is one Linear; add `modality_submodules_spec["audio"]`, audio/video token ids, frame collate. Small increment on Phase 1's MIMO plumbing.

## Hardest risks (ranked)

1. **Per-layer head_dim (512 vs 256) + global MQA-1 + K=V** — the structural hazard. **Reuse/extend** (mechanism exists in `Gemma4DenseSelfAttention` per-layer clone + K=V path; the *extend* is populating `num_global_key_value_heads=1`).
2. **Compound numeric parity** — proportional RoPE + softcap + plain-weight RMSNorm + softmax-scale-1.0 + `layer_scalar`. Each solved/trivial individually; they compound in the logit-parity check. Mostly **reuse**, a little **new** (`layer_scalar`, norm-offset, scale=1.0).
3. **Bidirectional vision mask (sliding-only, blockwise)** — **new**; not the default causal mask, and even Gemma4-VL isn't a drop-in (it's all-layer). Phase 1+.
4. **Vision projector key naming + patched EOI/EOA rows** — **extend**, but gated on reading the real checkpoint's keys.
5. **`layer_scalar` buffer / PLE-off path** — small **new/extend** items, easy to forget → parity drift.

Everything in the text tower is **reuse/extend, not new-kernel**. Genuinely new code is concentrated in Phases 1–2.

## Dev loop on the single H200

12B bf16 fits for inference/LoRA/parity. Loop: implement bridge/provider → `AutoBridge` convert HF→Megatron → `examples/conversion/compare_hf_and_megatron/compare.py` for logit parity vs HF on a text prompt → fix compound-numerics → short LoRA/SFT smoke. Text-only parity needs no multimodal inputs. TP=1/PP=1/CP=1.

## Recommendation

- **Hand-roll Phase 0 now** for text LoRA/SFT of gemma-4-12B — do not wait for upstream. Residual risk is a handful of parity-catchable items.
- **Before writing code**, verify (cheap, in the installed source; no checkpoint needed): (a) does `Gemma4DenseProvider` no-op PLE when `hidden_size_per_layer_input=0`; (b) does `modeling_gemma4` RMSNorm use plain weight or `+1`; (c) does the dense weight-map already carry a per-layer scalar; (d) proportional-RoPE at head_dim 512 / factor 0.25. Then read the real `state_dict().keys()` to lock projector naming for Phase 1.
- Start Phase 1 (vision) only after the checkpoint's vision key naming is confirmed; treat Phase 2 (audio) as a small increment.

## Caveats

Detailed line-referenced internals came from Megatron-Bridge `main`; the container is 0.6.1 (slightly older) — reconfirm the fork points against the pinned version. Numerical parity of the *existing* Gemma4 code to HF `gemma-4-12B` was not run; the UNVERIFIED items above must be checked before locking effort.
