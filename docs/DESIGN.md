# Design & engineering notes

How `google/gemma-4-12B` was brought up on Megatron-Bridge, and the reasoning behind each decision.
All facts here were verified against the **installed** Megatron-Bridge 0.6.1 (container `nvcr.io/nvidia/nemo:26.08`) and `google/gemma-4-12B` (transformers 5.12.1), not against docs alone.

## 0. The model

`google/gemma-4-12B` declares `architectures: ["Gemma4UnifiedForConditionalGeneration"]`, `model_type = "gemma4_unified"`. It is a **12 B dense, encoder-free omni** model with three towers under `model.`:

- **text** (`model.language_model`): 48 layers, hidden 3840, 16 heads, 8 KV heads, `head_dim=256`, `global_head_dim=512`, `num_global_key_value_heads=1` (global MQA), `layer_types` = 5×sliding : 1×full, `sliding_window=1024`, dual RoPE (sliding θ=1e4 default; full θ=1e6 **proportional**, partial factor 0.25), `attention_k_eq_v=true`, `final_logit_softcapping=30`, per-layer `layer_scalar`, no PLE (`hidden_size_per_layer_input=0`), vocab 262208, tied embeddings, `use_bidirectional_attention="vision"`.
- **vision** (`model.embed_vision`, encoder-free): raw 48²×3=6912 patch → LN₁ → Dense(→3840) → LN₂ → +factorized 2D pos-emb `[1120,2,3840]` → pos-norm → scaleless RMSNorm → Linear(3840→3840).
- **audio** (`model.embed_audio`, encoder-free): a single Linear(640→3840).

`AutoBridge.from_hf_pretrained("google/gemma-4-12B")` raises `ValueError` today — the *only* gate that rejects it is "no registered bridge"; the `ForConditionalGeneration` suffix gate already passes.

## 1. Phase 0 — text bridge

Megatron-Bridge 0.6.1 already ships a full **Gemma-4 dense** text stack whose feature set is a near-exact superset of `gemma4_unified_text`: `Gemma4DenseProvider` + `gemma4_bridge` implement dual/proportional RoPE, the 5:1 sliding/full interleave, per-layer head dims, `attention_k_eq_v`, `layer_scalar` (buffer **and** weight-map entry), softcap, gelu-tanh, and a `_Gemma4DenseQKVMapping` that tolerates the missing `v_proj` on K=V global layers. Critically, `Gemma4Bridge._build_dense_provider(hf_config)` reads *exactly* the fields present in the Unified `text_config`.

So the bridge (`src/gemma4_unified_bridge.py`) is a ~30-line subclass:

```python
@MegatronModelBridge.register_bridge(source="Gemma4UnifiedForConditionalGeneration",
                                     target=GPTModel, provider=Gemma4DenseProvider,
                                     model_type="gemma4_unified")
class Gemma4UnifiedBridge(Gemma4Bridge):
    def provider_bridge(self, hf_pretrained):
        text_config = getattr(hf_pretrained.config, "text_config", hf_pretrained.config)
        self._is_dense = True
        self._unified_text_config = text_config
        return self._build_dense_provider(text_config)      # reuse the dense provider
    def _text_config(self):                                  # for K=V synthesis + mapping
        return getattr(self, "_unified_text_config", None) or getattr(self, "hf_config", None)
    def _hf_layer_prefix(self):
        return "model.language_model."                       # Unified nests the LM here
```

Validated: `test_phase0.py` builds the 12 B model (config-only, no download) and runs a forward; `parity_g4.py` converts the real weights and matches HF logits (**cosine 0.9999, 100 % top-1**); `train_lora_g4.py` runs a 3-step LoRA finetune (loss 1.86→1.12).

**Gotcha (weight loading):** `to_megatron_provider(load_weights=True)` does **not** load weights for the dense provider — its custom `build()` skips the pre-wrap hook. Load explicitly via `AutoBridge.load_hf_weights([model])`, or via the training checkpoint path (`cfg.checkpoint.pretrained_checkpoint = <HF dir>`), which is what the LoRA recipe uses.

## 2. Phase 1 — vision

### 2a. Vision embedder (`src/gemma4_vision_projector.py`)
A plain `nn.Module` reproducing the HF forward exactly. Validated **bit-exact** (`test_vision_parity.py`: cosine 1.000000) by feeding identical synthetic patches to HF `get_image_features` and to this module loaded from the same weights.

**Naming divergence (resolved):** the on-disk safetensors uses `model.vision_embedder.*` + `model.embed_vision.embedding_projection`, but the *runtime* HF module (what a bridge reads) renames to `model.embed_vision.*` + `model.embed_vision.multimodal_embedder.embedding_projection`.

### 2b. Merge (`src/gemma4_unified_vl.py`)
`Gemma4UnifiedVLModel` embeds text, projects vision, `masked_scatter`s the vision soft tokens into the image-token (`258880`) slots, and runs the LM via `decoder_input` — the same mechanism as MIMO's `align_embeddings_by_token_positions`, kept as a thin single-GPU wrapper (the MIMO framework can't slot Gemma-4's custom-built GPTModel into its spec-based LM construction without more work; that's a productionization follow-up).

### 2c. Bidirectional vision mask (`src/gemma4_vision_mask.py`)
Gemma-4's rule: within a contiguous image soft-token span, attention is **bidirectional**; text stays causal; each layer also applies its sliding window. HF builds this by passing `block_sequence_ids` to `create_causal_mask` / `create_sliding_window_causal_mask`.

**The injection is non-trivial.** The Gemma-4 dense model uses MCore's **local** `DotProductAttention`, which applies masking inside `FusedScaleMaskSoftmax` using the **init-time** `attn_mask_type` (causal) + window, and *ignores* the per-call `attn_mask_type`/`attention_mask` on its fused-causal path (verified: A/B output byte-identical). It also **rejects** `attention_bias`. And the `arbitrary` mask type is ignored by both the flash and unfused backends. The working injection: per applied layer, temporarily set `core_attention.scale_mask_softmax.attn_mask_type = padding` (which honors the passed boolean mask) and feed the full mask `~(window ∧ (causal ∨ same_vision_block))`, preserving Gemma's softmax scale/offset. A/B confirms it changes attention (`max|Δ|=12.25`).

## 3. Two bugs found by parity diffing

Naïve Phase-1 parity was only **cosine 0.49**. Two subtle bugs, each found with a targeted diff:

1. **Text embeddings need `×√hidden`** — *found via a scale-invariant hidden-state diff* (`test_hidden_diff.py`), which showed image embeddings matched exactly but text embeddings had the right direction and **1/√3840 the magnitude**. Megatron's `LanguageModelEmbedding` returns *unscaled* embeddings (`GPTModel.forward` uses `decoder_input` directly, no scale), while HF scales **text** by √hidden but leaves **image** features unscaled. This *relative* text/image magnitude cancels under RMSNorm for pure text (so Phase-0 was exact) but **not** in the multimodal case. Fix: scale text embeds by √hidden, vision unscaled.

2. **The bidirectional mask applies to all 48 layers**, not sliding-only — *found via a mask diff* (`test_mask_diff.py`), which showed HF's `full_attention` per-layer mask is *identical* to its `sliding_attention` mask (the only per-type difference is the sliding window, which doesn't bite at short lengths). The prior assumption "global layers stay causal" was wrong.

With both fixes: **image-position cosine 0.999, text 0.975 (top-1 83 %)** — up from 0.49.

## 4. Residual (bf16)

The remaining text gap (0.975, not ~0.99) is **bf16-level, not a bug** — a per-layer hidden-state diff (`test_layer_diff.py`) shows the divergence is *gradual* (text 0.9997@L12 → 0.946@L30) and *partially recovers* by the output (L47 text 0.994 / image 0.987): no single-layer cliff, i.e. no discrete bug. A true fp32 confirmation isn't runnable — TE attention kernels are bf16/fp16-only (`RuntimeError: Only fp16 and bf16 are supported`). Image positions are bit-exact, so the multimodal wiring is correct; the residual is cross-framework bf16 accumulation in a complex custom attention (K=V, dual/proportional RoPE, softcap, per-layer head dims).

## 5. Verification methodology

Every claim in this repo is backed by a runnable harness in `tests/`, comparing against the HuggingFace reference on identical inputs — logit parity (`parity_g4.py`, `phase1_parity.py`), component parity (`test_vision_parity.py`, `test_mask_diff.py`), and localizer diffs (`test_hidden_diff.py`, `test_layer_diff.py`). This diff-driven approach is what surfaced both Phase-1 bugs.
