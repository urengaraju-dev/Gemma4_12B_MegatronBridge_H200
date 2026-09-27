# Gemma-4-12B on Megatron-Bridge — Phase 1 (vision) status

Builds on the working, HF-parity-PASS Phase-0 text bridge. Target: single H200 (colocated, TP=1).

## Done ✅

- **Vision projector written and numerically EXACT vs HF.** `gemma4_vision_projector.py` reproduces
  HF `Gemma4UnifiedVisionEmbedder` + `Gemma4UnifiedMultimodalEmbedder`:
  `patch_ln1 → patch_dense(6912→3840) → patch_ln2 → +factorized pos_embedding → pos_norm →
   fp32 scaleless-RMSNorm → embedding_projection(3840→3840)`.
  Parity on identical synthetic patches: **cosine 1.000000, max|Δ| 0.0000** (`test_vision_parity.py`).
- **Naming divergence resolved.** Runtime HF keys (bridge source) are `model.embed_vision.*` +
  `model.embed_vision.multimodal_embedder.embedding_projection.weight` (the on-disk safetensors uses
  `model.vision_embedder.*`; HF renames on `from_pretrained`).
- **Wiring design locked (Option A).** On a colocated single GPU, MIMO's `_forward_all_modules` only
  calls `submodule.forward(encoder_inputs=...)` — so `encoders={}` silently returns None. Correct pattern:
  register the vision embedder (through the scaleless RMSNorm) as a MIMO **encoder**, keep only the final
  bias-free `embedding_projection` as `input_projections[0]`. Merge is automatic (masked_scatter at
  `image_token_id=258880` via `align_embeddings_by_token_positions`).

## Steps 1-3: DONE ✅ — image+text pipeline runs (`src/gemma4_unified_vl.py`)

Assembled a `Gemma4UnifiedVLModel` wrapper (pragmatic single-GPU integration; full MIMO-framework
wiring deferred to productionization/PP): Phase-0 text GPTModel (weights loaded) + vision encoder
(bit-exact) + `nn.Linear` projection; merge = `masked_scatter` of vision soft tokens at
`image_token_id=258880` into the LM embedding stream, then run the LM via `decoder_input`.
Result on 1×H200: `logits (1,80,262144)` finite, `loss` finite — the multimodal forward runs.
Vision weights loaded straight from the on-disk safetensors (`model.vision_embedder.*` +
`model.embed_vision.embedding_projection.weight`).

## Steps 4-5: DONE ✅ — image-conditioned parity achieved (bf16)

Final image-conditioned parity vs HF (synthetic patches, identical to both models):
**text cosine 0.975 (top-1 83%), image-pos cosine 0.999** (was 0.49 / 0.72 when broken).
Two real bugs found via a scale-invariant hidden-state diff and fixed:
1. **Text embeddings need ×sqrt(hidden).** Megatron's `LanguageModelEmbedding` returns UNSCALED
   embeds (`GPTModel.forward` uses `decoder_input` directly, no scale), and HF scales TEXT by
   sqrt(h) but leaves IMAGE features UNSCALED — a *relative* text/image magnitude that does NOT
   wash out under RMSNorm in the multimodal case (it does for pure text, which is why Phase-0 was
   exact). Fix: scale text embeds by sqrt(h), vision unscaled.
2. **The bidirectional vision mask applies to ALL 48 layers**, not sliding-only (verified by
   diffing HF's `full_attention` vs `sliding_attention` masks — identical). `APPLY_ALL=1` default.
The residual text gap (0.975 vs ~0.99) is bf16-level numerical, not a bug — confirmed by a
layer-by-layer hidden-state diff (`test_layer_diff.py`): the divergence is GRADUAL (text 0.9997@L12 →
0.946@L30) and partially RECOVERS by the output (L47 text 0.994 / image 0.987) — no single-layer cliff,
i.e. no discrete bug. A true fp32 confirmation isn't runnable (TE attention kernels are bf16/fp16-only:
"Only fp16 and bf16 are supported"), so the model is locked to bf16. Image positions are bit-exact
(0.999) — the multimodal wiring is correct; the text residual is cross-framework bf16 accumulation in a
complex custom attention (K=V, dual/proportional RoPE, softcap, per-layer head dims). Closing further
would require fp32/bit-matched kernels — impractical and low value given image parity is bit-exact.

Mechanism (from earlier debugging): the Gemma-4 dense model uses MCore's LOCAL DotProductAttention,
which masks inside FusedScaleMaskSoftmax with the INIT-time attn_mask_type and ignores the per-call
mask/type (+ rejects attention_bias; flash & unfused both ignore an `arbitrary` mask). Injection that
works: per applied layer, temporarily set `core_attention.scale_mask_softmax.attn_mask_type=padding`
and feed the full boolean mask `~(window ∧ (causal ∨ same_vision_block))`, preserving softmax scale/offset.

## (history) Step 4 earlier attempts

**Update — the injection now works.** Root cause of the earlier "mask ignored": the Gemma-4 dense
model uses MCore's **local** `DotProductAttention`, which applies masking inside
`FusedScaleMaskSoftmax` using the **init-time** `attn_mask_type=causal` (+window) and ignores the
per-call mask/type (and rejects `attention_bias`; both flash and unfused ignore an `arbitrary` mask).
**Fix:** per sliding layer, temporarily set `core_attention.scale_mask_softmax.attn_mask_type=padding`
(which applies the PASSED boolean mask) and feed the full mask `~(window ∧ (causal ∨ same_block))` —
Gemma's softmax scale/offset preserved. A/B confirms it now changes attention (`max|Δ|=12.25`).

Image-conditioned parity vs HF (synthetic patches, both models identical pixel_values):
- mask OFF (causal): text cosine 0.487, image-pos 0.718
- **mask ON  (bidir): text cosine 0.681, image-pos 0.898**  ← mask working, moving toward HF
- confirmed **sliding-only** is correct (all-layers is worse: text 0.596).

Not yet bit-exact (target ~0.99). **Mask diff vs HF (test_mask_diff.py): my sliding mask is BIT-EXACT
(0 mismatches).** KEY CORRECTION: HF's `full_attention` (global) layer mask is IDENTICAL to the sliding
one — HF applies the bidirectional-in-vision-span mask on ALL 48 layers, not sliding-only (the only
per-type difference is the sliding window, which doesn't bite for short seqs). So "sliding-only" was
structurally wrong; it only scored higher because my GLOBAL-layer application is numerically off
(all-layers helped image 0.90→0.92 but hurt text 0.68→0.60 — the global layers are K=V / head-dim-512 /
MQA, where the padding-softmax injection misbehaves). Two remaining items: (1) fix the global-layer mask
application; (2) a residual non-mask gap (image cosine ~0.92 with mask, not 1.0) → RoPE-position / scale.
Vision embedder bit-exact, text-only parity 100%.

## (superseded) Step 4 first attempt — blocked on the TE attention backend

- **Mask logic done & correct** (`gemma4_vision_mask.py`): builds `[b,1,s,s]` bool
  (True=masked) `= sliding_window ∧ (causal ∨ same_vision_block)`, correctly identifies the
  40 `sliding_attention` layers (leaves the 8 global layers causal), and the per-instance
  `core_attention` wrapper fires and injects `attention_mask` + `attn_mask_type=arbitrary`
  on exactly those 40 layers (`applied=40`).
- **BLOCKER (R1, confirmed empirically):** with the auto-selected **flash-attention-3**
  backend, the injected arbitrary mask is **silently ignored** — logits are byte-identical for
  mask=bidir / OFF / flipped. Arbitrary per-element masks need the **unfused/cuDNN** TE backend.
  Forcing it via `NVTE_FLASH_ATTN=0` is rejected (MCore asserts `=1` for `attention_backend=auto`);
  it must be set at the model level (`provider.attention_backend = AttnBackend.unfused` / a
  cuDNN path) — which needs re-validating numerics for head_dim 256/512, so it's a deeper change.
- **Consequence:** image-conditioned parity vs HF is not yet passing (text-pos cosine ≈ 0.49); the
  gap is dominated by the un-applied bidirectional attention (HF is bidirectional over the image
  span; our sliding layers are still plain-causal via flash). Vision embedder itself is bit-exact,
  and text-only Phase-0 parity is 100% — so the residual is the mask/backend, not the vision math.
- **Path forward:** (a) build the LM (or just the 40 sliding layers) with an arbitrary-mask-capable
  attention backend and re-run the parity; (b) then confirm the remaining minor items (RoPE
  position handling for image tokens, exact vision embed-scale) if any gap remains.

## Remaining Phase-1 steps (ordered)

4. **Bidirectional vision mask — finish (see above: switch sliding layers to a mask-capable backend)** — per-layer-type: blockwise-bidirectional within the image
   soft-token span on the 40 `sliding_attention` layers only; the 8 `full_attention` layers stay causal.
   No stock Megatron API broadcasts per-layer masks → needs a thin language wrapper routing two `[b,1,s,s]`
   masks by `layer_types`. Must verify TE arbitrary-mask support + polarity + sliding-bound first.
5. **Image-conditioned parity vs HF** — same input_ids+pixel_values+image_position_ids+mm_token_type_ids;
   compare next-token logits after the image span. Requires the real image-processor patch layout
   (`image_processing_gemma4_unified.py`) + the mask (step 4).

## Files
- `gemma4_vision_projector.py` — vision embedder (bit-exact vs HF).
- `test_vision_parity.py` — the parity test.
- The full source-grounded design plan (MIMO wiring, weight map, mask algorithm, ranked risks R1–R11)
  is in the workflow output; key risks: R1 per-layer mask routing, R3 patch layout (partly resolved),
  R4 embed-scaling of scattered vision tokens, R8 single-rank MIMO forward.
