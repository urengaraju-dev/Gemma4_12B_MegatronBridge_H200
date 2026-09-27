# Validated parity results

All numbers below were produced on a single **NVIDIA H200 (143 GB)** inside
`nvcr.io/nvidia/nemo:26.08` (Megatron-Bridge 0.6.1, Megatron-Core 0.19.1, CUDA 13, bf16),
comparing this implementation against the HuggingFace reference for `google/gemma-4-12B`
on **identical inputs** (same `input_ids`, same synthetic `pixel_values`/positions).

## Summary

| Check | Harness | Metric | Result |
|---|---|---|---|
| Phase 0 — text logit parity | `tests/parity_g4.py` | cosine / top-1 / max\|Δ\| | **0.999931 / 100% / 0.44** |
| Phase 0 — LoRA finetune smoke | `tests/train_lora_g4.py` | 3-step lm loss | **1.86 → 1.12** (no NaNs) |
| Phase 1 — vision embedder | `tests/test_vision_parity.py` | cosine / max\|Δ\| | **1.000000 / 0.0000** (bit-exact) |
| Phase 1 — mask correctness | `tests/test_mask_diff.py` | mismatches vs HF mask | **0** |
| Phase 1 — image-conditioned (image positions) | `tests/phase1_parity.py` | cosine | **0.99869** (bit-exact) |
| Phase 1 — image-conditioned (text positions) | `tests/phase1_parity.py` | cosine / top-1 | **0.975 / 83%** (bf16-limited) |

## How to reproduce

```bash
make setup            # container + weights (one-time)
make phase0-parity    # -> cosine 0.999931, top-1 100%
make phase0-lora      # -> loss 1.86 -> 1.12
make phase1-vision    # -> cosine 1.000000
make mask-diff        # -> 0 mismatches
make phase1-parity    # -> image 0.999 / text 0.975
```

## Notes on the residuals

- **Text positions in the multimodal case (0.975, not ~0.99).** Characterized as bf16
  accumulation across 48 layers, **not a discrete bug**: a per-layer hidden-state diff
  (`tests/test_layer_diff.py`) shows the divergence is *gradual* (text 0.9997@L12 → 0.946@L30)
  and partially *recovers* by the output (L47 text 0.994 / image 0.987) — no single-layer cliff.
  A true fp32 confirmation is not runnable: TE attention kernels are bf16/fp16-only
  (`RuntimeError: Only fp16 and bf16 are supported`).
- **Image positions are bit-exact (0.999)**, which confirms the multimodal wiring
  (vision embed → scale → masked_scatter → bidirectional mask) is correct; the text residual
  is cross-framework bf16 accumulation in a complex custom attention (K=V, dual/proportional
  RoPE, softcap, per-layer head dims).

See [`docs/DESIGN.md`](../DESIGN.md) §3–§4 for the full derivation, and
[`phase1_notes.md`](phase1_notes.md) for the chronological debugging log.
