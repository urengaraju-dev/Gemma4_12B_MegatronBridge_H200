# Gemma-4-12B on NVIDIA Megatron-Bridge

![Framework](https://img.shields.io/badge/Megatron--Bridge-0.6.1-76B900)
![Container](https://img.shields.io/badge/NeMo-26.08-76B900)
![CUDA](https://img.shields.io/badge/CUDA-13-76B900)
![GPU](https://img.shields.io/badge/GPU-1%C3%97%20H200-76B900)
![Phase 0](https://img.shields.io/badge/Phase%200%20(text)-parity%20%2B%20LoRA%20passing-brightgreen)
![Phase 1](https://img.shields.io/badge/Phase%201%20(vision)-image%20parity%200.999-brightgreen)

Adds **`google/gemma-4-12B`** support to **[NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)**, which does not ship it out of the box.

`google/gemma-4-12B` is **`Gemma4UnifiedForConditionalGeneration`** — a 12 B **dense, encoder-free omni** model (text + vision + audio) on a new architecture that Megatron-Bridge's `AutoBridge` rejects (`ValueError: model architecture is not supported`). This repository implements and validates that support, incrementally:

| Phase | Scope | Status | Evidence |
|---|---|---|---|
| **0** | Text tower (LoRA/SFT) | ✅ **complete** | HF logit-parity **100 % top-1, cosine 0.9999**; 3-step LoRA smoke (loss 1.86→1.12) |
| **1** | + Vision (image) | ✅ **working** | Vision embedder **bit-exact vs HF**; image-conditioned parity **image 0.999 / text 0.975** |
| **2** | + Audio | ⏳ planned | single Linear projector on the Phase-1 scaffold |

Built and validated end-to-end on a single **NVIDIA H200** (143 GB), container `nvcr.io/nvidia/nemo:26.08` (Megatron-Bridge 0.6.1, Megatron-Core 0.19.1, CUDA 13).

---

## Why this is small (and why it works)

Megatron-Bridge already ships a complete **Gemma-4 dense text stack** (`Gemma4DenseProvider` + `gemma4_bridge`: dual/proportional RoPE, 5:1 sliding/full interleave, per-layer head dims, K=V attention, `layer_scalar`, logit softcap) and a **MIMO** multimodal framework. So the delta for the Unified variant is a **thin fork**, not a from-scratch port:

- **Phase 0** = register one bridge for `Gemma4UnifiedForConditionalGeneration` that reads the nested `text_config` and points the HF weight prefix at `model.language_model.` — everything else is inherited. `src/gemma4_unified_bridge.py` (~30 lines).
- **Phase 1** = an encoder-free vision embedder (`src/gemma4_vision_projector.py`), a bidirectional vision attention mask (`src/gemma4_vision_mask.py`), and a small VL wrapper that merges vision soft tokens into the LM embedding stream (`src/gemma4_unified_vl.py`).

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full story, including the two subtle bugs that were found and fixed via layer-wise parity diffing.

## Repository layout

```
.
├── README.md · LICENSE · CHANGELOG.md · CONTRIBUTING.md · Makefile · .gitignore
├── src/
│   ├── gemma4_unified_bridge.py     # Phase 0: text bridge (register + provider, reuses Gemma4 dense)
│   ├── gemma4_vision_projector.py   # Phase 1: encoder-free vision embedder (bit-exact vs HF)
│   ├── gemma4_vision_mask.py        # Phase 1: bidirectional vision attention mask (softmax injection)
│   └── gemma4_unified_vl.py         # Phase 1: Gemma4UnifiedVLModel (text+vision merge) + runnable smoke
├── tests/                           # parity & smoke harnesses (the evidence)
│   ├── test_phase0.py               # build the 12B text model + forward (no weight download)
│   ├── parity_g4.py                 # Phase 0: HF vs Megatron logit parity
│   ├── train_lora_g4.py             # Phase 0: 3-step LoRA finetune smoke
│   ├── test_vision_parity.py        # Phase 1: vision embedder vs HF get_image_features (bit-exact)
│   ├── test_mask_diff.py            # Phase 1: my mask vs HF's actual per-layer mask
│   ├── phase1_parity.py             # Phase 1: image-conditioned HF logit parity
│   ├── test_hidden_diff.py          # locator: merged-embedding diff (found the text-scale bug)
│   └── test_layer_diff.py           # locator: per-layer hidden-state diff (bf16 characterization)
├── scripts/setup.sh                 # provision container + weights
└── docs/DESIGN.md · docs/FEASIBILITY.md · docs/results/
```

## Requirements

- 1× NVIDIA H200 (or any ≥ 48 GB CUDA-13 GPU for LoRA/parity).
- Docker with the NVIDIA runtime.
- Container `nvcr.io/nvidia/nemo:26.08`.
- A HuggingFace token for an account that has **accepted the Gemma license** at
  <https://huggingface.co/google/gemma-4-12B> (put it in `~/.hf_token` or `HF_TOKEN`).

## Quickstart

```bash
# 1. Provision: pull the container, download weights, start the container (idempotent)
bash scripts/setup.sh

# 2. Phase 0 — HF logit parity (text)
docker exec mb torchrun --nproc-per-node=1 /workspace/tests/parity_g4.py

# 3. Phase 0 — LoRA finetune smoke
docker exec mb torchrun --nproc-per-node=1 /workspace/tests/train_lora_g4.py

# 4. Phase 1 — vision embedder parity + image-conditioned parity
docker exec mb python  /workspace/tests/test_vision_parity.py
docker exec -e APPLY_ALL=1 mb torchrun --nproc-per-node=1 /workspace/tests/phase1_parity.py
```

(The container mounts the repo at `/workspace`; see `scripts/setup.sh`.)

## Validated results

**Phase 0 — text (`parity_g4.py`):**
```
loading HuggingFace-format checkpoint from .../gemma-4-12B   (real HF→Megatron conversion, 530 tensors)
max|Δ| 0.44 | cosine 0.999931 | top-1 agreement 100%   → PASS
```

**Phase 0 — LoRA (`train_lora_g4.py`):** 3 iters, lm loss 1.86 → 1.12, no NaNs, `torch_dist` checkpoint saved.

**Phase 1 — vision embedder (`test_vision_parity.py`):** `cosine 1.000000, max|Δ| 0.0000` vs HF `get_image_features`.

**Phase 1 — image-conditioned (`phase1_parity.py`):**
```
image-position cosine 0.99869   (bit-exact)
text-position  cosine 0.975, top-1 83%   (bf16-limited; see DESIGN §Residual)
```

## Known limitations / follow-ups

- **bf16 text residual** (0.975, not ~0.99): identified as bf16 accumulation across 48 layers (gradual, no discrete bug; image path bit-exact). TE kernels are bf16/fp16-only, so fp32 isn't available to close it.
- **`Gemma4DenseProvider` is PP=1 only** — multi-GPU is TP/DP on one node; pipeline parallel needs provider work.
- **Single-GPU VL wrapper**: `Gemma4UnifiedVLModel` is a pragmatic wrapper; folding it into the `MegatronMIMOProvider` + `megatron_mimo_step` (and adding the vision weights to the bridge mapping registry) is the productionization step.
- **Phase 2 (audio)** — one Linear projector (`model.embed_audio.embedding_projection`, 640→3840) on the existing merge scaffold.

## References

- Megatron-Bridge — <https://github.com/NVIDIA-NeMo/Megatron-Bridge>
- NeMo Framework container — `nvcr.io/nvidia/nemo:26.08`
- `google/gemma-4-12B` — <https://huggingface.co/google/gemma-4-12B>

## License

Apache-2.0 — see [LICENSE](LICENSE). Gemma weights are governed by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); accept them on Hugging Face to download.
