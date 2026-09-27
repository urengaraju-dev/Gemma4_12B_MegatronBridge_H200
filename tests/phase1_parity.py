#!/usr/bin/env python
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# Phase-1 step 4/5: validate the bidirectional vision mask by image-conditioned logit
# parity vs HF, on IDENTICAL synthetic patches (both models get the same pixel_values).
# Run: torchrun --nproc-per-node=1 /workspace/scripts/phase1_parity.py
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch

import gemma4_unified_bridge
from gemma4_unified_vl import Gemma4UnifiedVLModel, load_vision_from_safetensors
import gemma4_vision_mask as gm
from megatron.bridge import AutoBridge

MODEL = "google/gemma-4-12B"; IMG = 258880
def hr(t): print("\n" + "="*18 + " " + t + " " + "="*18, flush=True)

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
from megatron.core import parallel_state
if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(1, 1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(1234)

# ---- inputs (shared) ----
torch.manual_seed(0)
N, n_txt = 32, 24
S = N + n_txt
input_ids = torch.randint(0, 250000, (1, S), device="cuda")
input_ids[0, :N] = IMG
mm_type = torch.zeros(1, S, dtype=torch.long, device="cuda"); mm_type[0, :N] = 1
pixel_values = torch.randn(N, 6912, dtype=torch.bfloat16, device="cuda")
image_position_ids = torch.stack([torch.arange(N) % 8, torch.arange(N) // 8], -1).to("cuda")

# ---- build my VL model (Phase-0 text + vision), load weights, install mask ----
hr("build + load my VL model")
br = AutoBridge.from_hf_pretrained(MODEL)
provider = br.to_megatron_provider(load_weights=False)
provider.tensor_model_parallel_size = 1; provider.pipeline_model_parallel_size = 1; provider.seq_length = 512
# (dense Gemma4 uses MCore local DotProductAttention regardless of backend; mask injected at its softmax)
lm = provider.provide().cuda(); br.load_hf_weights([lm]); del br.hf_pretrained; torch.cuda.empty_cache()
enc, proj = load_vision_from_safetensors(); enc = enc.cuda().eval(); proj = proj.cuda().eval()
vl = Gemma4UnifiedVLModel(lm, enc, proj).cuda().eval()
with torch.no_grad():
    _t = torch.tensor([[100]], device="cuda"); _p = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    _raw = lm.embedding.word_embeddings.weight[100].float()
    _sc = lm.embedding(input_ids=_t, position_ids=_p)[0, 0].float()
    print(f"  [scale] megatron embed ratio={(_sc.norm()/_raw.norm()).item():.4f}  sqrt(h)={3840**0.5:.4f}")
gm.install_sliding_arbitrary_mask(vl.language_model)
print("  installed bidirectional mask (APPLY_ALL=%s)" % gm._APPLY_ALL)

# ---- HF reference ----
hr("HF reference forward")
from transformers import AutoModelForImageTextToText
hf = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    hf_logits = hf(input_ids=input_ids, pixel_values=pixel_values.unsqueeze(0),
                   image_position_ids=image_position_ids.unsqueeze(0),
                   mm_token_type_ids=mm_type).logits.float().cpu()
del hf; torch.cuda.empty_cache()
print("  HF logits:", tuple(hf_logits.shape))

# ---- my forward WITH mask ----
hr("my VL forward (with bidirectional mask)")
pos = torch.arange(S, device="cuda").unsqueeze(0)
def run(mask):
    gm._CTX["mask"] = mask
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = vl(input_ids=input_ids, position_ids=pos, pixel_values=pixel_values,
                 image_position_ids=image_position_ids, attention_mask=None).float().cpu()
    gm._CTX["mask"] = None
    return out

def cmp(tag, b):
    a = hf_logits[0, N:]; bb = b[0, N:]
    maxd = (a-bb).abs().max().item(); cos = torch.nn.functional.cosine_similarity(a, bb, dim=-1).mean().item()
    t1 = (a.argmax(-1) == bb.argmax(-1)).float().mean().item()
    # image-position parity too
    ai = hf_logits[0, :N]; bi = b[0, :N]
    cosi = torch.nn.functional.cosine_similarity(ai, bi, dim=-1).mean().item()
    print(f"  [{tag}] text: max|Δ| {maxd:.3f} cosine {cos:.5f} top1 {t1*100:.1f}%  | image-pos cosine {cosi:.5f}")

hr("PARITY diagnostics")
mask = gm.build_local_mask(mm_type)
cmp("mask=bidir ", run(mask))
print("  _STATS after 1 forward:", gm._STATS)
cmp("mask=OFF   ", run(None))
