#!/usr/bin/env python
# Logit-parity: HF google/gemma-4-12B (text) vs Megatron (converted via our bridge).
# Run: torchrun --nproc-per-node=1 /workspace/scripts/parity_g4.py
import os, sys, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch

MODEL = "google/gemma-4-12B"
PROMPT = "The capital of France is"

def hr(t): print("\n" + "="*20 + " " + t + " " + "="*20, flush=True)

import gemma4_unified_bridge  # registers the bridge
from megatron.bridge import AutoBridge

# --- distributed / MCore init ---
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
from megatron.core import parallel_state
if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(1, 1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(1234)

# --- tokenize ---
hr("tokenize")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
T = ids.shape[1]
print("  prompt:", repr(PROMPT), "| input_ids:", ids.tolist(), "| T =", T)

# --- HF reference logits (text-only) ---
hr("HF reference forward")
from transformers import AutoModelForImageTextToText
hf = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager")
hf = hf.cuda().eval()
with torch.no_grad():
    hf_out = hf(input_ids=ids)
    hf_logits = (hf_out.logits if hasattr(hf_out, "logits") else hf_out[0]).float().cpu()
print("  HF logits:", tuple(hf_logits.shape), "| dtype:", hf_logits.dtype)
del hf
torch.cuda.empty_cache()

# --- convert HF -> Megatron (loads real weights via our bridge) ---
hr("HF -> Megatron conversion")
br = AutoBridge.from_hf_pretrained(MODEL)
# NOTE: Gemma4DenseProvider.build() is a custom builder that does NOT run the
# pre_wrap_hook that load_weights=True registers, so we build then load explicitly.
provider = br.to_megatron_provider(load_weights=False)
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.seq_length = 8192
model = provider.provide()
model = model.cuda()
emb0 = model.embedding.word_embeddings.weight.detach().float().norm().item()
br.load_hf_weights([model])  # explicit HF->Megatron weight streaming via our bridge mapping
emb1 = model.embedding.word_embeddings.weight.detach().float().norm().item()
model = model.eval()
n = sum(p.numel() for p in model.parameters())
print(f"  Megatron model built + weights loaded — params: {n:,} ({n/1e9:.2f} B)")
print(f"  embedding weight norm before/after load: {emb0:.2f} -> {emb1:.2f}  (must change)")

# --- Megatron logits on same input ---
hr("Megatron forward")
pos = torch.arange(T, device="cuda").unsqueeze(0)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    mg = model(input_ids=ids, position_ids=pos, attention_mask=None)
mg_logits = mg.float().cpu()
print("  MG logits:", tuple(mg_logits.shape))

# --- compare ---
hr("PARITY")
V = min(hf_logits.shape[-1], mg_logits.shape[-1])
a = hf_logits[..., :V].reshape(T, V)
b = mg_logits[..., :V].reshape(T, V)
maxdiff = (a - b).abs().max().item()
meandiff = (a - b).abs().mean().item()
cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
hf_top1 = a.argmax(-1)
mg_top1 = b.argmax(-1)
top1_agree = (hf_top1 == mg_top1).float().mean().item()
# top-5 overlap on the last position
k = 5
hf_top5 = set(a[-1].topk(k).indices.tolist())
mg_top5 = set(b[-1].topk(k).indices.tolist())
print(f"  vocab compared: {V}")
print(f"  max|Δ|        : {maxdiff:.4f}")
print(f"  mean|Δ|       : {meandiff:.4f}")
print(f"  cosine(mean)  : {cos:.6f}")
print(f"  top-1 agree   : {top1_agree*100:.1f}%  (HF {hf_top1.tolist()} vs MG {mg_top1.tolist()})")
print(f"  last-tok top-5 overlap: {len(hf_top5 & mg_top5)}/{k}")
print(f"  HF next-token : {tok.decode(hf_top1[-1])!r} | MG next-token: {tok.decode(mg_top1[-1])!r}")
verdict = "PASS" if (top1_agree == 1.0 and cos > 0.99) else ("CLOSE" if cos > 0.95 else "FAIL")
print(f"\n  VERDICT: {verdict}")
