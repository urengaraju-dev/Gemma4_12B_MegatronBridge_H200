#!/usr/bin/env python
# Phase-0 validation for the Gemma4Unified text bridge — no 24GB weight download.
# Run: torchrun --nproc-per-node=1 /workspace/scripts/test_phase0.py
import os, sys, json, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch

def hr(t): print("\n" + "="*22 + " " + t + " " + "="*22, flush=True)

# --- register the bridge (decorator runs on import) ---
import gemma4_unified_bridge as g4u
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

hr("registration")
found = False
for attr in dir(MegatronModelBridge):
    v = getattr(MegatronModelBridge, attr, None)
    if isinstance(v, dict) and any("Gemma4Unified" in str(k) for k in v):
        print(f"  registry '{attr}' contains Gemma4UnifiedForConditionalGeneration"); found = True
print("  bridge class:", g4u.Gemma4UnifiedBridge.__name__, "| registry-probe found:", found)

# --- load config.json only (few KB, gated) ---
hr("config")
from huggingface_hub import hf_hub_download
cfg_path = hf_hub_download("google/gemma-4-12B", "config.json")
d = json.load(open(cfg_path))
class Cfg:
    def __init__(self, dd):
        for k, v in dd.items(): setattr(self, k, v)
top = Cfg(d)
top.text_config = Cfg(d["text_config"])
class FP:
    def __init__(self, c): self.config = c
print("  architectures:", d.get("architectures"), "| model_type:", d.get("model_type"))

# --- translate text_config -> provider ---
hr("provider_bridge (config translation)")
br = g4u.Gemma4UnifiedBridge()
provider = br.provider_bridge(FP(top))
for f in ("num_layers","hidden_size","ffn_hidden_size","num_attention_heads","num_query_groups",
          "kv_channels","global_kv_channels","num_global_query_groups","attention_k_eq_v",
          "sliding_window_rope_base","full_attention_rope_base","full_attention_rope_partial_factor",
          "num_kv_shared_layers","per_layer_embed_dim","window_size","window_attn_skip_freq",
          "final_logit_softcapping","vocab_size","seq_length"):
    print(f"   provider.{f} = {getattr(provider, f, 'N/A')}")

# --- build the real 12B model (random init) + forward ---
hr("build model + forward (random init, on GPU)")
try:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    lr = int(os.environ.get("LOCAL_RANK", "0")); torch.cuda.set_device(lr)
    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)

    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.seq_length = 8192  # cap (real max_position_embeddings=262144) for build speed
    model = provider.provide()
    n = sum(p.numel() for p in model.parameters())
    print(f"  MODEL BUILT OK — parameters: {n:,} ({n/1e9:.2f} B)")
    print("  model class:", type(model).__name__)

    model = model.cuda().eval()
    ids = torch.randint(0, 262144, (1, 16), device="cuda")
    pos = torch.arange(16, device="cuda").unsqueeze(0)
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, position_ids=pos, attention_mask=None)
        print("  FORWARD OK — logits shape:", tuple(out.shape), "| dtype:", out.dtype)
    except Exception as e:
        print("  FORWARD FAILED:", type(e).__name__, str(e)[:400])
        traceback.print_exc()
except Exception as e:
    print("  BUILD FAILED:", type(e).__name__, str(e)[:400])
    traceback.print_exc()

hr("done")
