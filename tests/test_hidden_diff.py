#!/usr/bin/env python
# Locate the dominant multimodal bug: compare the MERGED decoder input (HF vs mine),
# scale-invariant (cosine per position), at image vs text positions.
# Run: torchrun --nproc-per-node=1 /workspace/scripts/test_hidden_diff.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import gemma4_unified_bridge
from gemma4_unified_vl import Gemma4UnifiedVLModel, load_vision_from_safetensors
from megatron.bridge import AutoBridge

MODEL = "google/gemma-4-12B"; IMG = 258880
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
from megatron.core import parallel_state
if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(1, 1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(1234)

torch.manual_seed(0)
N, n_txt = 32, 24; S = N + n_txt
input_ids = torch.randint(0, 250000, (1, S), device="cuda"); input_ids[0, :N] = IMG
mm_type = (input_ids == IMG).long()
pixel_values = torch.randn(N, 6912, dtype=torch.bfloat16, device="cuda")
image_position_ids = torch.stack([torch.arange(N) % 8, torch.arange(N) // 8], -1).to("cuda")
pos = torch.arange(S, device="cuda").unsqueeze(0)

# my merged embedding
br = AutoBridge.from_hf_pretrained(MODEL)
provider = br.to_megatron_provider(load_weights=False)
provider.tensor_model_parallel_size = 1; provider.seq_length = 512
lm = provider.provide().cuda(); br.load_hf_weights([lm]); del br.hf_pretrained; torch.cuda.empty_cache()
enc, proj = load_vision_from_safetensors(); enc = enc.cuda().eval(); proj = proj.cuda().eval()
vl = Gemma4UnifiedVLModel(lm, enc, proj).cuda().eval()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    _ = vl(input_ids=input_ids, position_ids=pos, pixel_values=pixel_values,
           image_position_ids=image_position_ids, attention_mask=None)
my_merged = vl.last_merged[0].float().cpu()   # [S,H]

# HF merged (inputs_embeds into the text model)
from transformers import AutoModelForImageTextToText
hf = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
cap = {}
def hook(mod, args, kwargs):
    cap["ie"] = kwargs.get("inputs_embeds", None)
hf.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
with torch.no_grad():
    hf(input_ids=input_ids, pixel_values=pixel_values.unsqueeze(0),
       image_position_ids=image_position_ids.unsqueeze(0), mm_token_type_ids=mm_type)
hf_merged = cap["ie"][0].float().cpu()        # [S,H]

def stats(name, a, b):
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    ratio = (a.norm(dim=-1) / b.norm(dim=-1))
    print(f"  {name}: cosine mean={cos.mean():.5f} min={cos.min():.5f} | |a|/|b| mean={ratio.mean():.3f}")

print("=== merged decoder input: HF vs mine ===")
stats("image pos", my_merged[:N], hf_merged[:N])
stats("text  pos", my_merged[N:], hf_merged[N:])
