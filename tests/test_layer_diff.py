#!/usr/bin/env python
# Layer-by-layer hidden-state cosine diff (HF vs mine) to localize the residual text gap.
# Run: torchrun --nproc-per-node=1 /workspace/scripts/test_layer_diff.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import gemma4_unified_bridge
from gemma4_unified_vl import Gemma4UnifiedVLModel, load_vision_from_safetensors
import gemma4_vision_mask as gm
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

br = AutoBridge.from_hf_pretrained(MODEL)
provider = br.to_megatron_provider(load_weights=False)
provider.tensor_model_parallel_size = 1; provider.seq_length = 512
lm = provider.provide().cuda(); br.load_hf_weights([lm]); del br.hf_pretrained; torch.cuda.empty_cache()
enc, proj = load_vision_from_safetensors(); enc = enc.cuda().eval(); proj = proj.cuda().eval()
vl = Gemma4UnifiedVLModel(lm, enc, proj).cuda().eval()
gm.install_sliding_arbitrary_mask(vl.language_model)

my_h = {}
def mk(i):
    def h(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out    # [s,b,h]
        my_h[i] = t.detach().float().transpose(0, 1)[0].cpu()   # [s,h]
    return h
for i, l in enumerate(vl.language_model.decoder.layers):
    l.register_forward_hook(mk(i))

gm._CTX["mask"] = gm.build_local_mask(mm_type)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    _ = vl(input_ids=input_ids, position_ids=pos, pixel_values=pixel_values,
           image_position_ids=image_position_ids, attention_mask=None)
gm._CTX["mask"] = None

from transformers import AutoModelForImageTextToText
hf = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
hf_h = {}
def mkh(i):
    def h(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out    # [b,s,h]
        hf_h[i] = t.detach().float()[0].cpu()            # [s,h]
    return h
for i, l in enumerate(hf.model.language_model.layers):
    l.register_forward_hook(mkh(i))
with torch.no_grad():
    hf(input_ids=input_ids, pixel_values=pixel_values.unsqueeze(0),
       image_position_ids=image_position_ids.unsqueeze(0), mm_token_type_ids=mm_type)

print("layer | text_cos  img_cos | text_|a|/|b|")
for i in range(len(vl.language_model.decoder.layers)):
    if i not in my_h or i not in hf_h: continue
    a, b = my_h[i], hf_h[i]
    tc = torch.nn.functional.cosine_similarity(a[N:], b[N:], dim=-1).mean().item()
    ic = torch.nn.functional.cosine_similarity(a[:N], b[:N], dim=-1).mean().item()
    rr = (a[N:].norm(dim=-1) / b[N:].norm(dim=-1)).mean().item()
    if i < 6 or i % 6 == 0 or i >= 44:
        print(f"  {i:2d}  | {tc:.5f}  {ic:.5f} | {rr:.3f}")
