#!/usr/bin/env python
# Diff my sliding mask against HF's ACTUAL per-layer attention mask on a tiny sequence.
# Run: python /workspace/scripts/test_mask_diff.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import gemma4_vision_mask as gm

MODEL = "google/gemma-4-12B"; IMG = 258880
def boolmat(t):  # -> [s,s] python for printing, 1=masked
    return "\n".join("".join("#" if v else "." for v in row) for row in t.tolist())

# tiny input: text .. image[3:7] .. text
S = 12
input_ids = torch.tensor([[10, 11, 12, IMG, IMG, IMG, IMG, 20, 21, 22, 23, 24]], device="cuda")
mm_type = (input_ids == IMG).long()
pixel_values = torch.randn(4, 6912, dtype=torch.bfloat16, device="cuda")
image_position_ids = torch.stack([torch.arange(4) % 2, torch.arange(4) // 2], -1).to("cuda")

from transformers import AutoModelForImageTextToText
hf = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()

# find the text decoder layers + capture attention_mask per layer
tm = hf.model.language_model
print("layer_types[:12]:", tm.config.layer_types[:12])
cap = {}
def mk_hook(i):
    def hook(mod, args, kwargs):
        cap[i] = kwargs.get("attention_mask", None)
    return hook
for i, layer in enumerate(tm.layers):
    layer.register_forward_pre_hook(mk_hook(i), with_kwargs=True)

with torch.no_grad():
    hf(input_ids=input_ids, pixel_values=pixel_values.unsqueeze(0),
       image_position_ids=image_position_ids.unsqueeze(0), mm_token_type_ids=mm_type)

def to_bool_masked(m):
    if m is None: return None
    t = m
    if t.dtype == torch.bool:
        masked = ~t                      # transformers bool: True=attend
    else:
        masked = t < -1.0                # additive: large-negative = masked
    return masked[0, 0]                  # [s,s]

# pick a sliding layer (0) and a full layer (first index in layer_types == full_attention)
sl_i = next(i for i, t in enumerate(tm.config.layer_types) if t == "sliding_attention")
fu_i = next(i for i, t in enumerate(tm.config.layer_types) if t == "full_attention")
hf_sl = to_bool_masked(cap[sl_i]); hf_fu = to_bool_masked(cap[fu_i])
my_sl = gm.build_local_mask(mm_type)[0, 0]

print(f"\n=== HF sliding (layer {sl_i}) mask type={cap[sl_i].dtype if cap[sl_i] is not None else None} shape={tuple(cap[sl_i].shape) if cap[sl_i] is not None else None} ===")
print(boolmat(hf_sl))
print("\n=== MY sliding mask ===")
print(boolmat(my_sl))
print("\n=== DIFF (X where HF != mine) ===")
diff = (hf_sl != my_sl)
print(boolmat(diff))
print("  sliding mismatches:", int(diff.sum().item()))
print(f"\n=== HF full (layer {fu_i}) mask ===")
print(boolmat(hf_fu))
print("  (my global layers = plain causal; compare visually)")
