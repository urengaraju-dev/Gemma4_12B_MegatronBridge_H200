#!/usr/bin/env python
# Vision-embedder numeric parity: my GemmaVisionProjector vs HF get_image_features,
# fed IDENTICAL synthetic patches (isolates forward math from the image processor).
# Run: python /workspace/scripts/test_vision_parity.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from gemma4_vision_projector import Gemma4UnifiedVisionProjector

MODEL = "google/gemma-4-12B"
def hr(t): print("\n" + "="*18 + " " + t + " " + "="*18, flush=True)

torch.manual_seed(0)
P, PATCH, DIM = 64, 6912, 3840          # 64 valid patches (no padding)
pixel_values = torch.randn(1, P, PATCH, dtype=torch.bfloat16).cuda()
# valid (x,y) positions, all < mm_posemb_size=1120, none == -1
image_position_ids = torch.stack([torch.arange(P) % 40, torch.arange(P) // 40], dim=-1).unsqueeze(0).cuda()

hr("load HF model (for reference + weights)")
from transformers import AutoModelForImageTextToText
hf = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
sd = hf.state_dict()
print("  vision keys present:", all(k in sd for k in [
    "model.vision_embedder.patch_dense.weight", "model.embed_vision.embedding_projection.weight"]))

hr("HF reference: get_image_features")
with torch.no_grad():
    ref = hf.model.get_image_features(pixel_values, image_position_ids).pooler_output  # [P,3840]
print("  HF vision features:", tuple(ref.shape), ref.dtype)

hr("resolve runtime vision key names (naming divergence)")
vis_keys = sorted(k for k in sd if any(t in k for t in
    ("patch_ln", "patch_dense", "pos_embedding", "pos_norm", "embedding_projection", "vision_embedder", "embed_vision")))
for k in vis_keys:
    print("   ", k, tuple(sd[k].shape))

def find(suffix):
    hits = [k for k in sd if k.endswith(suffix)]
    # prefer vision-scoped keys, exclude audio
    hits = [k for k in hits if "audio" not in k]
    return hits[0] if hits else None

hr("my projector (loaded by suffix-matched keys)")
proj = Gemma4UnifiedVisionProjector(dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    proj.patch_ln1.weight.copy_(sd[find("patch_ln1.weight")]); proj.patch_ln1.bias.copy_(sd[find("patch_ln1.bias")])
    proj.patch_dense.weight.copy_(sd[find("patch_dense.weight")]); proj.patch_dense.bias.copy_(sd[find("patch_dense.bias")])
    proj.patch_ln2.weight.copy_(sd[find("patch_ln2.weight")]); proj.patch_ln2.bias.copy_(sd[find("patch_ln2.bias")])
    proj.pos_embedding.copy_(sd[find("pos_embedding")])
    proj.pos_norm.weight.copy_(sd[find("pos_norm.weight")]); proj.pos_norm.bias.copy_(sd[find("pos_norm.bias")])
    # vision embedding_projection (exclude audio already handled by find)
    ep = [k for k in sd if k.endswith("embedding_projection.weight") and "audio" not in k][0]
    print("   embedding_projection <-", ep)
    proj.embedding_projection.weight.copy_(sd[ep])
with torch.no_grad():
    mine = proj(pixel_values, image_position_ids).reshape(P, DIM)  # [P,3840]
print("  my vision features:", tuple(mine.shape), mine.dtype)

hr("PARITY")
a, b = ref.float(), mine.float()
maxd = (a - b).abs().max().item()
meand = (a - b).abs().mean().item()
cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
rel = ((a - b).norm() / a.norm()).item()
print(f"  max|Δ| {maxd:.4f} | mean|Δ| {meand:.4f} | cosine {cos:.6f} | rel-L2 {rel:.5f}")
print("  VERDICT:", "PASS" if cos > 0.999 and rel < 0.02 else ("CLOSE" if cos > 0.99 else "FAIL"))
