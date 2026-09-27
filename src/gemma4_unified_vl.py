#!/usr/bin/env python
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# Phase-1: Gemma4UnifiedVLModel — assemble the Gemma-4-12B image+text model.
# Reuses the validated Phase-0 text GPTModel + bit-exact vision embedder; merges vision
# soft tokens into the LM embedding stream via masked_scatter at image_token_id (as MIMO does).
# Imported as a module by the harnesses; exercised via tests/phase1_parity.py.
import os, sys, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch, torch.nn as nn

import gemma4_unified_bridge  # registers the Phase-0 text bridge
from gemma4_vision_projector import Gemma4UnifiedVisionProjector, ScalelessRMSNorm
from megatron.bridge import AutoBridge

MODEL = "google/gemma-4-12B"
IMG_TOKEN = 258880
DIM = 3840
def hr(t): print("\n" + "="*20 + " " + t + " " + "="*20, flush=True)


class Gemma4UnifiedVLModel(nn.Module):
    """Phase-1 image+text wrapper: Phase-0 text GPTModel + encoder-free vision embedder.

    forward merges vision soft tokens into the token-embedding stream at IMG_TOKEN
    positions (masked_scatter), then runs the LM with decoder_input.
    NOTE: no bidirectional vision mask yet (Phase-1 step 4) -> image tokens attend causally.
    """
    def __init__(self, language_model, vision_encoder, vision_projection):
        super().__init__()
        self.language_model = language_model            # Phase-0 Gemma4 text GPTModel
        self.vision_encoder = vision_encoder            # patch stack -> scaleless RMSNorm
        self.vision_projection = vision_projection      # nn.Linear(3840,3840,bias=False)

    def forward(self, input_ids, position_ids, pixel_values, image_position_ids,
                attention_mask=None, labels=None):
        # Verified vs HF (hidden-state diff): HF scales TEXT embeds by sqrt(h) but leaves IMAGE
        # features UNSCALED — a relative magnitude that does NOT wash out under RMSNorm in the
        # multimodal case. lm.embedding here returns UNSCALED text (ratio 1.0), so scale text only.
        # 1) text embeds [s, b, h], scaled by sqrt(h)
        text_embeds = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)
        text_embeds = text_embeds * (self.language_model.config.hidden_size ** 0.5)
        # 2) UNSCALED vision soft tokens [N, h]
        vis = self.vision_projection(self.vision_encoder(pixel_values, image_position_ids))  # [N,h]
        # 3) masked_scatter vision into image-token slots (work in [b,s,h])
        embeds_bsh = text_embeds.transpose(0, 1).contiguous()          # [b,s,h]
        mask = (input_ids == IMG_TOKEN).unsqueeze(-1).expand_as(embeds_bsh)
        n_slots = int((input_ids == IMG_TOKEN).sum().item())
        assert n_slots == vis.shape[0], f"image slots {n_slots} != vision tokens {vis.shape[0]}"
        embeds_bsh = embeds_bsh.masked_scatter(mask, vis.to(embeds_bsh.dtype))
        self.last_merged = embeds_bsh.detach()                          # [b,s,h] for diagnostics
        decoder_input = embeds_bsh.transpose(0, 1).contiguous()        # [s,b,h]
        # 4) run the LM with the merged embeddings
        return self.language_model(input_ids=None, position_ids=position_ids,
                                   attention_mask=attention_mask, decoder_input=decoder_input,
                                   labels=labels)


def load_vision_from_safetensors(dtype=torch.bfloat16):
    import glob
    from safetensors import safe_open
    st = glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--google--gemma-4-12B",
                                "snapshots", "*", "model.safetensors"))[0]
    V, E = "model.vision_embedder.", "model.embed_vision."   # on-disk (safetensors) keys
    with safe_open(st, framework="pt", device="cpu") as f:
        g = lambda k: f.get_tensor(k)
        enc = Gemma4UnifiedVisionProjector(dtype=dtype)
        with torch.no_grad():
            enc.patch_ln1.weight.copy_(g(V+"patch_ln1.weight")); enc.patch_ln1.bias.copy_(g(V+"patch_ln1.bias"))
            enc.patch_dense.weight.copy_(g(V+"patch_dense.weight")); enc.patch_dense.bias.copy_(g(V+"patch_dense.bias"))
            enc.patch_ln2.weight.copy_(g(V+"patch_ln2.weight")); enc.patch_ln2.bias.copy_(g(V+"patch_ln2.bias"))
            enc.pos_embedding.copy_(g(V+"pos_embedding"))
            enc.pos_norm.weight.copy_(g(V+"pos_norm.weight")); enc.pos_norm.bias.copy_(g(V+"pos_norm.bias"))
            ep_w = g(E+"embedding_projection.weight")
            enc.embedding_projection.weight.copy_(ep_w)

    # split: encoder = everything up to & including the scaleless RMSNorm; projection = the Linear
    class _Encoder(nn.Module):
        def __init__(self, e):
            super().__init__()
            self.patch_ln1, self.patch_dense, self.patch_ln2 = e.patch_ln1, e.patch_dense, e.patch_ln2
            self.pos_embedding, self.pos_norm = e.pos_embedding, e.pos_norm
            self.rms = ScalelessRMSNorm(e.embedding_pre_projection_norm.eps)
        def forward(self, pixel_values, image_position_ids):
            h = self.patch_ln1(pixel_values.to(self.patch_dense.weight.dtype))
            h = self.patch_ln2(self.patch_dense(h))
            clamped = image_position_ids.clamp(min=0).long()
            valid = (image_position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
            axes = torch.arange(2, device=image_position_ids.device)
            h = h + (self.pos_embedding[clamped, axes] * valid).sum(-2)
            return self.rms(self.pos_norm(h))
    proj = nn.Linear(DIM, DIM, bias=False, dtype=dtype); proj.weight.data.copy_(ep_w)
    return _Encoder(enc), proj


if __name__ == "__main__":
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)

    hr("build + load Phase-0 text GPTModel")
    br = AutoBridge.from_hf_pretrained(MODEL)
    provider = br.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1; provider.pipeline_model_parallel_size = 1
    provider.seq_length = 512
    lm = provider.provide().cuda()
    br.load_hf_weights([lm])
    print("  text GPTModel loaded:", sum(p.numel() for p in lm.parameters()), "params")

    hr("load vision embedder + projection")
    enc, proj = load_vision_from_safetensors()
    enc = enc.cuda().eval(); proj = proj.cuda().eval()
    torch.cuda.empty_cache()
    print("  vision encoder + projection loaded")

    model = Gemma4UnifiedVLModel(lm, enc, proj).cuda().eval()

    hr("synthetic image+text batch")
    torch.manual_seed(0)
    N, n_txt = 64, 16
    S = N + n_txt
    input_ids = torch.randint(0, 250000, (1, S), device="cuda")
    input_ids[0, :N] = IMG_TOKEN
    position_ids = torch.arange(S, device="cuda").unsqueeze(0)
    pixel_values = torch.randn(N, 6912, dtype=torch.bfloat16, device="cuda")
    image_position_ids = torch.stack([torch.arange(N) % 40, torch.arange(N) // 40], -1).to("cuda")
    labels = torch.roll(input_ids, -1, dims=1)
    labels[input_ids == IMG_TOKEN] = -100
    print(f"  S={S} image_tokens={N} text_tokens={n_txt}")

    hr("forward")
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, position_ids=position_ids,
                        pixel_values=pixel_values, image_position_ids=image_position_ids,
                        attention_mask=None, labels=None)
        print("  FORWARD OK — logits:", tuple(out.shape), "| finite:", bool(torch.isfinite(out).all()))
        # also with labels -> loss
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, position_ids=position_ids,
                         pixel_values=pixel_values, image_position_ids=image_position_ids,
                         attention_mask=None, labels=labels)
        print("  LOSS forward OK — loss shape:", tuple(loss.shape), "| mean:", float(loss.float().mean()))
        print("\n  [phase1] STEPS 1-3 DONE — image+text pipeline runs end-to-end (causal; mask=Phase-1 step 4).")
    except Exception as e:
        print("  FORWARD FAILED:", type(e).__name__, str(e)[:400]); traceback.print_exc()
