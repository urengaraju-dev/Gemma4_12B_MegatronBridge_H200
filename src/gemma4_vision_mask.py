# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 step 4: Gemma-4 bidirectional vision attention (sliding layers only).

The Gemma-4 dense model uses MCore's LOCAL DotProductAttention, which applies masking
inside FusedScaleMaskSoftmax using the INIT-TIME attn_mask_type (causal) + window_size,
ignoring the per-call attn_mask_type/attention_mask on its fused-causal path (and it
rejects attention_bias). So we inject at the softmax: for each sliding layer, temporarily
switch its scale_mask_softmax to attn_mask_type=padding (which applies the PASSED boolean
mask via masked_fill) and feed our full mask [causal ∨ same_vision_block, within window].
Gemma's softmax scale/offset are preserved (we only change the mask). Global layers untouched.
"""
import os
import types
import torch
from megatron.core.transformer.enums import AttnMaskType
from megatron.bridge.models.gemma.modeling_gemma4 import _is_gemma4_sliding_layer

_CTX = {"mask": None}                       # [1,1,s,s] bool True=masked, set before LM forward
_STATS = {"sliding": 0, "wrapped": 0, "fired": 0, "applied": 0}
_APPLY_ALL = os.environ.get("APPLY_ALL", "1") == "1"   # HF applies the bidir vision mask on ALL layers (verified)


def build_block_ids(mm_token_type_ids):
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    prev = torch.zeros_like(is_vision); prev[:, 1:] = is_vision[:, :-1]
    group = (is_vision & ~prev).long().cumsum(dim=1)
    return torch.where(is_vision, group, torch.full_like(group, -1))


def build_local_mask(mm_token_type_ids, window: int = 1024) -> torch.Tensor:
    """[1,1,s,s] bool, True = masked-out, for sliding (local) layers."""
    b, s = mm_token_type_ids.shape
    dev = mm_token_type_ids.device
    i = torch.arange(s, device=dev)[:, None]; j = torch.arange(s, device=dev)[None, :]
    causal = j <= i; sliding = j > (i - window)
    block = build_block_ids(mm_token_type_ids)
    same_blk = (block[:, :, None] == block[:, None, :]) & (block[:, :, None] >= 0)
    allow = sliding[None] & (causal[None] | same_blk)          # [b,s,s]
    return (~allow).unsqueeze(1)                                # [b,1,s,s]


def install_sliding_arbitrary_mask(model):
    for layer in model.decoder.layers:
        sa = getattr(layer, "self_attention", None)
        if sa is None:
            continue
        ca = sa.core_attention
        ca._true_sliding = bool(_is_gemma4_sliding_layer(layer.config, layer.layer_number))
        ca._is_sliding = True if _APPLY_ALL else ca._true_sliding     # whether to APPLY the mask
        _STATS["sliding"] += int(ca._true_sliding)
        if getattr(ca, "_g4_mask_wrapped", False):
            continue
        ca._orig_forward = ca.forward

        def wrapped(self, *args, **kwargs):
            _STATS["fired"] += 1
            m = _CTX["mask"]
            if m is not None and getattr(self, "_is_sliding", False):
                smsm = self.scale_mask_softmax
                saved = smsm.attn_mask_type
                smsm.attn_mask_type = AttnMaskType.padding      # apply the PASSED mask fully
                nargs = list(args)
                if len(nargs) >= 4:
                    nargs[3] = m
                else:
                    kwargs = dict(kwargs); kwargs["attention_mask"] = m
                _STATS["applied"] += 1
                _tag = "sliding" if getattr(self, "_true_sliding", True) else "GLOBAL"
                if not _STATS.get("ab_" + _tag):
                    _STATS["ab_" + _tag] = True
                    with torch.no_grad():
                        om = self._orig_forward(*tuple(nargs), **kwargs)
                        smsm.attn_mask_type = saved
                        oo = self._orig_forward(*args, **kwargs)
                        smsm.attn_mask_type = AttnMaskType.padding
                        om = om[0] if isinstance(om, tuple) else om
                        oo = oo[0] if isinstance(oo, tuple) else oo
                        print(f"    [A/B {_tag}] core_attn max|Δ|(mask vs causal)="
                              f"{(om.float()-oo.float()).abs().max().item():.5f}")
                try:
                    return self._orig_forward(*tuple(nargs), **kwargs)
                finally:
                    smsm.attn_mask_type = saved
            return self._orig_forward(*args, **kwargs)

        ca.forward = types.MethodType(wrapped, ca)
        ca._g4_mask_wrapped = True
        _STATS["wrapped"] += 1
    print(f"  [mask install] sliding={_STATS['sliding']} wrapped={_STATS['wrapped']}")
    return model
