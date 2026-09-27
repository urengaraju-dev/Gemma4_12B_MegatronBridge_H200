#!/usr/bin/env python
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# 3-step LoRA smoke test of google/gemma-4-12B (text tower) on Megatron-Bridge,
# using the Phase-0 Gemma4Unified bridge. Real HF->Megatron weight load happens
# through cfg.checkpoint.pretrained_checkpoint (our bridge).
# Run: torchrun --nproc-per-node=1 /workspace/scripts/train_lora_g4.py
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import gemma4_unified_bridge  # registers Gemma4UnifiedBridge
from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _peft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

MODEL = "google/gemma-4-12B"
SEQ = 512
OUT = "/workspace/outputs/g4_12b_lora_smoke"


def _snapshot(model_id):
    hub = "models--" + model_id.replace("/", "--")
    hits = sorted(glob.glob(os.path.join(os.environ.get("HF_HOME", ""), "hub", hub, "snapshots", "*")))
    for s in hits:
        if glob.glob(os.path.join(s, "*.safetensors")):
            return s
    raise FileNotFoundError(model_id)


def build_config():
    cfg = _peft_common()
    cfg.peft = default_peft_config("lora")

    # Gemma-4-12B text provider (Phase-0 bridge), random-init here; weights load
    # from pretrained_checkpoint via the training harness.
    cfg.model = AutoBridge.from_hf_pretrained(MODEL).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = SEQ
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None

    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = MODEL

    # tiny, fast dataset: SQuAD, unpacked, few samples, no val/test
    cfg.dataset.seq_length = SEQ
    cfg.dataset.enable_offline_packing = False
    cfg.dataset.offline_packing_specs = None
    cfg.dataset.max_train_samples = 32
    cfg.dataset.do_validation = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.do_test = False

    cfg.train.train_iters = 3
    cfg.train.global_batch_size = 1
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 10_000
    cfg.validation.eval_iters = 0
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = 3
    cfg.logger.log_interval = 1

    snap = _snapshot(MODEL)
    print(f"[smoke] pretrained_checkpoint = {snap}", flush=True)
    cfg.checkpoint.pretrained_checkpoint = snap
    cfg.checkpoint.save = os.path.join(OUT, "checkpoints")
    cfg.checkpoint.load = os.path.join(OUT, "checkpoints")
    cfg.checkpoint.save_interval = 3
    return cfg


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cfg = build_config()
    print("[smoke] launching finetune() for Gemma-4-12B text LoRA (seq=512, iters=3, TP=PP=CP=1)", flush=True)
    finetune(config=cfg, forward_step_func=forward_step)
    print("[smoke] DONE — Gemma-4-12B LoRA pipeline ran end-to-end with no errors.", flush=True)
