# =============================================================================
# Gemma-4-12B on Megatron-Bridge — developer entry points.
# All targets run inside the NeMo container ($(CONTAINER), default `mb`),
# which is created/started by `make setup`. Repo is mounted at /workspace.
# =============================================================================
CONTAINER ?= mb
DEXEC      = docker exec $(CONTAINER)
TORCHRUN   = $(DEXEC) torchrun --nproc-per-node=1

.DEFAULT_GOAL := help
.PHONY: help setup phase0-parity phase0-lora phase0-build phase1-vision phase1-parity mask-diff shell clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Pull container, download gemma-4-12B, start the container (idempotent)
	bash scripts/setup.sh

phase0-build: ## Phase 0: build the 12B text model + forward (config only, no download)
	$(TORCHRUN) /workspace/tests/test_phase0.py

phase0-parity: ## Phase 0: HF vs Megatron logit parity (text) — expect cosine 0.9999, top-1 100%
	$(TORCHRUN) /workspace/tests/parity_g4.py

phase0-lora: ## Phase 0: 3-step LoRA finetune smoke — expect loss 1.86 -> 1.12
	$(TORCHRUN) /workspace/tests/train_lora_g4.py

phase1-vision: ## Phase 1: vision embedder vs HF get_image_features — expect cosine 1.000000
	$(DEXEC) python /workspace/tests/test_vision_parity.py

mask-diff: ## Phase 1: my bidirectional mask vs HF's actual per-layer mask (0 mismatches)
	$(DEXEC) python /workspace/tests/test_mask_diff.py

phase1-parity: ## Phase 1: image-conditioned HF logit parity — expect image 0.999 / text 0.975
	$(DEXEC) -e APPLY_ALL=1 torchrun --nproc-per-node=1 /workspace/tests/phase1_parity.py

shell: ## Open a shell in the container
	$(DEXEC) -it bash

clean: ## Remove the container (weights/caches on the host are kept)
	-docker rm -f $(CONTAINER)
