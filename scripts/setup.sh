#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh — provision the Megatron-Bridge environment for Gemma-4-12B
#                    on a single NVIDIA GPU (H200 here). Idempotent.
#
# Steps: verify HF token -> pull NeMo container -> download gemma-4-12B ->
#        start a long-lived container with GPU + caches + this repo mounted.
#
# Prereqs: Docker + NVIDIA runtime; an HF token (Gemma license accepted) in
#          ~/.hf_token or $HF_TOKEN.  Usage:  bash scripts/setup.sh
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-nvcr.io/nvidia/nemo:26.08}"          # bundles Megatron-Bridge 0.6.1
CONTAINER="${CONTAINER:-mb}"
HF_CACHE="${HF_CACHE:-/ephemeral/cache/huggingface}" # persistent host HF cache
MODEL="${MODEL:-google/gemma-4-12B}"

echo "==> repo : ${REPO_DIR}"
echo "==> image: ${IMAGE}"
mkdir -p "${HF_CACHE}"

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf_token" ]]; then
  HF_TOKEN="$(cat "${HOME}/.hf_token")"
fi
: "${HF_TOKEN:?ERROR: set HF_TOKEN (or ~/.hf_token) for a Gemma-licensed HF account}"
export HF_TOKEN HF_HOME="${HF_CACHE}"

echo "==> Pulling ${IMAGE} (large, one-time)..."
docker pull "${IMAGE}"

echo "==> Downloading ${MODEL} (~23 GB) into ${HF_CACHE} ..."
if command -v hf >/dev/null 2>&1; then
  HF_HOME="${HF_CACHE}" hf download "${MODEL}" --repo-type model
else
  HF_HOME="${HF_CACHE}" python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"
fi

echo "==> (Re)starting container '${CONTAINER}' ..."
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -d --name "${CONTAINER}" \
  --gpus all --ipc=host --shm-size=16g \
  -e HF_HOME=/hf_cache -e HF_TOKEN \
  -v "${REPO_DIR}":/workspace \
  -v "${HF_CACHE}":/hf_cache \
  -w /workspace \
  "${IMAGE}" sleep infinity

echo "==> Verifying megatron.bridge + GPU inside the container ..."
docker exec "${CONTAINER}" python -c \
  "import torch, importlib.metadata as m; print('megatron-bridge', m.version('megatron-bridge')); \
   print('cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

echo
echo "==> Ready. Try:"
echo "    make phase0-parity   # HF logit parity (text)"
echo "    make phase1-parity   # image-conditioned parity"
