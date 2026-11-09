#!/usr/bin/env bash
# GCE startup script for a runner VM: waits for the NVIDIA driver, makes sure
# Docker can use the GPU, then runs vLLM on the runner port.
# Parameters come from instance metadata (model, vllm-image, runner-port).
set -euo pipefail
md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
MODEL="$(md model)"
VLLM_IMAGE="$(md vllm-image)"
PORT="$(md runner-port)"

for _ in $(seq 1 120); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi

if ! command -v docker >/dev/null; then
  # Some DLVM images ship the NVIDIA stack without Docker.
  apt-get update && apt-get install -y docker.io
  systemctl enable --now docker
fi

if ! docker info 2>/dev/null | grep -qi 'nvidia'; then
  if ! command -v nvidia-ctk >/dev/null; then
    apt-get update && apt-get install -y nvidia-container-toolkit
  fi
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

mkdir -p /var/lib/hf-cache
docker rm -f vllm >/dev/null 2>&1 || true
docker run -d --name vllm --restart unless-stopped --gpus all --ipc=host \
  -p "${PORT}:8000" -v /var/lib/hf-cache:/root/.cache/huggingface \
  "${VLLM_IMAGE}" --model "${MODEL}" --max-model-len 8192 --gpu-memory-utilization 0.85
