#!/bin/bash

# Use the CONTAINER_RUNTIME from the environment, or default to docker if it's not set.
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
echo "Using container tool: ${CONTAINER_RUNTIME}"

# Set a default EPP_TAG if not provided
EPP_TAG="${EPP_TAG:-dev}"
# Set a default VLLM_SIMULATOR_TAG if not provided
VLLM_SIMULATOR_TAG="${VLLM_SIMULATOR_TAG:-latest}"
# Set the default routing side car image tag
SIDECAR_TAG="${SIDECAR_TAG:-dev}"
# Set the default UDS tokenizer image tag
UDS_TOKENIZER_TAG="${UDS_TOKENIZER_TAG:-dev}"

export EPP_IMAGE="${EPP_IMAGE:-ghcr.io/llm-d/llm-d-router-endpoint-picker:${EPP_TAG}}"
export VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/llm-d/llm-d-inference-sim:${VLLM_SIMULATOR_TAG}}"
export SIDECAR_IMAGE="${SIDECAR_IMAGE:-ghcr.io/llm-d/llm-d-router-disagg-sidecar:${SIDECAR_TAG}}"
export UDS_TOKENIZER_IMAGE="${UDS_TOKENIZER_IMAGE:-ghcr.io/llm-d/llm-d-uds-tokenizer:${UDS_TOKENIZER_TAG}}"
export VLLM_RENDER_IMAGE="${VLLM_RENDER_IMAGE:-vllm/vllm-openai-cpu:v0.19.1}"

TARGETOS="${TARGETOS:-linux}"
TARGETARCH="${TARGETARCH:-$(go env GOARCH)}"

# --- Helper Function to Ensure Image Availability ---
# This function checks the registry first, then falls back to a local-only check.
ensure_image() {
  local image_name="$1"
  echo "Checking for image: ${image_name}"

  if [ -n "$(${CONTAINER_RUNTIME} images -q "${image_name}")" ]; then
    echo " -> Found local image. Proceeding."
  elif ${CONTAINER_RUNTIME} manifest inspect "${image_name}" > /dev/null 2>&1; then
    echo " -> Image found on registry. Pulling..."
    if ! ${CONTAINER_RUNTIME} pull --platform ${TARGETOS}/${TARGETARCH} "${image_name}"; then
        echo "    ❌ ERROR: Failed to pull image '${image_name}'."
        exit 1
    fi
    echo "    ✅ Successfully pulled image."
  else
      echo "    ❌ ERROR: Image '${image_name}' is not available locally and could not be found on the registry."
      exit 1
  fi
}

# --- Print Final Images and Pull Dependencies ---
echo "--- Using the following images ---"
echo "Scheduler Image:     ${EPP_IMAGE}"
echo "Simulator Image:     ${VLLM_IMAGE}"
echo "Sidecar Image:       ${SIDECAR_IMAGE}"
echo "UDS Tokenizer Image: ${UDS_TOKENIZER_IMAGE}"
echo "vLLM Render Image:   ${VLLM_RENDER_IMAGE}"
echo "----------------------------------------------------"

echo "Pulling dependencies..."
ensure_image "${EPP_IMAGE}"
ensure_image "${VLLM_IMAGE}"
ensure_image "${SIDECAR_IMAGE}"
ensure_image "${UDS_TOKENIZER_IMAGE}"
ensure_image "${VLLM_RENDER_IMAGE}"
echo "Successfully pulled dependencies"
