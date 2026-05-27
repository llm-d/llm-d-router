# Centralized image registry + version tags for the dev environment.
# Override any of these via `make VAR=... env-dev-kind` or
# `export VAR=... && make env-dev-kind`.

IMAGE_REGISTRY            ?= ghcr.io/llm-d

# Image names (this repo's source-of-truth — only override for a fork)
EPP_IMAGE_NAME            ?= llm-d-router-endpoint-picker
SIDECAR_IMAGE_NAME        ?= llm-d-router-disagg-sidecar
VLLM_SIMULATOR_IMAGE_NAME ?= llm-d-inference-sim
UDS_TOKENIZER_IMAGE_NAME  ?= llm-d-uds-tokenizer
BUILDER_IMAGE_NAME        ?= llm-d-builder

# Image tags
EPP_TAG                   ?= dev
SIDECAR_TAG               ?= dev
VLLM_SIMULATOR_TAG        ?= v0.9.0
UDS_TOKENIZER_TAG         ?= dev
BUILDER_TAG               ?= dev
COORDINATOR_TAG           ?= dev

# Image bases (derived; cluster targets reference *_TAG_BASE directly)
EPP_IMAGE_TAG_BASE        ?= $(IMAGE_REGISTRY)/$(EPP_IMAGE_NAME)
SIDECAR_IMAGE_TAG_BASE    ?= $(IMAGE_REGISTRY)/$(SIDECAR_IMAGE_NAME)
VLLM_SIMULATOR_TAG_BASE   ?= $(IMAGE_REGISTRY)/$(VLLM_SIMULATOR_IMAGE_NAME)
UDS_TOKENIZER_TAG_BASE    ?= $(IMAGE_REGISTRY)/$(UDS_TOKENIZER_IMAGE_NAME)
BUILDER_TAG_BASE          ?= $(IMAGE_REGISTRY)/$(BUILDER_IMAGE_NAME)

# Full image references (override only if you need a non-standard repo)
export EPP_IMAGE           ?= $(EPP_IMAGE_TAG_BASE):$(EPP_TAG)
export SIDECAR_IMAGE       ?= $(SIDECAR_IMAGE_TAG_BASE):$(SIDECAR_TAG)
export VLLM_IMAGE          ?= $(VLLM_SIMULATOR_TAG_BASE):$(VLLM_SIMULATOR_TAG)
export UDS_TOKENIZER_IMAGE ?= $(UDS_TOKENIZER_TAG_BASE):$(UDS_TOKENIZER_TAG)
export BUILDER_IMAGE       ?= $(BUILDER_TAG_BASE):$(BUILDER_TAG)

# CPU-only vLLM image that exposes `vllm launch render`
export VLLM_RENDER_IMAGE   ?= vllm/vllm-openai-cpu:v0.21.0

# Images consumed only by the e-p-d-pools env (DISAGG_POOLS_TOPOLOGY=true).
export COORDINATOR_IMAGE     ?= ghcr.io/revit13/llm-d-coordinator:$(COORDINATOR_TAG)
export MOCK_DOWNLOADER_HTTP_IMAGE ?= python:3.10-slim
export MOCK_DOWNLOADER_INIT_IMAGE ?= busybox:1.36
