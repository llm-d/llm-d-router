SHELL := /usr/bin/env bash

# Image registry + dev-environment image tags (single source of truth).
include versions.mk

# Export all dev-env image references so scripts/kind-dev-env.sh sees them.
export IMAGE_REGISTRY VLLM_SIMULATOR_TAG EPP_TAG SIDECAR_TAG UDS_TOKENIZER_TAG
export VLLM_IMAGE EPP_IMAGE SIDECAR_IMAGE UDS_TOKENIZER_IMAGE

PROJECT_NAME ?= llm-d-coordinator
COORDINATOR_TAG ?= dev
export COORDINATOR_IMAGE ?= $(IMAGE_REGISTRY)/$(PROJECT_NAME):$(COORDINATOR_TAG)

CONTAINER_RUNTIME := $(shell { command -v docker >/dev/null 2>&1 && echo docker; } || { command -v podman >/dev/null 2>&1 && echo podman; } || echo "")
export CONTAINER_RUNTIME

.PHONY: help
help: ## Print help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ EPP Image

.PHONY: image-build-epp
image-build-epp: ## Clone llm-d-inference-scheduler at pinned commit and build EPP image
	scripts/build-epp-image.sh

##@ Kind Development Environment

.PHONY: env-dev-kind
env-dev-kind: image-build-epp ## Deploy dev environment on a local Kind cluster
	scripts/kind-dev-env.sh

.PHONY: clean-env-dev-kind
clean-env-dev-kind: ## Delete the Kind dev cluster
	kind delete cluster --name llm-d-coordinator-dev

##@ Go Coordinator Service

BINARY := coordinator
BUILD_DIR := bin

.PHONY: build
build: ## Build the coordinator Go binary
	@mkdir -p $(BUILD_DIR)
	go build -o $(BUILD_DIR)/$(BINARY) ./cmd/coordinator

.PHONY: test
test: ## Run Go tests
	go test ./...

.PHONY: lint
lint: ## Run Go linter
	golangci-lint run ./...

.PHONY: run
run: build ## Build and run the coordinator locally
	$(BUILD_DIR)/$(BINARY) --config configs/coordinator.yaml

.PHONY: tidy
tidy: ## Run go mod tidy
	go mod tidy

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf $(BUILD_DIR)
