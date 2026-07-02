SHELL := /usr/bin/env bash

# Image registry + dev-environment image tags (single source of truth).
include versions.mk

# Export all dev-env image references so the e2e suite sees them.
export IMAGE_REGISTRY COORDINATOR_TAG VLLM_SIMULATOR_TAG EPP_TAG
export COORDINATOR_IMAGE VLLM_IMAGE EPP_IMAGE

# Defaults
TARGETOS ?= $(shell command -v go >/dev/null 2>&1 && go env GOOS || uname -s | tr '[:upper:]' '[:lower:]')
TARGETARCH ?= $(shell command -v go >/dev/null 2>&1 && go env GOARCH || uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/; s/armv7l/arm/')
PROJECT_NAME ?= llm-d-coordinator
BUILDER_IMAGE_NAME ?= llm-d-coordinator-builder
IMAGE_REGISTRY ?= ghcr.io/llm-d

# Internal variable mappings for the generic image-build-% target.
coordinator_IMAGE = $(COORDINATOR_IMAGE)
epp_IMAGE         = $(EPP_IMAGE)

BUILDER_TAG ?= dev
BUILDER_TAG_BASE ?= $(IMAGE_REGISTRY)/$(BUILDER_IMAGE_NAME)
export BUILDER_IMAGE ?= $(BUILDER_TAG_BASE):$(BUILDER_TAG)

CONTAINER_RUNTIME := $(shell { command -v docker >/dev/null 2>&1 && echo docker; } || { command -v podman >/dev/null 2>&1 && echo podman; } || echo "")
export CONTAINER_RUNTIME

GIT_COMMIT_SHA ?= $(shell git rev-parse HEAD 2>/dev/null)
ROOT_RELEASE_TAG_MATCH ?= v[0-9]*
BUILD_REF ?= $(shell git describe --tags --match '$(ROOT_RELEASE_TAG_MATCH)' --abbrev=0 2>/dev/null)

# Named volumes for Go module and build caches, persisted across container runs and image rebuilds.
GO_MOD_CACHE_VOL ?= llm-d-gomodcache
GO_BUILD_CACHE_VOL ?= llm-d-gobuildcache

LDFLAGS ?= -s -w
LINT_NEW_ONLY ?= false

# Optional: override the runtime base image used in container builds.
BASE_IMAGE ?=

TEST_PACKAGES = $$(go list ./... | grep -v /test/ | tr '\n' ' ')

# Common flags for running the builder container: mounts source, Go caches, and runs as current user.
# Podman rootless requires --userns=keep-id to correctly map host UID; docker uses -u directly.
ifeq ($(CONTAINER_RUNTIME),podman)
PODMAN_ROOTLESS := $(shell podman info --format '{{.Host.Security.Rootless}}' 2>/dev/null)
ifeq ($(PODMAN_ROOTLESS),true)
BUILDER_USER_FLAGS = --userns=keep-id
else
BUILDER_USER_FLAGS =
endif
else
BUILDER_USER_FLAGS = -u $$(id -u):$$(id -g)
endif

BUILDER_RUN_FLAGS = --rm $(BUILDER_USER_FLAGS) \
	-v $$(pwd):/app:Z -w /app \
	-v $(GO_MOD_CACHE_VOL):/go/pkg/mod \
	-v $(GO_BUILD_CACHE_VOL):/go/cache

BUILDER_RUN = $(CONTAINER_RUNTIME) run $(BUILDER_RUN_FLAGS) $(BUILDER_IMAGE) sh -c

BUILDER_STAMP = build/.builder.stamp

.PHONY: help
help: ## Print help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: builder-shell
builder-shell: image-build-builder ## Open a shell in the builder container
	$(CONTAINER_RUNTIME) run -it $(BUILDER_RUN_FLAGS) $(BUILDER_IMAGE) bash


.PHONY: install-hooks
install-hooks: ## Install git hooks
	git config core.hooksPath hooks

.PHONY: presubmit
presubmit: LINT_NEW_ONLY=true
presubmit: git-branch-check signed-commits-check go-mod-check format lint ## Run all pre-submit checks

.PHONY: git-branch-check
git-branch-check:
	@branch=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$branch" = "main" ]; then \
		echo "ERROR: Direct push to 'main' is not allowed."; \
		echo "Create a branch and open a PR instead."; \
		exit 1; \
	fi

.PHONY: signed-commits-check
signed-commits-check:
	@./scripts/check-commits.sh origin/main

.PHONY: go-mod-check
go-mod-check: image-build-builder
	@echo "Checking go.mod/go.sum are clean..."
	$(BUILDER_RUN) 'go mod tidy'
	@git diff --exit-code go.mod go.sum || \
	( echo "ERROR: go.mod/go.sum are not tidy. Run 'go mod tidy' and commit."; exit 1 )

.PHONY: tidy
tidy: ## Tidy go modules
	go mod tidy

.PHONY: clean
clean: ## Clean build artifacts, tools and caches
	rm -rf bin build $(BUILDER_STAMP)
	-$(BUILDER_RUN) 'go clean -testcache -cache'

.PHONY: format
format: image-build-builder ## Format Go source files
	@printf "\033[33;1m==== Running go fmt ====\033[0m\n"
	$(BUILDER_RUN) 'gofmt -l -w . && golangci-lint fmt --config=./.golangci.yml'

.PHONY: lint
lint: image-build-builder ## Run lint (use LINT_NEW_ONLY=true to only check new code)
	$(eval LINT_ARGS := --config=./.golangci.yml$(if $(filter true,$(LINT_NEW_ONLY)), --new))
	@printf "\033[33;1m==== Running linting ====\033[0m\n"
	$(BUILDER_RUN) 'GOFLAGS=-buildvcs=false golangci-lint run $(LINT_ARGS) && typos'

.PHONY: test
test: test-unit ## Run all tests

.PHONY: test-unit
test-unit: image-build-builder
	@printf "\033[33;1m==== Running Unit Tests ====\033[0m\n"
	$(BUILDER_RUN) "go test -v -race $(TEST_PACKAGES)"

.PHONY: test-e2e-coordinator
test-e2e-coordinator: image-build-coordinator image-build-epp image-build-builder image-pull ## Run coordinator e2e tests against a new kind cluster
	test/scripts/run_e2e_coordinator.sh

.PHONY: build
build: image-build-builder ## Build the coordinator binary
	@printf "\033[33;1m==== Building coordinator ====\033[0m\n"
	$(BUILDER_RUN) 'go build -ldflags "$(LDFLAGS)" -o bin/coordinator ./cmd/coordinator/...'

COORDINATOR_CONFIG ?= config/coordinator/coordinator.yaml
.PHONY: run
run: build ## Build and run the coordinator with $(COORDINATOR_CONFIG)
	./bin/coordinator --config $(COORDINATOR_CONFIG)

##@ Container image Build

.PHONY: check-container-tool
check-container-tool:
	@if [ -z "$(CONTAINER_RUNTIME)" ]; then \
		echo "ERROR: No container tool detected. Please install docker or podman."; \
		exit 1; \
	else \
		echo "Container tool '$(CONTAINER_RUNTIME)' found."; \
	fi

.PHONY: image-build
image-build: image-build-epp image-build-coordinator ## Build all container images (epp + coordinator)

.PHONY: image-build-%
image-build-%: check-container-tool ## Build container image using $(CONTAINER_RUNTIME) (e.g. image-build-coordinator, image-build-epp)
	@printf "\033[33;1m==== Building Docker image $($*_IMAGE) ====\033[0m\n"
	$(CONTAINER_RUNTIME) build \
		--platform linux/$(TARGETARCH) \
		--build-arg TARGETOS=linux \
		--build-arg TARGETARCH=$(TARGETARCH) \
		--build-arg COMMIT_SHA=$(GIT_COMMIT_SHA) \
		--build-arg BUILD_REF=$(BUILD_REF) \
		--build-arg LDFLAGS="$(LDFLAGS)" \
		$(if $(BASE_IMAGE),--build-arg BASE_IMAGE="$(BASE_IMAGE)") \
		-t $($*_IMAGE) -f Dockerfile.$* .

.PHONY: image-build-builder
image-build-builder: check-container-tool ## Build builder image if missing locally, stamp missing, or Dockerfile.builder newer than stamp
	@if ! $(CONTAINER_RUNTIME) image inspect $(BUILDER_IMAGE) >/dev/null 2>&1 || \
	    [ ! -f $(BUILDER_STAMP) ] || \
	    [ Dockerfile.builder -nt $(BUILDER_STAMP) ]; then \
		printf "\033[33;1m==== Building image $(BUILDER_IMAGE) ====\033[0m\n"; \
		$(CONTAINER_RUNTIME) build -f Dockerfile.builder -t $(BUILDER_IMAGE) .; \
		mkdir -p $(dir $(BUILDER_STAMP)); \
		touch $(BUILDER_STAMP); \
	fi

.PHONY: image-build-epp
image-build-epp: ## Clone llm-d-inference-scheduler at pinned commit and build EPP image
	scripts/build-epp-image.sh

.PHONY: image-pull
image-pull: check-container-tool ## Pull all related images using $(CONTAINER_RUNTIME)
	@printf "\033[33;1m==== Pulling Container images ====\033[0m\n"
	./scripts/pull_images.sh

