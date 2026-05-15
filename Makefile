SHELL := /usr/bin/env bash

LOCALBIN ?= $(shell pwd)/bin
HELM ?= $(LOCALBIN)/helm
KUBECTL_VALIDATE ?= $(LOCALBIN)/kubectl-validate
YQ ?= $(LOCALBIN)/yq

# Tool checks (container runtime, kubectl, etc.) are defined in Makefile.tools.mk.
include Makefile.tools.mk
# Cluster (Kubernetes/OpenShift) specific targets are defined in Makefile.cluster.mk.
include Makefile.cluster.mk
# Kind specific targets are defined in Makefile.kind.mk.
include Makefile.kind.mk
# Code generation targets are defined in Makefile.gen.mk
include Makefile.gen.mk

# Defaults
TARGETOS ?= $(shell command -v go >/dev/null 2>&1 && go env GOOS || uname -s | tr '[:upper:]' '[:lower:]')
TARGETARCH ?= $(shell command -v go >/dev/null 2>&1 && go env GOARCH || uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/; s/armv7l/arm/')
PROJECT_NAME ?= llm-d-router
# TODO: rename EPP_IMAGE_NAME to router-endpoint-picker
EPP_IMAGE_NAME ?= llm-d-inference-scheduler
SIDECAR_IMAGE_NAME ?= llm-d-routing-sidecar
VLLM_SIMULATOR_IMAGE_NAME ?= llm-d-inference-sim
UDS_TOKENIZER_IMAGE_NAME ?= llm-d-uds-tokenizer
SIDECAR_NAME ?= pd-sidecar
BUILDER_IMAGE_NAME ?= llm-d-builder
IMAGE_REGISTRY ?= ghcr.io/llm-d

EPP_IMAGE_TAG_BASE ?= $(IMAGE_REGISTRY)/$(EPP_IMAGE_NAME)
EPP_TAG ?= dev
export EPP_IMAGE ?= $(EPP_IMAGE_TAG_BASE):$(EPP_TAG)

SIDECAR_IMAGE_TAG_BASE ?= $(IMAGE_REGISTRY)/$(SIDECAR_IMAGE_NAME)
SIDECAR_TAG ?= dev
export SIDECAR_IMAGE ?= $(SIDECAR_IMAGE_TAG_BASE):$(SIDECAR_TAG)

VLLM_SIMULATOR_TAG ?= v0.8.2
VLLM_SIMULATOR_TAG_BASE ?= $(IMAGE_REGISTRY)/$(VLLM_SIMULATOR_IMAGE_NAME)
export VLLM_IMAGE ?= $(VLLM_SIMULATOR_TAG_BASE):$(VLLM_SIMULATOR_TAG)

UDS_TOKENIZER_TAG ?= dev
UDS_TOKENIZER_TAG_BASE ?= $(IMAGE_REGISTRY)/$(UDS_TOKENIZER_IMAGE_NAME)
export UDS_TOKENIZER_IMAGE ?= $(UDS_TOKENIZER_TAG_BASE):$(UDS_TOKENIZER_TAG)

# CPU-only vLLM image that exposes `vllm launch render` for the token-producer
# plugin's HTTP backend.
export VLLM_RENDER_IMAGE ?= vllm/vllm-openai-cpu:v0.19.1

BUILDER_TAG ?= dev
BUILDER_TAG_BASE ?= $(IMAGE_REGISTRY)/$(BUILDER_IMAGE_NAME)
export BUILDER_IMAGE ?= $(BUILDER_TAG_BASE):$(BUILDER_TAG)

NAMESPACE ?= hc4ai-operator
LINT_NEW_ONLY ?= false # Set to true to only lint new code, false to lint all code (default matches CI behavior)

CONTAINER_RUNTIME ?= $(shell { command -v docker >/dev/null 2>&1 && echo docker; } || { command -v podman >/dev/null 2>&1 && echo podman; } || echo "")
export CONTAINER_RUNTIME

GIT_COMMIT_SHA ?= $(shell git rev-parse HEAD 2>/dev/null)
# Match only root-level release tags (v[0-9]*) so submodule tags don't leak into image versions.
ROOT_RELEASE_TAG_MATCH ?= v[0-9]*
BUILD_REF ?= $(shell git describe --tags --match '$(ROOT_RELEASE_TAG_MATCH)' --abbrev=0 2>/dev/null)

# Named volumes for Go module and build caches, persisted across container runs and image rebuilds.
GO_MOD_CACHE_VOL ?= llm-d-gomodcache
GO_BUILD_CACHE_VOL ?= llm-d-gobuildcache

# Common flags for running the builder container: mounts source, Go caches, and runs as current user.
# Podman rootless requires --userns=keep-id to correctly map host UID; docker uses -u directly.
# Rootful Podman (e.g. Podman machine on macOS) does not support --userns=keep-id with --network=host.
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

# Respect host KUBECONFIG if set; fall back to ~/.kube/config.
# Note: if KUBECONFIG is a colon-separated list, only the first file is mounted.
HOST_KUBECONFIG ?= $(or $(KUBECONFIG),$(HOME)/.kube/config)

# Flags for targets that need host network and kubeconfig (integration tests, benchmarks).
BUILDER_CLUSTER_FLAGS = --network=host \
	-v $(HOST_KUBECONFIG):/.kube/config:ro \
	-e KUBECONFIG=/.kube/config

# Mount the container runtime socket and set CONTAINER_HOST so podman --remote
# inside the builder can talk to the host's container runtime.
ifeq ($(CONTAINER_RUNTIME),podman)
CONTAINER_SOCK ?= $(or $(shell podman info --format '{{.Host.RemoteSocket.Path}}' 2>/dev/null | sed 's|^unix://||'),/run/podman/podman.sock)
BUILDER_SOCK_FLAGS = --security-opt label=disable \
	-v $(CONTAINER_SOCK):$(CONTAINER_SOCK) \
	-e CONTAINER_HOST=unix://$(CONTAINER_SOCK) \
	-e DOCKER_HOST=unix://$(CONTAINER_SOCK) \
	-e CONTAINER_RUNTIME=podman \
	-e KIND_EXPERIMENTAL_PROVIDER=podman
else
CONTAINER_SOCK ?= /var/run/docker.sock
ifeq ($(TARGETOS),darwin)
DOCKER_SOCK_GID := $(shell stat -f '%g' $(CONTAINER_SOCK) 2>/dev/null)
else
DOCKER_SOCK_GID := $(shell stat -c '%g' $(CONTAINER_SOCK) 2>/dev/null)
endif
BUILDER_SOCK_FLAGS = --group-add $(DOCKER_SOCK_GID) \
	-v $(CONTAINER_SOCK):$(CONTAINER_SOCK) \
	-e DOCKER_HOST=unix://$(CONTAINER_SOCK) \
	-e CONTAINER_RUNTIME=docker
endif

# Env vars forwarded into the e2e test container.
# Add new image vars here so they are automatically passed through.
# Should we pass ALL env vars here?
E2E_ENV_VARS = EPP_IMAGE VLLM_IMAGE SIDECAR_IMAGE UDS_TOKENIZER_IMAGE VLLM_RENDER_IMAGE \
               E2E_KEEP_CLUSTER_ON_FAILURE E2E_PORT E2E_METRICS_PORT K8S_CONTEXT READY_TIMEOUT
BUILDER_E2E_ENV_FLAGS = $(foreach v,$(E2E_ENV_VARS),$(if $($(v)),-e $(v)=$($(v))))
ifneq ($(filter command line environment,$(origin NAMESPACE)),)
BUILDER_E2E_ENV_FLAGS += -e NAMESPACE=$(NAMESPACE)
endif

# GAIE e2e test variables (for test-e2e-gaie target).
# GAIE_E2E_MANIFEST_PATH: path to the model server manifest inside the builder container
# (/app/... prefix). Defaults to the sim-deployment in testdata. Override to use a GPU manifest.
GAIE_E2E_MANIFEST_PATH ?=
GAIE_E2E_IMAGE         ?= $(EPP_IMAGE)

# When K8S_CONTEXT is set, mount the host kubeconfig so the e2e suite can call
# config.GetConfigWithContext(K8S_CONTEXT) against an existing cluster instead of
# creating a new kind cluster.
ifdef K8S_CONTEXT
BUILDER_E2E_KUBECONFIG_FLAGS = -v $(HOST_KUBECONFIG):/.kube/config:ro -e KUBECONFIG=/.kube/config
else
BUILDER_E2E_KUBECONFIG_FLAGS =
endif

# E2e tests create their own kind cluster, need host network (for NodePort access)
# and the container socket (for kind), but not the host kubeconfig.
BUILDER_E2E_FLAGS = --network=host $(BUILDER_SOCK_FLAGS) $(BUILDER_E2E_ENV_FLAGS) $(BUILDER_E2E_KUBECONFIG_FLAGS)

# Builder container invocations. Always use sh -c so commands with shell expansions
# (pipes, $(), etc.) run inside the container, not on the host.
BUILDER_RUN = $(CONTAINER_RUNTIME) run $(BUILDER_RUN_FLAGS) $(BUILDER_IMAGE) sh -c
BUILDER_RUN_CLUSTER = $(CONTAINER_RUNTIME) run $(BUILDER_RUN_FLAGS) $(BUILDER_CLUSTER_FLAGS) $(BUILDER_IMAGE) sh -c

# Linker flags for go build inside Docker images.
# Default strips debug symbols for smaller production images.
# Override locally to build a debuggable image: LDFLAGS="" make image-build-epp
LDFLAGS ?= -s -w

# Optional: override the runtime base image used in container builds.
# When set, passed as --build-arg BASE_IMAGE=<value> to the container build.
# Example: BASE_IMAGE=registry.access.redhat.com/ubi9/ubi-micro:9.7 make image-build-epp
BASE_IMAGE ?=

# test packages
epp_TEST_PACKAGES = $$(go list ./... | grep -v /test/ | grep -v ./pkg/sidecar/ | tr '\n' ' ')
sidecar_TEST_PACKAGES = ./pkg/sidecar/...

# Internal variables for generic targets
epp_IMAGE = $(EPP_IMAGE)
sidecar_IMAGE = $(SIDECAR_IMAGE)
epp_NAME = epp
sidecar_NAME = $(SIDECAR_NAME)


.PHONY: help
help: ## Print help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: builder-shell
builder-shell: image-build-builder ## Open a shell in the builder container
	$(CONTAINER_RUNTIME) run -it $(BUILDER_RUN_FLAGS) $(BUILDER_IMAGE) bash

.PHONY: builder-cluster-shell
builder-cluster-shell: image-build-builder ## Open a shell with cluster access
	$(CONTAINER_RUNTIME) run -it $(BUILDER_RUN_FLAGS) $(BUILDER_CLUSTER_FLAGS) $(BUILDER_IMAGE) bash

.PHONY: builder-e2e-shell
builder-e2e-shell: image-build-builder ## Open a shell with e2e test access
	$(CONTAINER_RUNTIME) run -it $(BUILDER_RUN_FLAGS) $(BUILDER_E2E_FLAGS) $(BUILDER_IMAGE) bash

.PHONY: install-hooks
install-hooks: ## Install git hooks
	git config core.hooksPath hooks

.PHONY: upgrade-deps
upgrade-deps: ## Upgrade all Go dependencies to latest minor/patch versions and tidy; review diff before committing
	go get -u ./...
	go mod tidy

.PHONY: vulncheck
vulncheck: image-build-builder ## Run govulncheck for known vulnerabilities
	@printf "\033[33;1m==== Running govulncheck ====\033[0m\n"
	$(BUILDER_RUN) 'govulncheck ./...'

.PHONY: presubmit
presubmit: LINT_NEW_ONLY=true
presubmit: git-branch-check signed-commits-check go-mod-check format lint vulncheck

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
	@./scripts/check-commits.sh upstream/main

.PHONY: go-mod-check
go-mod-check: image-build-builder
	@echo "Checking go.mod/go.sum are clean..."
	$(BUILDER_RUN) 'go mod tidy'
	@git diff --exit-code go.mod go.sum || \
	( echo "ERROR: go.mod/go.sum are not tidy. Run 'go mod tidy' and commit."; exit 1 )

.PHONY: tidy
tidy:
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
test: test-unit test-e2e ## Run all tests (unit and e2e)

.PHONY: test-unit
test-unit: test-unit-epp test-unit-sidecar ## Run unit tests

.PHONY: test-unit-%
test-unit-%: image-build-builder
	@mkdir -p $(COVERAGE_DIR)
	@printf "\033[33;1m==== Running $* Unit Tests ====\033[0m\n"
	$(BUILDER_RUN) "go test -v -race -coverprofile=$(COVERAGE_DIR)/$*.out -covermode=atomic $($*_TEST_PACKAGES)"
	$(BUILDER_RUN) 'go tool cover -func=$(COVERAGE_DIR)/$*.out | tail -1'

# Smoke test against llm-d-kv-cache main (or any ref via KVCACHE_UPSTREAM_REF).
# Pins the dep in a temp worktree so the developer's go.mod is not mutated.
# Auto-discovers every package that imports llm-d-kv-cache so new consumers
# are covered without touching this target. See issue #1056.
.PHONY: test-kvcache-upstream-smoke
test-kvcache-upstream-smoke: ## Run all kv-cache-consuming tests against llm-d-kv-cache main
	@set -e; \
	ref="$${KVCACHE_UPSTREAM_REF:-main}"; \
	wt=$$(mktemp -d -t kvcache-smoke.XXXXXX); trap "rm -rf $$wt" EXIT; \
	tar --exclude=.git -C . -cf - . | tar -C $$wt -xf -; \
	cd $$wt; \
	go mod edit -replace github.com/llm-d/llm-d-kv-cache=github.com/llm-d/llm-d-kv-cache@$$ref; \
	go mod tidy; \
	go list -m github.com/llm-d/llm-d-kv-cache; \
	pkgs=$$(go list -f '{{.ImportPath}} {{join .Imports " "}} {{join .TestImports " "}} {{join .XTestImports " "}}' ./... \
	  | awk '/github.com\/llm-d\/llm-d-kv-cache/ {print $$1}'); \
	echo "Testing packages:"; printf '  %s\n' $$pkgs; \
	go test -count=1 -timeout 10m $$pkgs

.PHONY: test-filter
test-filter: image-build-builder ## Run filtered unit tests (usage: make test-filter PATTERN=TestName TYPE=epp)
	@if [ -z "$(PATTERN)" ]; then \
		echo "ERROR: PATTERN is required. Usage: make test-filter PATTERN=TestName [TYPE=epp|sidecar]"; \
		exit 1; \
	fi
	@TEST_TYPE="$(if $(TYPE),$(TYPE),epp)"; \
	printf "\033[33;1m==== Running Filtered Tests (pattern: $(PATTERN), type: $$TEST_TYPE) ====\033[0m\n"; \
	if [ "$$TEST_TYPE" = "epp" ]; then \
		$(BUILDER_RUN) "go test -v -run \"$(PATTERN)\" $(epp_TEST_PACKAGES)"; \
	else \
		$(BUILDER_RUN) "go test -v -run \"$(PATTERN)\" $(sidecar_TEST_PACKAGES)"; \
	fi

.PHONY: test-integration
test-integration: image-build-builder ## Run integration tests (requires KUBECONFIG and running cluster)
	@mkdir -p $(COVERAGE_DIR)
	@printf "\033[33;1m==== Running Integration Tests ====\033[0m\n"
	$(BUILDER_RUN_CLUSTER) 'go test -v -race -tags=integration_tests -coverprofile=$(COVERAGE_DIR)/integration.out -covermode=atomic ./test/integration/'
	$(BUILDER_RUN) 'go tool cover -func=$(COVERAGE_DIR)/integration.out | tail -1'

.PHONY: test-integration-hermetic
test-integration-hermetic: image-build-builder ## Run hermetic integration tests (envtest, no cluster required)
	@mkdir -p $(COVERAGE_DIR)
	@printf "\033[33;1m==== Running Hermetic Integration Tests ====\033[0m\n"
	$(BUILDER_RUN) 'CGO_ENABLED=1 KUBEBUILDER_ASSETS="$$(setup-envtest use $$ENVTEST_K8S_VERSION --bin-dir $$ENVTEST_ASSETS_DIR -p path)" go test -v -race -coverprofile=$(COVERAGE_DIR)/integration-hermetic.out -covermode=atomic ./test/integration/...'
	$(BUILDER_RUN) 'go tool cover -func=$(COVERAGE_DIR)/integration-hermetic.out | tail -1'

.PHONY: test-e2e
test-e2e: image-build-builder image-build image-pull ## Run end-to-end tests against a new kind cluster
	@printf "\033[33;1m==== Running GAIE End to End Tests ====\033[0m\n"
	$(CONTAINER_RUNTIME) run $(BUILDER_RUN_FLAGS) $(BUILDER_E2E_FLAGS) \
		-e EPP_IMAGE=$(GAIE_E2E_IMAGE) \
		-e USE_KIND=true \
		$(BUILDER_IMAGE) ./hack/test-e2e.sh
	@printf "\033[33;1m==== Running End to End Tests ====\033[0m\n"
	$(CONTAINER_RUNTIME) run $(BUILDER_RUN_FLAGS) $(BUILDER_E2E_FLAGS) \
		$(BUILDER_IMAGE) ./test/scripts/run_e2e.sh


.PHONY: bench-tokenizer
bench-tokenizer: image-build-builder ## Run external tokenizer + scorer benchmark (requires kind cluster with EPP deployed)
	@printf "\033[33;1m==== Running External Tokenizer Benchmark ====\033[0m\n"
	@printf "Ensure the kind cluster is running with the external tokenizer config.\n"
	@printf "Run 'EXTERNAL_TOKENIZER_ENABLED=true KV_CACHE_ENABLED=true make env-dev-kind' first.\n\n"
	$(BUILDER_RUN_CLUSTER) 'go test -bench=. -benchmem -count=5 -timeout=5m ./test/profiling/tokenizerbench/'

.PHONY: post-deploy-test
post-deploy-test: ## Run post deployment tests
	@echo "Success!"
	@echo "Post-deployment tests passed."

##@ Helm

.PHONY: verify-helm-charts
verify-helm-charts: helm-install kubectl-validate ## Render and validate Helm charts.
	HELM="$(HELM)" KUBECTL_VALIDATE="$(KUBECTL_VALIDATE)" hack/verify-helm.sh $(MODE)

.PHONY: verify-manifests
verify-manifests: kubectl-validate ## Validate deployment manifests.
	KUBECTL_VALIDATE="$(KUBECTL_VALIDATE)" hack/verify-manifests.sh

.PHONY: inferencepool-helm-chart-push
inferencepool-helm-chart-push: yq helm-install ## Package and push the InferencePool Helm chart.
	CHART=inferencepool EXTRA_TAG="$(EXTRA_TAG)" YQ="$(YQ)" HELM="$(HELM)" ./hack/push-chart.sh

.PHONY: standalone-helm-chart-push
standalone-helm-chart-push: yq helm-install ## Package and push the standalone EPP Helm chart.
	CHART=standalone EXTRA_TAG="$(EXTRA_TAG)" YQ="$(YQ)" HELM="$(HELM)" ./hack/push-chart.sh


##@ Coverage

COVERAGE_DIR       ?= coverage
COVERAGE_THRESHOLD ?= 0
COVERAGE_LABEL     ?= main
BASE_REF           ?= main

.PHONY: test-coverage
test-coverage: test-unit-epp test-unit-sidecar ## Run all unit tests with coverage (alias for test-unit)

.PHONY: test-coverage-integration
test-coverage-integration: test-integration ## Run integration tests with coverage (alias for test-integration)

.PHONY: coverage-report
coverage-report: image-build-builder ## Generate HTML coverage reports (open coverage/*.html in browser)
	$(BUILDER_RUN) 'for f in $(COVERAGE_DIR)/*.out; do \
	    name=$$(basename "$$f" .out); \
	    go tool cover -html="$$f" -o "$(COVERAGE_DIR)/$$name.html"; \
	    printf "  $$name → $(COVERAGE_DIR)/$$name.html\n"; \
	done'

.PHONY: coverage-compare
coverage-compare: image-build-builder ## Compare coverage vs baseline (BASELINE_DIR=path or BASE_REF=git-ref, default main; COVERAGE_LABEL=label)
	@if [ -n "$(BASELINE_DIR)" ]; then \
	    ./scripts/compare-coverage.sh "$(BASELINE_DIR)" "$(COVERAGE_DIR)" "$(COVERAGE_THRESHOLD)" "$(COVERAGE_LABEL)"; \
	else \
	    printf "\033[33;1m==== Building Baseline Coverage from $(BASE_REF) ====\033[0m\n"; \
	    EXISTING=$$(git worktree list --porcelain \
	        | awk '/^worktree /{wt=$$2} /^branch refs\/heads\/$(BASE_REF)$$/{print wt}'); \
	    if [ -n "$$EXISTING" ]; then \
	        WORKTREE="$$EXISTING"; CLEANUP=0; \
	    else \
	        WORKTREE=$$(mktemp -u /tmp/cov-baseline-XXXXXX); \
	        git worktree add --quiet "$$WORKTREE" "$(BASE_REF)"; \
	        CLEANUP=1; \
	    fi; \
	    mkdir -p "$(COVERAGE_DIR)/baseline"; \
	    $(CONTAINER_RUNTIME) run $(BUILDER_RUN_FLAGS) \
	        -v "$$WORKTREE":/baseline:Z \
	        $(BUILDER_IMAGE) sh -c " \
	            cd /baseline && \
	            go test -race -coverprofile=/app/$(COVERAGE_DIR)/baseline/epp.out -covermode=atomic \
	                $$(go list ./... | grep -v /test/ | grep -v ./pkg/sidecar/ | tr '\n' ' ') && \
	            go test -race -coverprofile=/app/$(COVERAGE_DIR)/baseline/sidecar.out -covermode=atomic \
	                ./pkg/sidecar/..."; \
	    [ "$$CLEANUP" -eq 1 ] && git worktree remove --force "$$WORKTREE"; \
	    ./scripts/compare-coverage.sh "$(COVERAGE_DIR)/baseline" "$(COVERAGE_DIR)" "$(COVERAGE_THRESHOLD)" "$(COVERAGE_LABEL)"; \
	fi


##@ Build

.PHONY: build
build: build-epp build-sidecar ## Build the project for both epp and sidecar

.PHONY: build-%
build-%: image-build-builder ## Build the project
	@printf "\033[33;1m==== Building $* ====\033[0m\n"
	$(BUILDER_RUN) 'go build -o bin/$($*_NAME) cmd/$($*_NAME)/main.go'

##@ Container image Build/Push/Pull

.PHONY:	image-build
image-build: image-build-epp image-build-sidecar ## Build Container image using $(CONTAINER_RUNTIME)

.PHONY: image-build-%
image-build-%: check-container-tool ## Build Container image using $(CONTAINER_RUNTIME)
	@printf "\033[33;1m==== Building Docker image $($*_IMAGE) ====\033[0m\n"
	$(CONTAINER_RUNTIME) build \
		--platform linux/$(TARGETARCH) \
		--build-arg TARGETOS=linux \
		--build-arg TARGETARCH=$(TARGETARCH) \
		--build-arg COMMIT_SHA=${GIT_COMMIT_SHA} \
		--build-arg BUILD_REF=${BUILD_REF} \
		--build-arg LDFLAGS="$(LDFLAGS)" \
		$(if $(BASE_IMAGE),--build-arg BASE_IMAGE="$(BASE_IMAGE)") \
		-t $($*_IMAGE) -f Dockerfile.$* .

BUILDER_STAMP = build/.builder.stamp

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

.PHONY: image-push
image-push: image-push-epp image-push-sidecar ## Push container images to registry using $(CONTAINER_RUNTIME)

.PHONY: image-push-%
image-push-%: check-container-tool ## Push container image to registry using $(CONTAINER_RUNTIME)
	@printf "\033[33;1m==== Pushing Container image $($*_IMAGE) ====\033[0m\n"
	$(CONTAINER_RUNTIME) push $($*_IMAGE)

.PHONY: image-pull
image-pull: check-container-tool ## Pull all related images using $(CONTAINER_RUNTIME)
	@printf "\033[33;1m==== Pulling Container images ====\033[0m\n"
	./scripts/pull_images.sh

##@ Container Run

.PHONY: run-container
run-container: check-container-tool ## Run app in container using $(CONTAINER_RUNTIME)
	@echo "Starting container with $(CONTAINER_RUNTIME)..."
	$(CONTAINER_RUNTIME) run -d --name $(PROJECT_NAME)-container $(EPP_IMAGE)
	@echo "$(CONTAINER_RUNTIME) started successfully."
	@echo "To use $(PROJECT_NAME), run:"
	@echo "alias $(PROJECT_NAME)='$(CONTAINER_RUNTIME) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)'"

.PHONY: stop-container
stop-container: check-container-tool ## Stop and remove container
	@echo "Stopping and removing container..."
	$(CONTAINER_RUNTIME) stop $(PROJECT_NAME)-container && $(CONTAINER_RUNTIME) rm $(PROJECT_NAME)-container
	@echo "$(CONTAINER_RUNTIME) stopped and removed. Remove alias if set: unalias $(PROJECT_NAME)"

##@ Environment
.PHONY: env
env: ## Print environment variables
	@echo "TARGETOS=$(TARGETOS)"
	@echo "TARGETARCH=$(TARGETARCH)"
	@echo "CONTAINER_RUNTIME=$(CONTAINER_RUNTIME)"
	@echo "EPP_TAG=$(EPP_TAG)"
	@echo "EPP_IMAGE=$(EPP_IMAGE)"
	@echo "SIDECAR_TAG=$(SIDECAR_TAG)"
	@echo "SIDECAR_IMAGE=$(SIDECAR_IMAGE)"
	@echo "VLLM_SIMULATOR_TAG=$(VLLM_SIMULATOR_TAG)"
	@echo "VLLM_IMAGE=$(VLLM_IMAGE)"
	@echo "UDS_TOKENIZER_TAG=$(UDS_TOKENIZER_TAG)"
	@echo "UDS_TOKENIZER_IMAGE=$(UDS_TOKENIZER_IMAGE)"
	@echo "VLLM_RENDER_IMAGE=$(VLLM_RENDER_IMAGE)"
	@echo "BUILDER_IMAGE=$(BUILDER_IMAGE)"

.PHONY: print-namespace
print-namespace: ## Print the current namespace
	@echo "$(NAMESPACE)"

.PHONY: print-project-name
print-project-name: ## Print the current project name
	@echo "$(PROJECT_NAME)"

##@ Deprecated aliases for backwards compatibility
.PHONY: install-docker
install-docker: ## DEPRECATED: Use 'make run-container' instead
	@echo "WARNING: 'make install-docker' is deprecated. Use 'make run-container' instead."
	@$(MAKE) run-container

.PHONY: uninstall-docker
uninstall-docker: ## DEPRECATED: Use 'make stop-container' instead
	@echo "WARNING: 'make uninstall-docker' is deprecated. Use 'make stop-container' instead."
	@$(MAKE) stop-container

.PHONY: install
install: ## DEPRECATED: Use 'make run-container' instead
	@echo "WARNING: 'make install' is deprecated. Use 'make run-container' instead."
	@$(MAKE) run-container

.PHONY: uninstall
uninstall: ## DEPRECATED: Use 'make stop-container' instead
	@echo "WARNING: 'make uninstall' is deprecated. Use 'make stop-container' instead."
	@$(MAKE) stop-container
