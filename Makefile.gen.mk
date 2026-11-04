##@ Code generation targets

# Extract upstream CRD versions from go.mod so references stay in sync with Go dependencies.
GIE_VERSION ?= $(shell go list -m -f '{{.Version}}' sigs.k8s.io/gateway-api-inference-extension)
GATEWAY_API_VERSION ?= $(shell go list -m -f '{{.Version}}' sigs.k8s.io/gateway-api)
# Pinned sgl-project/sglang ref (tag or commit SHA) for verbatim .proto schemas.
SGLANG_REF ?= 93a936d007ada66544cb5ac855de540a596c2ca4

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	[ -d $@ ] || mkdir -p $@

CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
PROTOC ?= $(LOCALBIN)/protoc-$(PROTOC_VERSION)/bin/protoc
PROTOC_GEN_GO = $(LOCALBIN)/protoc-gen-go
PROTOC_GEN_GO_GRPC = $(LOCALBIN)/protoc-gen-go-grpc

## Tool Versions
CONTROLLER_TOOLS_VERSION ?= v0.19.0
PROTOC_VERSION ?= 3.19.6
PROTOC_GEN_GO_VERSION ?= v1.34.2
PROTOC_GEN_GO_GRPC_VERSION ?= v1.5.1

.PHONY: sync-upstream-versions
sync-upstream-versions: ## Update upstream CRD version references to match go.mod once version bumped run this target
	@echo "Syncing GIE version to $(GIE_VERSION)"
	@sed -i 's|gateway-api-inference-extension/releases/download/.*/v1-manifests.yaml|gateway-api-inference-extension/releases/download/$(GIE_VERSION)/v1-manifests.yaml|' test/e2e/artifacts/components/crds-gie/kustomization.yaml
	@sed -i 's|GIE_VERSION="$${GIE_VERSION:-.*}"|GIE_VERSION="$${GIE_VERSION:-$(GIE_VERSION)}"|' hack/verify-helm.sh hack/verify-manifests.sh
	@echo "Syncing Gateway API version to $(GATEWAY_API_VERSION)"
	@sed -i 's|gateway-api/config/crd?ref=.*|gateway-api/config/crd?ref=$(GATEWAY_API_VERSION)|' test/e2e/artifacts/components/crds-gateway-api/kustomization.yaml
	@sed -i 's|GATEWAY_API_VERSION="$${GATEWAY_API_VERSION:-.*}"|GATEWAY_API_VERSION="$${GATEWAY_API_VERSION:-$(GATEWAY_API_VERSION)}"|' hack/verify-helm.sh hack/verify-manifests.sh

.PHONY: generate
generate: controller-gen code-generator yq tidy sync-upstream-versions ## Generate WebhookConfiguration, ClusterRole, CustomResourceDefinition objects, code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	$(CONTROLLER_GEN) object:headerFile="/dev/null" paths="./..."
	$(CONTROLLER_GEN) crd output:dir="./config/crd/bases" paths="./..."
	./hack/update-codegen.sh $(LOCALBIN)
	@mkdir -p test/integration/epp/testdata/crd test/integration/epp/testdata/crd-dual test/integration/epp/testdata/crd-dual-v1storage
	@tmp=$$(mktemp -d); \
	$(CONTROLLER_GEN) crd output:dir="$$tmp" paths="./apix/..."; \
	$(YQ) eval 'del(.spec.versions[] | select(.name != "v1")) | (.spec.versions[0].served) = true | (.spec.versions[0].storage) = true' \
		"$$tmp/llm-d.ai_inferenceobjectives.yaml" > test/integration/epp/testdata/crd/llm-d.ai_inferenceobjectives.yaml; \
	$(YQ) eval '(.spec.versions[] | select(.name == "v1") | .served) = true' \
		"$$tmp/llm-d.ai_inferenceobjectives.yaml" > test/integration/epp/testdata/crd-dual/llm-d.ai_inferenceobjectives.yaml; \
	$(YQ) eval '(.spec.versions[] | select(.name == "v1") | .served) = true | (.spec.versions[] | select(.name == "v1") | .storage) = true | (.spec.versions[] | select(.name == "v1alpha2") | .storage) = false' \
		"$$tmp/llm-d.ai_inferenceobjectives.yaml" > test/integration/epp/testdata/crd-dual-v1storage/llm-d.ai_inferenceobjectives.yaml; \
	rm -rf "$$tmp"; \
	test "$$($(YQ) '.spec.versions[0].name' test/integration/epp/testdata/crd/llm-d.ai_inferenceobjectives.yaml)" = "v1" || \
		{ echo "ERROR: apix/v1 InferenceObjective missing from controller-gen output"; exit 1; }; \
	test "$$($(YQ) '[.spec.versions[] | select(.served)] | length' test/integration/epp/testdata/crd-dual/llm-d.ai_inferenceobjectives.yaml)" = "2" || \
		{ echo "ERROR: dual-serving CRD must serve both versions"; exit 1; }; \
	test "$$($(YQ) '[.spec.versions[] | select(.storage)] | length' test/integration/epp/testdata/crd-dual-v1storage/llm-d.ai_inferenceobjectives.yaml)" = "1" || \
		{ echo "ERROR: v1-storage CRD must have exactly one storage version"; exit 1; }

# Use same code-generator version as k8s.io/api
CODEGEN_VERSION := $(shell go list -m -f '{{.Version}}' k8s.io/api)
CODEGEN = $(LOCALBIN)/code-generator
CODEGEN_ROOT = $(shell go env GOMODCACHE)/k8s.io/code-generator@$(CODEGEN_VERSION)
.PHONY: code-generator
code-generator:
	@GOBIN=$(LOCALBIN) GO111MODULE=on go install k8s.io/code-generator/cmd/client-gen@$(CODEGEN_VERSION)
	cp -f $(CODEGEN_ROOT)/generate-groups.sh $(LOCALBIN)/
	cp -f $(CODEGEN_ROOT)/generate-internal-groups.sh $(LOCALBIN)/
	cp -f $(CODEGEN_ROOT)/kube_codegen.sh $(LOCALBIN)/

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen,$(CONTROLLER_TOOLS_VERSION))

SGLANG_GRPC_DIR := pkg/common/sglang/grpc
SGLANG_PROTO_DIR := $(SGLANG_GRPC_DIR)/proto
SGLANG_PROTO_FILES := \
	sglang/json/v1/options.proto \
	sglang/api/v1/common.proto \
	sglang/api/v1/finish.proto \
	sglang/api/v1/generate.proto \
	sglang/api/v1/hicache.proto \
	sglang/api/v1/sampling.proto \
	sglang/api/v1/service.proto
SGLANG_GO_MODULE := github.com/llm-d/llm-d-router
SGLANG_GEN_PKG := $(SGLANG_GO_MODULE)/$(SGLANG_GRPC_DIR)/gen
SGLANG_PROTO_GO_OPTS := \
	$(foreach p,$(filter sglang/json/v1/%,$(SGLANG_PROTO_FILES)),--go_opt=M$(p)=$(SGLANG_GEN_PKG)/jsonv1 --go-grpc_opt=M$(p)=$(SGLANG_GEN_PKG)/jsonv1) \
	$(foreach p,$(filter sglang/api/v1/%,$(SGLANG_PROTO_FILES)),--go_opt=M$(p)=$(SGLANG_GEN_PKG)/apiv1 --go-grpc_opt=M$(p)=$(SGLANG_GEN_PKG)/apiv1)

.PHONY: sync-sglang-proto
sync-sglang-proto: ## Sync verbatim SGLang .proto schemas from sgl-project/sglang@$(SGLANG_REF) and regenerate Go bindings.
	@echo "Syncing SGLang .proto files from sgl-project/sglang@$(SGLANG_REF)"
	@set -e; \
	for f in $(SGLANG_PROTO_FILES); do \
		mkdir -p "$(SGLANG_PROTO_DIR)/$$(dirname "$$f")"; \
		curl -sSfL "https://raw.githubusercontent.com/sgl-project/sglang/$(SGLANG_REF)/proto/$$f" -o "$(SGLANG_PROTO_DIR)/$$f"; \
	done
	$(MAKE) generate-proto-sglang

.PHONY: generate-proto generate-proto-vllm generate-proto-sglang
generate-proto: generate-proto-vllm generate-proto-sglang ## Generate Golang code from protobuf files.

generate-proto-vllm: protoc protoc-gen-go protoc-gen-go-grpc
	PATH="$(LOCALBIN):$$PATH" $(PROTOC) \
		-I pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/proto \
		-I . \
		--go_out=module=github.com/llm-d/llm-d-router:. \
		--go-grpc_out=module=github.com/llm-d/llm-d-router:. \
		pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/proto/*.proto

generate-proto-sglang: protoc protoc-gen-go protoc-gen-go-grpc
	PATH="$(LOCALBIN):$$PATH" $(PROTOC) \
		-I $(SGLANG_PROTO_DIR) \
		-I . \
		$(SGLANG_PROTO_GO_OPTS) \
		--go_out=module=$(SGLANG_GO_MODULE):. \
		--go-grpc_out=module=$(SGLANG_GO_MODULE):. \
		$(addprefix $(SGLANG_PROTO_DIR)/,$(SGLANG_PROTO_FILES))

.PHONY: verify-sglang-proto
verify-sglang-proto: sync-sglang-proto ## Check that vendored SGLang schemas and Go bindings match sgl-project/sglang@$(SGLANG_REF).
	@git diff --exit-code -- $(SGLANG_GRPC_DIR) || { \
		echo "ERROR: $(SGLANG_GRPC_DIR) differs from sgl-project/sglang@$(SGLANG_REF). Run 'make sync-sglang-proto' and commit the result."; \
		exit 1; \
	}
	@untracked="$$(git ls-files --others --exclude-standard -- $(SGLANG_GRPC_DIR))"; \
	if [ -n "$$untracked" ]; then \
		echo "ERROR: untracked generated files under $(SGLANG_GRPC_DIR):"; \
		echo "$$untracked"; \
		exit 1; \
	fi

PROTOC_OS := $(shell uname -s | sed 's/Linux/linux/; s/Darwin/osx/')
PROTOC_ARCH := $(shell uname -m | sed 's/amd64/x86_64/; s/arm64/aarch_64/; s/aarch64/aarch_64/')

.PHONY: protoc
protoc: $(PROTOC) ## Download protoc locally if necessary.
$(PROTOC): | $(LOCALBIN)
	@set -e; \
	dir="$(LOCALBIN)/protoc-$(PROTOC_VERSION)"; \
	zip="$$dir.zip"; \
	echo "Downloading protoc v$(PROTOC_VERSION) ($(PROTOC_OS)-$(PROTOC_ARCH))"; \
	curl -sSfL "https://github.com/protocolbuffers/protobuf/releases/download/v$(PROTOC_VERSION)/protoc-$(PROTOC_VERSION)-$(PROTOC_OS)-$(PROTOC_ARCH).zip" -o "$$zip"; \
	mkdir -p "$$dir"; \
	unzip -q -o "$$zip" -d "$$dir"; \
	rm -f "$$zip"

.PHONY: protoc-gen-go
protoc-gen-go: $(PROTOC_GEN_GO) ## Download protoc-gen-go locally if necessary.
$(PROTOC_GEN_GO): $(LOCALBIN)
	$(call go-install-tool,$(PROTOC_GEN_GO),google.golang.org/protobuf/cmd/protoc-gen-go,$(PROTOC_GEN_GO_VERSION))

.PHONY: protoc-gen-go-grpc
protoc-gen-go-grpc: $(PROTOC_GEN_GO_GRPC) ## Download protoc-gen-go-grpc locally if necessary.
$(PROTOC_GEN_GO_GRPC): $(LOCALBIN)
	$(call go-install-tool,$(PROTOC_GEN_GO_GRPC),google.golang.org/grpc/cmd/protoc-gen-go-grpc,$(PROTOC_GEN_GO_GRPC_VERSION))

# go-install-tool will 'go install' any package with custom target and name of binary, if it doesn't exist
# $1 - target path with name of binary
# $2 - package url which can be installed
# $3 - specific version of package
define go-install-tool
@[ -f "$(1)-$(3)" ] || { \
set -e; \
package=$(2)@$(3) ;\
echo "Downloading $${package}" ;\
rm -f $(1) || true ;\
GOBIN=$(LOCALBIN) go install $${package} ;\
mv $(1) $(1)-$(3) ;\
} ;\
ln -sf $(1)-$(3) $(1)
endef
