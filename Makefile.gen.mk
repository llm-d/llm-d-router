##@ Code generation targets

# Extract upstream CRD versions from go.mod so references stay in sync with Go dependencies.
GIE_VERSION ?= $(shell go list -m -f '{{.Version}}' sigs.k8s.io/gateway-api-inference-extension)
GATEWAY_API_VERSION ?= $(shell go list -m -f '{{.Version}}' sigs.k8s.io/gateway-api)
# Pinned vllm-project/vllm ref (tag or commit SHA) for upstream vLLM .proto schemas (rust/proto).
VLLM_REF ?= 2fb5111bbac7a2e31c0c19c1eb0aa01df5ecb664
# Pinned sgl-project/sglang ref (tag or commit SHA) for SGLang .proto schemas.
SGLANG_REF ?= 9877a3076f68e119eff4da664630efbe71d74ffa

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	[ -d $@ ] || mkdir -p $@

CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
BUF ?= $(LOCALBIN)/buf
PROTOC ?= protoc
PROTOC_GEN_GO = $(LOCALBIN)/protoc-gen-go
PROTOC_GEN_GO_GRPC = $(LOCALBIN)/protoc-gen-go-grpc

## Tool Versions
CONTROLLER_TOOLS_VERSION ?= v0.19.0
BUF_VERSION ?= v1.75.0
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

VLLM_GRPC_DIR := pkg/common/grpc/vllm
SGLANG_GRPC_DIR := pkg/common/grpc/sglang

.PHONY: generate-proto generate-proto-vllm generate-proto-vllm-engine generate-proto-vllm-upstream generate-proto-sglang
generate-proto: generate-proto-vllm generate-proto-sglang ## Generate Golang code from protobuf files.

generate-proto-vllm: generate-proto-vllm-engine generate-proto-vllm-upstream

generate-proto-vllm-engine: protoc-gen-go protoc-gen-go-grpc ## Generate Go bindings for the legacy Python VllmEngine .proto.
	PATH="$(LOCALBIN):$$PATH" $(PROTOC) \
		-I pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/proto \
		-I . \
		--go_out=module=github.com/llm-d/llm-d-router:. \
		--go-grpc_out=module=github.com/llm-d/llm-d-router:. \
		pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/proto/*.proto

generate-proto-vllm-upstream: buf protoc-gen-go protoc-gen-go-grpc ## Generate upstream vLLM Go bindings from vllm-project/vllm@$(VLLM_REF).
	PATH="$(LOCALBIN):$$PATH" $(BUF) generate \
		--template $(VLLM_GRPC_DIR)/buf.gen.yaml \
		--path inference.proto \
		--path control.proto \
		"https://github.com/vllm-project/vllm.git#ref=$(VLLM_REF),subdir=rust/proto,depth=1"

.PHONY: verify-vllm-proto
verify-vllm-proto: generate-proto-vllm-upstream ## Check that upstream vLLM Go bindings match vllm-project/vllm@$(VLLM_REF).
	@status="$$(git status --porcelain -- $(VLLM_GRPC_DIR))"; \
	if [ -n "$$status" ]; then \
		echo "ERROR: $(VLLM_GRPC_DIR) differs from vllm-project/vllm@$(VLLM_REF). Run 'make generate-proto-vllm-upstream' and commit the result:"; \
		echo "$$status"; \
		git diff -- $(VLLM_GRPC_DIR); \
		exit 1; \
	fi

generate-proto-sglang: buf protoc-gen-go protoc-gen-go-grpc ## Generate SGLang Go bindings from sgl-project/sglang@$(SGLANG_REF).
	PATH="$(LOCALBIN):$$PATH" $(BUF) generate \
		--template $(SGLANG_GRPC_DIR)/buf.gen.yaml \
		--path sglang/api/v1 \
		--path sglang/json/v1 \
		"https://github.com/sgl-project/sglang.git#ref=$(SGLANG_REF),subdir=proto,depth=1"

.PHONY: verify-sglang-proto
verify-sglang-proto: generate-proto-sglang ## Check that SGLang Go bindings match sgl-project/sglang@$(SGLANG_REF).
	@status="$$(git status --porcelain -- $(SGLANG_GRPC_DIR))"; \
	if [ -n "$$status" ]; then \
		echo "ERROR: $(SGLANG_GRPC_DIR) differs from sgl-project/sglang@$(SGLANG_REF). Run 'make generate-proto-sglang' and commit the result:"; \
		echo "$$status"; \
		git diff -- $(SGLANG_GRPC_DIR); \
		exit 1; \
	fi

.PHONY: buf
buf: $(BUF) ## Download buf locally if necessary.
$(BUF): $(LOCALBIN)
	$(call go-install-tool,$(BUF),github.com/bufbuild/buf/cmd/buf,$(BUF_VERSION))

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
