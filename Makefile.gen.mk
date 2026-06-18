##@ Code generation targets

# Extract upstream CRD versions from go.mod so references stay in sync with Go dependencies.
GOMODCACHE ?= $(shell go env GOMODCACHE)
GIE_VERSION ?= $(shell go list -m -f '{{.Version}}' sigs.k8s.io/gateway-api-inference-extension)
GIE_MOD := $(GOMODCACHE)/sigs.k8s.io/gateway-api-inference-extension@$(GIE_VERSION)
GATEWAY_API_VERSION ?= $(shell go list -m -f '{{.Version}}' sigs.k8s.io/gateway-api)
GATEWAY_API_MOD := $(GOMODCACHE)/sigs.k8s.io/gateway-api@$(GATEWAY_API_VERSION)

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	[ -d $@ ] || mkdir -p $@

CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
PROTOC ?= protoc
PROTOC_GEN_GO = $(LOCALBIN)/protoc-gen-go
PROTOC_GEN_GO_GRPC = $(LOCALBIN)/protoc-gen-go-grpc

## Tool Versions
CONTROLLER_TOOLS_VERSION ?= v0.19.0
PROTOC_GEN_GO_VERSION ?= v1.34.2
PROTOC_GEN_GO_GRPC_VERSION ?= v1.5.1

.PHONY: sync-upstream-crds
sync-upstream-crds: ## Copy upstream-owned CRDs from the Go module cache into deploy/components. Need to run after bumping version in go.mod.
	@go mod download sigs.k8s.io/gateway-api-inference-extension sigs.k8s.io/gateway-api
	@echo "Syncing GIE CRDs from $(GIE_VERSION)"
	@install -m 0644 \
		$(GIE_MOD)/config/crd/bases/inference.networking.k8s.io_inferencepools.yaml \
		$(GIE_MOD)/config/crd/bases/inference.networking.x-k8s.io_inferencepoolimports.yaml \
		deploy/components/crds-gie/
	@echo "Syncing Gateway API CRDs from $(GATEWAY_API_VERSION)"
	@install -m 0644 \
		$(GATEWAY_API_MOD)/config/crd/standard/gateway.networking.k8s.io_gatewayclasses.yaml \
		$(GATEWAY_API_MOD)/config/crd/standard/gateway.networking.k8s.io_gateways.yaml \
		$(GATEWAY_API_MOD)/config/crd/standard/gateway.networking.k8s.io_httproutes.yaml \
		deploy/components/crds-gateway-api/

.PHONY: generate
generate: controller-gen code-generator tidy sync-upstream-crds ## Generate WebhookConfiguration, ClusterRole, CustomResourceDefinition objects, code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	$(CONTROLLER_GEN) object:headerFile="/dev/null" paths="./..."
	$(CONTROLLER_GEN) crd output:dir="./config/crd/bases" paths="./..."
	./hack/update-codegen.sh $(LOCALBIN)

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

.PHONY: generate-proto
generate-proto: protoc-gen-go protoc-gen-go-grpc ## Generate Golang code from protobuf files.
	PATH="$(LOCALBIN):$$PATH" $(PROTOC) \
		-I pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/proto \
		-I . \
		--go_out=module=github.com/llm-d/llm-d-router:. \
		--go-grpc_out=module=github.com/llm-d/llm-d-router:. \
		pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/proto/*.proto

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
