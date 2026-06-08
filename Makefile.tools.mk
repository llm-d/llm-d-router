##@ Tool Checks

.PHONY: check-all-tools
check-all-tools: check-container-tool check-kubectl check-kustomize check-envsubst helm-install yq kubectl-validate ## Check and install all required tools
	@echo "All tools are available and ready!"

HELM_VERSION ?= v3.17.1
KUBECTL_VALIDATE_VERSION ?= v0.0.4
YQ_VERSION ?= v4.45.1

$(LOCALBIN):
	[ -d $@ ] || mkdir -p $@

define go-install-tool
@[ -f "$(1)-$(3)" ] || { \
set -e; \
package=$(2)@$(3) ;\
echo "Downloading $${package}" ;\
rm -f $(1) || true ;\
GOBIN=$(LOCALBIN) go install $${package} ;\
mv $(1) $(1)-$(3) ;\
} ;\
ln -sf $(notdir $(1))-$(3) $(1)
endef

.PHONY: helm-install
helm-install: $(HELM) ## Download helm locally if necessary.
$(HELM): | $(LOCALBIN)
	$(call go-install-tool,$(HELM),helm.sh/helm/v3/cmd/helm,$(HELM_VERSION))

.PHONY: kubectl-validate
kubectl-validate: $(KUBECTL_VALIDATE) ## Download kubectl-validate locally if necessary.
$(KUBECTL_VALIDATE): | $(LOCALBIN)
	$(call go-install-tool,$(KUBECTL_VALIDATE),sigs.k8s.io/kubectl-validate,$(KUBECTL_VALIDATE_VERSION))

.PHONY: yq
yq: $(YQ) ## Download yq locally if necessary.
$(YQ): | $(LOCALBIN)
	$(call go-install-tool,$(YQ),github.com/mikefarah/yq/v4,$(YQ_VERSION))

.PHONY: check-container-tool
check-container-tool:
	@if [ -z "$(CONTAINER_RUNTIME)" ]; then \
		echo "ERROR: No container tool detected. Please install docker or podman."; \
		exit 1; \
	else \
		echo "Container tool '$(CONTAINER_RUNTIME)' found."; \
	fi

.PHONY: check-kubectl
check-kubectl:
	@command -v kubectl >/dev/null 2>&1 || { \
	  echo "ERROR: kubectl is not installed. Install it from https://kubernetes.io/docs/tasks/tools/"; exit 1; }

.PHONY: check-kustomize
check-kustomize:
	@kubectl kustomize --help 2>&1 | grep -q -- '--enable-helm' || { \
	  echo "ERROR: kubectl kustomize does not support --enable-helm."; \
	  echo "Upgrade kubectl to v1.25 or later (https://kubernetes.io/docs/tasks/tools/)"; \
	  exit 1; }

.PHONY: check-envsubst
check-envsubst:
	@command -v envsubst >/dev/null 2>&1 || { \
	  echo "ERROR: envsubst is not installed. It is part of gettext."; \
	  echo "Try: sudo apt install gettext OR brew install gettext"; exit 1; }
