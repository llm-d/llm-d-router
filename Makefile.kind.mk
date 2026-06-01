##@ Kind Development Environments

KIND_CLUSTER_NAME ?= $(PROJECT_NAME)-dev
KIND_GATEWAY_HOST_PORT ?= 30080

.PHONY: image-kind
image-kind: ## Reload the EPP and SideCar images into the kind cluster $(KIND_CLUSTER_NAME)
	KIND_CLUSTER_NAME=$(KIND_CLUSTER_NAME) ./scripts/load_image.sh $(EPP_IMAGE) $(SIDECAR_IMAGE)

.PHONY: env-dev-kind
env-dev-kind: image-build image-pull ## Run under kind ($(KIND_CLUSTER_NAME))
	@if [ "$$PD_ENABLED" = "true" ] && [ "$$KV_CACHE_ENABLED" = "true" ]; then \
		echo "Error: Both PD_ENABLED and KV_CACHE_ENABLED are true. Skipping env-dev-kind."; \
		exit 1; \
	else \
		CLUSTER_NAME=$(KIND_CLUSTER_NAME) \
		GATEWAY_HOST_PORT=$(KIND_GATEWAY_HOST_PORT) \
		IMAGE_REGISTRY=$(IMAGE_REGISTRY) \
		./scripts/kind-dev-env.sh; \
	fi

.PHONY: clean-env-dev-kind
clean-env-dev-kind: ## Cleanup kind setup (delete cluster $(KIND_CLUSTER_NAME))
	@echo "INFO: cleaning up kind cluster $(KIND_CLUSTER_NAME)"
	kind delete cluster --name $(KIND_CLUSTER_NAME)

##@ Alias checking
.PHONY: check-alias
check-alias: check-container-tool ## Check if the container alias works correctly
	@echo "Checking alias functionality for container '$(PROJECT_NAME)-container'..."
	@if ! $(CONTAINER_RUNTIME) exec $(PROJECT_NAME)-container /app/$(PROJECT_NAME) --help >/dev/null 2>&1; then \
	  echo "WARNING: The container '$(PROJECT_NAME)-container' is running, but the alias might not work."; \
	  echo "Try: $(CONTAINER_RUNTIME) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)"; \
	else \
	  echo "Alias is likely to work: alias $(PROJECT_NAME)='$(CONTAINER_RUNTIME) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)'"; \
	fi
