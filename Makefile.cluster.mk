##@ Cluster Development Environments

.PHONY: env-dev-kubernetes
env-dev-kubernetes: check-kubectl check-envsubst ## Deploy full dev environment (vLLM + Gateway + EPP) to K8s/OpenShift cluster
	IMAGE_REGISTRY=$(IMAGE_REGISTRY) ./scripts/kubernetes-dev-env.sh 2>&1

# Kubernetes Development Environment - Teardown
.PHONY: clean-env-dev-kubernetes
clean-env-dev-kubernetes: check-kubectl check-envsubst ## Clean up full dev environment from K8s/OpenShift cluster
	@CLEAN=true ./scripts/kubernetes-dev-env.sh 2>&1
	@echo "INFO: Finished cleanup of development environment for namespace $(NAMESPACE)"


##@ RBAC Targets

.PHONY: install-rbac
install-rbac: check-kubectl check-envsubst ## Apply RBAC configuration to cluster
	@echo "Applying RBAC configuration from deploy/rbac..."
	kubectl kustomize deploy/environments/kubernetes-base/rbac | envsubst '$$PROJECT_NAME $$NAMESPACE $$VERSION' | kubectl apply -f -

.PHONY: uninstall-rbac
uninstall-rbac: check-kubectl check-envsubst ## Remove RBAC configuration from cluster
	@echo "Removing RBAC configuration from deploy/rbac..."
	kubectl kustomize deploy/environments/kubernetes-base/rbac | envsubst '$$PROJECT_NAME $$NAMESPACE $$VERSION' | kubectl delete -f - || true

##@ Kubernetes Targets

.PHONY: install-k8s
install-k8s: check-kubectl check-envsubst ## Deploy resources to Kubernetes
	@echo "Creating namespace (if needed) and setting context to $(NAMESPACE)..."
	kubectl create namespace $(NAMESPACE) 2>/dev/null || true
	kubectl config set-context --current --namespace=$(NAMESPACE)
	@echo "Deploying resources from deploy/ ..."
	# Build the kustomization from deploy, substitute variables, and apply the YAML
	kubectl kustomize deploy/environments/kubernetes-base | envsubst | kubectl apply -f -
	@echo "Waiting for pod to become ready..."
	sleep 5
	@POD=$$(kubectl get pod -l app=$(PROJECT_NAME)-statefulset -o jsonpath='{.items[0].metadata.name}'); \
	echo "Kubernetes installation complete."; \
	echo "To use the app, run:"; \
	echo "alias $(PROJECT_NAME)='kubectl exec -n $(NAMESPACE) -it $$POD -- /app/$(PROJECT_NAME)'"

.PHONY: uninstall-k8s
uninstall-k8s: check-kubectl check-envsubst ## Remove resources from Kubernetes
	@echo "Removing resources from Kubernetes..."
	kubectl kustomize deploy/environments/kubernetes-base | envsubst | kubectl delete --force -f - || true
	POD=$$(kubectl get pod -l app=$(PROJECT_NAME)-statefulset -o jsonpath='{.items[0].metadata.name}'); \
	echo "Deleting pod: $$POD"; \
	kubectl delete pod "$$POD" --force --grace-period=0 || true; \
	echo "Kubernetes uninstallation complete. Remove alias if set: unalias $(PROJECT_NAME)"

##@ OpenShift Targets

.PHONY: install-openshift
install-openshift: check-kubectl check-envsubst ## Deploy resources to OpenShift
	@echo $$PROJECT_NAME $$NAMESPACE $$EPP_IMAGE_TAG_BASE $$VERSION
	@echo "Creating namespace $(NAMESPACE)..."
	kubectl create namespace $(NAMESPACE) 2>/dev/null || true
	@echo "Deploying common resources from deploy/ ..."
	# Build and substitute the base manifests from deploy, then apply them
	kubectl kustomize deploy/environments/kubernetes-base | envsubst '$$PROJECT_NAME $$NAMESPACE $$EPP_IMAGE $$VERSION' | kubectl apply -n $(NAMESPACE) -f -
	@echo "Waiting for pod to become ready..."
	sleep 5
	@POD=$$(kubectl get pod -l app=$(PROJECT_NAME)-statefulset -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}'); \
	echo "OpenShift installation complete."; \
	echo "To use the app, run:"; \
	echo "alias $(PROJECT_NAME)='kubectl exec -n $(NAMESPACE) -it $$POD -- /app/$(PROJECT_NAME)'"

.PHONY: uninstall-openshift
uninstall-openshift: check-kubectl check-envsubst ## Remove resources from OpenShift
	@echo "Removing resources from OpenShift..."
	kubectl kustomize deploy/environments/kubernetes-base | envsubst '$$PROJECT_NAME $$NAMESPACE $$EPP_IMAGE $$VERSION' | kubectl delete --force -f - || true
	# @if kubectl api-resources --api-group=route.openshift.io | grep -q Route; then \
	#   envsubst '$$PROJECT_NAME $$NAMESPACE $$VERSION' < deploy/openshift/route.yaml | kubectl delete --force -f - || true; \
	# fi
	@POD=$$(kubectl get pod -l app=$(PROJECT_NAME)-statefulset -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}'); \
	echo "Deleting pod: $$POD"; \
	kubectl delete pod "$$POD" --force --grace-period=0 || true; \
	echo "OpenShift uninstallation complete. Remove alias if set: unalias $(PROJECT_NAME)"
