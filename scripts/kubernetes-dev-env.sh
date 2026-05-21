#!/bin/bash

# This shell script deploys a Kubernetes  cluster with an
# KGateway-based Gateway API implementation fully configured. It deploys the
# vllm, which it exposes with a Gateway -> HTTPRoute -> InferencePool.
# The Gateway is configured with the a filter for the ext_proc endpoint picker.

set -eux

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLEAN="${CLEAN:-false}"

# Validate required inputs
if [[ -z "${NAMESPACE:-}" ]]; then
  echo "ERROR: NAMESPACE environment variable is not set."
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN environment variable is not set."
  exit 1
fi

export VLLM_CHART_DIR="${VLLM_CHART_DIR:-../llm-d-kv-cache/vllm-setup-helm}"
# Check that Chart.yaml exists
if [[ ! -f "$VLLM_CHART_DIR/Chart.yaml" ]]; then
  echo "Chart.yaml not found in $VLLM_CHART_DIR"
  exit 1
fi

# Default image registry for pulling deployment images
export IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io/llm-d}"

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

# Full Hugging Face model name to deploy
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"

# Extract model family (e.g., "meta-llama" from "meta-llama/Llama-3.1-8B-Instruct")
export MODEL_FAMILY="${MODEL_NAME%%/*}"

# Extract model ID (e.g., "Llama-3.1-8B-Instruct")
export MODEL_ID="${MODEL_NAME##*/}"

# Safe model name for Kubernetes resources (lowercase, hyphenated)
export MODEL_NAME_SAFE=$(echo "${MODEL_ID}" | tr '[:upper:]' '[:lower:]' | tr ' /_.' '-')

# -----------------------------------------------------------------------------
# GIE Configuration
# -----------------------------------------------------------------------------

# Kubernetes service type for the gateway (e.g., NodePort, ClusterIP, LoadBalancer)
export GATEWAY_SERVICE_TYPE="${GATEWAY_SERVICE_TYPE:-NodePort}"

# Host port that maps to Gateway's service (used for NodePort)
export GATEWAY_HOST_PORT="${GATEWAY_HOST_PORT:-30080}"

# Inference pool name to register model-serving pods under
export POOL_NAME="${POOL_NAME:-${MODEL_NAME_SAFE}-inference-pool}"


# -----------------------------------------------------------------------------
# EPP Configuration
# -----------------------------------------------------------------------------

# Endpoint Picker (EPP) deployment name
export EPP_NAME="${EPP_NAME:-${MODEL_NAME_SAFE}-endpoint-picker}"

# EPP image tag
export EPP_TAG="${EPP_TAG:-v0.1.0}"

# EPP container image (full reference including tag)
export EPP_IMAGE="${EPP_IMAGE:-${IMAGE_REGISTRY}/llm-d-router-endpoint-picker:${EPP_TAG}}"

# Whether P/D mode is enabled for this deployment
export PD_ENABLED="\"${PD_ENABLED:-false}\""

# Token length threshold to trigger P/D logic
export PD_PROMPT_LEN_THRESHOLD="\"${PD_PROMPT_LEN_THRESHOLD:-10}\""

export EPP_CONFIG="${EPP_CONFIG:-deploy/config/epp-prefix-cache-tracking-config.yaml}"

# Redis deployment name
export REDIS_DEPLOYMENT_NAME="${REDIS_DEPLOYMENT_NAME:-lookup-server}"

# Redis service name
export REDIS_SVC_NAME="${REDIS_SVC_NAME:-${REDIS_DEPLOYMENT_NAME}-service}"

# Redis FQDN for internal Kubernetes communication
export REDIS_HOST="${REDIS_HOST:-vllm-${REDIS_SVC_NAME}.${NAMESPACE}.svc.cluster.local}"

# Redis port
export REDIS_PORT="${REDIS_PORT:-8100}"

# Key in the secret under which the HF token is stored
export HF_SECRET_KEY="${HF_SECRET_KEY:-token}"

# -----------------------------------------------------------------------------
# vLLM Configuration
# -----------------------------------------------------------------------------

# Helm release name for vLLM
export VLLM_HELM_RELEASE_NAME="${VLLM_HELM_RELEASE_NAME:-vllm}"

# Persistent volume claim size
export PVC_SIZE="${PVC_SIZE:-40Gi}"

# CPU request per vLLM replica
export VLLM_CPU_RESOURCES="${VLLM_CPU_RESOURCES:-10}"

# Memory request per vLLM replica
export VLLM_MEMORY_RESOURCES="${VLLM_MEMORY_RESOURCES:-40Gi}"

# GPU memory utilization (optional, default is null)
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-null}"

# Number of vLLM replicas
export VLLM_REPLICA_COUNT="${VLLM_REPLICA_COUNT:-3}"

# Tensor parallel size (optional, default is null)
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-null}"

# Number of GPU per vLLM
export VLLM_GPU_COUNT_PER_INSTANCE="${VLLM_GPU_COUNT_PER_INSTANCE:-1}"

# vLLM deployment name (derived from release + model)
export VLLM_DEPLOYMENT_NAME="${VLLM_HELM_RELEASE_NAME}-${MODEL_NAME_SAFE}"

# ------------------------------------------------------------------------------
# Deployment
# ------------------------------------------------------------------------------

kubectl create namespace ${NAMESPACE} 2>/dev/null || true

# Hack to better deal with kgateway on OpenShift
export PROXY_UID=$(kubectl get namespace ${NAMESPACE} -o json | jq -e -r '.metadata.annotations["openshift.io/sa.scc.uid-range"]' | perl -F'/' -lane 'print $F[0]+1');

# Detect if the cluster is OpenShift by checking for the 'route.openshift.io' API group
IS_OPENSHIFT=false
if kubectl api-resources 2>/dev/null | grep -q 'route.openshift.io'; then
  IS_OPENSHIFT=true
fi

set -o pipefail

if [[ "$CLEAN" == "true" ]]; then
  echo "INFO: CLEANING environment in namespace ${NAMESPACE}"
  # Delete the ConfigMAp created for the EPP configuration
  kubectl -n "${NAMESPACE}" delete --ignore-not-found=true ConfigMap epp-config
  # Delete inference schedulare and gateway resources.
  kubectl kustomize deploy/environments/dev/kubernetes-kgateway | envsubst | kubectl -n "${NAMESPACE}" delete --ignore-not-found=true -f -
  # Delete vllm resources.
  helm uninstall vllm --namespace ${NAMESPACE} --ignore-not-found
  exit 0
fi

echo "INFO: Deploying vLLM Environment in namespace ${NAMESPACE} ${EPP_IMAGE=}"
if [[ "${IS_OPENSHIFT}" == "true" ]]; then
  if command -v oc &>/dev/null; then
    # Grant the 'default' service account permission to run containers as any user (disables UID restrictions)
    oc adm policy add-scc-to-user anyuid -z default -n "${NAMESPACE}"
    echo "INFO: OpenShift cluster detected – granted 'anyuid' SCC to the 'default' service account in namespace '${NAMESPACE}'."
  fi
fi

# Run Helm upgrade/install vllm
echo "INFO: Deploying vLLM Environment in namespace ${NAMESPACE}, ${POOL_NAME}"
helm upgrade --install "$VLLM_HELM_RELEASE_NAME" "$VLLM_CHART_DIR" \
  --namespace="$NAMESPACE" \
  --set secret.create=true \
  --set secret.hfTokenValue="$HF_TOKEN" \
  --set vllm.poolLabelValue="$POOL_NAME" \
  --set vllm.model.name="$MODEL_NAME" \
  --set vllm.model.label="$MODEL_NAME_SAFE" \
  --set vllm.replicaCount="$VLLM_REPLICA_COUNT" \
  --set vllm.resources.requests.cpu="$VLLM_CPU_RESOURCES" \
  --set vllm.resources.requests.memory="$VLLM_MEMORY_RESOURCES" \
  --set vllm.resources.requests."nvidia\.com/gpu"="$VLLM_GPU_COUNT_PER_INSTANCE" \
  --set vllm.resources.limits."nvidia\.com/gpu"="$VLLM_GPU_COUNT_PER_INSTANCE" \
  --set vllm.gpuMemoryUtilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  --set vllm.tensorParallelSize="${VLLM_TENSOR_PARALLEL_SIZE}" \
  --set persistence.enabled=true \
  --set persistence.size="$PVC_SIZE" \
  --set lmcache.redis.enabled=true \
  --set lmcache.redis.nameSuffix="$REDIS_DEPLOYMENT_NAME" \
  --set lmcache.redis.service.nameSuffix="$REDIS_SVC_NAME" \
  --set lmcache.redis.service.port="$REDIS_PORT"

echo "INFO: Deploying Gateway Environment in namespace ${NAMESPACE}, ${POOL_NAME}"
kubectl -n "${NAMESPACE}" create configmap epp-config --from-file=epp-config.yaml=<(envsubst < "${EPP_CONFIG}") --dry-run=client -o yaml | kubectl apply -f -
kubectl kustomize deploy/environments/dev/kubernetes-kgateway | envsubst | kubectl -n "${NAMESPACE}" apply -f -
echo "INFO: Waiting for resources in namespace ${NAMESPACE} to become ready"
# Wait for gateway resources
kubectl -n "${NAMESPACE}" wait deployment/${EPP_NAME} --for=condition=Available --timeout=60s
kubectl -n "${NAMESPACE}" wait gateway/inference-gateway --for=condition=Programmed --timeout=60s
kubectl -n "${NAMESPACE}" wait deployment/inference-gateway --for=condition=Available --timeout=60s

# Wait for vllm resources
kubectl -n "${NAMESPACE}" wait deployment/${VLLM_HELM_RELEASE_NAME}-redis-${REDIS_DEPLOYMENT_NAME} --for=condition=Available --timeout=60s
kubectl -n "${NAMESPACE}" wait deployment/${VLLM_HELM_RELEASE_NAME}-${VLLM_DEPLOYMENT_NAME} --for=condition=Available --timeout=240s

cat <<EOF
-----------------------------------------
Deployment to namespace ${NAMESPACE} completed!

Status:

* The vllm is running and exposed via InferencePool
* The Gateway is exposing the InferencePool via HTTPRoute
* The Endpoint Picker is loaded into the Gateway via ext_proc

You can watch the Endpoint Picker logs with:

  $ kubectl logs -f deployments/${EPP_NAME}

With that running in the background, you can make requests using port-forward and curl:

  $ kubectl port-forward service/inference-gateway 8080:80 -n "${NAMESPACE}" &
  $ curl -s -w '\n' http://localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{"model":"${MODEL_NAME}","prompt":"hi","max_tokens":10,"temperature":0}' | jq

See DEVELOPMENT.md for additional access methods if the above fails.
-----------------------------------------
EOF
