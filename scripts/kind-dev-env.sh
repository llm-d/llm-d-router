#!/bin/bash

# This shell script deploys a kind cluster with an Istio-based Gateway API
# implementation fully configured. It deploys the vllm simulator, which it
# exposes with a Gateway -> HTTPRoute -> InferencePool. The Gateway is
# configured with the a filter for the ext_proc endpoint picker.

set -eo pipefail

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set a default CLUSTER_NAME if not provided
: "${CLUSTER_NAME:=llm-d-router-dev}"

# Set the host port to map to the Gateway's inbound port (30080)
: "${GATEWAY_HOST_PORT:=30080}"

# Set the default IMAGE_REGISTRY if not provided
: "${IMAGE_REGISTRY:=ghcr.io/llm-d}"

# Set a default VLLM_SIMULATOR_TAG if not provided
export VLLM_SIMULATOR_TAG="${VLLM_SIMULATOR_TAG:-v0.8.2}"

# VLLM_IMAGE: the vLLM container image to deploy. Can be a simulator or real vLLM image
# (e.g., vllm/vllm-openai:v0.16.0 for production). Defaults to the simulator image.
export VLLM_IMAGE="${VLLM_IMAGE:-${IMAGE_REGISTRY}/llm-d-inference-sim:${VLLM_SIMULATOR_TAG}}"

# Set a default EPP_TAG if not provided
export EPP_TAG="${EPP_TAG:-dev}"

# Set a default EPP_IMAGE if not provided
EPP_IMAGE="${EPP_IMAGE:-${IMAGE_REGISTRY}/llm-d-inference-scheduler:${EPP_TAG}}"
export EPP_IMAGE

# Set the model name to deploy.
# When Encode disaggregation is enabled (multimodal pipeline), default to a
# multimodal model. Otherwise use the standard text-only model.
# Note: DISAGG_E/DISAGG_P are set later in this script, so read the raw env vars here.
if [ "${DISAGG_E:-false}" == "true" ] || [ "${EPD_ENABLED:-false}" == "true" ] || [ "${EPD_ENABLED:-false}" == "\"true\"" ]; then
  export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
else
  export MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
fi
# Extract model family (e.g., "meta-llama" from "meta-llama/Llama-3.1-8B-Instruct")
export MODEL_FAMILY="${MODEL_NAME%%/*}"
# Extract model ID (e.g., "Llama-3.1-8B-Instruct")
export MODEL_ID="${MODEL_NAME##*/}"
# Safe model name for Kubernetes resources (lowercase, hyphenated)
export MODEL_NAME_SAFE=$(echo "${MODEL_ID}" | tr '[:upper:]' '[:lower:]' | tr ' /_.' '-')

# Set the endpoint-picker to deploy
export EPP_NAME="${EPP_NAME:-${MODEL_NAME_SAFE}-endpoint-picker}"

# Set the default routing side car image tag
export SIDECAR_TAG="${SIDECAR_TAG:-dev}"

# Set a default SIDECAR_IMAGE if not provided
SIDECAR_IMAGE="${SIDECAR_IMAGE:-${IMAGE_REGISTRY}/llm-d-routing-sidecar:${SIDECAR_TAG}}"
export SIDECAR_IMAGE

# Set the default UDS tokenizer image tag
export UDS_TOKENIZER_TAG="${UDS_TOKENIZER_TAG:-dev}"

# Set a default UDS_TOKENIZER_IMAGE if not provided
UDS_TOKENIZER_IMAGE="${UDS_TOKENIZER_IMAGE:-${IMAGE_REGISTRY}/llm-d-uds-tokenizer:${UDS_TOKENIZER_TAG}}"
export UDS_TOKENIZER_IMAGE

# Set a default VLLM_RENDER_IMAGE if not provided (CPU-only vLLM image that
# runs `vllm launch render` for the token-producer plugin's HTTP backend).
export VLLM_RENDER_IMAGE="${VLLM_RENDER_IMAGE:-vllm/vllm-openai-cpu:v0.19.1}"

# Set the inference pool name for the deployment
export POOL_NAME="${POOL_NAME:-${MODEL_NAME_SAFE}-inference-pool}"


# By default we are not deploying Prometheus monitoring
export PROM_ENABLED="${PROM_ENABLED:-false}"

# Set the host port to map to the Prometheus NodePort (30090)
: "${PROM_HOST_PORT:=30090}"

# Disaggregation flags (independent boolean options):
#   DISAGG_E=true  — deploy a separate Encoder pod
#   DISAGG_P=true  — deploy a separate Prefill pod
#
# Combinations:
#   DISAGG_E=false DISAGG_P=false  → EPD (no disaggregation, default)
#   DISAGG_E=false DISAGG_P=true   → P/D
#   DISAGG_E=true  DISAGG_P=false  → E/PD
#   DISAGG_E=true  DISAGG_P=true   → E/P/D
export DISAGG_E="${DISAGG_E:-false}"
export DISAGG_P="${DISAGG_P:-false}"

# Backward compatibility: PD_ENABLED and EPD_ENABLED are deprecated.
# Use DISAGG_P=true and DISAGG_E=true instead.
PD_ENABLED="${PD_ENABLED:-false}"
EPD_ENABLED="${EPD_ENABLED:-false}"
if [ "${EPD_ENABLED}" == "true" ] || [ "${EPD_ENABLED}" == "\"true\"" ]; then
  echo "WARNING: EPD_ENABLED is deprecated. Use DISAGG_E=true DISAGG_P=true instead." >&2
  DISAGG_E="true"
  DISAGG_P="true"
elif [ "${PD_ENABLED}" == "true" ] || [ "${PD_ENABLED}" == "\"true\"" ]; then
  echo "WARNING: PD_ENABLED is deprecated. Use DISAGG_P=true instead." >&2
  DISAGG_P="true"
fi

# By default we are not setting up for KV cache
export KV_CACHE_ENABLED="${KV_CACHE_ENABLED:-false}"

# By default we are not setting up for external tokenizer
export EXTERNAL_TOKENIZER_ENABLED="${EXTERNAL_TOKENIZER_ENABLED:-false}"

# Replica counts for E (Encode), P (Prefill), and D (Decode)
export VLLM_REPLICA_COUNT_E="${VLLM_REPLICA_COUNT_E:-1}"
export VLLM_REPLICA_COUNT_P="${VLLM_REPLICA_COUNT_P:-1}"
export VLLM_REPLICA_COUNT_D="${VLLM_REPLICA_COUNT_D:-1}"

# Data Parallel size
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"

# vLLM mode: echo for simulator, empty for real vLLM
export VLLM_SIM_MODE="${VLLM_SIM_MODE:-echo}"

# Role label for the vllm-d pod in the EPD (no-disaggregation) scenario.
# Empty by default — Kubernetes accepts empty label values, and the EPD patch
# uses this to optionally mark the unified pod's role.
export DECODE_ROLE="${DECODE_ROLE:-}"

# Kubernetes namespace for all deployed resources
export NAMESPACE="${NAMESPACE:-default}"

# Metrics endpoint auth (false for dev/test, true for production)
export METRICS_ENDPOINT_AUTH="${METRICS_ENDPOINT_AUTH:-false}"

# HuggingFace token for model downloads (empty for simulator)
export HF_TOKEN="${HF_TOKEN:-}"

# Extra vLLM args per pod type (empty by default). Use --flag=value format.
# Example: VLLM_EXTRA_ARGS_D="--tensor-parallel-size=2"
export VLLM_EXTRA_ARGS_E="${VLLM_EXTRA_ARGS_E:-}"
export VLLM_EXTRA_ARGS_P="${VLLM_EXTRA_ARGS_P:-}"
export VLLM_EXTRA_ARGS_D="${VLLM_EXTRA_ARGS_D:-}"

# Connector types — derived from DISAGG_E and DISAGG_P
# KV connector: needed when P is disaggregated (P/D or E/P/D)
if [ "${DISAGG_P}" == "true" ]; then
  export CONNECTOR_TYPE="${CONNECTOR_TYPE:-nixlv2}"
  export KV_CONNECTOR_TYPE="${KV_CONNECTOR_TYPE:-nixlv2}"
else
  export CONNECTOR_TYPE="${CONNECTOR_TYPE:-}"
  export KV_CONNECTOR_TYPE="${KV_CONNECTOR_TYPE:-}"
fi
# EC connector: needed when E is disaggregated (E/PD or E/P/D)
if [ "${DISAGG_E}" == "true" ]; then
  export EC_CONNECTOR_TYPE="${EC_CONNECTOR_TYPE:-ec-example}"
else
  export EC_CONNECTOR_TYPE="${EC_CONNECTOR_TYPE:-}"
fi

# Determine EPP config file based on disaggregation flags
# KV cache and data parallel are independent options that work with any mode
if [ "${EXTERNAL_TOKENIZER_ENABLED}" == "true" ]; then
  DEFAULT_EPP_CONFIG="deploy/config/sim-epp-external-tokenizer-config.yaml"
elif [ "${KV_CACHE_ENABLED}" == "true" ]; then
  DEFAULT_EPP_CONFIG="deploy/config/sim-epp-kvcache-config.yaml"
elif [ "${DISAGG_E}" == "true" ] && [ "${DISAGG_P}" == "true" ]; then
  DEFAULT_EPP_CONFIG="deploy/config/sim-e-p-d-epp-config.yaml"
elif [ "${DISAGG_E}" == "true" ]; then
  DEFAULT_EPP_CONFIG="deploy/config/sim-e-pd-epp-config.yaml"
elif [ "${DISAGG_P}" == "true" ]; then
  DEFAULT_EPP_CONFIG="deploy/config/sim-pd-epp-config.yaml"
else
  DEFAULT_EPP_CONFIG="deploy/config/sim-epp-config.yaml"
fi

export EPP_CONFIG="${EPP_CONFIG:-${DEFAULT_EPP_CONFIG}}"

# ------------------------------------------------------------------------------
# Setup & Requirement Checks
# ------------------------------------------------------------------------------

# Check for a supported container runtime if an explicit one was not set
if [ -z "${CONTAINER_RUNTIME}" ]; then
  if command -v docker &> /dev/null; then
    CONTAINER_RUNTIME="docker"
  elif command -v podman &> /dev/null; then
    CONTAINER_RUNTIME="podman"
  else
    echo "Neither docker nor podman could be found in PATH" >&2
    exit 1
  fi
fi

set -u

# Check for required programs
for cmd in kind kubectl ${CONTAINER_RUNTIME}; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is not installed or not in the PATH."
        exit 1
    fi
done

# Prometheus config-reloader needs sufficient inotify resources
if [ "${PROM_ENABLED}" == "true" ]; then
  INOTIFY_INSTANCES=$(cat /proc/sys/fs/inotify/max_user_instances)
  if [ "${INOTIFY_INSTANCES}" -lt 512 ]; then
    echo "Error: fs.inotify.max_user_instances is ${INOTIFY_INSTANCES} (need >= 512) for Prometheus."
    echo ""
    echo "  sudo sysctl -w fs.inotify.max_user_instances=512"
    echo ""
    echo "To persist: echo 'fs.inotify.max_user_instances=512' | sudo tee /etc/sysctl.d/99-inotify.conf"
    exit 1
  fi
fi

# TARGET_PORTS is substituted directly into the `targetPorts: ${TARGET_PORTS}` field
# in deploy/components/inference-gateway/inference-pools.yaml. Each item must be
# indented with exactly 2 spaces to match the indentation of that field. If the
# field is ever reindented in inference-pools.yaml, update the indentation here too.
NEW_LINE=$'\n'
TARGET_PORTS="${NEW_LINE}  - number: 8000"
for ((i = 1; i < VLLM_DATA_PARALLEL_SIZE; ++i)); do
    EXTRA_PORT=$((8000 + i))
    TARGET_PORTS="${TARGET_PORTS}${NEW_LINE}  - number: ${EXTRA_PORT}"
done

export TARGET_PORTS

# ------------------------------------------------------------------------------
# Cluster Deployment
# ------------------------------------------------------------------------------

# Check if the cluster already exists
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    echo "Cluster '${CLUSTER_NAME}' already exists, re-using"
else
    EXTRA_PORT_MAPPINGS=""
    if [ "${PROM_ENABLED}" == "true" ]; then
      EXTRA_PORT_MAPPINGS="  - containerPort: 30090
    hostPort: ${PROM_HOST_PORT}
    protocol: TCP"
    fi

    kind create cluster --name "${CLUSTER_NAME}" --config - << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  # Pin to Kubernetes 1.31+ for Gateway API v1.5.1 compatibility
  # (requires isIP() CEL function and ValidatingAdmissionPolicy)
  image: kindest/node:v1.31.12
  extraPortMappings:
  - containerPort: 30080
    hostPort: ${GATEWAY_HOST_PORT}
    protocol: TCP
${EXTRA_PORT_MAPPINGS}
EOF
fi

# Set the kubectl context to the kind cluster
KUBE_CONTEXT="kind-${CLUSTER_NAME}"
kubectl config set-context ${KUBE_CONTEXT} --namespace=default

set -x

# Hotfix for https://github.com/kubernetes-sigs/kind/issues/3880
CONTAINER_NAME="${CLUSTER_NAME}-control-plane"
${CONTAINER_RUNTIME} exec ${CONTAINER_NAME} /bin/bash -c "sysctl net.ipv4.conf.all.arp_ignore=0"

# Wait for all pods to be ready
kubectl --context ${KUBE_CONTEXT} -n kube-system wait --for=condition=Ready --all pods --timeout=300s

echo "Waiting for local-path-storage pods to be created..."
deadline=$(( $(date +%s) + 120 ))
until kubectl --context ${KUBE_CONTEXT} -n local-path-storage get pods -o name 2>/dev/null | grep -q pod/; do
  if (( $(date +%s) >= deadline )); then
    echo "ERROR: local-path-storage pods did not appear within 120s" >&2
    kubectl --context ${KUBE_CONTEXT} get namespaces >&2 || true
    kubectl --context ${KUBE_CONTEXT} -n local-path-storage get pods >&2 || true
    exit 1
  fi
  sleep 2
done
kubectl --context ${KUBE_CONTEXT} -n local-path-storage wait --for=condition=Ready --all pods --timeout=300s

# ------------------------------------------------------------------------------
# Load Container Images
# ------------------------------------------------------------------------------

LINUX_ARCH="$(uname -m)"
case "${LINUX_ARCH}" in
    x86_64) LINUX_ARCH="amd64" ;;
    aarch64|arm64) LINUX_ARCH="arm64" ;;
esac

PLATFORM_ARGS=()
SAVE_ARGS=()
if [ "${CONTAINER_RUNTIME}" == "docker" ]; then
    PLATFORM_ARGS=("--platform" "linux/${LINUX_ARCH}")
elif [ "${CONTAINER_RUNTIME}" == "podman" ]; then
    SAVE_ARGS=("--format=docker-archive")
fi

pull_image() {
    local image="$1"
    if ! "${CONTAINER_RUNTIME}" image inspect "${image}" > /dev/null 2>&1; then
        echo "Image ${image} not found locally, pulling..."
        "${CONTAINER_RUNTIME}" pull ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} "${image}"
    fi
}

load_image() {
    local image="$1"
    echo "Loading ${image} into kind cluster..."
    if [ "${CONTAINER_RUNTIME}" == "docker" ]; then
        # KIND's `kind load` uses `ctr import --all-platforms` internally, which
        # fails when only the target architecture's layers are locally cached
        # (e.g. after `docker pull --platform linux/amd64` of a multi-arch image).
        # Bypass this by piping directly to `ctr import` without --all-platforms.
        docker save "${image}" | \
            docker exec --privileged -i "${CLUSTER_NAME}-control-plane" \
            ctr --namespace=k8s.io images import --digests --snapshotter=overlayfs -
    else
        "${CONTAINER_RUNTIME}" save ${SAVE_ARGS[@]+"${SAVE_ARGS[@]}"} "${image}" | kind --name "${CLUSTER_NAME}" load image-archive /dev/stdin
    fi
}

for IMAGE in "${VLLM_IMAGE}" "${EPP_IMAGE}" "${SIDECAR_IMAGE}" "${UDS_TOKENIZER_IMAGE}"; do
    pull_image "${IMAGE}"
    load_image "${IMAGE}"
done

# ------------------------------------------------------------------------------
# CRD Deployment (Gateway API + GIE)
# ------------------------------------------------------------------------------

# apply_crds retries the kustomize+apply pipeline up to 3 times with a 5-second
# backoff. etcd occasionally times out on large CRD sets (e.g. Istio); retrying
# is safe because --server-side --force-conflicts is idempotent.
apply_crds() {
    local kustomize_extra_flags="$1"
    local kustomize_dir="$2"
    local attempt max_attempts=3
    for attempt in $(seq 1 ${max_attempts}); do
        if kubectl kustomize ${kustomize_extra_flags} "${kustomize_dir}" \
               | kubectl --context ${KUBE_CONTEXT} apply --server-side --force-conflicts -f -; then
            return 0
        fi
        if [ "${attempt}" -lt "${max_attempts}" ]; then
            echo "CRD apply failed (attempt ${attempt}/${max_attempts}), retrying in 5s..." >&2
            sleep 5
        fi
    done
    echo "Error: CRD apply failed after ${max_attempts} attempts: ${kustomize_dir}" >&2
    return 1
}

apply_crds ""               deploy/components/crds-gateway-api
apply_crds ""               deploy/components/crds-gie
apply_crds "--enable-helm"  deploy/components/crds-istio

# ------------------------------------------------------------------------------
# Development Environment
# ------------------------------------------------------------------------------

ENV_BASE="deploy/environments/dev"

if [ "${DISAGG_E}" == "true" ] && [ "${DISAGG_P}" == "true" ]; then
  KUSTOMIZE_DIR="${ENV_BASE}/e-p-d"
elif [ "${DISAGG_E}" == "true" ]; then
  KUSTOMIZE_DIR="${ENV_BASE}/e-pd"
elif [ "${DISAGG_P}" == "true" ]; then
  KUSTOMIZE_DIR="${ENV_BASE}/p-d"
else
  KUSTOMIZE_DIR="${ENV_BASE}/epd"
fi

TEMP_FILE=$(mktemp)
# Ensure that the temporary file is deleted now matter what happens in the script
trap "rm -f \"${TEMP_FILE}\"" EXIT

kubectl --context ${KUBE_CONTEXT} delete configmap epp-config --ignore-not-found
envsubst '$MODEL_NAME' < ${EPP_CONFIG} > ${TEMP_FILE}
kubectl --context ${KUBE_CONTEXT} create configmap epp-config --from-file=epp-config.yaml=${TEMP_FILE}

# Deploy Istio base (shared infrastructure)
kubectl kustomize --enable-helm deploy/environments/dev/base-kind-istio \
  | envsubst '${POOL_NAME} ${MODEL_NAME} ${MODEL_NAME_SAFE} ${EPP_NAME} ${EPP_IMAGE} ${VLLM_IMAGE} \
  ${SIDECAR_IMAGE} ${UDS_TOKENIZER_IMAGE} ${VLLM_RENDER_IMAGE} ${TARGET_PORTS} ${NAMESPACE} ${METRICS_ENDPOINT_AUTH} \
${VLLM_REPLICA_COUNT_E} ${VLLM_REPLICA_COUNT_P} ${VLLM_REPLICA_COUNT_D} ${VLLM_DATA_PARALLEL_SIZE}' \
  | kubectl --context ${KUBE_CONTEXT} apply -f -

# Deploy scenario-specific vLLM components
kubectl kustomize --enable-helm ${KUSTOMIZE_DIR} \
  | envsubst '${POOL_NAME} ${MODEL_NAME} ${MODEL_NAME_SAFE} ${EPP_NAME} ${EPP_IMAGE} ${VLLM_IMAGE} \
  ${SIDECAR_IMAGE} ${UDS_TOKENIZER_IMAGE} ${TARGET_PORTS} ${NAMESPACE} \
  ${VLLM_REPLICA_COUNT_E} ${VLLM_REPLICA_COUNT_P} ${VLLM_REPLICA_COUNT_D} ${VLLM_DATA_PARALLEL_SIZE} \
  ${KV_CONNECTOR_TYPE} ${EC_CONNECTOR_TYPE} ${CONNECTOR_TYPE} ${KV_CACHE_ENABLED} ${HF_TOKEN} ${VLLM_SIM_MODE} \
  ${DECODE_ROLE} ${VLLM_EXTRA_ARGS_E} ${VLLM_EXTRA_ARGS_P} ${VLLM_EXTRA_ARGS_D}' \
  | awk '
    /^[[:space:]]*-[[:space:]]+".*"[[:space:]]*$/ {
      match($0, /^[[:space:]]*/); indent = substr($0, 1, RLENGTH)
      content = $0
      sub(/^[[:space:]]*-[[:space:]]+"/, "", content)
      sub(/"[[:space:]]*$/, "", content)
      if (content == "") { next }
      if (substr(content, 1, 2) == "--") {
        n = split(content, flags, " --")
        for (i = 1; i <= n; i++) {
          flag = flags[i]
          if (i > 1) flag = "--" flag
          if (flag != "") print indent "- \"" flag "\""
        }
        next
      }
    }
    { print }
  ' \
  | kubectl --context ${KUBE_CONTEXT} apply -f -

# ------------------------------------------------------------------------------
# Check & Verify
# ------------------------------------------------------------------------------

# Wait for all control-plane deployments to be ready
kubectl --context ${KUBE_CONTEXT} -n llm-d-istio-system wait --for=condition=available --timeout=600s deployment --all

# Wait for all deployments to be ready
kubectl --context ${KUBE_CONTEXT} -n default wait --for=condition=available --timeout=600s deployment --all

# Wait for the gateway to be ready
kubectl --context ${KUBE_CONTEXT} wait gateway/inference-gateway --for=condition=Programmed --timeout=600s

# ------------------------------------------------------------------------------
# Prometheus Monitoring (optional)
# ------------------------------------------------------------------------------

if [ "${PROM_ENABLED}" == "true" ]; then
  echo "Deploying Prometheus monitoring stack..."

  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
  helm repo update prometheus-community

  # Install kube-prometheus-stack (Prometheus only)
  helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring --create-namespace \
    --set grafana.enabled=false \
    --set alertmanager.enabled=false \
    --set kubeControllerManager.enabled=false \
    --set kubeEtcd.enabled=false \
    --set kubeProxy.enabled=false \
    --set kubeScheduler.enabled=false \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.resources.requests.memory=512Mi \
    --set prometheus.prometheusSpec.resources.limits.memory=1Gi \
    --set prometheus.service.type=NodePort \
    --set prometheus.service.nodePort=30090 \
    --kube-context ${KUBE_CONTEXT} \
    --wait --timeout 300s

  kubectl kustomize deploy/components/monitoring \
    | envsubst '${EPP_NAME} ${POOL_NAME}' \
    | kubectl --context ${KUBE_CONTEXT} apply -f -

  echo "Prometheus monitoring deployed."
fi

cat <<EOF
-----------------------------------------
Deployment completed!

* Kind Cluster Name: ${CLUSTER_NAME}
* Kubectl Context: ${KUBE_CONTEXT}

Status:

* The vllm simulator is running and exposed via InferencePool
* The Gateway is exposing the InferencePool via HTTPRoute
* The Endpoint Picker is loaded into the Gateway via ext_proc

You can watch the Endpoint Picker logs with:

  $ kubectl --context ${KUBE_CONTEXT} logs -f deployments/${EPP_NAME}

With that running in the background, you can make requests:

  $ curl -s -w '\n' http://localhost:${GATEWAY_HOST_PORT}/v1/completions -H 'Content-Type: application/json' -d '{"model":"${MODEL_NAME}","prompt":"hi","max_tokens":10,"temperature":0}' | jq

See DEVELOPMENT.md for additional access methods if the above fails.

-----------------------------------------
EOF

if [ "${PROM_ENABLED}" == "true" ]; then
cat <<EOF

Monitoring:

* Prometheus: http://localhost:${PROM_HOST_PORT}

EOF
fi
