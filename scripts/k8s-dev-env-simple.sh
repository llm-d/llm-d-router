#!/bin/bash

# This shell script deploys a Kubernetes cluster with real vllm instances
# and a proxy in standalone mode as a sidecar to the EPP.
# The proxy communicates over localhost with the EPP and the traffic is
# sent directly to the EPP pod's externally exposed port.

set -eu

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
build_io_config() {
  local role="$1"

  jq -cn \
    --arg connector "$IO_CONNECTOR_CLASS" \
    --arg role "$role" \
    --arg port "$IO_PORT" \
    --argjson max_io_threads "$MAX_IO_THREADS" \
    '{
      io_connector: $connector,
      io_role: $role,
      io_connector_extra_config: {
        host: "$(POD_IP)",
        port: $port,
        max_io_threads: $max_io_threads
      }
    }'
}

build_pooling_cache_config() {
  jq -cn \
    --argjson pre_process_caching "$([ "${PRE_CACHING}" = "true" ] && echo true || echo false)" \
    --argjson inference_caching "$([ "${INF_CACHING}" = "true" ] && echo true || echo false)" \
    --argjson post_process_caching "$([ "${POST_CACHING}" = "true" ] && echo true || echo false)" \
    '{
      pre_process_caching: $pre_process_caching,
      inference_caching: $inference_caching,
      post_process_caching: $post_process_caching
    }'
}

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Validate required inputs
if [[ -z "${NAMESPACE:-}" ]]; then
  echo "ERROR: NAMESPACE environment variable is not set."
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN environment variable is not set."
  exit 1
fi

# Default image registry for pulling deployment images
export IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io/llm-d}"

# Example image url
export EXAMPLE_IMG_URL="https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/examples/India_900498_S2Hand.tif"

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

# Full Hugging Face model name to deploy
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"

# Extract model ID (e.g., "Llama-3.1-8B-Instruct")
export MODEL_ID="${MODEL_NAME##*/}"

# Safe model name for Kubernetes resources (lowercase, hyphenated)
export MODEL_NAME_SAFE=$(echo "${MODEL_ID}" | tr '[:upper:]' '[:lower:]' | tr ' /_.' '-')

# Inference app name to register model-serving pods under
export APP_NAME="${APP_NAME:-${MODEL_NAME_SAFE}-inference-app}"

# -----------------------------------------------------------------------------
# EPP Configuration
# -----------------------------------------------------------------------------

# Endpoint Picker (EPP) deployment name
export EPP_NAME="${EPP_NAME:-${MODEL_NAME_SAFE}-endpoint-picker}"

# EPP image tag
export EPP_TAG="${EPP_TAG:-v0.1.0}"

# EPP container image (full reference including tag)
export EPP_IMAGE="${EPP_IMAGE:-${IMAGE_REGISTRY}/llm-d-router-endpoint-picker:${EPP_TAG}}"

# EPP config file path
export EPP_CONFIG="${EPP_CONFIG:-test/e2e/artifacts/config/epp-full-io-disagg-config.yaml}"

# Number of EPP replicas
export EPP_REPLICA_COUNT="${EPP_REPLICA_COUNT:-1}"

# EPP log level (default: 2)
export EPP_LOG_LEVEL="\"${EPP_LOG_LEVEL:-2}\""

# Key in the secret under which the HF token is stored
export HF_SECRET_KEY="${HF_SECRET_KEY:-token}"

# -----------------------------------------------------------------------------
# vLLM Configuration
# -----------------------------------------------------------------------------

export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.1}"
export VLLM_CPU_IMAGE="${VLLM_CPU_IMAGE:-vllm/vllm-openai-cpu:v0.19.1}"
export SIDECAR_IMAGE="${SIDECAR_IMAGE:-ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.9.0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-96}"
export OMP_NUM_THREADS_STR="\"${OMP_NUM_THREADS}\""

export TARGET_PORT="${TARGET_PORT:-8000}"
export TARGET_PORT_STR="\"${TARGET_PORT}\""

export IO_PORT="${IO_PORT:-5000}"
export IO_PORT_STR="\"${IO_PORT}\""

export IO_CONNECTOR_TYPE="${IO_CONNECTOR_TYPE:-io-p2p}"
export IO_CONNECTOR_CLASS="${IO_CONNECTOR_CLASS:-IOP2PConnector}"

export MAX_IO_THREADS="${MAX_IO_THREADS:-16}"

export PRE_IO_TRANSFER_CONFIG=$(build_io_config "io_sender")
export INF_IO_TRANSFER_CONFIG=$(build_io_config "io_both")
export POST_IO_TRANSFER_CONFIG=$(build_io_config "io_receiver")

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

export INF_MAX_NUM_SEQS="${INF_MAX_NUM_SEQS:-32}"
export INF_MAX_NUM_SEQS_STR="\"${INF_MAX_NUM_SEQS}\""

export PRE_RENDERER_NUM_WORKERS="${PRE_RENDERER_NUM_WORKERS:-1}"
export PRE_RENDERER_NUM_WORKERS_STR="\"${PRE_RENDERER_NUM_WORKERS}\""
export INF_RENDERER_NUM_WORKERS="${INF_RENDERER_NUM_WORKERS:-1}"
export INF_RENDERER_NUM_WORKERS_STR="\"${INF_RENDERER_NUM_WORKERS}\""
export POST_RENDERER_NUM_WORKERS="${POST_RENDERER_NUM_WORKERS:-1}"
export POST_RENDERER_NUM_WORKERS_STR="\"${POST_RENDERER_NUM_WORKERS}\""

# Python dependencies for the terratorch plugin
export IO_PROCESSOR_PLUGIN="${IO_PROCESSOR_PLUGIN:-terratorch_segmentation}"

# Preprocessing instances
export PRE_DEPLOYMENT_NAME="${PRE_DEPLOYMENT_NAME:-vllm-pre}"
export PRE_REPLICA_COUNT="${PRE_REPLICA_COUNT:-1}"
export PRE_ROLE="${PRE_ROLE:-preprocess}"

export PRE_CPU_RESOURCES="${PRE_CPU_RESOURCES:-4}"
export PRE_MEMORY_RESOURCES="${PRE_MEMORY_RESOURCES:-2G}"
export PRE_CPU_RESOURCES_LIM="${PRE_CPU_RESOURCES_LIM:-8}"
export PRE_MEMORY_RESOURCES_LIM="${PRE_MEMORY_RESOURCES_LIM:-4G}"

# Inference instances
export INF_DEPLOYMENT_NAME="${INF_DEPLOYMENT_NAME:-vllm-inf}"
export INF_REPLICA_COUNT="${INF_REPLICA_COUNT:-1}"
export INF_ROLE="${INF_ROLE:-inference}"

export INF_CPU_RESOURCES="${INF_CPU_RESOURCES:-32}"
export INF_MEMORY_RESOURCES="${INF_MEMORY_RESOURCES:-32G}"
export INF_GPU_RESOURCES="${INF_GPU_RESOURCES:-1}"
export INF_CPU_RESOURCES_LIM="${INF_CPU_RESOURCES_LIM:-32}"
export INF_MEMORY_RESOURCES_LIM="${INF_MEMORY_RESOURCES_LIM:-64G}"
export INF_GPU_RESOURCES_LIM="${INF_GPU_RESOURCES_LIM:-1}"

# Postprocessing instances
export POST_DEPLOYMENT_NAME="${POST_DEPLOYMENT_NAME:-vllm-post}"
export POST_REPLICA_COUNT="${POST_REPLICA_COUNT:-1}"
export POST_ROLE="${POST_ROLE:-postprocess}"

export POST_CPU_RESOURCES="${POST_CPU_RESOURCES:-4}"
export POST_MEMORY_RESOURCES="${POST_MEMORY_RESOURCES:-2G}"
export POST_CPU_RESOURCES_LIM="${POST_CPU_RESOURCES_LIM:-8}"
export POST_MEMORY_RESOURCES_LIM="${POST_MEMORY_RESOURCES_LIM:-4G}"

export POST_TARGET_PORT="${POST_TARGET_PORT:-8200}"
export POST_TARGET_PORT_STR="\"${POST_TARGET_PORT}\""

# Telemetry
export TRACING_EPP="${TRACING_EPP:-false}"
export TRACING_SIDECAR="${TRACING_SIDECAR:-false}"
export JAEGER_IP="${JAEGER_IP:-127.0.0.1}"
export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://${JAEGER_IP}:4317}"
export OTEL_TRACES_SAMPLER_ARG="${OTEL_TRACES_SAMPLER_ARG:-0.1}"
export OTEL_TRACES_SAMPLER_ARG_STR="\"${OTEL_TRACES_SAMPLER_ARG}\""

export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="${OTEL_EXPORTER_OTLP_TRACES_PROTOCOL:-http/protobuf}"
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://${JAEGER_IP}:4318/v1/traces}"

# Caching
export PRE_CACHING="${PRE_CACHING:-false}"
export INF_CACHING="${INF_CACHING:-false}"
export POST_CACHING="${POST_CACHING:-false}"
export POOLING_CACHE_CONFIG="'$(build_pooling_cache_config)'"

# ------------------------------------------------------------------------------
# Deployment
# ------------------------------------------------------------------------------

kubectl create namespace ${NAMESPACE} 2>/dev/null || true

set -o pipefail

echo "INFO: Deploying vLLM Environment in namespace ${NAMESPACE}, ${APP_NAME}"

# envsubst must run before kubectl kustomize so that patch name: selectors
# (which kustomize evaluates at build time) resolve to actual resource names.
KUST_TMP=$(mktemp -d)
trap 'rm -rf "$KUST_TMP"' EXIT

find deploy -type f | while read -r f; do
  dest="${KUST_TMP}/${f}"
  mkdir -p "$(dirname "$dest")"
  envsubst < "$f" > "$dest"
done

kubectl kustomize "${KUST_TMP}/deploy/environments/dev/io-disagg" | kubectl apply -f -

echo "INFO: Deploying Proxy and Endpoint Picker Environment in namespace ${NAMESPACE}, ${APP_NAME}"
kubectl -n "${NAMESPACE}" create configmap epp-config --from-file=epp-config.yaml="${KUST_TMP}/${EPP_CONFIG}" --dry-run=client -o yaml | kubectl apply -f -

kubectl kustomize "${KUST_TMP}/deploy/components/inference-gateway-minimal" | kubectl apply -f -

echo "INFO: Waiting for resources in namespace ${NAMESPACE} to become ready"
# Wait for gateway resources
kubectl -n "${NAMESPACE}" wait deployment/${EPP_NAME} --for=condition=Available --timeout=60s

# Wait for vllm resources
kubectl -n "${NAMESPACE}" wait deployment/${PRE_DEPLOYMENT_NAME} --for=condition=Available --timeout=240s
kubectl -n "${NAMESPACE}" wait deployment/${POST_DEPLOYMENT_NAME} --for=condition=Available --timeout=240s
kubectl -n "${NAMESPACE}" wait deployment/${INF_DEPLOYMENT_NAME} --for=condition=Available --timeout=240s

cat <<EOF
-----------------------------------------
Deployment to namespace ${NAMESPACE} completed!

Status:

* The vllm is running and exposed via label selector app: ${APP_NAME}.
* The Envoy proxy is exposing the model servers via service/inference-gateway-minimal.
* The Endpoint Picker is up and running.

You can watch the Endpoint Picker logs with:

  $ kubectl logs -f deployments/${EPP_NAME}

With that running in the background, you can make requests using port-forward and curl:

  $ kubectl port-forward service/inference-gateway-minimal 8080:80 -n "${NAMESPACE}" &
  $ curl -s -w '\n' http://localhost:8080/pooling -H 'Content-Type: application/json' -d '{
        "data": {
            "data": "${EXAMPLE_IMG_URL}",
            "data_format": "url",
            "image_format": "tiff",
            "out_data_format": "b64_json"
        },
        "model":"${MODEL_NAME}"
    }' | jq

Resources created in namespace ${NAMESPACE}:

  Deployments:       ${PRE_DEPLOYMENT_NAME} ${INF_DEPLOYMENT_NAME} ${POST_DEPLOYMENT_NAME} ${EPP_NAME}
  Services:          ${PRE_DEPLOYMENT_NAME} ${INF_DEPLOYMENT_NAME} ${POST_DEPLOYMENT_NAME} inference-gateway-minimal
  ConfigMaps:        envoy epp-config
  ServiceAccounts:   ${EPP_NAME}
  Roles:             ${EPP_NAME}
  RoleBindings:      ${EPP_NAME}-binding
  PVC:               disagg-data-pvc

To delete all of the above:

  kubectl -n ${NAMESPACE} delete deployment ${PRE_DEPLOYMENT_NAME} ${INF_DEPLOYMENT_NAME} ${POST_DEPLOYMENT_NAME} ${EPP_NAME}
  kubectl -n ${NAMESPACE} delete service ${PRE_DEPLOYMENT_NAME} ${INF_DEPLOYMENT_NAME} ${POST_DEPLOYMENT_NAME} inference-gateway-minimal
  kubectl -n ${NAMESPACE} delete configmap envoy epp-config
  kubectl -n ${NAMESPACE} delete serviceaccount ${EPP_NAME}
  kubectl -n ${NAMESPACE} delete role ${EPP_NAME}
  kubectl -n ${NAMESPACE} delete rolebinding ${EPP_NAME}-binding
  kubectl -n ${NAMESPACE} delete pvc disagg-data-pvc

-----------------------------------------
EOF