#!/bin/bash

# Deploys multiple inference pools that share a single set of pre/post process
# workers. Each pool gets its own EPP and inference deployment, labeled with a
# pool-specific app name. The shared pre/post pods carry a label for every pool
# so each EPP can reach them via its --endpoint-selector.

set -eu

# Pool definitions: "app_name:epp_name:epp_config:served_model_name:inf_replica_count:model_name"
# model_name is optional; omitting it falls back to SHARED_MODEL_NAME.
# inf_replica_count is optional; omitting it falls back to INF_REPLICA_COUNT below.
POOL_DEFS=(
  "terratorch-flood:epp-flood:deploy/config/epp-full-io-disagg-config.yaml:terratorch-flood:1:ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
)

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

# Builds the POOL_APP_LABELS block: one "        <app_name>: \"true\"" line per
# pool, indented to sit under the pod template labels: map (8 spaces).
build_pool_app_labels() {
  local labels=""
  for def in "${POOL_DEFS[@]}"; do
    local pool_app
    pool_app="$(echo "$def" | cut -d: -f1)"
    labels+="        ${pool_app}: \"true\""$'\n'
  done
  # Strip trailing newline so envsubst produces clean YAML
  printf '%s' "${labels%$'\n'}"
}

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

# Validate required inputs
if [[ -z "${NAMESPACE:-}" ]]; then
  echo "ERROR: NAMESPACE environment variable is not set."
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN environment variable is not set."
  exit 1
fi

export IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io/llm-d}"

export EXAMPLE_IMG_URL="https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/examples/India_900498_S2Hand.tif"

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

export SHARED_MODEL_NAME="${SHARED_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export MODEL_ID="${SHARED_MODEL_NAME##*/}"
export MODEL_NAME_SAFE=$(echo "${MODEL_ID}" | tr '[:upper:]' '[:lower:]' | tr ' /_.' '-')

export SHARED_APP_NAME="${SHARED_APP_NAME:-${MODEL_NAME_SAFE}-shared}"

# Build SHARED_SERVED_MODEL_NAME as a YAML args snippet: one value entry per
# unique name, collecting served_model_name (field 4) and model_name (field 6)
# from each pool. Indented at 12 spaces to sit inside the containers[0].args list.
_seen=()
_snippet=""
_add_name() {
  local _n="$1"
  [[ -z "$_n" ]] && return
  for _s in "${_seen[@]:-}"; do [[ "$_s" == "$_n" ]] && return; done
  _seen+=("$_n")
  _snippet+="            - \"${_n}\""$'\n'
}
for _def in "${POOL_DEFS[@]}"; do
  _add_name "$(echo "$_def" | cut -d: -f4)"
  _add_name "$(echo "$_def" | cut -d: -f6-)"
done
export SHARED_SERVED_MODEL_NAME="${_snippet%$'\n'}"
unset _seen _snippet _def
unset -f _add_name

# -----------------------------------------------------------------------------
# EPP Configuration
# -----------------------------------------------------------------------------

export EPP_TAG="${EPP_TAG:-v0.1.0}"
export EPP_IMAGE="${EPP_IMAGE:-${IMAGE_REGISTRY}/llm-d-router-endpoint-picker:${EPP_TAG}}"
export EPP_REPLICA_COUNT="${EPP_REPLICA_COUNT:-1}"
export EPP_LOG_LEVEL="\"${EPP_LOG_LEVEL:-2}\""
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

export IO_PROCESSOR_PLUGIN="${IO_PROCESSOR_PLUGIN:-terratorch_segmentation}"

# Shared preprocessing/postprocessing deployment names
export PRE_DEPLOYMENT_NAME="${PRE_DEPLOYMENT_NAME:-vllm-pre}"
export PRE_REPLICA_COUNT="${PRE_REPLICA_COUNT:-1}"
export PRE_ROLE="${PRE_ROLE:-preprocess}"

export PRE_CPU_RESOURCES="${PRE_CPU_RESOURCES:-4}"
export PRE_MEMORY_RESOURCES="${PRE_MEMORY_RESOURCES:-2G}"
export PRE_CPU_RESOURCES_LIM="${PRE_CPU_RESOURCES_LIM:-8}"
export PRE_MEMORY_RESOURCES_LIM="${PRE_MEMORY_RESOURCES_LIM:-4G}"

export POST_DEPLOYMENT_NAME="${POST_DEPLOYMENT_NAME:-vllm-post}"
export POST_REPLICA_COUNT="${POST_REPLICA_COUNT:-1}"
export POST_ROLE="${POST_ROLE:-postprocess}"

export POST_CPU_RESOURCES="${POST_CPU_RESOURCES:-4}"
export POST_MEMORY_RESOURCES="${POST_MEMORY_RESOURCES:-2G}"
export POST_CPU_RESOURCES_LIM="${POST_CPU_RESOURCES_LIM:-8}"
export POST_MEMORY_RESOURCES_LIM="${POST_MEMORY_RESOURCES_LIM:-4G}"

export POST_TARGET_PORT="${POST_TARGET_PORT:-8200}"
export POST_TARGET_PORT_STR="\"${POST_TARGET_PORT}\""

# Per-pool inference resource defaults (can be overridden per-pool below)
export INF_ROLE="${INF_ROLE:-inference}"
export INF_CPU_RESOURCES="${INF_CPU_RESOURCES:-32}"
export INF_MEMORY_RESOURCES="${INF_MEMORY_RESOURCES:-32G}"
export INF_GPU_RESOURCES="${INF_GPU_RESOURCES:-1}"
export INF_CPU_RESOURCES_LIM="${INF_CPU_RESOURCES_LIM:-32}"
export INF_MEMORY_RESOURCES_LIM="${INF_MEMORY_RESOURCES_LIM:-64G}"
export INF_GPU_RESOURCES_LIM="${INF_GPU_RESOURCES_LIM:-1}"

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

# Build POOL_APP_LABELS for injection into pre/post deployment templates
export POOL_APP_LABELS="$(build_pool_app_labels)"

# ------------------------------------------------------------------------------
# Deployment
# ------------------------------------------------------------------------------

kubectl create namespace "${NAMESPACE}" 2>/dev/null || true

set -o pipefail

# envsubst must run before kubectl kustomize so that patch name: selectors
# (which kustomize evaluates at build time) resolve to actual resource names.
KUST_TMP=$(mktemp -d)
trap 'rm -rf "$KUST_TMP"' EXIT

echo "INFO: Expanding templates into ${KUST_TMP}"
find deploy -type f | while read -r f; do
  dest="${KUST_TMP}/${f}"
  mkdir -p "$(dirname "$dest")"
  envsubst < "$f" > "$dest"
done

# ---------------------------------------------------------------------------
# Step 1: Deploy shared pre/post workers (once, with all pool labels)
# ---------------------------------------------------------------------------

echo "INFO: Deploying shared pre/post workers in namespace ${NAMESPACE}"
kubectl kustomize "${KUST_TMP}/deploy/environments/dev/io-disagg-multi-pool" | kubectl apply -f -

# Create the shared envoy configmap (used by all EPP deployments)
echo "INFO: Applying shared Envoy configmap"
kubectl -n "${NAMESPACE}" apply -f "${KUST_TMP}/deploy/components/inference-gateway-minimal/envoy-cm.yaml"

# ---------------------------------------------------------------------------
# Step 2: Deploy one inference deployment + EPP per pool
# ---------------------------------------------------------------------------

for def in "${POOL_DEFS[@]}"; do
  INF_APP_NAME="$(echo "$def"         | cut -d: -f1)"
  EPP_NAME="$(echo "$def"             | cut -d: -f2)"
  EPP_CONFIG="$(echo "$def"           | cut -d: -f3)"
  local_served="$(echo "$def"         | cut -d: -f4)"
  local_replicas="$(echo "$def"       | cut -d: -f5)"
  local_model="$(echo "$def"          | cut -d: -f6-)"

  # Build INF_SERVED_MODEL_NAME as a YAML args snippet (12-space indent),
  # deduplicating served_model_name and model_name when they are identical.
  INF_SERVED_MODEL_NAME="            - \"${local_served}\""
  if [[ -n "$local_model" && "$local_model" != "$local_served" ]]; then
    INF_SERVED_MODEL_NAME+=$'\n'"            - \"${local_model}\""
  fi

  # Inference deployment name: vllm-inf-<pool-app>
  INF_DEPLOYMENT_NAME="vllm-inf-${INF_APP_NAME}"

  # Per-pool overrides; fall back to shared defaults when unset
  export INF_REPLICA_COUNT="${local_replicas:-${INF_REPLICA_COUNT}}"
  export MODEL_NAME="${local_model:-${SHARED_MODEL_NAME}}"

  export INF_APP_NAME EPP_NAME EPP_CONFIG INF_SERVED_MODEL_NAME INF_DEPLOYMENT_NAME

  echo "INFO: Deploying pool '${INF_APP_NAME}'"

  # Re-expand templates with the per-pool variables now set
  POOL_TMP=$(mktemp -d)
  find deploy -type f | while read -r f; do
    dest="${POOL_TMP}/${f}"
    mkdir -p "$(dirname "$dest")"
    envsubst < "$f" > "$dest"
  done

  # Deploy inference deployment + EPP (service accounts, RBAC, services, deployment)
  kubectl kustomize "${POOL_TMP}/deploy/environments/dev/io-disagg-pool" | kubectl apply -f -
  kubectl kustomize "${POOL_TMP}/deploy/components/inference-gateway-pool" | kubectl apply -f -

  # Create per-pool EPP configmap
  kubectl -n "${NAMESPACE}" create configmap "${EPP_NAME}-config" \
    --from-file=epp-config.yaml="${POOL_TMP}/${EPP_CONFIG}" \
    --dry-run=client -o yaml | kubectl apply -f -

  rm -rf "$POOL_TMP"
done

# ---------------------------------------------------------------------------
# Step 3: Wait for everything to be ready
# ---------------------------------------------------------------------------

echo "INFO: Waiting for shared workers in namespace ${NAMESPACE} to become ready"
kubectl -n "${NAMESPACE}" wait deployment/"${PRE_DEPLOYMENT_NAME}" --for=condition=Available --timeout=240s
kubectl -n "${NAMESPACE}" wait deployment/"${POST_DEPLOYMENT_NAME}" --for=condition=Available --timeout=240s

for def in "${POOL_DEFS[@]}"; do
  INF_APP_NAME="$(echo "$def" | cut -d: -f1)"
  EPP_NAME="$(echo "$def"     | cut -d: -f2)"
  INF_DEPLOYMENT_NAME="vllm-inf-${INF_APP_NAME}"

  echo "INFO: Waiting for pool '${INF_APP_NAME}'"
  kubectl -n "${NAMESPACE}" wait deployment/"${EPP_NAME}"             --for=condition=Available --timeout=60s
  kubectl -n "${NAMESPACE}" wait deployment/"${INF_DEPLOYMENT_NAME}"  --for=condition=Available --timeout=240s
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

cat <<EOF
-----------------------------------------
Multi-pool deployment to namespace ${NAMESPACE} completed!

Shared workers:
  pre:  deployment/${PRE_DEPLOYMENT_NAME}
  post: deployment/${POST_DEPLOYMENT_NAME}
  served-model-name:
${SHARED_SERVED_MODEL_NAME}

Pools:
EOF

# Accumulate per-pool resource names for the teardown section.
_del_deployments="${PRE_DEPLOYMENT_NAME} ${POST_DEPLOYMENT_NAME}"
_del_services=""
_del_configmaps="envoy"
_del_serviceaccounts=""
_del_roles=""
_del_rolebindings=""

for def in "${POOL_DEFS[@]}"; do
  INF_APP_NAME="$(echo "$def" | cut -d: -f1)"
  EPP_NAME="$(echo "$def"     | cut -d: -f2)"
  INF_SERVED_MODEL_NAME="$(echo "$def" | cut -d: -f4)"
  INF_DEPLOYMENT_NAME="vllm-inf-${INF_APP_NAME}"

  _del_deployments="${_del_deployments} ${INF_DEPLOYMENT_NAME} ${EPP_NAME}"
  _del_services="${_del_services} ${INF_DEPLOYMENT_NAME} ${EPP_NAME}"
  _del_configmaps="${_del_configmaps} ${EPP_NAME}-config"
  _del_serviceaccounts="${_del_serviceaccounts} ${EPP_NAME}"
  _del_roles="${_del_roles} ${EPP_NAME}"
  _del_rolebindings="${_del_rolebindings} ${EPP_NAME}-binding"

  cat <<EOF
  pool: ${INF_APP_NAME}
    inference:  deployment/${INF_DEPLOYMENT_NAME}
    epp:        deployment/${EPP_NAME}  (service/${EPP_NAME})
    model name: ${INF_SERVED_MODEL_NAME}

    port-forward:
      kubectl port-forward service/${EPP_NAME} 8080:80 -n ${NAMESPACE} &
    request:
      curl -s http://localhost:8080/pooling -H 'Content-Type: application/json' -d '{
          "data": {"data": "${EXAMPLE_IMG_URL}", "data_format": "url", "image_format": "tiff", "out_data_format": "b64_json"},
          "model": "${INF_SERVED_MODEL_NAME}"
      }' | jq

EOF
done

cat <<EOF
Resources created in namespace ${NAMESPACE}:

  Deployments:       ${_del_deployments# }
  Services:          ${_del_services# }
  ConfigMaps:        ${_del_configmaps}
  ServiceAccounts:   ${_del_serviceaccounts# }
  Roles:             ${_del_roles# }
  RoleBindings:      ${_del_rolebindings# }
  PVC:               disagg-data-pvc

To delete all of the above:

  kubectl -n ${NAMESPACE} delete deployment ${_del_deployments# }
  kubectl -n ${NAMESPACE} delete service ${_del_services# }
  kubectl -n ${NAMESPACE} delete configmap ${_del_configmaps}
  kubectl -n ${NAMESPACE} delete serviceaccount ${_del_serviceaccounts# }
  kubectl -n ${NAMESPACE} delete role ${_del_roles# }
  kubectl -n ${NAMESPACE} delete rolebinding ${_del_rolebindings# }
  kubectl -n ${NAMESPACE} delete pvc disagg-data-pvc

-----------------------------------------
EOF
