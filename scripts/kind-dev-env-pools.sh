#!/bin/bash

# Deploys a kind cluster for the coordinator-driven E/P/D-with-Pools topology:
# coordinator + vllm-render sidecar + mock downloaders + three phase-specific
# InferencePools (encode/prefill/decode), each backed by its own EPP.
# Use `make coordinator-epd-pools-env-dev-kind` to invoke this script.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------------------
# Topology-specific variables (resolved before sourcing the common library).
# ------------------------------------------------------------------------------

: "${COORDINATOR_HOST_PORT:=30081}"

# Pools topology defaults to a multimodal model.
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"

export COORDINATOR_IMAGE="${COORDINATOR_IMAGE:-ghcr.io/llm-d/llm-d-coordinator:dev}"
export MOCK_DOWNLOADER_HTTP_IMAGE="${MOCK_DOWNLOADER_HTTP_IMAGE:-python:3.10-slim}"
export MOCK_DOWNLOADER_INIT_IMAGE="${MOCK_DOWNLOADER_INIT_IMAGE:-busybox:1.36}"

# Pools use per-phase EPP configs, so the top-level connectors stay empty.
export CONNECTOR_TYPE="${CONNECTOR_TYPE:-}"
export KV_CONNECTOR_TYPE="${KV_CONNECTOR_TYPE:-}"
export EC_CONNECTOR_TYPE="${EC_CONNECTOR_TYPE:-}"

# EPP config for the shared top-level configmap (not phase-specific).
export EPP_CONFIG="${EPP_CONFIG:-deploy/config/sim-epp-config.yaml}"

# Inject the coordinator NodePort mapping into the kind config; the common
# library appends the Prometheus mapping (if enabled) before creating the cluster.
export EXTRA_PORT_MAPPINGS="  - containerPort: 30081
    hostPort: ${COORDINATOR_HOST_PORT}
    protocol: TCP"

# ------------------------------------------------------------------------------
# Common defaults + stage functions
# ------------------------------------------------------------------------------

. "${SCRIPT_DIR}/kind-dev-env-common.sh"

kind_setup_checks
kind_compute_target_ports
kind_create_cluster
kind_post_cluster_setup
kind_pull_and_load_images \
    "${VLLM_IMAGE}" "${EPP_IMAGE}" "${SIDECAR_IMAGE}" "${VLLM_RENDER_IMAGE}" \
    "${COORDINATOR_IMAGE}" "${MOCK_DOWNLOADER_HTTP_IMAGE}" "${MOCK_DOWNLOADER_INIT_IMAGE}"
kind_apply_standard_crds

# ------------------------------------------------------------------------------
# Development Environment
# ------------------------------------------------------------------------------

TEMP_FILE=$(mktemp)
trap "rm -f \"${TEMP_FILE}\"" EXIT

kubectl --context ${KUBE_CONTEXT} delete configmap epp-config --ignore-not-found
envsubst '$MODEL_NAME' < ${EPP_CONFIG} > ${TEMP_FILE}
kubectl --context ${KUBE_CONTEXT} create configmap epp-config --from-file=epp-config.yaml=${TEMP_FILE}

for phase in encode prefill decode; do
  kubectl --context ${KUBE_CONTEXT} delete configmap "epp-config-${phase}" --ignore-not-found
  envsubst '$MODEL_NAME' < "deploy/config/sim-epp-${phase}-config.yaml" > "${TEMP_FILE}"
  kubectl --context ${KUBE_CONTEXT} create configmap "epp-config-${phase}" \
    --from-file=epp-config.yaml="${TEMP_FILE}"
done

kubectl kustomize --enable-helm "deploy/environments/dev/base-kind-istio/epd-pools" \
  | envsubst '${POOL_NAME} ${MODEL_NAME} ${MODEL_NAME_SAFE} ${EPP_NAME} ${EPP_IMAGE} ${VLLM_IMAGE} \
  ${SIDECAR_IMAGE} ${VLLM_RENDER_IMAGE} ${TARGET_PORTS} ${NAMESPACE} ${METRICS_ENDPOINT_AUTH} \
${VLLM_REPLICA_COUNT_E} ${VLLM_REPLICA_COUNT_P} ${VLLM_REPLICA_COUNT_D} ${VLLM_DATA_PARALLEL_SIZE}' \
  | kubectl --context ${KUBE_CONTEXT} apply -f -

kubectl kustomize --enable-helm "deploy/environments/dev/coordinator-e-p-d-pools" \
  | envsubst '${POOL_NAME} ${MODEL_NAME} ${MODEL_NAME_SAFE} ${EPP_NAME} ${EPP_IMAGE} ${VLLM_IMAGE} \
  ${SIDECAR_IMAGE} ${VLLM_RENDER_IMAGE} ${TARGET_PORTS} ${NAMESPACE} ${METRICS_ENDPOINT_AUTH} \
  ${VLLM_REPLICA_COUNT_E} ${VLLM_REPLICA_COUNT_P} ${VLLM_REPLICA_COUNT_D} ${VLLM_DATA_PARALLEL_SIZE} \
  ${KV_CONNECTOR_TYPE} ${EC_CONNECTOR_TYPE} ${CONNECTOR_TYPE} ${KV_CACHE_ENABLED} ${HF_TOKEN} ${VLLM_SIM_MODE} \
  ${DECODE_ROLE} ${VLLM_EXTRA_ARGS_E} ${VLLM_EXTRA_ARGS_P} ${VLLM_EXTRA_ARGS_D} ${COORDINATOR_IMAGE}' \
  | awk '
    /^[[:space:]]*-[[:space:]]+"--[^"]*"[[:space:]]*$/ {
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

kind_wait_deployments_and_gateway
kind_deploy_prometheus_if_enabled

cat <<EOF
-----------------------------------------
Deployment completed!

* Kind Cluster Name: ${CLUSTER_NAME}
* Kubectl Context: ${KUBE_CONTEXT}

Status (coordinator-epd-pools-env-dev-kind):

* Coordinator drives the multimodal pipeline (replace-media-urls →
  render → encode → prefill → decode) and exposes :8080 via the
  llm-d-coordinator NodePort (host port ${COORDINATOR_HOST_PORT}).
* Three phase-specific InferencePools (encode/prefill/decode) are wired
  to per-phase EPPs via header-based HTTPRoutes (EPP-Phase: <phase>).
* mock-downloader1/2 stand in for media URLs.

Watch the coordinator logs with:

  \$ kubectl --context ${KUBE_CONTEXT} logs -f deployments/llm-d-coordinator -c coordinator

-----------------------------------------
EOF

if [ "${PROM_ENABLED}" == "true" ]; then
cat <<EOF

Monitoring:

* Prometheus: http://localhost:${PROM_HOST_PORT}

EOF
fi
