#!/bin/bash

# This shell script deploys a kind cluster with an Istio-based Gateway API
# implementation fully configured. It deploys the vllm simulator, which it
# exposes with a Gateway -> HTTPRoute -> InferencePool. The Gateway is
# configured with the a filter for the ext_proc endpoint picker.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------------------
# Topology-specific variables (resolved before sourcing the common library so
# they can drive the model and EPP selections below).
# ------------------------------------------------------------------------------

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

# When Encode disaggregation is enabled (multimodal pipeline), default to a
# multimodal model. Otherwise use the standard text-only model.
if [ "${DISAGG_E}" == "true" ]; then
  export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
else
  export MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
fi

export EXTERNAL_TOKENIZER_ENABLED="${EXTERNAL_TOKENIZER_ENABLED:-false}"

# KV connector: needed when P is disaggregated (P/D or E/P/D).
if [ "${DISAGG_P}" == "true" ]; then
  export CONNECTOR_TYPE="${CONNECTOR_TYPE:-nixlv2}"
  export KV_CONNECTOR_TYPE="${KV_CONNECTOR_TYPE:-nixlv2}"
else
  export CONNECTOR_TYPE="${CONNECTOR_TYPE:-}"
  export KV_CONNECTOR_TYPE="${KV_CONNECTOR_TYPE:-}"
fi
# EC connector: needed when E is disaggregated (E/PD or E/P/D).
if [ "${DISAGG_E}" == "true" ]; then
  export EC_CONNECTOR_TYPE="${EC_CONNECTOR_TYPE:-ec-example}"
else
  export EC_CONNECTOR_TYPE="${EC_CONNECTOR_TYPE:-}"
fi

# EPP config selection. KV cache and external tokenizer are independent options
# that work with any disaggregation mode. KV_CACHE_ENABLED is read with a
# default here because the common library has not yet been sourced.
if [ "${EXTERNAL_TOKENIZER_ENABLED}" == "true" ]; then
  DEFAULT_EPP_CONFIG="deploy/config/sim-epp-external-tokenizer-config.yaml"
elif [ "${KV_CACHE_ENABLED:-false}" == "true" ]; then
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

# ------------------------------------------------------------------------------
# Common defaults + stage functions
# ------------------------------------------------------------------------------

. "${SCRIPT_DIR}/kind-dev-env-common.sh"

kind_setup_checks
kind_compute_target_ports
kind_create_cluster
kind_post_cluster_setup
kind_pull_and_load_images "${VLLM_IMAGE}" "${EPP_IMAGE}" "${SIDECAR_IMAGE}" "${VLLM_RENDER_IMAGE}"
kind_apply_standard_crds

# ------------------------------------------------------------------------------
# Development Environment
# ------------------------------------------------------------------------------

TEMP_FILE=$(mktemp)
# Ensure that the temporary file is deleted now matter what happens in the script
trap "rm -f \"${TEMP_FILE}\"" EXIT

kubectl --context ${KUBE_CONTEXT} delete configmap epp-config --ignore-not-found
envsubst '$MODEL_NAME' < ${EPP_CONFIG} > ${TEMP_FILE}
kubectl --context ${KUBE_CONTEXT} create configmap epp-config --from-file=epp-config.yaml=${TEMP_FILE}

kubectl kustomize --enable-helm "deploy/environments/dev/base-kind-istio/single-pool" \
  | envsubst '${POOL_NAME} ${MODEL_NAME} ${MODEL_NAME_SAFE} ${EPP_NAME} ${EPP_IMAGE} ${VLLM_IMAGE} \
  ${SIDECAR_IMAGE} ${VLLM_RENDER_IMAGE} ${TARGET_PORTS} ${NAMESPACE} ${METRICS_ENDPOINT_AUTH} \
${VLLM_REPLICA_COUNT_E} ${VLLM_REPLICA_COUNT_P} ${VLLM_REPLICA_COUNT_D} ${VLLM_DATA_PARALLEL_SIZE}' \
  | kubectl --context ${KUBE_CONTEXT} apply -f -

# Deploy scenario-specific vLLM components
kubectl kustomize --enable-helm ${KUSTOMIZE_DIR} \
  | envsubst '${POOL_NAME} ${MODEL_NAME} ${MODEL_NAME_SAFE} ${EPP_NAME} ${EPP_IMAGE} ${VLLM_IMAGE} \
  ${SIDECAR_IMAGE} ${VLLM_RENDER_IMAGE} ${TARGET_PORTS} ${NAMESPACE} ${METRICS_ENDPOINT_AUTH} \
  ${VLLM_REPLICA_COUNT_E} ${VLLM_REPLICA_COUNT_P} ${VLLM_REPLICA_COUNT_D} ${VLLM_DATA_PARALLEL_SIZE} \
  ${KV_CONNECTOR_TYPE} ${EC_CONNECTOR_TYPE} ${CONNECTOR_TYPE} ${KV_CACHE_ENABLED} ${HF_TOKEN} ${VLLM_SIM_MODE} \
  ${DECODE_ROLE} ${VLLM_EXTRA_ARGS_E} ${VLLM_EXTRA_ARGS_P} ${VLLM_EXTRA_ARGS_D}' \
  | awk '
    # Match only flag-shaped quoted list items ("--foo" or "--foo --bar"); leave
    # other quoted lists (e.g. RBAC apiGroups: - "") alone so legitimate
    # empty-string entries are not silently dropped.
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
