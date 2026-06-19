#!/usr/bin/env bash

# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_ROOT=$(dirname "${BASH_SOURCE}")/..
GATEWAY_API_VERSION="${GATEWAY_API_VERSION:-v1.5.1}"
GKE_GATEWAY_API_VERSION="${GKE_GATEWAY_API_VERSION:-v1.4.0}"
GIE_VERSION="${GIE_VERSION:-v1.5.0}"
HELM="${HELM:-${SCRIPT_ROOT}/bin/helm}"
KUBECTL_VALIDATE="${KUBECTL_VALIDATE:-${SCRIPT_ROOT}/bin/kubectl-validate}"
TEMP_DIR=$(mktemp -d)

make kubectl-validate

cleanup() {
  rm -rf "${TEMP_DIR}" || true
}
trap cleanup EXIT

fetch_crds() {
  local url="$1"
  curl -sL "${url}" -o "${TEMP_DIR}/$(basename "${url}")"
}

# Use local 'config/crd/bases', run "make generate" to regenerate llm-d CRDs
cp "${SCRIPT_ROOT}/config/crd/bases/"*.yaml "${TEMP_DIR}/"
# GIE (Gateway API Inference Extension) CRDs - owned by upstream GIE
fetch_crds "https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${GIE_VERSION}/config/crd/bases/inference.networking.k8s.io_inferencepools.yaml"
fetch_crds "https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${GIE_VERSION}/config/crd/bases/inference.networking.x-k8s.io_inferencepoolimports.yaml"
# GW API CRD
fetch_crds "https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/refs/tags/${GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_httproutes.yaml"
# GKE CRD
fetch_crds "https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/refs/tags/${GKE_GATEWAY_API_VERSION}/config/crd/networking.gke.io_gcpbackendpolicies.yaml"
fetch_crds "https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/refs/tags/${GKE_GATEWAY_API_VERSION}/config/crd/networking.gke.io_healthcheckpolicies.yaml"

# Read the first argument, default to "ci" if not provided
MODE=${1:-ci}

if [ "$MODE" == "local" ]; then
  # Local Mode: Permissive. Updates lock file automatically.
  DEP_CMD="update"
  echo "🔸 MODE: Local (Dev) - Using 'helm dependency update'"
else
  # CI/CD Mode (Default): Strict. Fails if lock file is out of sync.
  DEP_CMD="build"
  echo "🔹 MODE: CI/CD (Strict) - Using 'helm dependency build'"
fi

declare -A test_cases_llm_d_router_gateway

# llm_d_router_gateway Helm Chart test cases
test_cases_llm_d_router_gateway["basic"]="--set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_gateway["gke-provider"]="--set provider.name=gke --set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_gateway["multiple-replicas"]="--set router.replicas=3 --set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_gateway["latency-predictor"]="--set router.latencyPredictor.enabled=true --set router.modelServers.matchLabels.app=llm-instance-gateway"

# Run the install command in case this script runs from a different bash
# source (such as in the verify-all script)
make helm-install

echo "Processing dependencies for llm-d-router-gateway chart..."
${HELM} dependency ${DEP_CMD} ${SCRIPT_ROOT}/config/charts/llm-d-router-gateway
if [ $? -ne 0 ]; then
  echo "Helm dependency ${DEP_CMD} failed."
  exit 1
fi

# Running tests cases
echo "Running helm template command for llm-d-router-gateway chart..."
# Loop through the keys of the associative array
for key in "${!test_cases_llm_d_router_gateway[@]}"; do
  echo "Running test: ${key}"
  output_dir="${SCRIPT_ROOT}/bin/llm-d-router-gateway-${key}"
  command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-gateway ${test_cases_llm_d_router_gateway[$key]} --output-dir=${output_dir}"
  echo "Executing: ${command}"
  ${command}
  if [ $? -ne 0 ]; then
    echo "Helm template command failed for test: ${key}"
    exit 1
  fi

  ${KUBECTL_VALIDATE} ${output_dir} --local-crds "${TEMP_DIR}"
  if [ $? -ne 0 ]; then
    echo "Kubectl validation failed for test: ${key}"
    exit 1
  fi

  if [ "${key}" == "triton" ]; then
    if ! grep -q "passthrough-parser" "${output_dir}/llm-d-router-gateway/templates/inferenceextension.yaml"; then
      echo "Validation failed: passthrough-parser not found in rendered output for test: ${key}"
      exit 1
    fi
  fi

  echo "Test case ${key} passed validation."
done

declare -A test_cases_llm_d_router_standalone

# llm_d_router_standalone Helm Chart test cases
test_cases_llm_d_router_standalone["basic"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false"
test_cases_llm_d_router_standalone["gke-provider"]="--set provider.name=gke --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false"
test_cases_llm_d_router_standalone["latency-predictor"]="--set router.latencyPredictor.enabled=true --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false"
test_cases_llm_d_router_standalone["llm-d-router-gateway"]="--set router.inferencePool.create=true --set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_standalone["agentgateway"]="--set router.proxy.proxyType=agentgateway --set router.proxy.agentgateway.service.name=llm-instance-gateway --set 'router.proxy.agentgateway.service.ports[0]=8000' --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000'"
test_cases_llm_d_router_standalone["proxy-service"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.replicas=3"
test_cases_llm_d_router_standalone["triton"]="--set router.modelServers.type=triton --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false"


echo "Processing dependencies for llm-d-router-standalone chart..."
${HELM} dependency ${DEP_CMD} ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone
if [ $? -ne 0 ]; then
  echo "Helm dependency ${DEP_CMD} failed."
  exit 1
fi

# Running tests cases
echo "Running helm template command for llm-d-router-standalone chart..."
# Loop through the keys of the associative array
for key in "${!test_cases_llm_d_router_standalone[@]}"; do
  echo "Running test: ${key}"
  output_dir="${SCRIPT_ROOT}/bin/llm-d-router-standalone-${key}"
  command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone ${test_cases_llm_d_router_standalone[$key]} --output-dir=${output_dir}"
  echo "Executing: ${command}"
  ${command}
  if [ $? -ne 0 ]; then
    echo "Helm template command failed for test: ${key}"
    exit 1
  fi
  ${KUBECTL_VALIDATE} ${output_dir} --local-crds "${TEMP_DIR}"
  if [ $? -ne 0 ]; then
    echo "Kubectl validation failed for test: ${key}"
    exit 1
  fi
  echo "Test case ${key} passed validation."
done

echo "Running llm-d-router-standalone negative validation tests..."
missing_endpoint_selector_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.inferencePool.create=false --set router.modelServers.type=vllm --set 'router.modelServers.targetPorts[0].number=8000' >/dev/null"
echo "Executing: ${missing_endpoint_selector_command}"
if eval "${missing_endpoint_selector_command}"; then
  echo "Helm template unexpectedly succeeded for inferencePool.create=false without modelServers.matchLabels"
  exit 1
fi

invalid_proxy_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.proxyType=bogus >/dev/null"
echo "Executing: ${invalid_proxy_command}"
if eval "${invalid_proxy_command}"; then
  echo "Helm template unexpectedly succeeded for invalid proxyType"
  exit 1
fi

missing_agentgateway_service_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.proxyType=agentgateway >/dev/null"
echo "Executing: ${missing_agentgateway_service_command}"
if eval "${missing_agentgateway_service_command}"; then
  echo "Helm template unexpectedly succeeded for missing agentgateway service.name"
  exit 1
fi

unsupported_agentgateway_llm_d_router_gateway_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.proxy.proxyType=agentgateway --set router.proxy.agentgateway.service.name=llm-instance-gateway --set 'router.proxy.agentgateway.service.ports[0]=8000' --set router.inferencePool.create=true --set router.modelServers.matchLabels.app=llm-instance-gateway >/dev/null"
echo "Executing: ${unsupported_agentgateway_llm_d_router_gateway_command}"
if eval "${unsupported_agentgateway_llm_d_router_gateway_command}"; then
  echo "Helm template unexpectedly succeeded for unsupported agentgateway createInferencePool=true configuration"
  exit 1
fi

mismatched_agentgateway_ports_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.proxy.proxyType=agentgateway --set router.proxy.agentgateway.service.name=llm-instance-gateway --set 'router.proxy.agentgateway.service.ports[0]=8001' --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000' >/dev/null"
echo "Executing: ${mismatched_agentgateway_ports_command}"
if eval "${mismatched_agentgateway_ports_command}"; then
  echo "Helm template unexpectedly succeeded for mismatched agentgateway service.ports"
  exit 1
fi

unsupported_agentgateway_listener_port_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.proxy.proxyType=agentgateway --set router.proxy.agentgateway.service.name=llm-instance-gateway --set 'router.proxy.agentgateway.service.ports[0]=8000' --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000' --set 'router.extraServicePorts[0].name=proxy' --set 'router.extraServicePorts[0].port=9000' --set 'router.extraServicePorts[0].protocol=TCP' --set 'router.extraServicePorts[0].targetPort=9000' >/dev/null"
echo "Executing: ${unsupported_agentgateway_listener_port_command}"
if eval "${unsupported_agentgateway_listener_port_command}"; then
  echo "Helm template unexpectedly succeeded without an agentgateway listener Service port named http"
  exit 1
fi

mismatched_agentgateway_listener_target_port_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.proxy.proxyType=agentgateway --set router.proxy.agentgateway.service.name=llm-instance-gateway --set 'router.proxy.agentgateway.service.ports[0]=8000' --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000' --set 'router.extraServicePorts[0].name=http' --set 'router.extraServicePorts[0].port=9000' --set 'router.extraServicePorts[0].protocol=TCP' --set 'router.extraServicePorts[0].targetPort=9001' >/dev/null"
echo "Executing: ${mismatched_agentgateway_listener_target_port_command}"
if eval "${mismatched_agentgateway_listener_target_port_command}"; then
  echo "Helm template unexpectedly succeeded for an agentgateway listener targetPort that does not match port"
  exit 1
fi

unsupported_proxy_service_agentgateway_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.proxyType=agentgateway --set router.proxy.agentgateway.service.name=llm-instance-gateway --set 'router.proxy.agentgateway.service.ports[0]=8000' --set 'router.modelServers.targetPorts[0].number=8000' >/dev/null"
echo "Executing: ${unsupported_proxy_service_agentgateway_command}"
if eval "${unsupported_proxy_service_agentgateway_command}"; then
  echo "Helm template unexpectedly succeeded for proxy mode=service with proxyType=agentgateway"
  exit 1
fi

invalid_failopen_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.failOpen=notabool >/dev/null"
echo "Executing: ${invalid_failopen_command}"
if eval "${invalid_failopen_command}"; then
  echo "Helm template unexpectedly succeeded for non-boolean router.proxy.failOpen"
  exit 1
fi

echo "Verifying llm-d-router-standalone extra flags render as --flag=value..."
flag_render_output="${TEMP_DIR}/llm-d-router-standalone-flag-render.yaml"
flag_render_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set-string router.epp.flags.secure-serving=false > ${flag_render_output}"
echo "Executing: ${flag_render_command}"
eval "${flag_render_command}"
if ! grep -q -- '--secure-serving=false' "${flag_render_output}"; then
  echo "Helm template did not render extra flags as --flag=value"
  exit 1
fi

echo "Verifying llm-d-router-standalone agentgateway renders plaintext EPP and custom listener ports..."
agentgateway_render_output="${TEMP_DIR}/llm-d-router-standalone-agentgateway-render.yaml"
agentgateway_render_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.proxy.proxyType=agentgateway --set router.proxy.agentgateway.service.name=llm-instance-gateway --set 'router.proxy.agentgateway.service.ports[0]=8000' --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000' --set 'router.extraServicePorts[0].name=http' --set 'router.extraServicePorts[0].port=9000' --set 'router.extraServicePorts[0].protocol=TCP' --set 'router.extraServicePorts[0].targetPort=http' > ${agentgateway_render_output}"
echo "Executing: ${agentgateway_render_command}"
eval "${agentgateway_render_command}"
if ! grep -q -- '--secure-serving=false' "${agentgateway_render_output}"; then
  echo "Agentgateway Helm template did not render plaintext EPP serving"
  exit 1
fi
if ! grep -q -- 'containerPort: 9000' "${agentgateway_render_output}"; then
  echo "Agentgateway Helm template did not render the custom listener containerPort"
  exit 1
fi
if ! grep -A1 -- 'containerPort: 9000' "${agentgateway_render_output}" | grep -q -- 'name: http'; then
  echo "Agentgateway Helm template did not render the listener containerPort named http"
  exit 1
fi
if ! grep -q -- '    - port: 9000' "${agentgateway_render_output}"; then
  echo "Agentgateway Helm template did not render the custom listener bind port"
  exit 1
fi
if ! grep -q -- 'destinationMode: passthrough' "${agentgateway_render_output}"; then
  echo "Agentgateway Helm template did not render passthrough destination mode"
  exit 1
fi

agentgateway_service_block="${TEMP_DIR}/llm-d-router-standalone-agentgateway-service.yaml"
sed -n '/^# Source: llm-d-router-standalone\/templates\/agentgateway-service.yaml/,/^---/p' "${agentgateway_render_output}" > "${agentgateway_service_block}"
if ! grep -q -- 'app.kubernetes.io/component: agentgateway-model-service' "${agentgateway_service_block}"; then
  echo "Agentgateway model Service did not render its component label"
  exit 1
fi
agentgateway_selector_block="${TEMP_DIR}/llm-d-router-standalone-agentgateway-service-selector.yaml"
sed -n '/^  selector:$/,/^  ports:$/p' "${agentgateway_service_block}" > "${agentgateway_selector_block}"
if ! grep -Eq -- '^[[:space:]]+"?app"?:[[:space:]]+"?llm-instance-gateway"?[[:space:]]*$' "${agentgateway_selector_block}"; then
  echo "Agentgateway model Service did not render selector labels from router.modelServers.matchLabels"
  exit 1
fi
if grep -q -- 'app.kubernetes.io/name:' "${agentgateway_service_block}"; then
  echo "Agentgateway model Service rendered an app.kubernetes.io/name label"
  exit 1
fi

echo "Verifying llm-d-router-standalone proxy mode=service renders a separate proxy Deployment and Service..."
proxy_service_render_output="${TEMP_DIR}/llm-d-router-standalone-proxy-service-render.yaml"
proxy_service_render_command="${HELM} template proxy-svc ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --namespace proxy-ns --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.replicas=3 > ${proxy_service_render_output}"
echo "Executing: ${proxy_service_render_command}"
eval "${proxy_service_render_command}"
if ! grep -q -- 'name: proxy-svc-proxy' "${proxy_service_render_output}"; then
  echo "Proxy service mode did not render the separate proxy Deployment/Service named proxy-svc-proxy"
  exit 1
fi
if ! grep -q -- 'replicas: 3' "${proxy_service_render_output}"; then
  echo "Proxy service mode did not honor router.proxy.replicas"
  exit 1
fi
if ! grep -q -- 'type: STRICT_DNS' "${proxy_service_render_output}"; then
  echo "Proxy service mode did not switch the ext_proc cluster to STRICT_DNS"
  exit 1
fi
if ! grep -q -- 'address: proxy-svc-epp.proxy-ns.svc.cluster.local' "${proxy_service_render_output}"; then
  echo "Proxy service mode did not point ext_proc at the EPP Service FQDN"
  exit 1
fi
if ! grep -q -- 'failure_mode_allow: true' "${proxy_service_render_output}"; then
  echo "Proxy service mode did not enable fail-open on the ext_proc filter"
  exit 1
fi
# The proxy listener must be exposed by the separate proxy Service.
if ! grep -q -- 'port: 8081' "${proxy_service_render_output}"; then
  echo "Proxy service mode did not expose the proxy listener Service port"
  exit 1
fi
# In service mode the envoy-proxy container lives only in the standalone proxy
# Deployment, never as a sidecar in the EPP Deployment, so it appears once.
proxy_container_count=$(grep -c -- '- name: envoy-proxy$' "${proxy_service_render_output}")
if [ "${proxy_container_count}" -ne 1 ]; then
  echo "Proxy service mode rendered ${proxy_container_count} envoy-proxy containers, expected exactly 1 (in the proxy Deployment)"
  exit 1
fi
# The proxy Deployment selector/labels must be distinct from the EPP selector.
if ! grep -q -- 'llm-d-router-proxy: proxy-svc-proxy' "${proxy_service_render_output}"; then
  echo "Proxy service mode did not render the dedicated proxy selector label"
  exit 1
fi
