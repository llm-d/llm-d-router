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
GATEWAY_API_VERSION="${GATEWAY_API_VERSION:-v1.6.2}"
GKE_GATEWAY_API_VERSION="${GKE_GATEWAY_API_VERSION:-v1.4.0}"
GIE_VERSION="${GIE_VERSION:-v1.6.2}"
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
test_cases_llm_d_router_gateway["active-active-redis"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.epp.pluginsConfigFile=payload-agnostic.yaml --set router.epp.crossReplica.enabled=true --set router.epp.crossReplica.redis.address=redis.example:6379 --set router.epp.crossReplica.redis.passwordSecretRef.name=redis-auth --set router.epp.crossReplica.redis.passwordSecretRef.key=password --set router.epp.replicas=3"
test_cases_llm_d_router_gateway["active-active-redis-hpa"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.epp.pluginsConfigFile=payload-agnostic.yaml --set router.epp.crossReplica.enabled=true --set router.epp.crossReplica.redis.address=redis.example:6379 --set router.epp.crossReplica.redis.passwordSecretRef.name=redis-auth --set router.epp.crossReplica.redis.passwordSecretRef.key=password --set router.epp.autoscaling.enabled=true --set router.epp.autoscaling.minReplicas=2"
test_cases_llm_d_router_gateway["basic-kubernetes-1.30"]="--kube-version 1.30.0 --set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_gateway["gke-provider"]="--set provider.name=gke --set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_gateway["pdb"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.epp.replicas=2 --set router.epp.podDisruptionBudget.enabled=true --set router.epp.podDisruptionBudget.minAvailable=1"
test_cases_llm_d_router_gateway["multiple-replicas"]="--set router.replicas=3 --set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_gateway["latency-predictor"]="--set router.latencyPredictor.enabled=true --set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_gateway["latency-predictor-service"]="--set router.latencyPredictor.enabled=true --set router.latencyPredictor.mode=service --set router.latencyPredictor.predictionServers.count=3 --set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_gateway["tokenizer-python"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.tokenizer.enabled=true --set router.tokenizer.modelName=test-model"
test_cases_llm_d_router_gateway["tokenizer-rust"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.tokenizer.enabled=true --set router.tokenizer.flavor=rust --set router.tokenizer.modelName=test-model"

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

  if [ "${key}" == "pdb" ]; then
    epp_yaml="${output_dir}/llm-d-router-gateway/templates/epp.yaml"
    if ! grep -q "kind: PodDisruptionBudget" "${epp_yaml}"; then
      echo "Validation failed: PodDisruptionBudget not found in rendered output for test: ${key}"
      exit 1
    fi
    if ! grep -q "minAvailable: 1" "${epp_yaml}"; then
      echo "Validation failed: PodDisruptionBudget minAvailable not rendered for test: ${key}"
      exit 1
    fi
  fi

  if [ "${key}" == "latency-predictor-service" ]; then
    epp_yaml="${output_dir}/llm-d-router-gateway/templates/epp.yaml"
    for resource in Deployment/release-name-latency-training Service/release-name-latency-training \
                    Deployment/release-name-latency-prediction Service/release-name-latency-prediction; do
      # Match kind and metadata.name inside the same YAML document.
      if ! awk -v kind="${resource%%/*}" -v name="${resource#*/}" '
        /^---/ { if (k && n) found = 1; k = 0; n = 0; next }
        $0 == "kind: " kind { k = 1 }
        $0 == "  name: " name { n = 1 }
        END { if (k && n) found = 1; exit !found }' "${epp_yaml}"; then
        echo "Validation failed: ${resource} not found in rendered output for test: ${key}"
        exit 1
      fi
    done
    if ! grep -q 'value: "http://release-name-latency-prediction.default.svc.cluster.local:8001"' "${epp_yaml}"; then
      echo "Validation failed: EPP PREDICTION_SERVER_URL does not point at the prediction Service for test: ${key}"
      exit 1
    fi
    if grep -q "localhost:8001" "${epp_yaml}"; then
      echo "Validation failed: sidecar prediction URL rendered in service mode for test: ${key}"
      exit 1
    fi
  fi

  if [ "${key}" == "tokenizer-rust" ]; then
    if ! grep -q "vllm-rs" "${output_dir}/llm-d-router-gateway/templates/epp.yaml"; then
      echo "Validation failed: vllm-rs not found in rendered output for test: ${key}"
      exit 1
    fi
  fi
  if [ "${key}" == "tokenizer-python" ]; then
    if ! grep -q "vllm" "${output_dir}/llm-d-router-gateway/templates/epp.yaml" || ! grep -q "launch" "${output_dir}/llm-d-router-gateway/templates/epp.yaml"; then
      echo "Validation failed: vllm launch not found in rendered output for test: ${key}"
      exit 1
    fi
  fi

  echo "Test case ${key} passed validation."
done

echo "Verifying latency predictor service mode keeps EPP config checksum and scheduling..."
lp_service_args="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.latencyPredictor.enabled=true --set router.latencyPredictor.mode=service"
epp_checksum() {
  ${HELM} template lp ${SCRIPT_ROOT}/config/charts/llm-d-router-gateway ${lp_service_args} "$@" \
    | awk '/^kind: Deployment/{d=1} /^---/{d=0} d && /^  name: lp-epp$/{e=1} e && /checksum\/config:/{print $2; exit}'
}
lp_base_checksum=$(epp_checksum)
if [ -z "${lp_base_checksum}" ] || [ "${lp_base_checksum}" != "$(epp_checksum --set router.latencyPredictor.predictionServers.config.UVICORN_WORKERS=4)" ]; then
  echo "Latency predictor service mode: a prediction-server-only config change altered the EPP config checksum"
  exit 1
fi
lp_tolerations_render="${TEMP_DIR}/latency-predictor-service-tolerations.yaml"
${HELM} template lp ${SCRIPT_ROOT}/config/charts/llm-d-router-gateway ${lp_service_args} \
  --set 'router.epp.tolerations[0].key=dedicated' --set 'router.epp.tolerations[0].operator=Exists' > "${lp_tolerations_render}"
if [ "$(grep -c -- '- key: dedicated' "${lp_tolerations_render}")" -ne 3 ]; then
  echo "Latency predictor service mode: predictor Deployments did not inherit router.epp.tolerations"
  exit 1
fi

echo "Verifying GKE Gateway monitoring defaults to Prometheus Operator..."
gke_monitoring_render_output="${TEMP_DIR}/llm-d-router-gateway-gke-monitoring-render.yaml"
gke_monitoring_render_command="${HELM} template gke-monitoring ${SCRIPT_ROOT}/config/charts/llm-d-router-gateway --set provider.name=gke --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.monitoring.prometheus.enabled=true --set router.monitoring.prometheus.auth.enabled=false > ${gke_monitoring_render_output}"
echo "Executing: ${gke_monitoring_render_command}"
eval "${gke_monitoring_render_command}"
if ! grep -q -- '^kind: ServiceMonitor$' "${gke_monitoring_render_output}"; then
  echo "GKE Gateway monitoring did not render a ServiceMonitor when the monitoring provider was unset"
  exit 1
fi
if grep -q -- '^kind: PodMonitoring$' "${gke_monitoring_render_output}"; then
  echo "GKE Gateway monitoring unexpectedly rendered PodMonitoring when the monitoring provider was unset"
  exit 1
fi
if grep -Eq -- '^kind: ClusterRole(Binding)?$' "${gke_monitoring_render_output}"; then
  echo "GKE Gateway monitoring unexpectedly rendered cluster RBAC when Prometheus authentication was disabled"
  exit 1
fi

echo "Verifying Prometheus authentication renders cluster RBAC..."
prometheus_auth_render_output="${TEMP_DIR}/llm-d-router-gateway-prometheus-auth-render.yaml"
prometheus_auth_render_command="${HELM} template prometheus-auth ${SCRIPT_ROOT}/config/charts/llm-d-router-gateway --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.monitoring.prometheus.enabled=true --set router.monitoring.prometheus.auth.enabled=true > ${prometheus_auth_render_output}"
echo "Executing: ${prometheus_auth_render_command}"
eval "${prometheus_auth_render_command}"
if ! grep -q -- '^kind: ClusterRole$' "${prometheus_auth_render_output}"; then
  echo "Prometheus authentication did not render a ClusterRole"
  exit 1
fi
if ! grep -q -- '^kind: ClusterRoleBinding$' "${prometheus_auth_render_output}"; then
  echo "Prometheus authentication did not render a ClusterRoleBinding"
  exit 1
fi

echo "Verifying explicit GMP monitoring renders PodMonitoring..."
gmp_monitoring_render_output="${TEMP_DIR}/llm-d-router-gateway-gmp-monitoring-render.yaml"
gmp_monitoring_render_command="${HELM} template gmp-monitoring ${SCRIPT_ROOT}/config/charts/llm-d-router-gateway --set provider.name=gke --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.monitoring.provider.name=gmp --set router.monitoring.prometheus.enabled=true --set router.monitoring.prometheus.auth.enabled=true > ${gmp_monitoring_render_output}"
echo "Executing: ${gmp_monitoring_render_command}"
eval "${gmp_monitoring_render_command}"
if ! grep -q -- '^kind: PodMonitoring$' "${gmp_monitoring_render_output}"; then
  echo "Explicit GMP monitoring did not render PodMonitoring"
  exit 1
fi
if grep -q -- '^kind: ServiceMonitor$' "${gmp_monitoring_render_output}"; then
  echo "Explicit GMP monitoring unexpectedly rendered ServiceMonitor"
  exit 1
fi

declare -A test_cases_llm_d_router_standalone

# llm_d_router_standalone Helm Chart test cases
test_cases_llm_d_router_standalone["basic"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false"
test_cases_llm_d_router_standalone["pdb"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.epp.replicas=2 --set router.epp.podDisruptionBudget.enabled=true --set router.epp.podDisruptionBudget.minAvailable=1"
test_cases_llm_d_router_standalone["pdb-proxy-service"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.podDisruptionBudget.enabled=true --set router.proxy.podDisruptionBudget.maxUnavailable=1"
test_cases_llm_d_router_standalone["active-active-redis"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.epp.pluginsConfigFile=payload-agnostic.yaml --set router.epp.crossReplica.enabled=true --set router.epp.crossReplica.redis.address=redis.example:6379 --set router.epp.crossReplica.redis.passwordSecretRef.name=redis-auth --set router.epp.crossReplica.redis.passwordSecretRef.key=password --set router.epp.replicas=3"
test_cases_llm_d_router_standalone["active-active-redis-hpa"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.epp.pluginsConfigFile=payload-agnostic.yaml --set router.epp.crossReplica.enabled=true --set router.epp.crossReplica.redis.address=redis.example:6379 --set router.epp.crossReplica.redis.passwordSecretRef.name=redis-auth --set router.epp.crossReplica.redis.passwordSecretRef.key=password --set router.epp.autoscaling.enabled=true --set router.epp.autoscaling.minReplicas=2"
test_cases_llm_d_router_standalone["gke-provider"]="--set provider.name=gke --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false"
test_cases_llm_d_router_standalone["latency-predictor"]="--set router.latencyPredictor.enabled=true --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false"
test_cases_llm_d_router_standalone["latency-predictor-service"]="--set router.latencyPredictor.enabled=true --set router.latencyPredictor.mode=service --set router.latencyPredictor.predictionServers.count=3 --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false"
test_cases_llm_d_router_standalone["llm-d-router-gateway"]="--set router.inferencePool.create=true --set router.modelServers.matchLabels.app=llm-instance-gateway"
test_cases_llm_d_router_standalone["agentgateway"]="--set router.proxy.proxyType=agentgateway --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000'"
test_cases_llm_d_router_standalone["proxy-service"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.replicas=3"
test_cases_llm_d_router_standalone["proxy-service-kubernetes-1.30"]="--kube-version 1.30.0 --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service"
test_cases_llm_d_router_standalone["lifecycle-override"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.epp.lifecycle.preStop.exec.command[0]=/epp-drain --set router.proxy.lifecycle.preStop.exec.command[0]=/proxy-drain"
test_cases_llm_d_router_standalone["lifecycle-override-service"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.lifecycle.preStop.exec.command[0]=/proxy-drain"
test_cases_llm_d_router_standalone["proxy-autoscaling"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.autoscaling.enabled=true --set router.proxy.autoscaling.minReplicas=2 --set router.proxy.autoscaling.maxReplicas=6 --set router.proxy.autoscaling.targetCPUUtilizationPercentage=70"
test_cases_llm_d_router_standalone["agentgateway-service"]="--set router.proxy.proxyType=agentgateway --set router.proxy.mode=service --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000'"
test_cases_llm_d_router_standalone["triton"]="--set router.modelServers.type=triton --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false"
test_cases_llm_d_router_standalone["tokenizer-python"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.tokenizer.enabled=true --set router.tokenizer.modelName=test-model"
test_cases_llm_d_router_standalone["tokenizer-rust"]="--set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.tokenizer.enabled=true --set router.tokenizer.flavor=rust --set router.tokenizer.modelName=test-model"


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
  if [ "${key}" == "tokenizer-rust" ]; then
    if ! grep -q "vllm-rs" "${output_dir}/llm-d-router-standalone/templates/epp.yaml"; then
      echo "Validation failed: vllm-rs not found in rendered output for test: ${key}"
      exit 1
    fi
  fi
  if [ "${key}" == "tokenizer-python" ]; then
    if ! grep -q "vllm" "${output_dir}/llm-d-router-standalone/templates/epp.yaml" || ! grep -q "launch" "${output_dir}/llm-d-router-standalone/templates/epp.yaml"; then
      echo "Validation failed: vllm launch not found in rendered output for test: ${key}"
      exit 1
    fi
  fi
  if [ "${key}" == "pdb" ]; then
    epp_yaml="${output_dir}/llm-d-router-standalone/templates/epp.yaml"
    if ! grep -q "kind: PodDisruptionBudget" "${epp_yaml}"; then
      echo "Validation failed: PodDisruptionBudget not found in rendered output for test: ${key}"
      exit 1
    fi
    if ! grep -q "minAvailable: 1" "${epp_yaml}"; then
      echo "Validation failed: PodDisruptionBudget minAvailable not rendered for test: ${key}"
      exit 1
    fi
  fi
  if [ "${key}" == "pdb-proxy-service" ]; then
    epp_yaml="${output_dir}/llm-d-router-standalone/templates/epp.yaml"
    if ! grep -q "kind: PodDisruptionBudget" "${epp_yaml}"; then
      echo "Validation failed: proxy PodDisruptionBudget not found in service-mode render for test: ${key}"
      exit 1
    fi
    if ! grep -q "maxUnavailable: 1" "${epp_yaml}"; then
      echo "Validation failed: proxy PodDisruptionBudget maxUnavailable not rendered for test: ${key}"
      exit 1
    fi
  fi
  if [ "${key}" == "lifecycle-override" ]; then
    epp_yaml="${output_dir}/llm-d-router-standalone/templates/epp.yaml"
    for hook in /epp-drain /proxy-drain; do
      if ! grep -q -- "${hook}" "${epp_yaml}"; then
        echo "Validation failed: lifecycle hook ${hook} not found in rendered output for test: ${key}"
        exit 1
      fi
    done
    if grep -q "seconds: 5" "${epp_yaml}"; then
      echo "Validation failed: default preStop sleep rendered alongside a lifecycle override for test: ${key}"
      exit 1
    fi
  fi
  if [ "${key}" == "lifecycle-override-service" ]; then
    epp_yaml="${output_dir}/llm-d-router-standalone/templates/epp.yaml"
    if ! grep -q -- "/proxy-drain" "${epp_yaml}"; then
      echo "Validation failed: proxy lifecycle override not found in service-mode render for test: ${key}"
      exit 1
    fi
  fi
  if [ "${key}" == "proxy-autoscaling" ]; then
    if ! grep -q "name: release-name-proxy" "${output_dir}/llm-d-router-standalone/templates/epp.yaml" || ! grep -q "kind: HorizontalPodAutoscaler" "${output_dir}/llm-d-router-standalone/templates/epp.yaml"; then
      echo "Validation failed: proxy HorizontalPodAutoscaler not found in rendered output for test: ${key}"
      exit 1
    fi
  fi
  echo "Test case ${key} passed validation."
done

echo "Verifying PodDisruptionBudget validations..."
verify_pdb_validation_failure() {
  local description="$1"
  shift
  local output
  if output=$("${HELM}" template pdb-validation "${SCRIPT_ROOT}/config/charts/llm-d-router-standalone" \
      --namespace pdb-test --set router.modelServers.matchLabels.app=llm-instance-gateway \
      --set router.inferencePool.create=false "$@" 2>&1); then
    echo "PDB validation failed to reject: ${description}"
    exit 1
  fi
  if ! grep -q "podDisruptionBudget" <<<"${output}"; then
    echo "PDB validation rejected ${description} without naming the values path: ${output}"
    exit 1
  fi
}
verify_pdb_validation_failure "minAvailable and maxUnavailable together" \
  --set router.epp.replicas=2 --set router.epp.podDisruptionBudget.enabled=true \
  --set router.epp.podDisruptionBudget.minAvailable=1 --set router.epp.podDisruptionBudget.maxUnavailable=1
verify_pdb_validation_failure "enabled without a disruption field" \
  --set router.epp.replicas=2 --set router.epp.podDisruptionBudget.enabled=true
verify_pdb_validation_failure "minAvailable with a single replica" \
  --set router.epp.podDisruptionBudget.enabled=true --set router.epp.podDisruptionBudget.minAvailable=1
echo "PodDisruptionBudget validation checks passed."

echo "Verifying leader-election RBAC..."
verify_leader_election_rbac() {
  local chart="$1" expected="$2"
  shift 2
  local output="${TEMP_DIR}/${chart}-leader-election.yaml"
  local args=()
  if [ "${chart}" == "llm-d-router-standalone" ]; then
    args+=(--set router.inferencePool.create=false)
  fi
  if ! "${HELM}" template leader-election "${SCRIPT_ROOT}/config/charts/${chart}" \
    --namespace election-test --set router.modelServers.matchLabels.app=llm-instance-gateway \
    "${args[@]}" "$@" > "${output}"; then
    echo "Leader-election rendering failed for ${chart}: $*"
    exit 1
  fi
  local name actual
  for name in leader-election-epp-leader-election leader-election-epp-leader-election-binding; do
    if grep -q -- "^  name: ${name}$" "${output}"; then
      actual=true
    else
      actual=false
    fi
    if [ "${actual}" != "${expected}" ]; then
      echo "${chart}: expected ${name} present=${expected}, got ${actual}; flags: $*"
      exit 1
    fi
  done
  if [ "${expected}" == "true" ]; then
    if ! grep -Fq -- 'resources: [ "leases" ]' "${output}"; then
      echo "${chart}: leader-election Role is missing lease permissions"
      exit 1
    fi
  fi
}

for chart in llm-d-router-gateway llm-d-router-standalone; do
  verify_leader_election_rbac "${chart}" false
  verify_leader_election_rbac "${chart}" true --set router.epp.replicas=2
  verify_leader_election_rbac "${chart}" false --set router.epp.replicas=2 --set router.epp.flags.ha-enable-leader-election=false
  verify_leader_election_rbac "${chart}" true --set router.epp.replicas=1 --set router.epp.flags.ha-enable-leader-election=true
  verify_leader_election_rbac "${chart}" false --set router.epp.replicas=1 --set router.epp.flags.ha-enable-leader-election=false
  for value in true True TRUE t T 1; do
    verify_leader_election_rbac "${chart}" true --set-string "router.epp.flags.ha-enable-leader-election=${value}"
  done
  for value in false False FALSE f F 0; do
    verify_leader_election_rbac "${chart}" false --set-string "router.epp.flags.ha-enable-leader-election=${value}"
  done
  if [ "${chart}" == "llm-d-router-gateway" ]; then
    mode_flags=(--set provider.name=gke --set provider.gke.preferredBackends.enabled=true)
  else
    mode_flags=(--set router.proxy.mode=service --set router.proxy.priorityRouting.enabled=true --set router.inferencePool.create=false)
  fi
  verify_leader_election_rbac "${chart}" false --set router.epp.replicas=2 "${mode_flags[@]}"
  verify_leader_election_rbac "${chart}" true --set router.epp.replicas=2 "${mode_flags[@]}" --set router.epp.flags.ha-enable-leader-election=true
  verify_leader_election_rbac "${chart}" false --set router.epp.replicas=2 "${mode_flags[@]}" --set router.epp.flags.ha-enable-leader-election=false
  verify_leader_election_rbac "${chart}" false --set router.epp.replicas=2 --set router.epp.autoscaling.enabled=true
  echo "Leader-election RBAC checks passed for ${chart}."
done

echo "Verifying EPP autoscaling (HPA) rendering and validations..."
hpa_out="${TEMP_DIR}/hpa-render.yaml"
hpa_deploy="${TEMP_DIR}/hpa-deployment.yaml"
render() {
  "${HELM}" template hpa "${SCRIPT_ROOT}/config/charts/${chart}" \
    --set router.modelServers.matchLabels.app=test-app \
    --set router.epp.autoscaling.enabled=true "${extra_args[@]}" "$@"
}
# Renders the chart and extracts the EPP Deployment document into ${hpa_deploy}.
render_ok() {
  render "$@" > "${hpa_out}" || { echo "${chart}: render failed: $*"; exit 1; }
  awk 'BEGIN{RS="---"} (/\nkind: Deployment/ || /^kind: Deployment/) && /name: hpa-epp/ {print}' "${hpa_out}" > "${hpa_deploy}"
  [ -s "${hpa_deploy}" ] || { echo "${chart}: EPP Deployment not rendered: $*"; exit 1; }
}
expect_fail() {
  if render "$@" >/dev/null 2>&1; then echo "${chart}: expected failure for $*"; exit 1; fi
}
require() { grep -q -- "$1" "$2" || { echo "${chart}: expected '$1' in $2"; exit 1; }; }
forbid() { ! grep -q -- "$1" "$2" || { echo "${chart}: unexpected '$1' in $2"; exit 1; }; }

for chart in llm-d-router-gateway llm-d-router-standalone; do
  extra_args=()
  if [ "${chart}" == "llm-d-router-gateway" ]; then
    mode_flags=(--set provider.name=gke --set provider.gke.preferredBackends.enabled=true)
  else
    extra_args+=(--set router.inferencePool.create=false)
    mode_flags=(--set router.proxy.mode=service --set router.proxy.priorityRouting.enabled=true)
  fi

  render_ok --set router.epp.autoscaling.enabled=false
  require '^  replicas: 1$' "${hpa_deploy}"
  forbid 'kind: HorizontalPodAutoscaler' "${hpa_out}"

  render_ok
  require 'kind: HorizontalPodAutoscaler' "${hpa_out}"
  require 'minReplicas: 1' "${hpa_out}"
  require 'maxReplicas: 5' "${hpa_out}"
  require 'averageUtilization: 80' "${hpa_out}"
  forbid '^  replicas:' "${hpa_deploy}"
  require 'maxUnavailable: 0' "${hpa_deploy}"
  require 'maxSurge: 1' "${hpa_deploy}"
  forbid 'ha-enable-leader-election' "${hpa_deploy}"

  render_ok --set router.epp.deploymentStrategy.type=Recreate
  require 'type: Recreate' "${hpa_deploy}"
  forbid 'maxSurge:' "${hpa_deploy}"

  render_ok --set router.epp.autoscaling.minReplicas=3 --set router.epp.autoscaling.maxReplicas=3
  require 'minReplicas: 3' "${hpa_out}"
  require 'maxReplicas: 3' "${hpa_out}"

  render_ok --set router.epp.autoscaling.behavior.scaleDown.stabilizationWindowSeconds=300
  require 'stabilizationWindowSeconds: 300' "${hpa_out}"

  expect_fail "${mode_flags[@]}"
  expect_fail --set router.epp.flags.ha-enable-leader-election=true
  expect_fail --set router.epp.autoscaling.minReplicas=5 --set router.epp.autoscaling.maxReplicas=2
  for v in 0 -1; do
    expect_fail --set router.epp.autoscaling.minReplicas="${v}"
    expect_fail --set router.epp.autoscaling.maxReplicas="${v}"
  done
  for v in 0 101; do
    expect_fail --set router.epp.autoscaling.targetCPUUtilizationPercentage="${v}"
    expect_fail --set router.epp.autoscaling.targetMemoryUtilizationPercentage="${v}"
  done

  echo "EPP autoscaling checks passed for ${chart}."
done

echo "Verifying metrics authentication RBAC..."
verify_metrics_auth_rbac() {
  local chart="$1" expected_delegation="$2" expected_metrics_reader="$3"
  shift 3
  local output="${TEMP_DIR}/${chart}-metrics-auth.yaml"
  local args=()
  if [ "${chart}" == "llm-d-router-standalone" ]; then
    args+=(--set router.inferencePool.create=false)
  fi
  if ! "${HELM}" template metrics-auth "${SCRIPT_ROOT}/config/charts/${chart}" \
    --namespace metrics-test --set router.modelServers.matchLabels.app=llm-instance-gateway \
    "${args[@]}" "$@" > "${output}"; then
    echo "Metrics authentication rendering failed for ${chart}: $*"
    exit 1
  fi
  local resource actual
  for resource in tokenreviews subjectaccessreviews; do
    if grep -q -- "^    - ${resource}$" "${output}"; then
      actual=true
    else
      actual=false
    fi
    if [ "${actual}" != "${expected_delegation}" ]; then
      echo "${chart}: expected ${resource} rule present=${expected_delegation}, got ${actual}; flags: $*"
      exit 1
    fi
  done
  # The EPP ClusterRole quotes the path; the GMP metrics reader ClusterRole does not.
  if grep -q -- '^    - "/metrics"$' "${output}"; then
    actual=true
  else
    actual=false
  fi
  if [ "${actual}" != "${expected_metrics_reader}" ]; then
    echo "${chart}: expected EPP /metrics rule present=${expected_metrics_reader}, got ${actual}; flags: $*"
    exit 1
  fi
}

for chart in llm-d-router-gateway llm-d-router-standalone; do
  verify_metrics_auth_rbac "${chart}" true false
  verify_metrics_auth_rbac "${chart}" true true --set router.monitoring.prometheus.enabled=true
  verify_metrics_auth_rbac "${chart}" false false --set router.monitoring.prometheus.auth.enabled=false
  verify_metrics_auth_rbac "${chart}" false false --set router.monitoring.prometheus.enabled=true --set router.monitoring.prometheus.auth.enabled=false
  verify_metrics_auth_rbac "${chart}" true false --set router.monitoring.prometheus.enabled=true --set router.monitoring.provider.name=gmp
  echo "Metrics authentication RBAC checks passed for ${chart}."
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

invalid_latency_predictor_mode_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.latencyPredictor.mode=bogus >/dev/null"
echo "Executing: ${invalid_latency_predictor_mode_command}"
if eval "${invalid_latency_predictor_mode_command}"; then
  echo "Helm template unexpectedly succeeded for invalid latencyPredictor.mode"
  exit 1
fi

deprecated_agentgateway_service_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.proxyType=agentgateway --set router.proxy.agentgateway.service.name=foo >/dev/null"
echo "Executing: ${deprecated_agentgateway_service_command}"
if eval "${deprecated_agentgateway_service_command}"; then
  echo "Helm template unexpectedly succeeded for deprecated agentgateway.service configuration"
  exit 1
fi

unsupported_agentgateway_llm_d_router_gateway_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.proxy.proxyType=agentgateway --set router.inferencePool.create=true --set router.modelServers.matchLabels.app=llm-instance-gateway >/dev/null"
echo "Executing: ${unsupported_agentgateway_llm_d_router_gateway_command}"
if eval "${unsupported_agentgateway_llm_d_router_gateway_command}"; then
  echo "Helm template unexpectedly succeeded for unsupported agentgateway createInferencePool=true configuration"
  exit 1
fi

unsupported_agentgateway_listener_port_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.proxy.proxyType=agentgateway --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000' --set 'router.extraServicePorts[0].name=proxy' --set 'router.extraServicePorts[0].port=9000' --set 'router.extraServicePorts[0].protocol=TCP' --set 'router.extraServicePorts[0].targetPort=9000' >/dev/null"
echo "Executing: ${unsupported_agentgateway_listener_port_command}"
if eval "${unsupported_agentgateway_listener_port_command}"; then
  echo "Helm template unexpectedly succeeded without an agentgateway listener Service port named http"
  exit 1
fi

mismatched_agentgateway_listener_target_port_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.proxy.proxyType=agentgateway --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000' --set 'router.extraServicePorts[0].name=http' --set 'router.extraServicePorts[0].port=9000' --set 'router.extraServicePorts[0].protocol=TCP' --set 'router.extraServicePorts[0].targetPort=9001' >/dev/null"
echo "Executing: ${mismatched_agentgateway_listener_target_port_command}"
if eval "${mismatched_agentgateway_listener_target_port_command}"; then
  echo "Helm template unexpectedly succeeded for an agentgateway listener targetPort that does not match port"
  exit 1
fi

invalid_failopen_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.failOpen=notabool >/dev/null"
echo "Executing: ${invalid_failopen_command}"
if eval "${invalid_failopen_command}"; then
  echo "Helm template unexpectedly succeeded for non-boolean router.proxy.failOpen"
  exit 1
fi

invalid_priority_routing_enabled_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set-string router.proxy.priorityRouting.enabled=false >/dev/null"
echo "Executing: ${invalid_priority_routing_enabled_command}"
if eval "${invalid_priority_routing_enabled_command}"; then
  echo "Helm template unexpectedly succeeded for non-boolean router.proxy.priorityRouting.enabled"
  exit 1
fi

invalid_priority_routing_health_checking_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.priorityRouting.enabled=true --set-string router.epp.flags.health-checking=false >/dev/null"
echo "Executing: ${invalid_priority_routing_health_checking_command}"
if eval "${invalid_priority_routing_health_checking_command}"; then
  echo "Helm template unexpectedly succeeded for non-boolean router.epp.flags.health-checking with priority routing"
  exit 1
fi

invalid_tokenizer_flavor_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.tokenizer.enabled=true --set router.tokenizer.modelName=test-model --set router.tokenizer.flavor=invalid >/dev/null"
echo "Executing: ${invalid_tokenizer_flavor_command}"
if eval "${invalid_tokenizer_flavor_command}"; then
  echo "Helm template unexpectedly succeeded for invalid router.tokenizer.flavor"
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

echo "Verifying router.epp.metricsDataSource.insecureSkipVerify renders the configured value..."
for chart in llm-d-router-gateway llm-d-router-standalone; do
  for skip_verify_case in "default:true" "true:true" "false:false"; do
    skip_verify_set="${skip_verify_case%%:*}"
    skip_verify_want="${skip_verify_case##*:}"
    skip_verify_args=""
    if [ "${skip_verify_set}" != "default" ]; then
      skip_verify_args="--set router.epp.metricsDataSource.insecureSkipVerify=${skip_verify_set}"
    fi
    skip_verify_render_output="${TEMP_DIR}/${chart}-insecure-skip-verify-${skip_verify_set}-render.yaml"
    skip_verify_render_command="${HELM} template ${SCRIPT_ROOT}/config/charts/${chart} --set router.modelServers.matchLabels.app=llm-instance-gateway ${skip_verify_args} > ${skip_verify_render_output}"
    echo "Executing: ${skip_verify_render_command}"
    eval "${skip_verify_render_command}"
    if ! grep -Eq -- "^[[:space:]]+insecureSkipVerify: ${skip_verify_want}$" "${skip_verify_render_output}"; then
      echo "${chart} did not render insecureSkipVerify: ${skip_verify_want} for router.epp.metricsDataSource.insecureSkipVerify=${skip_verify_set}"
      exit 1
    fi
  done
done

if ! HELM="${HELM}" bash "${SCRIPT_ROOT}/hack/verify-plugins-config.sh"; then
  echo "Structured plugins configuration validation failed"
  exit 1
fi

if ! HELM="${HELM}" bash "${SCRIPT_ROOT}/hack/verify-cross-replica-config.sh"; then
  echo "Cross-replica configuration validation failed"
  exit 1
fi

echo "Verifying standalone proxy sidecar preStop delays..."
proxy_pre_stop=$(printf '          lifecycle:\n            preStop:\n              sleep:\n                seconds: 5')
for proxy_type in envoy agentgateway; do
  for kube_version in 1.29.9 1.30.0; do
    proxy_sidecar_render_output="${TEMP_DIR}/proxy-sidecar-${proxy_type}-${kube_version}.yaml"
    if ! "${HELM}" template "${SCRIPT_ROOT}/config/charts/llm-d-router-standalone" \
      --kube-version "${kube_version}" --set router.proxy.mode=sidecar \
      --set router.proxy.proxyType="${proxy_type}" \
      --set router.modelServers.matchLabels.app=llm-instance-gateway \
      --set router.inferencePool.create=false \
      --set 'router.modelServers.targetPorts[0].number=8000' > "${proxy_sidecar_render_output}"; then
      echo "Proxy sidecar rendering failed for ${proxy_type} on Kubernetes ${kube_version}"
      exit 1
    fi
    if ! "${KUBECTL_VALIDATE}" "${proxy_sidecar_render_output}" --local-crds "${TEMP_DIR}"; then
      echo "Proxy sidecar validation failed for ${proxy_type} on Kubernetes ${kube_version}"
      exit 1
    fi
    for container_name in "${proxy_type}-proxy" epp; do
      container_spec=$(awk -v name="${container_name}" '
        $0 == "        - name: " name { in_container = 1; next }
        in_container && /^        - name:/ { exit }
        in_container { print }
      ' "${proxy_sidecar_render_output}")
      if [ -z "${container_spec}" ]; then
        echo "Container ${container_name} not found in proxy sidecar render"
        exit 1
      fi
      if [ "${kube_version}" == "1.30.0" ]; then
        if [ "$(grep -A3 -- '^          lifecycle:$' <<< "${container_spec}")" != "${proxy_pre_stop}" ]; then
          echo "Expected a five-second preStop sleep for ${container_name} on Kubernetes ${kube_version}"
          exit 1
        fi
      elif grep -q -- '^          lifecycle:$' <<< "${container_spec}"; then
        echo "Unexpected lifecycle hook for ${container_name} on Kubernetes ${kube_version}"
        exit 1
      fi
    done
  done
done

echo "Verifying llm-d-router-standalone agentgateway renders plaintext EPP and custom listener ports..."
agentgateway_render_output="${TEMP_DIR}/llm-d-router-standalone-agentgateway-render.yaml"
agentgateway_render_command="${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.proxy.proxyType=agentgateway --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000' --set 'router.extraServicePorts[0].name=http' --set 'router.extraServicePorts[0].port=9000' --set 'router.extraServicePorts[0].protocol=TCP' --set 'router.extraServicePorts[0].targetPort=http' > ${agentgateway_render_output}"
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
if ! grep -q -- 'name: "default/llm-instance-gateway"' "${agentgateway_render_output}"; then
  echo "Agentgateway Helm template did not derive the logical backend name from modelServers"
  exit 1
fi
if ! grep -q -- 'hostname: "llm-instance-gateway"' "${agentgateway_render_output}"; then
  echo "Agentgateway Helm template did not derive the logical backend hostname from modelServers"
  exit 1
fi
if grep -q -- '# Source: llm-d-router-standalone/templates/agentgateway-service.yaml' "${agentgateway_render_output}"; then
  echo "Agentgateway model Service unexpectedly rendered"
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
# Fail-open (the service-mode default) adds a fallback route to the model servers
# for requests EPP did not schedule, via a headless Service selecting the pool.
if ! grep -q -- 'cluster: inference_pool_fallback' "${proxy_service_render_output}"; then
  echo "Proxy service mode with fail-open did not render the fallback route"
  exit 1
fi
if ! grep -q -- 'present_match: true' "${proxy_service_render_output}"; then
  echo "Proxy service mode with fail-open did not gate original_destination_cluster on the destination header"
  exit 1
fi
if ! grep -q -- 'address: proxy-svc-epp-model-servers.proxy-ns.svc.cluster.local' "${proxy_service_render_output}"; then
  echo "Proxy service mode with fail-open did not point the fallback cluster at the model-server Service FQDN"
  exit 1
fi
fallback_service=$(awk '/^kind: Service$/{svc=1} /^---/{svc=0} svc' "${proxy_service_render_output}" | grep -A12 -- 'name: proxy-svc-epp-model-servers$' || true)
if ! grep -q -- 'clusterIP: None' <<<"${fallback_service}" || ! grep -q -- 'app: llm-instance-gateway' <<<"${fallback_service}" || ! grep -q -- 'port: 8000' <<<"${fallback_service}"; then
  echo "Proxy service mode with fail-open did not render the headless model-server Service"
  exit 1
fi
# Only EPP may set the destination header; client-supplied values are removed before routing.
if ! grep -q -- 'remove: x-gateway-destination-endpoint' "${proxy_service_render_output}"; then
  echo "Proxy service mode did not strip client-supplied x-gateway-destination-endpoint headers"
  exit 1
fi
# The fallback route skips EPP, so it removes the client routing and screening headers EPP would remove.
fallback_route=$(grep -A30 -- 'cluster: inference_pool_fallback$' "${proxy_service_render_output}" | head -30)
for header in x-llm-d-prefiller-host-port x-prefiller-host-port x-llm-d-encoder-hosts-ports x-encoder-hosts-ports \
  x-data-parallel-host-port x-llm-d-kv-cache-source-host-port x-kv-cache-source-host-port x-llm-d-pin-host-port; do
  if ! grep -q -- "- ${header}$" <<<"${fallback_route}"; then
    echo "Fail-open fallback route does not remove client-supplied ${header}"
    exit 1
  fi
done

echo "Verifying llm-d-router-standalone fail-open fallback follows router.proxy.failOpen and targetPorts..."
fail_closed_render_output="${TEMP_DIR}/llm-d-router-standalone-proxy-fail-closed-render.yaml"
${HELM} template proxy-svc ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --namespace proxy-ns --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.failOpen=false > "${fail_closed_render_output}"
if grep -q -- 'inference_pool_fallback\|proxy-svc-epp-model-servers' "${fail_closed_render_output}"; then
  echo "Fail-open fallback rendered with router.proxy.failOpen=false"
  exit 1
fi
if ! grep -q -- 'remove: x-gateway-destination-endpoint' "${fail_closed_render_output}"; then
  echo "Destination header strip missing with router.proxy.failOpen=false"
  exit 1
fi
fallback_port_render_output="${TEMP_DIR}/llm-d-router-standalone-proxy-fallback-port-render.yaml"
${HELM} template proxy-svc ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --namespace proxy-ns --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8200' --set 'router.modelServers.targetPorts[1].number=8201' > "${fallback_port_render_output}"
if ! grep -q -- 'port_value: 8200' "${fallback_port_render_output}" || ! grep -q -- 'targetPort: 8200' "${fallback_port_render_output}"; then
  echo "Fail-open fallback did not use the first router.modelServers.targetPorts entry"
  exit 1
fi
agentgateway_fallback_output="${TEMP_DIR}/llm-d-router-standalone-agentgateway-fallback-render.yaml"
${HELM} template ag-svc ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --namespace ag-ns --set router.proxy.proxyType=agentgateway --set router.proxy.mode=service --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000' > "${agentgateway_fallback_output}"
if grep -q -- 'ag-svc-epp-model-servers' "${agentgateway_fallback_output}"; then
  echo "Envoy fail-open fallback Service unexpectedly rendered for agentgateway"
  exit 1
fi
# Sidecar mode defaults to fail-open too, so it gets the fallback route and Service.
fallback_sidecar_output="${TEMP_DIR}/llm-d-router-standalone-fallback-sidecar-render.yaml"
${HELM} template proxy-sc ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --namespace proxy-ns --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=sidecar > "${fallback_sidecar_output}"
if ! grep -q -- 'cluster: inference_pool_fallback' "${fallback_sidecar_output}" || ! grep -q -- 'name: proxy-sc-epp-model-servers$' "${fallback_sidecar_output}"; then
  echo "Sidecar mode with default fail-open did not render the fallback route and Service"
  exit 1
fi
# The fallback Service selects every pool label, and the cluster FQDN follows router.clusterDomain.
fallback_selector_output="${TEMP_DIR}/llm-d-router-standalone-fallback-selector-render.yaml"
${HELM} template proxy-svc ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --namespace proxy-ns --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.modelServers.matchLabels.tier=decode --set router.inferencePool.create=false --set router.clusterDomain=example.internal > "${fallback_selector_output}"
fallback_selector=$(awk '/^kind: Service$/{svc=1} /^---/{svc=0} svc' "${fallback_selector_output}" | grep -A12 -- 'name: proxy-svc-epp-model-servers$' || true)
if ! grep -q -- 'app: llm-instance-gateway' <<<"${fallback_selector}" || ! grep -q -- 'tier: decode' <<<"${fallback_selector}"; then
  echo "Fail-open fallback Service did not select every router.modelServers.matchLabels entry"
  exit 1
fi
if ! grep -q -- 'address: proxy-svc-epp-model-servers.proxy-ns.svc.example.internal' "${fallback_selector_output}"; then
  echo "Fail-open fallback cluster did not use router.clusterDomain"
  exit 1
fi
# Priority routing keeps the fallback: it applies when every EPP replica is unreachable.
fallback_priority_output="${TEMP_DIR}/llm-d-router-standalone-fallback-priority-render.yaml"
${HELM} template proxy-svc ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --namespace proxy-ns --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set router.proxy.mode=service --set router.proxy.priorityRouting.enabled=true --set router.epp.replicas=2 > "${fallback_priority_output}"
if ! grep -q -- 'cluster: inference_pool_fallback' "${fallback_priority_output}"; then
  echo "Priority routing did not render the fail-open fallback route"
  exit 1
fi
# The gateway chart has no standalone proxy, so it renders neither the fallback nor the header strip.
fallback_gateway_output="${TEMP_DIR}/llm-d-router-gateway-fallback-render.yaml"
${HELM} template gw ${SCRIPT_ROOT}/config/charts/llm-d-router-gateway --namespace gw-ns --set router.modelServers.matchLabels.app=llm-instance-gateway > "${fallback_gateway_output}"
if grep -q -- 'inference_pool_fallback\|gw-epp-model-servers\|early_header_mutation' "${fallback_gateway_output}"; then
  echo "Gateway chart unexpectedly rendered the standalone fail-open fallback"
  exit 1
fi

echo "Verifying llm-d-router-standalone agentgateway in service mode reaches EPP over the Service..."
agentgateway_service_mode_output="${TEMP_DIR}/llm-d-router-standalone-agentgateway-service-render.yaml"
agentgateway_service_mode_command="${HELM} template ag-svc ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --namespace ag-ns --set router.proxy.proxyType=agentgateway --set router.proxy.mode=service --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.modelServers.targetPorts[0].number=8000' > ${agentgateway_service_mode_output}"
echo "Executing: ${agentgateway_service_mode_command}"
eval "${agentgateway_service_mode_command}"
if ! grep -q -- 'name: ag-svc-proxy' "${agentgateway_service_mode_output}"; then
  echo "Agentgateway service mode did not render the separate proxy Deployment/Service named ag-svc-proxy"
  exit 1
fi
# The agentgateway container lives only in the standalone proxy Deployment, not the EPP pod.
agentgateway_container_count=$(grep -c -- '- name: agentgateway-proxy$' "${agentgateway_service_mode_output}")
if [ "${agentgateway_container_count}" -ne 1 ]; then
  echo "Agentgateway service mode rendered ${agentgateway_container_count} agentgateway-proxy containers, expected exactly 1 (in the proxy Deployment)"
  exit 1
fi
# endpointPicker must target the EPP Service FQDN, not loopback.
if ! grep -q -- 'host: "ag-svc-epp.ag-ns.svc.cluster.local:9002"' "${agentgateway_service_mode_output}"; then
  echo "Agentgateway service mode did not point endpointPicker.host at the EPP Service FQDN"
  exit 1
fi
if grep -q -- 'host: "127.0.0.1:9002"' "${agentgateway_service_mode_output}"; then
  echo "Agentgateway service mode still points endpointPicker.host at loopback"
  exit 1
fi
# The agentgateway config must be mounted in the proxy Deployment.
if ! grep -q -- 'agentgateway-config-template' "${agentgateway_service_mode_output}"; then
  echo "Agentgateway service mode did not mount the agentgateway config in the proxy Deployment"
  exit 1
fi

echo "Verifying standalone proxy autoscaling (HPA) rendering and validations..."
proxy_hpa_out="${TEMP_DIR}/proxy-hpa-render.yaml"
proxy_hpa_deploy="${TEMP_DIR}/proxy-hpa-deployment.yaml"
render_proxy() {
  "${HELM}" template proxy-hpa "${SCRIPT_ROOT}/config/charts/llm-d-router-standalone" \
    --set router.modelServers.matchLabels.app=test-app \
    --set router.inferencePool.create=false \
    --set router.proxy.mode=service \
    --set router.proxy.autoscaling.enabled=true "$@"
}
render_proxy_ok() {
  render_proxy "$@" > "${proxy_hpa_out}" || { echo "llm-d-router-standalone: proxy render failed: $*"; exit 1; }
  awk 'BEGIN{RS="---"} (/\nkind: Deployment/ || /^kind: Deployment/) && /name: proxy-hpa-proxy/ {print}' "${proxy_hpa_out}" > "${proxy_hpa_deploy}"
  [ -s "${proxy_hpa_deploy}" ] || { echo "llm-d-router-standalone: Proxy Deployment not rendered: $*"; exit 1; }
}
expect_proxy_fail() {
  if render_proxy "$@" >/dev/null 2>&1; then echo "llm-d-router-standalone: expected proxy failure for $*"; exit 1; fi
}

render_proxy_ok --set router.proxy.autoscaling.enabled=false
require '^  replicas: 2$' "${proxy_hpa_deploy}"
forbid 'kind: HorizontalPodAutoscaler' "${proxy_hpa_out}"

render_proxy_ok
require 'name: proxy-hpa-proxy' "${proxy_hpa_out}"
require 'kind: HorizontalPodAutoscaler' "${proxy_hpa_out}"
require 'minReplicas: 1' "${proxy_hpa_out}"
require 'maxReplicas: 5' "${proxy_hpa_out}"
require 'averageUtilization: 80' "${proxy_hpa_out}"
forbid '^  replicas:' "${proxy_hpa_deploy}"
require 'terminationGracePeriodSeconds: 70' "${proxy_hpa_deploy}"

render_proxy_ok --set router.proxy.autoscaling.minReplicas=3 --set router.proxy.autoscaling.maxReplicas=3
require 'minReplicas: 3' "${proxy_hpa_out}"
require 'maxReplicas: 3' "${proxy_hpa_out}"

render_proxy_ok --set router.proxy.autoscaling.behavior.scaleDown.stabilizationWindowSeconds=300
require 'stabilizationWindowSeconds: 300' "${proxy_hpa_out}"

render_proxy_ok --set router.proxy.autoscaling.targetMemoryUtilizationPercentage=75
require 'averageUtilization: 75' "${proxy_hpa_out}"

render_proxy_ok --set router.proxy.autoscaling.targetCPUUtilizationPercentage=null --set 'router.proxy.autoscaling.metrics[0].type=Resource' --set 'router.proxy.autoscaling.metrics[0].resource.name=cpu' --set 'router.proxy.autoscaling.metrics[0].resource.target.type=Utilization' --set 'router.proxy.autoscaling.metrics[0].resource.target.averageUtilization=60'
require 'averageUtilization: 60' "${proxy_hpa_out}"

# Negative validations
expect_proxy_fail --set router.proxy.mode=sidecar
expect_proxy_fail --set router.proxy.enabled=false
expect_proxy_fail --set router.proxy.autoscaling.minReplicas=5 --set router.proxy.autoscaling.maxReplicas=2
for v in 0 -1; do
  expect_proxy_fail --set router.proxy.autoscaling.minReplicas="${v}"
  expect_proxy_fail --set router.proxy.autoscaling.maxReplicas="${v}"
done
for v in 0 101; do
  expect_proxy_fail --set router.proxy.autoscaling.targetCPUUtilizationPercentage="${v}"
  expect_proxy_fail --set router.proxy.autoscaling.targetMemoryUtilizationPercentage="${v}"
done

echo "Proxy autoscaling checks passed for llm-d-router-standalone."

echo "Verifying llm-d-router-standalone proxy preset args..."
arg_rendered() { if [ -z "$3" ]; then grep -q -- "- \"$2\"\$" "$1"; else grep -A1 -- "- \"$2\"\$" "$1" | grep -q -- "- \"$3\"\$"; fi; }
envoy_default_output="${TEMP_DIR}/llm-d-router-standalone-envoy-default-render.yaml"
${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false > "${envoy_default_output}" || exit 1
envoy_explicit_args_output="${TEMP_DIR}/llm-d-router-standalone-envoy-explicit-args-render.yaml"
${HELM} template ${SCRIPT_ROOT}/config/charts/llm-d-router-standalone --set router.modelServers.matchLabels.app=llm-instance-gateway --set router.inferencePool.create=false --set 'router.proxy.args={--concurrency,4,-c,/etc/envoy/envoy.yaml}' > "${envoy_explicit_args_output}" || exit 1
# name:file:flag:value:want (1 = rendered, 0 = absent; empty value = flag alone)
for c in "sidecar:${envoy_default_output}:--concurrency:8:1" "sidecar-cpuset:${envoy_default_output}:--cpuset-threads::0" \
         "service:${proxy_service_render_output}:--concurrency:8:1" \
         "agentgateway:${agentgateway_render_output}:--concurrency:8:0" "agentgateway-preset:${agentgateway_render_output}:-f:/config/config.yaml:1" \
         "explicit:${envoy_explicit_args_output}:--concurrency:4:1" "explicit-preset:${envoy_explicit_args_output}:--concurrency:8:0"; do
  IFS=: read -r name file flag value want <<< "${c}"
  if arg_rendered "${file}" "${flag}" "${value}"; then got=1; else got=0; fi
  if [ "${got}" != "${want}" ]; then
    echo "Proxy args for ${name}: ${flag} ${value} rendered=${got}, want ${want}"
    exit 1
  fi
done
