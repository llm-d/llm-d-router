#!/usr/bin/env bash

# Copyright 2026 The Kubernetes Authors.
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

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELM="${HELM:-${SCRIPT_ROOT}/bin/helm}"
YQ="${YQ:-${SCRIPT_ROOT}/bin/yq}"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "${TEMP_DIR}"' EXIT

assert_yaml() {
  if ! "${YQ}" -e "$2" "$1" > /dev/null; then
    echo "Assertion failed: $2 in $1" >&2
    exit 1
  fi
}

render() {
  "${HELM}" template sync-test "${chart_path}" "${args[@]}" "$@" > "${output}" || return $?
  "${YQ}" 'select(.kind == "Deployment" and .metadata.name == "sync-test-epp")' "${output}" > "${deployment}"
}

render_notes() {
  "${HELM}" template sync-test "${notes_chart}" --show-only templates/test-notes.yaml \
    "${args[@]}" "$@" | "${YQ}" -r '.data.notes' > "${TEMP_DIR}/notes.txt"
}

assert_rejected() {
  local message=$1
  shift
  if render "$@" 2> "${TEMP_DIR}/error.log"; then
    echo "Accepted invalid configuration: $*" >&2
    exit 1
  fi
  grep -Fq -- "${message}" "${TEMP_DIR}/error.log"
}

for chart in llm-d-router-gateway llm-d-router-standalone; do
  echo "Verifying cross-replica configuration for ${chart}..."
  chart_path="${SCRIPT_ROOT}/config/charts/${chart}"
  args=(--set router.modelServers.matchLabels.app=llm-instance-gateway)
  output="${TEMP_DIR}/render.yaml"
  deployment="${TEMP_DIR}/deployment.yaml"
  config="${TEMP_DIR}/config.yaml"
  "${HELM}" dependency build "${chart_path}"

  render --set router.epp.replicas=3
  cp "${output}" "${TEMP_DIR}/baseline.yaml"
  render --set router.epp.replicas=3 --set router.epp.crossReplica.enabled=false
  diff -u "${TEMP_DIR}/baseline.yaml" "${output}"

  render --set router.epp.replicas=3 --set router.epp.haMode=all-active
  cp "${output}" "${TEMP_DIR}/all-active.yaml"
  for value in 0 f F false FALSE False; do
    render --set router.epp.replicas=3 --set-string "router.epp.flags.ha-enable-leader-election=${value}"
    diff -u "${TEMP_DIR}/all-active.yaml" "${output}"
    render --set router.epp.replicas=1 --set-string "router.epp.flags.ha-enable-leader-election=${value}"
    assert_yaml "${deployment}" '.spec.strategy.type == "Recreate"'
  done
  for value in 1 t T true TRUE True; do
    render --set router.epp.replicas=3 --set-string "router.epp.flags.ha-enable-leader-election=${value}"
    diff -u "${TEMP_DIR}/baseline.yaml" "${output}"
    assert_rejected 'requires active-active' --set router.epp.autoscaling.enabled=true --set-string "router.epp.flags.ha-enable-leader-election=${value}"
  done
  assert_rejected 'ha-enable-leader-election must be a boolean' --set-string router.epp.flags.ha-enable-leader-election=invalid
  render --set router.epp.replicas=3 --set router.epp.flags.ha-enable-leader-election=false --set router.epp.deploymentStrategy.type=Recreate
  assert_yaml "${deployment}" '.spec.strategy.type == "Recreate"'

  sync=(--set router.epp.crossReplica.enabled=true
        --set router.epp.crossReplica.redis.address=redis.example:6379
        --set router.epp.replicas=3)
  render "${sync[@]}"
  assert_yaml "${deployment}" '.spec.replicas == 3 and .spec.strategy.type == "RollingUpdate" and .spec.strategy.rollingUpdate.maxUnavailable == 0 and .spec.strategy.rollingUpdate.maxSurge == 1'
  assert_yaml "${deployment}" '.spec.template.spec.containers[] | select(.name == "epp") | .args | contains(["--ha-enable-leader-election=false", "--allow-experimental-plugins=true"])'
  assert_yaml "${output}" 'select(.kind == "ConfigMap" and .metadata.name == "sync-test-epp") | .data."default-plugins.yaml" | from_yaml | .plugins[] | select(.type == "redis-syncer") | (.parameters | sort_keys(..) | to_json) == ({"address": "redis.example:6379", "coordinationTTL": "180s", "db": 0, "stateTTL": "2s"} | sort_keys(..) | to_json)'
  "${YQ}" -N 'select(.metadata.name == "sync-test-epp-leader-election" or .metadata.name == "sync-test-epp-leader-election-binding" or .kind == "StatefulSet" or .kind == "Secret")' "${output}" > "${TEMP_DIR}/unexpected.yaml"
  test ! -s "${TEMP_DIR}/unexpected.yaml"

  # Helm template omits NOTES.txt; render it through a temporary ConfigMap.
  notes_chart="${TEMP_DIR}/${chart}-notes"
  cp -R "${chart_path}" "${notes_chart}"
  {
    printf '{{- define "test.notes" -}}\n'
    cat "${chart_path}/templates/NOTES.txt"
    cat <<'EOF'
{{- end -}}
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-notes
data:
  notes: {{ include "test.notes" . | quote }}
EOF
  } > "${notes_chart}/templates/test-notes.yaml"

  # Helm cannot resolve the dependencies of custom EPP plugins.
  for values_file in "" "${SCRIPT_ROOT}/test/perf/config/router-configs/optimized-baseline.yaml"; do
    values=(--set router.epp.pluginsConfigFile=default-plugins.yaml)
    if [[ -n ${values_file} ]]; then values=(-f "${values_file}"); fi
    render_notes "${values[@]}" "${sync[@]}"
    grep -Fq 'WARNING: Redis configuration alone does not enable shared in-flight load.' "${TEMP_DIR}/notes.txt"
    grep -Fq 'default and optimized-baseline profiles do not consume in-flight load' "${TEMP_DIR}/notes.txt"
  done
  render_notes
  if grep -Fq 'WARNING: Redis configuration alone' "${TEMP_DIR}/notes.txt"; then
    echo 'Cross-replica warning emitted with synchronization disabled' >&2
    exit 1
  fi

  render "${sync[@]}" --set router.epp.autoscaling.enabled=true --set router.epp.autoscaling.minReplicas=2 --set router.epp.autoscaling.maxReplicas=7
  assert_yaml "${deployment}" 'has("spec") and (.spec | has("replicas") | not)'
  assert_yaml "${output}" 'select(.kind == "HorizontalPodAutoscaler") | .spec.minReplicas == 2 and .spec.maxReplicas == 7 and .spec.scaleTargetRef.name == "sync-test-epp"'
  render "${sync[@]}" --set router.epp.deploymentStrategy.type=Recreate
  assert_yaml "${deployment}" '(.spec.strategy | sort_keys(..) | to_json) == ({"type": "Recreate"} | sort_keys(..) | to_json)'

  # Only the selected configuration gains a syncer; its routing profile is preserved.
  for source in builtin structured raw; do
    for filename in default-plugins.yaml payload-agnostic.yaml custom.yaml; do
      if [[ ${source} == builtin && ${filename} == custom.yaml ]]; then continue; fi
      values=(--set "router.epp.pluginsConfigFile=${filename}")
      if [[ ${source} != builtin ]]; then
        "${YQ}" '.router.epp.pluginsConfig.plugins += [{"type": "metrics-data-source"}, {"type": "core-metrics-extractor"}] | .router.epp.pluginsConfig.dataLayer = {"injectDefaults": false, "sources": [{"pluginRef": "metrics-data-source", "extractors": [{"pluginRef": "core-metrics-extractor"}]}]} | del(.router.epp.pluginsCustomConfig)' \
          "${SCRIPT_ROOT}/hack/testdata/plugins-config-base.yaml" > "${TEMP_DIR}/values.yaml"
        if [[ ${source} == raw ]]; then
          CONFIG_FILE="${filename}" "${YQ}" -i '.router.epp.pluginsCustomConfig[strenv(CONFIG_FILE)] = (.router.epp.pluginsConfig | to_yaml) | del(.router.epp.pluginsConfig)' "${TEMP_DIR}/values.yaml"
        fi
        values=(-f "${TEMP_DIR}/values.yaml" "${values[@]}")
      fi
      render "${values[@]}"
      CONFIG_FILE="${filename}" "${YQ}" -o=json 'select(.kind == "ConfigMap" and .metadata.name == "sync-test-epp") | .data | del(.[strenv(CONFIG_FILE)]) | sort_keys(..)' "${output}" > "${TEMP_DIR}/other-before.json"
      CONFIG_FILE="${filename}" "${YQ}" 'select(.kind == "ConfigMap" and .metadata.name == "sync-test-epp") | .data[strenv(CONFIG_FILE)] | from_yaml' "${output}" > "${TEMP_DIR}/before.yaml"
      render "${values[@]}" "${sync[@]}" \
        --set router.epp.crossReplica.redis.db=4 \
        --set router.epp.crossReplica.redis.stateTTL=5s \
        --set router.epp.crossReplica.redis.coordinationTTL=60s \
        --set router.epp.crossReplica.syncInterval=100ms \
        --set router.epp.crossReplica.publishTimeout=500ms \
        --set router.epp.crossReplica.redis.passwordSecretRef.name=redis-auth \
        --set router.epp.crossReplica.redis.passwordSecretRef.key=password
      CONFIG_FILE="${filename}" "${YQ}" 'select(.kind == "ConfigMap" and .metadata.name == "sync-test-epp") | .data[strenv(CONFIG_FILE)] | from_yaml' "${output}" > "${config}"
      CONFIG_FILE="${filename}" "${YQ}" -o=json 'select(.kind == "ConfigMap" and .metadata.name == "sync-test-epp") | .data | del(.[strenv(CONFIG_FILE)]) | sort_keys(..)' "${output}" > "${TEMP_DIR}/other-after.json"
      diff -u "${TEMP_DIR}/other-before.json" "${TEMP_DIR}/other-after.json"
      assert_yaml "${config}" '(.dataLayer.crossReplica | sort_keys(..) | to_json) == ({"publishTimeout": "500ms", "syncInterval": "100ms", "syncerPluginRef": "redis-syncer"} | sort_keys(..) | to_json)'
      assert_yaml "${config}" '[.plugins[] | select(.type == "redis-syncer")] | length == 1'
      assert_yaml "${config}" '.plugins[] | select(.type == "redis-syncer") | (.parameters | sort_keys(..) | to_json) == ({"address": "redis.example:6379", "coordinationTTL": "60s", "db": 4, "passwordEnv": "EPP_REDIS_PASSWORD", "stateTTL": "5s"} | sort_keys(..) | to_json)'
      assert_yaml "${deployment}" '.spec.template.spec.containers[] | select(.name == "epp") | .env[] | select(.name == "EPP_REDIS_PASSWORD") | (.valueFrom.secretKeyRef | sort_keys(..) | to_json) == ({"key": "password", "name": "redis-auth"} | sort_keys(..) | to_json)'
      # Compare all original fields, allowing the generated dataLayer container.
      "${YQ}" -o=json 'del(.dataLayer.crossReplica) | .plugins |= map(select(.type != "redis-syncer")) | sort_keys(..)' "${config}" > "${TEMP_DIR}/after.yaml"
      "${YQ}" -o=json '.dataLayer = (.dataLayer // {}) | sort_keys(..)' "${TEMP_DIR}/before.yaml" > "${TEMP_DIR}/expected.yaml"
      diff -u "${TEMP_DIR}/expected.yaml" "${TEMP_DIR}/after.yaml"
    done
  done

  render --set router.epp.haMode=all-active --set router.epp.replicas=3
  assert_yaml "${deployment}" '.spec.template.spec.containers[] | select(.name == "epp") | .args | contains(["--ha-enable-leader-election=false"])'
  render --set router.epp.haMode=leader-elected
  assert_yaml "${output}" 'select(.metadata.name == "sync-test-epp-leader-election") | .kind == "Role"'

  assert_rejected 'router.epp.haMode must be' --set router.epp.haMode=invalid
  assert_rejected 'requires all-active' "${sync[@]}" --set router.epp.haMode=leader-elected
  assert_rejected 'conflicts with router.epp.haMode' "${sync[@]}" --set router.epp.flags.ha-enable-leader-election=true
  assert_rejected 'requires active-active' --set router.epp.haMode=leader-elected --set router.epp.autoscaling.enabled=true
  assert_rejected 'requires allow-experimental-plugins' "${sync[@]}" --set router.epp.flags.allow-experimental-plugins=false
  assert_rejected 'redis.address is required' --set router.epp.crossReplica.enabled=true
  assert_rejected 'passwordSecretRef requires name and key' "${sync[@]}" --set router.epp.crossReplica.redis.passwordSecretRef.name=redis-auth
  assert_rejected 'plaintext password' "${sync[@]}" --set router.epp.crossReplica.redis.password=never-inline
  assert_rejected 'reserved' "${sync[@]}" --set router.epp.crossReplica.redis.passwordSecretRef.name=redis-auth --set router.epp.crossReplica.redis.passwordSecretRef.key=password --set-json 'router.epp.env=[{"name":"EPP_REDIS_PASSWORD","value":"collision"}]'
  for flag in config-file config-text; do
    assert_rejected 'cannot override' "${sync[@]}" --set "router.epp.flags.${flag}=external"
  done
  assert_rejected 'already configures crossReplica' "${sync[@]}" -f "${SCRIPT_ROOT}/hack/testdata/plugins-config-base.yaml" --set router.epp.pluginsConfig.dataLayer.crossReplica.syncerPluginRef=manual
  assert_rejected 'already configures redis-syncer' "${sync[@]}" -f "${SCRIPT_ROOT}/hack/testdata/plugins-config-base.yaml" --set-json 'router.epp.pluginsConfig.plugins=[{"type":"redis-syncer"}]'
  assert_rejected 'requires an llm-d.ai/v1 EndpointPickerConfig' "${sync[@]}" -f "${SCRIPT_ROOT}/hack/testdata/plugins-config-base.yaml" --set router.epp.pluginsConfig.apiVersion=llm-d.ai/v1alpha1
  assert_rejected 'selected pluginsConfigFile' "${sync[@]}" --set router.epp.pluginsConfigFile=missing.yaml
  if [[ ${chart} == llm-d-router-gateway ]]; then
    assert_rejected 'incompatible with priority routing and GKE preferredBackends' "${sync[@]}" --set provider.name=gke --set provider.gke.preferredBackends.enabled=true
  else
    assert_rejected 'incompatible with priority routing and GKE preferredBackends' "${sync[@]}" --set router.proxy.mode=service --set router.proxy.priorityRouting.enabled=true
  fi
  echo "Cross-replica configuration checks passed for ${chart}."
done

echo "Verifying active-active Envoy endpoint discovery..."
chart_path="${SCRIPT_ROOT}/config/charts/llm-d-router-standalone"
args=(--set router.modelServers.matchLabels.app=llm-instance-gateway
      --set router.inferencePool.create=false --set router.proxy.mode=service)

render --set router.epp.replicas=3
"${YQ}" -o=json 'select(.kind == "Service" and .metadata.name == "sync-test-epp") | .spec' "${output}" > "${TEMP_DIR}/service-before.json"

for mode in redis redis-single all-active all-active-single flag autoscaling; do
  case "${mode}" in
    redis) mode_args=("${sync[@]}") ;;
    redis-single) mode_args=("${sync[@]}" --set router.epp.replicas=1) ;;
    all-active) mode_args=(--set router.epp.replicas=3 --set router.epp.haMode=all-active) ;;
    all-active-single) mode_args=(--set router.epp.replicas=1 --set router.epp.haMode=all-active) ;;
    flag) mode_args=(--set router.epp.replicas=3 --set router.epp.flags.ha-enable-leader-election=false) ;;
    autoscaling) mode_args=(--set router.epp.autoscaling.enabled=true) ;;
  esac
  render "${mode_args[@]}"
  assert_yaml "${deployment}" '.spec.strategy.type == "RollingUpdate" and .spec.strategy.rollingUpdate.maxUnavailable == 0'
  assert_yaml "${deployment}" '.spec.template.spec.containers[] | select(.name == "epp") | .args | contains(["--health-checking=true"])'
  assert_yaml "${output}" 'select(.kind == "Service" and .metadata.name == "sync-test-epp-discovery") | (.spec.clusterIP == "None" and .spec.publishNotReadyAddresses != true and .spec.selector."llm-d-router-standalone" == "sync-test-epp" and .spec.ports[0].port == 9002)'
  assert_yaml "${output}" 'select(.kind == "ConfigMap" and .metadata.name == "envoy") | .data."envoy.yaml" | from_yaml | .static_resources.clusters[] | select(.name == "ext_proc") | (.type == "STRICT_DNS" and .connect_timeout == "1s" and .load_assignment.endpoints[0].lb_endpoints[0].endpoint.address.socket_address.address == "sync-test-epp-discovery.default.svc.cluster.local")'
  "${YQ}" -o=json 'select(.kind == "Service" and .metadata.name == "sync-test-epp") | .spec' "${output}" > "${TEMP_DIR}/service-after.json"
  diff -u "${TEMP_DIR}/service-before.json" "${TEMP_DIR}/service-after.json"
done

render "${sync[@]}" --set router.epp.flags.health-checking=true
assert_yaml "${deployment}" '.spec.template.spec.containers[] | select(.name == "epp") | .args | contains(["--health-checking=true"])'
assert_rejected 'active-active Envoy discovery requires' "${sync[@]}" --set router.epp.flags.health-checking=false
assert_rejected 'health-checking must be a boolean' "${sync[@]}" --set-string router.epp.flags.health-checking=false

render "${sync[@]}" --namespace routing --set router.clusterDomain=example.internal
assert_yaml "${output}" 'select(.kind == "ConfigMap" and .metadata.name == "envoy") | .data."envoy.yaml" | from_yaml | .static_resources.clusters[] | select(.name == "ext_proc") | .load_assignment.endpoints[0].lb_endpoints[0].endpoint.address.socket_address.address == "sync-test-epp-discovery.routing.svc.example.internal"'

for mode in default leader flag single single-flag-off sidecar disabled priority gke agentgateway; do
  case "${mode}" in
    default) mode_args=(--set router.epp.replicas=3) ;;
    leader) mode_args=(--set router.epp.haMode=leader-elected) ;;
    flag) mode_args=(--set router.epp.flags.ha-enable-leader-election=true) ;;
    single) mode_args=(--set router.epp.replicas=1 --set router.epp.flags.health-checking=false) ;;
    single-flag-off) mode_args=(--set router.epp.replicas=1 --set-string router.epp.flags.ha-enable-leader-election=False --set router.epp.flags.health-checking=false) ;;
    sidecar) mode_args=("${sync[@]}" --set router.proxy.mode=sidecar) ;;
    disabled) mode_args=("${sync[@]}" --set router.proxy.mode=sidecar --set router.proxy.enabled=false) ;;
    priority) mode_args=(--set router.proxy.priorityRouting.enabled=true) ;;
    gke) mode_args=(--set provider.name=gke --set provider.gke.preferredBackends.enabled=true --set router.epp.flags.ha-enable-leader-election=false) ;;
    agentgateway) mode_args=("${sync[@]}" --set router.proxy.proxyType=agentgateway) ;;
  esac
  render "${mode_args[@]}"
  "${YQ}" eval-all -e '[select(.kind == "Service" and .metadata.name == "sync-test-epp-discovery")] | length == 0' "${output}" > /dev/null
  case "${mode}" in
    default|leader|flag|single|single-flag-off|gke)
      assert_yaml "${output}" 'select(.kind == "ConfigMap" and .metadata.name == "envoy") | .data."envoy.yaml" | from_yaml | .static_resources.clusters[] | select(.name == "ext_proc") | (.connect_timeout == "86400s" and .load_assignment.endpoints[0].lb_endpoints[0].endpoint.address.socket_address.address == "sync-test-epp.default.svc.cluster.local")'
      ;;
    priority)
      assert_yaml "${output}" 'select(.kind == "Service" and .metadata.name == "sync-test-epp-headless") | (.spec.clusterIP == "None" and .spec.publishNotReadyAddresses == true)'
      ;;
    sidecar)
      assert_yaml "${output}" 'select(.kind == "ConfigMap" and .metadata.name == "envoy") | .data."envoy.yaml" | from_yaml | .static_resources.clusters[] | select(.name == "ext_proc") | .load_assignment.endpoints[0].lb_endpoints[0].endpoint.address.socket_address.address == "127.0.0.1"'
      ;;
    agentgateway)
      assert_yaml "${output}" 'select(.kind == "ConfigMap" and .metadata.name == "agentgateway") | .data."config.yaml" | contains("sync-test-epp.default.svc.cluster.local:9002")'
      ;;
  esac
done
echo "Active-active Envoy endpoint discovery checks passed."
