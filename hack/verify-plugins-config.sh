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
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "${TEMP_DIR}"' EXIT

# Compare the selected ConfigMap literal, excluding the other plugin files.
extract_config() {
  local file=$1 key=$2
  awk -v key="${key}" '
    $0 == "  " key ": |" || $0 == "  \"" key "\": |" { found++; active=1; next }
    active && /^    / { sub(/^    /, ""); print; next }
    active { active=0 }
    END {
      if (found != 1) {
        print "Expected one ConfigMap entry for " key ", found " (found+0) > "/dev/stderr"
        exit 1
      }
    }
  ' "${file}"
}

assert_config_equal() {
  extract_config "$1" "$3" > "${TEMP_DIR}/expected.yaml"
  extract_config "$2" "$3" > "${TEMP_DIR}/actual.yaml"
  diff -u "${TEMP_DIR}/expected.yaml" "${TEMP_DIR}/actual.yaml"
}

for chart in llm-d-router-gateway llm-d-router-standalone; do
  echo "Verifying structured plugins configuration for ${chart}..."
  chart_path="${SCRIPT_ROOT}/config/charts/${chart}"
  args=(--set router.modelServers.matchLabels.app=llm-instance-gateway)
  if [[ ${chart} == llm-d-router-standalone ]]; then
    args+=(--set router.inferencePool.create=false)
  fi
  "${HELM}" dependency build "${chart_path}"
  baseline="${TEMP_DIR}/${chart}-baseline.yaml"
  output="${TEMP_DIR}/${chart}-render.yaml"
  "${HELM}" template plugins-test "${chart_path}" "${args[@]}" > "${baseline}"

  for empty in '{}' null; do
    "${HELM}" template plugins-test "${chart_path}" "${args[@]}" \
      --set-json "router.epp.pluginsConfig=${empty}" > "${output}"
    diff -u "${baseline}" "${output}"
  done

  values=(-f "${SCRIPT_ROOT}/hack/testdata/plugins-config-base.yaml"
          -f "${SCRIPT_ROOT}/hack/testdata/plugins-config-overlay.yaml")
  for filename in structured-plugins.yaml default-plugins.yaml payload-agnostic.yaml; do
    "${HELM}" template plugins-test "${chart_path}" "${values[@]}" "${args[@]}" \
      --set "router.epp.pluginsConfigFile=${filename}" > "${output}"
    extract_config "${output}" "${filename}" > "${TEMP_DIR}/actual.yaml"
    diff -u "${SCRIPT_ROOT}/hack/testdata/plugins-config-expected.yaml" "${TEMP_DIR}/actual.yaml"
    grep -Fq -- "\"/config/${filename}\"" "${output}"
    extract_config "${output}" extra-plugins.yaml > "${TEMP_DIR}/raw.yaml"
    diff -u <(printf 'apiVersion: llm-d.ai/v1alpha1\nkind: EndpointPickerConfig\n') "${TEMP_DIR}/raw.yaml"
    for builtin in default-plugins.yaml payload-agnostic.yaml; do
      if [[ ${filename} != "${builtin}" ]]; then
        assert_config_equal "${baseline}" "${output}" "${builtin}"
      fi
    done

    if "${HELM}" template plugins-test "${chart_path}" "${values[@]}" "${args[@]}" \
      --set "router.epp.pluginsConfigFile=${filename}" \
      --set-string "router.epp.pluginsCustomConfig.${filename//./\\.}=conflict" \
      > "${output}" 2> "${TEMP_DIR}/error.log"; then
      echo "Accepted conflicting pluginsConfig and pluginsCustomConfig for ${filename}"
      exit 1
    fi
    grep -Fq -- "router.epp.pluginsConfig and router.epp.pluginsCustomConfig cannot both define \"${filename}\"" "${TEMP_DIR}/error.log"
  done

  "${HELM}" template plugins-test "${chart_path}" "${values[@]}" "${args[@]}" \
    --set-json 'router.epp.pluginsConfig.plugins=[{"type":"queue-scorer"}]' > "${output}"
  extract_config "${output}" structured-plugins.yaml > "${TEMP_DIR}/actual.yaml"
  grep -Fxq -- '- type: queue-scorer' "${TEMP_DIR}/actual.yaml"
  if grep -Fq -- 'kv-cache-utilization-scorer' "${TEMP_DIR}/actual.yaml"; then
    echo "Plugin lists were merged instead of replaced"
    exit 1
  fi

  "${HELM}" template plugins-test "${chart_path}" "${values[@]}" "${args[@]}" \
    --set router.epp.pluginsConfig=null \
    --set router.epp.pluginsConfigFile=extra-plugins.yaml > "${output}"
  for builtin in default-plugins.yaml payload-agnostic.yaml; do
    assert_config_equal "${baseline}" "${output}" "${builtin}"
  done
  extract_config "${output}" extra-plugins.yaml > "${TEMP_DIR}/raw.yaml"
  diff -u <(printf 'apiVersion: llm-d.ai/v1alpha1\nkind: EndpointPickerConfig\n') "${TEMP_DIR}/raw.yaml"
  grep -Fq -- '"/config/extra-plugins.yaml"' "${output}"

  for invalid in '"raw YAML"' '[{"type":"queue-scorer"}]'; do
    if "${HELM}" template plugins-test "${chart_path}" "${args[@]}" \
      --set-json "router.epp.pluginsConfig=${invalid}" > "${output}" 2> "${TEMP_DIR}/error.log"; then
      echo "Accepted non-map pluginsConfig: ${invalid}"
      exit 1
    fi
    grep -Fq -- 'router.epp.pluginsConfig must be a map' "${TEMP_DIR}/error.log"
  done
  echo "Structured plugins configuration checks passed for ${chart}."
done
