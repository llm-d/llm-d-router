#!/usr/bin/env bash

# Copyright 2026 The llm-d Authors.
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

study_namespace=${NAMESPACE:-epp-tracing-overhead}
study_selector=${EPP_LABEL_SELECTOR:-llm-d-router-gateway=epp-tracing-epp}
study_output_dir=${OUTPUT_DIR:-./epp-pprof}
study_cpu_seconds=${CPU_SECONDS:-30}
study_base_port=${BASE_PORT:-19090}

mkdir -p "$study_output_dir"
study_pods=()
while IFS= read -r pod; do
  [[ -n "$pod" ]] && study_pods+=("$pod")
done < <(
  kubectl get pods -n "$study_namespace" -l "$study_selector" \
    --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
)

if (( ${#study_pods[@]} == 0 )); then
  echo "no running EPP pods matched selector $study_selector" >&2
  exit 1
fi

capture_pod() {
  local pod=$1
  local port=$2
  local pod_dir="$study_output_dir/$pod"
  local forward_log="$pod_dir/port-forward.log"
  mkdir -p "$pod_dir"

  kubectl port-forward -n "$study_namespace" "pod/$pod" \
    "$port:9090" >"$forward_log" 2>&1 &
  local forward_pid=$!

  trap 'kill "$forward_pid" 2>/dev/null || true
    wait "$forward_pid" 2>/dev/null || true' EXIT

  for _ in $(seq 1 50); do
    if curl --fail --silent "http://127.0.0.1:$port/debug/pprof/" \
      >/dev/null; then
      break
    fi
    sleep 0.2
  done

  curl --fail --silent --show-error \
    "http://127.0.0.1:$port/debug/pprof/profile?seconds=$study_cpu_seconds" \
    --output "$pod_dir/cpu.pprof"
  curl --fail --silent --show-error \
    "http://127.0.0.1:$port/debug/pprof/heap" \
    --output "$pod_dir/heap.pprof"

  test -s "$pod_dir/cpu.pprof"
  test -s "$pod_dir/heap.pprof"
}

study_pids=()
for index in "${!study_pods[@]}"; do
  capture_pod "${study_pods[$index]}" "$((study_base_port + index))" &
  study_pids+=("$!")
done

study_status=0
for pid in "${study_pids[@]}"; do
  if ! wait "$pid"; then
    study_status=1
  fi
done

exit "$study_status"
