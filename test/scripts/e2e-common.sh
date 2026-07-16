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

# Shared helpers for the e2e runner scripts. Source, do not execute.

# e2e_handle_interrupt deletes the named kind cluster on Ctrl-C and exits 130.
# An empty cluster name means there is nothing to delete (the caller did not
# create a cluster it owns). E2E_KEEP_CLUSTER_ON_FAILURE=true keeps the cluster.
e2e_handle_interrupt() {
  local cluster="$1"
  echo "Interrupted!"
  if [ -n "${cluster}" ] && [ "${E2E_KEEP_CLUSTER_ON_FAILURE:-false}" != "true" ]; then
    echo "Deleting kind cluster '${cluster}'"
    kind delete cluster --name "${cluster}" 2>/dev/null || true
  elif [ -n "${cluster}" ]; then
    echo "Keeping kind cluster '${cluster}' (E2E_KEEP_CLUSTER_ON_FAILURE=true)"
  fi
  exit 130 # SIGINT (Ctrl+C)
}

# run_ginkgo_suite runs the Ginkgo e2e suite in the given package directory,
# applying E2E_LABEL_FILTER when set.
run_ginkgo_suite() {
  local pkg="$1"
  if [ -n "${E2E_LABEL_FILTER:-}" ]; then
    echo "Label filter: ${E2E_LABEL_FILTER}"
    # Tests not being run in parallel using Ginkgo
    E2E_NUM_PROCS=1 go test -v -timeout 45m "${pkg}" -ginkgo.v -ginkgo.fail-fast "-ginkgo.label-filter=${E2E_LABEL_FILTER}"
  else
    ginkgo run --procs=${E2E_NUM_PROCS} --timeout 45m -v --fail-fast "${pkg}"
  fi
}
