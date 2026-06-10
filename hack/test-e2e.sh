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

set -euox pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EPP_IMAGE="${EPP_IMAGE:-ghcr.io/llm-d/llm-d-router-endpoint-picker:dev}"
SIM_IMAGE="${VLLM_IMAGE:-ghcr.io/llm-d/llm-d-inference-sim:v0.9.1}"
MANIFEST_PATH="${MANIFEST_PATH:-${DIR}/../test/testdata/sim-deployment.yaml}"
USE_KIND="${USE_KIND:-true}"
KIND_NODE_IMAGE="${KIND_NODE_IMAGE:-mirror.gcr.io/kindest/node:v1.32.2}"

KIND_CLUSTER_NAME="inference-e2e"

# Tracks whether this script created the cluster so cleanup knows
# whether to delete it (we never delete a cluster we didn't create ourselves).
CREATED_CLUSTER=""

install_kind() {
  if ! command -v kind &>/dev/null; then
    echo "kind not found, installing..."
    [ "$(uname -m)" = x86_64  ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.29.0/kind-linux-amd64
    [ "$(uname -m)" = aarch64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.29.0/kind-linux-arm64
    chmod +x ./kind
    mv ./kind /usr/local/bin/kind
  else
    echo "kind is already installed."
  fi
}

load_images() {
  local cluster="$1"
  echo "Loading EPP and sim images into kind cluster ${cluster}: ${EPP_IMAGE} ${SIM_IMAGE}"
  CLUSTER_NAME="${cluster}" ./scripts/load_image.sh "${EPP_IMAGE}" "${SIM_IMAGE}"
}

cleanup() {
  echo "Interrupted!"
  if [ -n "${CREATED_CLUSTER}" ] && [ "${E2E_KEEP_CLUSTER_ON_FAILURE:-false}" != "true" ]; then
    echo "Deleting kind cluster '${CREATED_CLUSTER}'"
    kind delete cluster --name "${CREATED_CLUSTER}" 2>/dev/null || true
  fi
  exit 130  # SIGINT (Ctrl+C)
}

# Normally kind cluster cleanup is done by AfterSuite; this trap only fires on
# interruption signals so that a Ctrl+C still cleans up the cluster we created.
trap cleanup INT TERM

if [ "${USE_KIND}" = "true" ]; then
  install_kind
  current_context=$(kubectl config current-context 2>/dev/null || true)
  if [ "${current_context}" = "kind-${KIND_CLUSTER_NAME}" ]; then
    echo "Found an active kind cluster '${KIND_CLUSTER_NAME}' for running the tests..."
    load_images "${KIND_CLUSTER_NAME}"
  else
    if [ -n "${current_context}" ]; then
      echo "WARNING: current kubecontext '${current_context}' is not kind-${KIND_CLUSTER_NAME}, wont use it." >&2
    fi
    # if the cluster already exists but isn't the current context
    if kind get clusters 2>/dev/null | grep -qx "${KIND_CLUSTER_NAME}"; then
      echo "Found existing kind cluster '${KIND_CLUSTER_NAME}', switching context..."
      kubectl config use-context "kind-${KIND_CLUSTER_NAME}"
    else
      echo "Creating new kind cluster '${KIND_CLUSTER_NAME}' for running the tests..."
      kind create cluster --name "${KIND_CLUSTER_NAME}" --image "${KIND_NODE_IMAGE}"
      CREATED_CLUSTER="${KIND_CLUSTER_NAME}"
    fi
    load_images "${KIND_CLUSTER_NAME}"
  fi
else
  # USE_KIND=false: caller is responsible for loading images into the cluster.
  # Useful for testing against a real GPU cluster or a pre-provisioned environment.
  if ! kubectl config current-context >/dev/null 2>&1; then
    echo "No active kubecontext found. Exiting..."
    exit 1
  fi
fi

echo "Running Go e2e tests in ./test/e2e/epp/..."
if [ -n "${E2E_LABEL_FILTER:-}" ]; then
  echo "Label filter: ${E2E_LABEL_FILTER}"
  MANIFEST_PATH="${MANIFEST_PATH}" E2E_IMAGE="${EPP_IMAGE}" \
    go test "${DIR}/../test/e2e/epp/" -v -timeout 45m -ginkgo.v -ginkgo.fail-fast "-ginkgo.label-filter=${E2E_LABEL_FILTER}"
else
  MANIFEST_PATH="${MANIFEST_PATH}" E2E_IMAGE="${EPP_IMAGE}" \
    go test "${DIR}/../test/e2e/epp/" -v -timeout 45m -ginkgo.v -ginkgo.fail-fast
fi
