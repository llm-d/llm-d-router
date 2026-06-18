#!/bin/bash

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

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_ROOT=$(dirname "${BASH_SOURCE}")/..
GOMODCACHE="${GOMODCACHE:-$(go env GOMODCACHE)}"

GATEWAY_API_VERSION="${GATEWAY_API_VERSION:-$(go list -m -f '{{.Version}}' sigs.k8s.io/gateway-api)}"
GATEWAY_API_MOD="${GOMODCACHE}/sigs.k8s.io/gateway-api@${GATEWAY_API_VERSION}"

GIE_VERSION="${GIE_VERSION:-$(go list -m -f '{{.Version}}' sigs.k8s.io/gateway-api-inference-extension)}"
GIE_MOD="${GOMODCACHE}/sigs.k8s.io/gateway-api-inference-extension@${GIE_VERSION}"

GKE_GATEWAY_API_VERSION="${GKE_GATEWAY_API_VERSION:-v1.4.0}"
ISTIO_VERSION="${ISTIO_VERSION:-1.26.2}"

KUBECTL_VALIDATE="${KUBECTL_VALIDATE:-${SCRIPT_ROOT}/bin/kubectl-validate}"
TEMP_DIR=$(mktemp -d)

cleanup() {
  rm -rf "${TEMP_DIR}" || true
}
trap cleanup EXIT

fetch_crds() {
  local url="$1"
  curl -sL "${url}" -o "${TEMP_DIR}/$(basename "${url}")"
}

copy_crds() {
  local src="$1"
  cp "${src}" "${TEMP_DIR}/$(basename "${src}")"
}

main() {
  go mod download sigs.k8s.io/gateway-api-inference-extension sigs.k8s.io/gateway-api
  # Use local 'config/crd/bases', run "make generate" to regenerate llm-d CRDs
  cp "${SCRIPT_ROOT}/config/crd/bases/"*.yaml "${TEMP_DIR}/"
  # GIE (Gateway API Inference Extension) CRDs - owned by upstream GIE
  copy_crds "${GIE_MOD}/config/crd/bases/inference.networking.k8s.io_inferencepools.yaml"
  copy_crds "${GIE_MOD}/config/crd/bases/inference.networking.x-k8s.io_inferencepoolimports.yaml"
  # GW API CRD
  copy_crds "${GATEWAY_API_MOD}/config/crd/standard/gateway.networking.k8s.io_gatewayclasses.yaml"
  copy_crds "${GATEWAY_API_MOD}/config/crd/standard/gateway.networking.k8s.io_gateways.yaml"
  copy_crds "${GATEWAY_API_MOD}/config/crd/standard/gateway.networking.k8s.io_httproutes.yaml"
  # GKE CRD
  fetch_crds "https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/refs/tags/${GKE_GATEWAY_API_VERSION}/config/crd/networking.gke.io_gcpbackendpolicies.yaml"
  fetch_crds "https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/refs/tags/${GKE_GATEWAY_API_VERSION}/config/crd/networking.gke.io_healthcheckpolicies.yaml"
  # Istio CRD
  fetch_crds "https://raw.githubusercontent.com/istio/istio/refs/tags/${ISTIO_VERSION}/manifests/charts/base/files/crd-all.gen.yaml"

  # Run the install command in case this script runs from a different bash
  # source (such as in the verify-all script)
  make kubectl-validate

  ${KUBECTL_VALIDATE} "${TEMP_DIR}"
  ${KUBECTL_VALIDATE} "${SCRIPT_ROOT}/config/manifests" --local-crds "${TEMP_DIR}"
}

main
