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

set -o errexit
set -o nounset
set -o pipefail

DEST_CHART_DIR=${DEST_CHART_DIR:-bin/}

EXTRA_TAG=${EXTRA_TAG:-$(git branch --show-current)}
CHART_VERSION=${CHART_VERSION:-"v0"}
IMAGE_REGISTRY=${IMAGE_REGISTRY:-ghcr.io/llm-d}
AGENTGATEWAY_TAG=${AGENTGATEWAY_TAG:-latest-dev}
CHART_SUFFIX=${CHART_SUFFIX:-""}
EPP_RELEASE_IMAGE_REPOSITORY=${EPP_RELEASE_IMAGE_REPOSITORY:-llm-d-router-endpoint-picker}
LATENCY_PREDICTOR_TAG=${LATENCY_PREDICTOR_TAG:-latest}
export EXTRA_TAG AGENTGATEWAY_TAG IMAGE_REGISTRY EPP_RELEASE_IMAGE_REPOSITORY LATENCY_PREDICTOR_TAG CHART_SUFFIX

HELM_CHART_REPO=${HELM_CHART_REPO:-${IMAGE_REGISTRY}/charts}
CHART=${CHART:-llm-d-router-gateway}

HELM=${HELM:-./bin/helm}

readonly semver_regex='^v([0-9]+)(\.[0-9]+){1,2}(-rc.[0-9]+)?$'

chart_versions=("${CHART_VERSION}")

# PR builds version charts as <latest release tag>-pr.<n>, overwritten on each
# push, plus an immutable <base>-pr.<n>+sha.<short sha> snapshot. The release
# tag base keeps PR charts sortable against production chart versions.
if [[ -n "${PR_NUMBER:-}" ]]; then
  : "${CHART_VERSION_BASE:?CHART_VERSION_BASE is required for PR builds}"
  : "${PR_SHA:?PR_SHA is required for PR builds}"

  # PR images are published to the staging registry, so the chart has to point
  # there instead of at the production defaults in values.yaml.
  ${YQ} -i \
    '.router.epp.image.registry=strenv(IMAGE_REGISTRY) |
     .router.epp.image.repository=strenv(EPP_RELEASE_IMAGE_REPOSITORY) |
     .router.epp.image.tag=strenv(EXTRA_TAG) |
     .router.epp.image.pullPolicy="IfNotPresent"' \
    config/charts/${CHART}/values.yaml

  chart_versions=("${CHART_VERSION_BASE}-pr.${PR_NUMBER}" "${CHART_VERSION_BASE}-pr.${PR_NUMBER}+sha.${PR_SHA}")
fi

if [[ ${EXTRA_TAG} =~ ${semver_regex} ]]
then
  ${YQ} -i \
    '.router.epp.image.registry=strenv(IMAGE_REGISTRY) |
     .router.epp.image.repository=strenv(EPP_RELEASE_IMAGE_REPOSITORY) |
     .router.epp.image.tag=strenv(EXTRA_TAG) |
     .router.epp.image.pullPolicy="IfNotPresent"' \
    config/charts/${CHART}/values.yaml
  if [[ ! ${LATENCY_PREDICTOR_TAG} =~ ${semver_regex} ]]; then
    echo "ERROR: LATENCY_PREDICTOR_TAG must be a semver value on a release branch, got '${LATENCY_PREDICTOR_TAG}'"
    exit 1
  fi
  ${YQ} -i \
    '.router.latencyPredictor.trainingServer.image.registry=strenv(IMAGE_REGISTRY) |
     .router.latencyPredictor.trainingServer.image.repository="llm-d-latency-predictor-training-server" |
     .router.latencyPredictor.trainingServer.image.tag=strenv(LATENCY_PREDICTOR_TAG) |
     .router.latencyPredictor.trainingServer.image.pullPolicy="IfNotPresent" |
     .router.latencyPredictor.predictionServers.image.registry=strenv(IMAGE_REGISTRY) |
     .router.latencyPredictor.predictionServers.image.repository="llm-d-latency-predictor-prediction-server" |
     .router.latencyPredictor.predictionServers.image.tag=strenv(LATENCY_PREDICTOR_TAG) |
     .router.latencyPredictor.predictionServers.image.pullPolicy="IfNotPresent"' \
    config/charts/${CHART}/values.yaml
  if [[ ${CHART} == "llm-d-router-standalone" ]]; then
    ${YQ} -i \
      '.router.proxy.presets.agentgateway.image="cr.agentgateway.dev/agentgateway:" + strenv(AGENTGATEWAY_TAG)' \
      config/charts/${CHART}/values.yaml
  fi
  chart_versions=("${EXTRA_TAG}")
fi

# If suffix is defined, dynamically rename the chart in Chart.yaml before packaging and ensure it gets reverted on exit
if [[ -n "${CHART_SUFFIX}" ]]; then
  cleanup() {
    echo "reverting Chart.yaml name back to ${CHART}..."
    ${YQ} -i ".name = \"${CHART}\"" "config/charts/${CHART}/Chart.yaml"
  }
  trap cleanup EXIT

  ${YQ} -i ".name = .name + \"${CHART_SUFFIX}\"" "config/charts/${CHART}/Chart.yaml"
fi

# Update dependencies
${HELM} dependency update "config/charts/${CHART}"

for version in "${chart_versions[@]}"; do
  # Create the package
  ${HELM} package --version "${version}" --app-version "${version}" "config/charts/${CHART}" -d "${DEST_CHART_DIR}"

  # Push the package
  echo "pushing chart to ${HELM_CHART_REPO}"
  ${HELM} push "${DEST_CHART_DIR}${CHART}${CHART_SUFFIX}-${version}.tgz" "oci://${HELM_CHART_REPO}"
done
