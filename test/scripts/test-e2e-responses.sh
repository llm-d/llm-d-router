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

# Run only the OpenResponses compliance e2e specs against the simulator.
# Identical to test-e2e-router.sh except ginkgo is invoked with --focus.

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# shellcheck source=test/scripts/e2e-common.sh
source "${DIR}/e2e-common.sh"

trap 'e2e_handle_interrupt "e2e-tests"' INT TERM

echo "Running OpenResponses compliance e2e tests (simulator backend)"

# Point the render deployment at the sim image so BeforeSuite createRender
# succeeds without pulling the multi-GB vllm-openai-cpu image.
export VLLM_RENDER_IMAGE="${VLLM_IMAGE}"
export LOAD_VLLM_RENDER_IMAGE=false

# Prepare the local chart dependency before parallel workers render it.
helm dependency build --skip-refresh "${DIR}/../../config/charts/llm-d-router-standalone"

ginkgo run \
  --procs="${E2E_NUM_PROCS:-1}" \
  --timeout 45m \
  -v \
  --fail-fast \
  --focus="OpenResponses compliance" \
  "${DIR}/../e2e/"
