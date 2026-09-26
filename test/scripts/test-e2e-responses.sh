#!/usr/bin/env bash
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

ginkgo run \
  --procs="${E2E_NUM_PROCS:-1}" \
  --timeout 45m \
  -v \
  --fail-fast \
  --focus="OpenResponses compliance" \
  "${DIR}/../e2e/"
