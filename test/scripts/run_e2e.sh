#!/bin/bash

set -euo pipefail

cleanup() {
    echo "Interrupted!"
    if [ "${E2E_KEEP_CLUSTER_ON_FAILURE:-false}" = "true" ]; then
        echo "Keeping kind cluster 'e2e-tests' (E2E_KEEP_CLUSTER_ON_FAILURE=true)"
    else
        echo "Deleting kind cluster 'e2e-tests'"
        kind delete cluster --name e2e-tests 2>/dev/null || true
    fi
    exit 130  # SIGINT (Ctrl+C)
}

# Set trap only for interruption signals
# Normally kind cluster cleanup is done by AfterSuite
trap cleanup INT TERM

echo "Running end to end tests"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -n "${E2E_LABEL_FILTER:-}" ]; then
  echo "Label filter: ${E2E_LABEL_FILTER}"
  go test -v -timeout 45m "${DIR}/../e2e/" -ginkgo.v -ginkgo.fail-fast "-ginkgo.label-filter=${E2E_LABEL_FILTER}"
else
  go test -v -timeout 45m "${DIR}/../e2e/" -ginkgo.v -ginkgo.fail-fast
fi
