#!/bin/bash

set -euo pipefail

cleanup() {
    echo "Interrupted!"
    if [ "${E2E_KEEP_CLUSTER_ON_FAILURE:-false}" = "true" ]; then
        echo "Keeping kind cluster 'e2e-coordinator-tests' (E2E_KEEP_CLUSTER_ON_FAILURE=true)"
    else
        echo "Deleting kind cluster 'e2e-coordinator-tests'"
        kind delete cluster --name e2e-coordinator-tests 2>/dev/null || true
    fi
    exit 130
}

trap cleanup INT TERM

echo "Running coordinator end-to-end tests"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NAMESPACE="${NAMESPACE:-default}" go test -v -timeout 120m ${DIR}/../e2e/coordinator/ -ginkgo.v -ginkgo.fail-fast
