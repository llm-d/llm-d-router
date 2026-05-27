#!/bin/bash
#
# Runs the e-p-d-pools end-to-end suite. Mirrors run_e2e.sh but targets
# test/e2e/epd_pools/ and uses a distinct kind cluster name
# (`e2e-pools-tests`) so it doesn't collide with `make test-e2e`.

set -euo pipefail

cleanup() {
    echo "Interrupted!"
    if [ "${E2E_KEEP_CLUSTER_ON_FAILURE:-false}" = "true" ]; then
        echo "Keeping kind cluster 'e2e-pools-tests' (E2E_KEEP_CLUSTER_ON_FAILURE=true)"
    else
        echo "Deleting kind cluster 'e2e-pools-tests'"
        kind delete cluster --name e2e-pools-tests 2>/dev/null || true
    fi
    exit 130
}

# INT/TERM only — Ginkgo's ReportAfterSuite owns the normal cleanup path.
trap cleanup INT TERM

echo "Running e-p-d-pools end-to-end tests"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
go test -v -timeout 45m ${DIR}/../e2e/epd_pools/ -ginkgo.v -ginkgo.fail-fast
