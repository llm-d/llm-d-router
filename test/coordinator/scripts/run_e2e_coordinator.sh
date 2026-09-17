#!/bin/bash

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
