#!/usr/bin/env bash

# Copyright 2026 The Kubernetes Authors.
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

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# shellcheck source=test/scripts/e2e-common.sh
source "${DIR}/e2e-common.sh"

# Set trap only for interruption signals.
# Normally kind cluster cleanup is done by AfterSuite; this trap deletes the
# e2e-tests cluster on Ctrl-C. The delete is unconditional and only meaningful
# when the suite created that cluster (K8S_CONTEXT unset); when running against
# an existing context it is a no-op unless a cluster named e2e-tests happens to
# exist.
trap 'e2e_handle_interrupt "e2e-tests"' INT TERM

echo "Running end to end tests"

run_ginkgo_suite "${DIR}/../e2e/"
