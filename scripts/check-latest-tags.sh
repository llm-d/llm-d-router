#!/usr/bin/env bash

# Copyright 2025 The llm-d Authors.
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

# Check that YAML files do not use the ':latest' tag for the simulator image.
# Simulator images should be pinned to a specific version so that builds and
# tests are reproducible.
#
# Usage:
#   ./scripts/check-latest-tags.sh [DIR ...]
#
# When no directories are given the entire repository is scanned.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The simulator image name to check.
SIMULATOR_IMAGE="llm-d-inference-sim"

if [[ $# -gt 0 ]]; then
  SCAN_DIRS=("$@")
else
  SCAN_DIRS=("${SCRIPT_ROOT}")
fi

violations=""

for dir in "${SCAN_DIRS[@]}"; do
  [[ -d "$dir" ]] || continue

  matches="$(grep -rn --include='*.yaml' --include='*.yml' \
    "${SIMULATOR_IMAGE}:latest" "$dir" \
    | grep -v '^\s*#' \
    || true)"

  if [[ -n "$matches" ]]; then
    violations="${violations}${matches}"$'\n'
  fi
done

if [[ -z "$violations" ]]; then
  echo "No '${SIMULATOR_IMAGE}:latest' references found."
  exit 0
fi

echo "ERROR: The following YAML files use '${SIMULATOR_IMAGE}:latest'."
echo "Pin the simulator image to a specific version (e.g. :v0.9.0)."
echo ""
echo "$violations"
exit 1
