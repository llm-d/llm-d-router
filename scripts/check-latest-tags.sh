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

# Check that YAML files do not use the ':latest' image tag.
# Container images should be pinned to a specific version so that builds
# and tests are reproducible.
#
# Usage:
#   ./scripts/check-latest-tags.sh [--warn] [DIR ...]
#
# Flags:
#   --warn   Print violations but exit 0 (warn-only mode).
#
# When no directories are given the entire repository is scanned.

set -o errexit
set -o nounset
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WARN_ONLY=false

# Parse flags.
args=()
for arg in "$@"; do
  case "$arg" in
    --warn) WARN_ONLY=true ;;
    *)      args+=("$arg") ;;
  esac
done
set -- "${args[@]+"${args[@]}"}"

if [[ $# -gt 0 ]]; then
  SCAN_DIRS=("$@")
else
  SCAN_DIRS=("${REPO_ROOT}")
fi

violations=""

for dir in "${SCAN_DIRS[@]}"; do
  [[ -d "$dir" ]] || continue

  # Find YAML files, pruning .git and vendor trees for speed.
  # Use grep -Hn (force filenames, no recursion) since find already
  # supplies explicit paths. Match "image:" as a YAML key (leading
  # whitespace, or column 0 after the grep line-number prefix) to
  # avoid substring hits in unrelated fields.
  matches="$(find "$dir" \
    -path '*/.git' -prune -o \
    -path '*/vendor' -prune -o \
    -path '*/node_modules' -prune -o \
    -type f \( -name '*.yaml' -o -name '*.yml' \) -print0 \
    | xargs -0 grep -Hn ':latest' \
    | grep -E '([[:space:]]image:|:[0-9]+:image:)' \
    | grep -v ':latest-' \
    | grep -v 'description:' \
    | grep -v '<your-registry>' \
    | awk -F: '{content = substr($0, index($0,$3)); if (content !~ /^[[:space:]]*#/) print}' \
    || true)"

  if [[ -n "$matches" ]]; then
    violations="${violations}${matches}"$'\n'
  fi
done

if [[ -z "$violations" ]]; then
  echo "No ':latest' image tags found in YAML files."
  exit 0
fi

if [[ "$WARN_ONLY" == true ]]; then
  echo "WARNING: The following YAML files use the ':latest' image tag."
else
  echo "ERROR: The following YAML files use the ':latest' image tag."
fi
echo "Pin images to a specific version for reproducible builds."
echo ""
echo "$violations"

if [[ "$WARN_ONLY" == true ]]; then
  exit 0
fi
exit 1
