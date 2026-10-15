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

set -euo pipefail

case "${1:-}" in
    update|verify) mode=$1 ;;
    *) echo "Usage: $0 update|verify" >&2; exit 2 ;;
esac
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
registry=pkg/common/observability/semconv/registry
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

bash hack/weaver.sh registry check --registry "$registry"
bash hack/weaver.sh registry generate router "$tmp" \
    --registry "$registry" --templates hack/telemetry/templates
gofmt -w "$tmp/pkg/common/observability/semconv/"*.go

status=0
for file in pkg/common/observability/semconv/llm_d.go pkg/common/observability/semconv/spans.go docs/telemetry.md; do
    if [[ "$mode" == update ]]; then
        cp "$tmp/$file" "$file"
    elif ! diff -u "$file" "$tmp/$file"; then
        status=1
    fi
done
if [[ "$status" != 0 ]]; then
    echo 'Generated telemetry is stale. Run make update-telemetry.' >&2
fi
exit "$status"
