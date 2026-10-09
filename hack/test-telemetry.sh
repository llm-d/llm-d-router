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

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$root/bin" "$tmp/hack" "$tmp/pkg/common/observability" "$tmp/docs"
cp "$root/hack/telemetry.sh" "$root/hack/weaver.sh" "$tmp/hack/"
cp -R "$root/hack/telemetry" "$tmp/hack/"
cp -R "$root/pkg/common/observability/semconv" "$tmp/pkg/common/observability/"
cp "$root/docs/telemetry.md" "$tmp/docs/"
# Reuse the platform-specific tool cache, but keep every edited file isolated.
ln -s "$root/bin" "$tmp/bin"

bash "$tmp/hack/telemetry.sh" verify
bash "$tmp/hack/telemetry.sh" update
bash "$tmp/hack/telemetry.sh" verify

expect_failure() {
    if bash "$tmp/hack/telemetry.sh" verify >"$tmp/failure.log" 2>&1; then
        echo "ERROR: verification accepted $1" >&2
        exit 1
    fi
}

for file in pkg/common/observability/semconv/llm_d.go pkg/common/observability/semconv/spans.go docs/telemetry.md; do
    cp "$tmp/$file" "$tmp/original"
    printf '\nmanual edit\n' >>"$tmp/$file"
    expect_failure "edited $file"
    grep -q 'Generated telemetry is stale' "$tmp/failure.log"
    grep -q 'manual edit' "$tmp/$file"
    mv "$tmp/original" "$tmp/$file"
done

mv "$tmp/docs/telemetry.md" "$tmp/original"
expect_failure 'missing generated catalog'
[[ ! -e "$tmp/docs/telemetry.md" ]]
mv "$tmp/original" "$tmp/docs/telemetry.md"

registry="$tmp/pkg/common/observability/semconv/registry/attributes.yaml"
cp "$registry" "$tmp/original"
# Changing a description must update both the Go documentation and catalog.
sed 's/Custom llm-d router trace attributes./Modified registry description./' "$registry" >"$tmp/changed"
mv "$tmp/changed" "$registry"
expect_failure 'registry changed without regeneration'
grep -q 'Generated telemetry is stale' "$tmp/failure.log"
mv "$tmp/original" "$registry"

printf '\ngroups: [invalid\n' >>"$registry"
expect_failure 'invalid registry'
echo 'Telemetry generation and drift checks passed.'
