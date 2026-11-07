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
mkdir -p "$tmp/hack" "$tmp/pkg/common/observability" "$tmp/pkg/epp/metrics" "$tmp/docs"
cp "$root/hack/telemetry.sh" "$tmp/hack/"
cp -R "$root/hack/telemetry" "$tmp/hack/"
cp -R "$root/pkg/common/observability/semconv" "$tmp/pkg/common/observability/"
cp "$root/pkg/epp/metrics/zz_generated.scheduler_metrics.go" "$tmp/pkg/epp/metrics/"
cp "$root/docs/telemetry.md" "$root/docs/metrics.md" "$tmp/docs/"

bash "$tmp/hack/telemetry.sh" verify
bash "$tmp/hack/telemetry.sh" update
bash "$tmp/hack/telemetry.sh" verify

expect_failure() {
    if bash "$tmp/hack/telemetry.sh" verify >"$tmp/failure.log" 2>&1; then
        echo "ERROR: verification accepted $1" >&2
        exit 1
    fi
}

for file in pkg/common/observability/semconv/llm_d.go pkg/common/observability/semconv/spans.go \
    pkg/epp/metrics/zz_generated.scheduler_metrics.go docs/telemetry.md; do
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

# Only the marked table is generated. Preserve edits to the rest of the catalog.
printf '\nManually maintained metric documentation.\n' >>"$tmp/docs/metrics.md"
bash "$tmp/hack/telemetry.sh" verify
bash "$tmp/hack/telemetry.sh" update
grep -q 'Manually maintained metric documentation' "$tmp/docs/metrics.md"
cp "$tmp/docs/metrics.md" "$tmp/original"
sed 's/Total number of scheduling attempts./Edited scheduling description./' "$tmp/original" >"$tmp/docs/metrics.md"
expect_failure 'edited scheduler table'
grep -q 'Generated telemetry is stale' "$tmp/failure.log"
grep -q 'Edited scheduling description' "$tmp/docs/metrics.md"
bash "$tmp/hack/telemetry.sh" update
cmp "$tmp/original" "$tmp/docs/metrics.md"

# Missing, duplicate, or reversed markers must fail rather than dropping prose.
for markers in missing duplicate reversed; do
    python3 - "$tmp/original" "$tmp/docs/metrics.md" "$markers" <<'PYMARKERS'
from pathlib import Path
import sys

doc = Path(sys.argv[1]).read_text()
start = "<!-- BEGIN GENERATED SCHEDULER METRICS -->"
end = "<!-- END GENERATED SCHEDULER METRICS -->"
if sys.argv[3] == "missing":
    doc = doc.replace(start, "")
elif sys.argv[3] == "duplicate":
    doc += "\n" + start + "\n"
else:
    doc = doc.replace(start, "TEMP_MARKER").replace(end, start).replace("TEMP_MARKER", end)
Path(sys.argv[2]).write_text(doc)
PYMARKERS
    cp "$tmp/docs/metrics.md" "$tmp/before"
    expect_failure "$markers scheduler markers"
    grep -q 'must contain one ordered pair' "$tmp/failure.log"
    cmp "$tmp/before" "$tmp/docs/metrics.md"
    if bash "$tmp/hack/telemetry.sh" update >"$tmp/failure.log" 2>&1; then
        echo "ERROR: update accepted $markers scheduler markers" >&2
        exit 1
    fi
    grep -q 'must contain one ordered pair' "$tmp/failure.log"
    cmp "$tmp/before" "$tmp/docs/metrics.md"
done
mv "$tmp/original" "$tmp/docs/metrics.md"

scheduler_registry="$tmp/pkg/common/observability/semconv/registry/scheduler-metrics.yaml"
cp "$scheduler_registry" "$tmp/original"
sed 's/Total number of scheduling attempts./Updated scheduling description./' "$tmp/original" >"$scheduler_registry"
expect_failure 'scheduler registry changed without regeneration'
grep -q 'Generated telemetry is stale' "$tmp/failure.log"
bash "$tmp/hack/telemetry.sh" update
grep -q 'Updated scheduling description' "$tmp/docs/metrics.md"
grep -q 'Updated scheduling description' "$tmp/pkg/epp/metrics/zz_generated.scheduler_metrics.go"
bash "$tmp/hack/telemetry.sh" verify
mv "$tmp/original" "$scheduler_registry"
bash "$tmp/hack/telemetry.sh" update

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
grep -Fq 'duplicate entry with key "groups"' "$tmp/failure.log"
echo 'Telemetry generation and drift checks passed.'
