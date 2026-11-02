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

weaver registry check --registry "$registry"
weaver registry generate router "$tmp" \
    --registry "$registry" --templates hack/telemetry/templates
gofmt -w "$tmp/pkg/common/observability/semconv/"*.go "$tmp/pkg/epp/metrics/"*.go

# Preserve the manually maintained metric catalog outside the generated table.
python3 - "$tmp" <<'PYDOC'
from pathlib import Path
import sys

tmp = Path(sys.argv[1])
doc = Path("docs/metrics.md").read_text()
start = "<!-- BEGIN GENERATED SCHEDULER METRICS -->"
end = "<!-- END GENERATED SCHEDULER METRICS -->"
if doc.count(start) != 1 or doc.count(end) != 1 or doc.index(start) > doc.index(end):
    sys.exit("docs/metrics.md must contain one ordered pair of scheduler metrics markers")
before, rest = doc.split(start)
_, after = rest.split(end)
table = (tmp / "scheduler-metrics.md").read_text()
(tmp / "docs/metrics.md").write_text(before + start + "\n\n" + table + "\n" + end + after)
PYDOC

status=0
for file in pkg/common/observability/semconv/llm_d.go pkg/common/observability/semconv/spans.go \
    pkg/epp/metrics/zz_generated.scheduler_metrics.go docs/telemetry.md docs/metrics.md; do
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
