#!/usr/bin/env bash
# compare-coverage.sh <baseline-dir> <current-dir> [threshold] [label] [max-regression]
#
# Compares Go coverage profiles between a baseline and the current run.
# Outputs a markdown table to stdout and, when running in GitHub Actions,
# appends it to $GITHUB_STEP_SUMMARY so it appears in the Job Summary.
#
# Usage:
#   ./scripts/compare-coverage.sh coverage/baseline coverage/ 0 main 2.0
#
# Arguments:
#   baseline-dir    Directory containing baseline *.out coverage profiles
#   current-dir     Directory containing current *.out coverage profiles
#   threshold       Optional minimum total coverage % (default: 0, report only)
#   label           Optional baseline label for the report heading (default: main)
#   max-regression  Optional maximum allowed regression in percentage points
#                   (default: 2.0). Regressions within this tolerance are
#                   reported but do not cause a non-zero exit.

set -euo pipefail

BASELINE_DIR="${1:?baseline-dir required}"
CURRENT_DIR="${2:?current-dir required}"
THRESHOLD="${3:-0}"
LABEL="${4:-main}"
MAX_REGRESSION="${5:-2.0}"

# extract_total <profile.out> → percentage as a bare number, e.g. "72.4"
extract_total() {
    local profile="$1"
    if [[ ! -s "$profile" ]]; then
        echo ""
        return
    fi
    go tool cover -func="$profile" 2>/dev/null \
        | awk '/^total:/{gsub(/%/,"",$NF); print $NF}' || true
}

# delta_str <base> <cur> → e.g. "+1.2" or "-0.5" or "0.0"
delta_str() {
    awk "BEGIN{printf \"%+.1f\", $2 - $1}"
}

# status_icon <base_pct> <cur_pct> <threshold>
status_icon() {
    local base="$1" cur="$2" threshold="$3"
    awk -v base="$base" -v cur="$cur" -v threshold="$threshold" 'BEGIN {
        if (cur == "" || base == "") { print "⚠️ missing data"; exit }
        if (threshold > 0 && cur+0 < threshold+0) { print "❌ below threshold"; exit }
        diff = cur - base
        if (diff < -0.05)                         { print "⬇️ regression"; exit }
        if (diff >  0.05)                         { print "⬆️ improvement"; exit }
        print "✅ no change"
    }'
}

any_regression=0
rows=""

# Find all .out files present in either directory
all_names=()
for f in "$BASELINE_DIR"/*.out "$CURRENT_DIR"/*.out; do
    [[ -f "$f" ]] || continue
    name=$(basename "$f" .out)
    # deduplicate
    if [[ ! " ${all_names[*]-} " =~ " ${name} " ]]; then
        all_names+=("$name")
    fi
done

if [[ ${#all_names[@]} -eq 0 ]]; then
    echo "No coverage profiles found in $BASELINE_DIR or $CURRENT_DIR"
    exit 0
fi

for name in "${all_names[@]}"; do
    base_pct=$(extract_total "$BASELINE_DIR/$name.out")
    cur_pct=$(extract_total  "$CURRENT_DIR/$name.out")

    if [[ -n "$base_pct" && -n "$cur_pct" ]]; then
        delta=$(delta_str "$base_pct" "$cur_pct")
        status=$(status_icon "$base_pct" "$cur_pct" "$THRESHOLD")
        base_fmt="${base_pct}%"
        cur_fmt="${cur_pct}%"
    elif [[ -z "$base_pct" && -n "$cur_pct" ]]; then
        delta="n/a"
        status="🆕 new"
        base_fmt="—"
        cur_fmt="${cur_pct}%"
    elif [[ -n "$base_pct" && -z "$cur_pct" ]]; then
        delta="n/a"
        status="⚠️ missing"
        base_fmt="${base_pct}%"
        cur_fmt="—"
    else
        delta="n/a"
        status="⚠️ missing data"
        base_fmt="—"
        cur_fmt="—"
    fi

    if [[ "$status" == *"below threshold"* ]]; then
        any_regression=1
    elif [[ "$status" == *"regression"* && -n "$base_pct" && -n "$cur_pct" ]]; then
        drop=$(awk "BEGIN{printf \"%.1f\", $base_pct - $cur_pct}")
        exceeds=$(awk -v d="$drop" -v m="$MAX_REGRESSION" 'BEGIN{print (d+0 > m+0) ? 1 : 0}')
        if [[ "$exceeds" -eq 1 ]]; then
            any_regression=1
        fi
    fi

    rows+="| \`$name\` | $base_fmt | $cur_fmt | $delta% | $status |\n"
done

output="$(printf '## Coverage Report vs %s\n\n| Component | Baseline | Current | Delta | Status |\n|-----------|----------|---------|-------|--------|\n%b' "$LABEL" "$rows")"
if [[ "$THRESHOLD" -gt 0 ]]; then
    output+="$(printf '\n> Minimum threshold: **%s%%**' "$THRESHOLD")"
fi
output+="$(printf '\n> Allowed regression tolerance: **%s%%**' "$MAX_REGRESSION")"

printf '%s\n' "$output"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf '%s\n' "$output" >> "$GITHUB_STEP_SUMMARY"
fi

exit $any_regression
