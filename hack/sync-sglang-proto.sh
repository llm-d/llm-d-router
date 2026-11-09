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

# Moves SGLANG_REF to the newest sgl-project/sglang commit that changed the
# tracked SGLang schemas, regenerates the Go bindings via buf, and writes a
# pull request description.
# Used by .github/workflows/sync-sglang-proto.yaml; also safe to run locally.
#
# Environment:
#   GH_TOKEN       token for the GitHub API (any token can read the public repo)
#   TARGET_REF     optional sgl-project/sglang ref to sync to instead of the
#                  newest commit that changed the tracked schemas
#   PR_BODY_FILE   where to write the pull request description
#                  (default: sglang-proto-sync-pr.md in the temp dir)
#   GITHUB_OUTPUT  set by GitHub Actions; receives changed, ref, and short_ref

set -o errexit
set -o nounset
set -o pipefail

UPSTREAM=sgl-project/sglang
UPSTREAM_BRANCH=main
# Upstream directories (under proto/) compiled by generate-proto-sglang.
SCHEMA_DIRS=(sglang/api/v1 sglang/json/v1)
MAKEFILE=Makefile.gen.mk
GRPC_DIR=pkg/common/grpc/sglang
PR_BODY_FILE="${PR_BODY_FILE:-${TMPDIR:-/tmp}/sglang-proto-sync-pr.md}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

output() {
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "$1=$2" >>"${GITHUB_OUTPUT}"
  fi
}

# Prints the newer of two commits on the upstream branch (linear history).
newer_commit() {
  local status
  status=$(gh api "repos/${UPSTREAM}/compare/$1...$2" --jq .status)
  case "${status}" in
    ahead) echo "$2" ;;
    behind | identical) echo "$1" ;;
    *)
      echo "ERROR: ${UPSTREAM} commits $1 and $2 have diverged (${status})." >&2
      return 1
      ;;
  esac
}

current=$(sed -n 's/^SGLANG_REF ?= //p' "${MAKEFILE}")
if [[ -z "${current}" ]]; then
  echo "ERROR: SGLANG_REF not found in ${MAKEFILE}." >&2
  exit 1
fi

if [[ -n "${TARGET_REF:-}" ]]; then
  latest=$(gh api "repos/${UPSTREAM}/commits/${TARGET_REF}" --jq .sha)
else
  latest=""
  for dir in "${SCHEMA_DIRS[@]}"; do
    sha=$(gh api "repos/${UPSTREAM}/commits?sha=${UPSTREAM_BRANCH}&path=proto/${dir}&per_page=1" --jq '.[0].sha')
    if [[ -z "${latest}" ]]; then
      latest="${sha}"
    else
      latest=$(newer_commit "${latest}" "${sha}")
    fi
  done
fi

if [[ "${latest}" == "${current}" ]]; then
  echo "SGLANG_REF is already the newest commit that changed the tracked schemas (${current})."
  output changed false
  exit 0
fi
echo "Syncing SGLANG_REF ${current} -> ${latest}"

sed -i.bak "s/^SGLANG_REF ?= .*/SGLANG_REF ?= ${latest}/" "${MAKEFILE}"
rm -f "${MAKEFILE}.bak"
make generate-proto-sglang

if git diff --quiet -- "${GRPC_DIR}" && [[ -z "$(git ls-files --others --exclude-standard -- "${GRPC_DIR}")" ]]; then
  echo "The generated Go bindings are unchanged at ${latest}; leaving SGLANG_REF alone."
  git checkout -- "${MAKEFILE}"
  output changed false
  exit 0
fi

# Flag breaking changes. Upstream checks them too, but a maintainer can
# approve one there, and the router has to adapt to it.
BUF_BIN="${LOCALBIN:-./bin}/buf"
if [[ ! -x "${BUF_BIN}" ]]; then
  BUF_BIN="buf"
fi
if command -v "${BUF_BIN}" >/dev/null; then
  path_flags=()
  for dir in "${SCHEMA_DIRS[@]}"; do
    path_flags+=(--path "${dir}")
  done
  if breaking=$("${BUF_BIN}" breaking "${path_flags[@]}" \
    "https://github.com/${UPSTREAM}.git#ref=${latest},subdir=proto,depth=1" \
    --against "https://github.com/${UPSTREAM}.git#ref=${current},subdir=proto,depth=1" 2>&1); then
    breaking_section="\`buf breaking\` (FILE rules) found no breaking changes."
  else
    breaking_section=$(printf '> [!WARNING]\n> `buf breaking` (FILE rules) reported breaking changes. Check that the router still handles these messages before merging.\n\n```\n%s\n```' "${breaking}")
  fi
else
  breaking_section="\`buf\` is not installed, so breaking changes were not checked."
fi

# Upstream pull requests that changed the tracked directories. The upstream
# branch has linear history, so commits after SGLANG_REF's date are the new ones.
since=$(gh api "repos/${UPSTREAM}/commits/${current}" --jq .commit.committer.date)
changes=$(for dir in "${SCHEMA_DIRS[@]}"; do
  gh api "repos/${UPSTREAM}/commits?sha=${latest}&path=proto/${dir}&since=${since}&per_page=100" \
    --jq ".[] | select(.sha != \"${current}\") | .sha + \" \" + (.commit.message | split(\"\n\")[0])"
done | awk '!seen[$1]++' | cut -d' ' -f2- |
  sed -E "s/ \(#([0-9]+)\)$/ (${UPSTREAM/\//\\/}#\1)/; s/^/- /")

short="${latest:0:12}"
cat >"${PR_BODY_FILE}" <<EOF
**What type of PR is this?**
/kind cleanup

**What this PR does / why we need it**:
Syncs \`SGLANG_REF\` and regenerates the SGLang \`sglang.api.v1\` and \`sglang.json.v1\` Go bindings under \`pkg/common/grpc/sglang/gen\` from ${UPSTREAM}@${short}, the newest upstream commit that changed those schemas. Generated by \`.github/workflows/sync-sglang-proto.yaml\`.

Upstream diff: https://github.com/${UPSTREAM}/compare/${current}...${latest}

Upstream changes:
${changes:-- (no matching commits found; see the diff above)}

${breaking_section}

This PR only updates the generated bindings. Using new fields or RPCs belongs in a follow-up PR.

**Release note**:
\`\`\`release-note
NONE
\`\`\`
EOF

echo "Wrote pull request description to ${PR_BODY_FILE}"
output changed true
output ref "${latest}"
output short_ref "${short}"
