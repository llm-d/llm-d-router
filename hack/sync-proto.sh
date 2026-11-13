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

# Moves the pinned upstream ref of one engine (SGLANG_REF or VLLM_REF in
# Makefile.gen.mk) to the newest upstream commit that changed the tracked
# schemas, regenerates the Go bindings via buf, and writes a pull request
# description. Used by .github/workflows/sync-proto.yaml. Safe to run locally.
#
# Usage:
#   hack/sync-proto.sh <sglang|vllm> [ref] [--dry-run]
#
#   ref        optional upstream ref to sync to instead of the newest commit
#              that changed the tracked schemas (overrides TARGET_REF)
#   --dry-run  resolve the target commit and list the upstream changes, but
#              do not modify any file
#
# Environment:
#   GH_TOKEN       token for the GitHub API (any token can read the public repo)
#   TARGET_REF     optional upstream ref, same as the ref argument
#   PR_BODY_FILE   where to write the pull request description
#                  (default: <engine>-proto-sync-pr.md in the temp dir)
#   GITHUB_OUTPUT  set by GitHub Actions; receives changed, ref, short_ref,
#                  title, branch, and grpc_dir

set -o errexit
set -o errtrace
set -o nounset
set -o pipefail

usage() {
  echo "usage: $0 <sglang|vllm> [ref] [--dry-run]" >&2
  exit 2
}

DRY_RUN=false
args=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=true ;;
    -*) usage ;;
    *) args+=("${arg}") ;;
  esac
done
if ((${#args[@]} < 1 || ${#args[@]} > 2)); then
  usage
fi
ENGINE="${args[0]}"
TARGET_REF="${args[1]:-${TARGET_REF:-}}"

# Per-engine configuration.
#   SCHEMA_PATHS  upstream paths, relative to the upstream repository root, that
#                 MAKE_TARGET compiles. Only commits that touch them move the
#                 pinned ref.
#   BUF_SUBDIR    the buf module root in the upstream repository. SCHEMA_PATHS
#                 must live under it.
UPSTREAM_BRANCH=main
case "${ENGINE}" in
  sglang)
    UPSTREAM=sgl-project/sglang
    REF_VAR=SGLANG_REF
    MAKE_TARGET=generate-proto-sglang
    GRPC_DIR=pkg/common/grpc/sglang
    BUF_SUBDIR=proto
    SCHEMA_PATHS=(proto/sglang/api/v1 proto/sglang/json/v1)
    BOT_BRANCH=bot/sync-sglang-proto
    DISPLAY_NAME=SGLang
    BINDINGS='SGLang `sglang.api.v1` and `sglang.json.v1`'
    ;;
  vllm)
    UPSTREAM=vllm-project/vllm
    REF_VAR=VLLM_REF
    MAKE_TARGET=generate-proto-vllm-upstream
    GRPC_DIR=pkg/common/grpc/vllm
    BUF_SUBDIR=rust/proto
    # rust/proto is also a Rust crate (Cargo.toml, build.rs, src/). Track the
    # two schema files so that Rust-only commits do not move VLLM_REF.
    SCHEMA_PATHS=(rust/proto/inference.proto rust/proto/control.proto)
    BOT_BRANCH=bot/sync-vllm-proto
    DISPLAY_NAME=vLLM
    BINDINGS='upstream vLLM `vllm.Inference` and `vllm.Control`'
    ;;
  *)
    echo "ERROR: unknown engine '${ENGINE}'." >&2
    usage
    ;;
esac
MAKEFILE=Makefile.gen.mk
PR_BODY_FILE="${PR_BODY_FILE:-${TMPDIR:-/tmp}/${ENGINE}-proto-sync-pr.md}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

output() {
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "$1=$2" >>"${GITHUB_OUTPUT}"
  fi
}

# Fails unless $1 is a full commit SHA. `gh api --jq '.[0].sha'` prints null
# for an empty list (renamed path, wrong branch), and null must not reach the
# Makefile.
require_sha() {
  if [[ ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: ${UPSTREAM} returned '$1' instead of a commit SHA for $2." >&2
    exit 1
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

# Prints a Markdown list of the upstream pull requests that changed a tracked
# path between the two commits. The upstream branch has linear history, so
# commits after the first commit's date are the new ones.
upstream_changes() {
  local from=$1 to=$2 since path
  since=$(gh api "repos/${UPSTREAM}/commits/${from}" --jq .commit.committer.date)
  for path in "${SCHEMA_PATHS[@]}"; do
    gh api "repos/${UPSTREAM}/commits?sha=${to}&path=${path}&since=${since}&per_page=100" \
      --jq ".[] | select(.sha != \"${from}\") | .sha + \" \" + (.commit.message | split(\"\n\")[0])"
  done | awk '!seen[$1]++' | cut -d' ' -f2- |
    sed -E "s/ \(#([0-9]+)\)$/ (${UPSTREAM/\//\\/}#\1)/; s/^/- /"
}

current=$(sed -n "s/^${REF_VAR} ?= //p" "${MAKEFILE}")
if [[ -z "${current}" ]]; then
  echo "ERROR: ${REF_VAR} not found in ${MAKEFILE}." >&2
  exit 1
fi

if [[ -n "${TARGET_REF}" ]]; then
  latest=$(gh api "repos/${UPSTREAM}/commits/${TARGET_REF}" --jq .sha)
  require_sha "${latest}" "ref ${TARGET_REF}"
else
  latest=""
  for path in "${SCHEMA_PATHS[@]}"; do
    sha=$(gh api "repos/${UPSTREAM}/commits?sha=${UPSTREAM_BRANCH}&path=${path}&per_page=1" --jq '.[0].sha')
    require_sha "${sha}" "path ${path}"
    if [[ -z "${latest}" ]]; then
      latest="${sha}"
    else
      latest=$(newer_commit "${latest}" "${sha}")
    fi
  done
fi

if [[ "${latest}" == "${current}" ]]; then
  echo "${REF_VAR} is already at ${current}. Nothing to sync."
  output changed false
  exit 0
fi

if [[ "${DRY_RUN}" == true ]]; then
  echo "Dry run: would sync ${REF_VAR} ${current} -> ${latest}"
  echo "Upstream diff: https://github.com/${UPSTREAM}/compare/${current}...${latest}"
  echo "Upstream changes:"
  upstream_changes "${current}" "${latest}"
  exit 0
fi
echo "Syncing ${REF_VAR} ${current} -> ${latest}"

sed -i.bak "s/^${REF_VAR} ?= .*/${REF_VAR} ?= ${latest}/" "${MAKEFILE}"
rm -f "${MAKEFILE}.bak"
# Do not leave a half-applied ref behind when a later step fails.
trap 'git checkout -- "${MAKEFILE}"' ERR
make "${MAKE_TARGET}"

if git diff --quiet -- "${GRPC_DIR}" && [[ -z "$(git ls-files --others --exclude-standard -- "${GRPC_DIR}")" ]]; then
  echo "The generated Go bindings are unchanged at ${latest}. Leaving ${REF_VAR} alone."
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
  # buf resolves --path relative to the module root (BUF_SUBDIR).
  path_flags=()
  for path in "${SCHEMA_PATHS[@]}"; do
    path_flags+=(--path "${path#"${BUF_SUBDIR}/"}")
  done
  if breaking=$("${BUF_BIN}" breaking "${path_flags[@]}" \
    "https://github.com/${UPSTREAM}.git#ref=${latest},subdir=${BUF_SUBDIR},depth=1" \
    --against "https://github.com/${UPSTREAM}.git#ref=${current},subdir=${BUF_SUBDIR},depth=1" 2>&1); then
    breaking_section="\`buf breaking\` (FILE rules) found no breaking changes."
  else
    breaking_section=$(printf '> [!WARNING]\n> `buf breaking` (FILE rules) reported breaking changes. Check that the router still handles these messages before merging.\n\n```\n%s\n```' "${breaking}")
  fi
else
  breaking_section="\`buf\` is not installed, so breaking changes were not checked."
fi

changes=$(upstream_changes "${current}" "${latest}")

short="${latest:0:12}"
if [[ -n "${TARGET_REF}" ]]; then
  origin="a ref selected by hand (\`${TARGET_REF}\`)"
else
  origin="the newest upstream commit that changed those schemas"
fi
cat >"${PR_BODY_FILE}" <<EOF
**What type of PR is this?**
/kind cleanup

**What this PR does / why we need it**:
Syncs \`${REF_VAR}\` and regenerates the ${BINDINGS} Go bindings under \`${GRPC_DIR}/gen\` from ${UPSTREAM}@${short}, ${origin}. Generated by \`.github/workflows/sync-proto.yaml\`.

Upstream diff: https://github.com/${UPSTREAM}/compare/${current}...${latest}

Upstream changes:
${changes:-- (no matching commits found; see the diff above)}

${breaking_section}

This PR only updates the generated bindings. Support for new fields or RPCs belongs in a follow-up PR.

**Release note**:
\`\`\`release-note
NONE
\`\`\`
EOF

echo "Wrote pull request description to ${PR_BODY_FILE}"
output changed true
output ref "${latest}"
output short_ref "${short}"
output title "Sync ${DISPLAY_NAME} protos to ${UPSTREAM}@${short}"
output branch "${BOT_BRANCH}"
output grpc_dir "${GRPC_DIR}"
