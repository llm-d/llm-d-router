#!/usr/bin/env bash
# Migrate one or more directories or files from the source repo into this repo,
# rewriting Go import paths throughout.
#
# Usage:
#   ./scripts/migrate-gaie-paths.sh [--since <ref>] <src1> <dest1> [<src2> <dest2> ...]
#
#   --since <ref>  tag or commit SHA marking the last sync point; when given,
#                  only commits after <ref> are cherry-picked (incremental sync).
#                  The ref to pass next time is printed in the PR title.
#                  Omit for the initial full migration.
#
# Requires: git filter-repo  (pip/pipx install git-filter-repo)
#
# NOTE
# Must merge the initial migration PR via "Create a merge commit" to preserve history.
# Incremental migrations can use "merge commit" (cleanest audit trail) or
# "rebase and merge" (for a slightly cleaner linear history).

set -euo pipefail

# Configuration
LOCAL_PATH="/tmp"  # override as needed

SOURCE_ORG="kubernetes-sigs"
SOURCE_REPO_NAME="gateway-api-inference-extension"
SOURCE_UPSTREAM_URL="git@github.com:${SOURCE_ORG}/${SOURCE_REPO_NAME}.git"

DEST_ORG="llm-d"
DEST_REPO_NAME="llm-d-router"
DEST_UPSTREAM_URL="git@github.com:${DEST_ORG}/${DEST_REPO_NAME}.git"

UPSTREAM_REMOTE="upstream"
ORIGIN_REMOTE="origin"
MAIN_BRANCH="main"

SOURCE_DIR="${LOCAL_PATH}/${SOURCE_REPO_NAME}"
DEST_DIR="${LOCAL_PATH}/${DEST_REPO_NAME}"
FILTER_WORK_DIR="${LOCAL_PATH}/${SOURCE_REPO_NAME}-filter-work"

usage() {
  cat <<EOF
Usage: $0 [--since <ref>] <src1> <dest1> [<src2> <dest2> ...]

  --since <ref>  tag or commit SHA of the last migration/sync (from the PR title)
  src            path relative to source repo root  (e.g. pkg/common)
  dest           path relative to dest repo root    (e.g. pkg/common/igw)

Repos are cloned to ${LOCAL_PATH} if not present, otherwise updated to ${UPSTREAM_REMOTE}/${MAIN_BRANCH}.
EOF
  exit 1
}

# Parse --since flag
SINCE_REF=""
if [[ "${1:-}" == "--since" ]]; then
  [[ $# -ge 2 ]] || usage
  SINCE_REF="$2"
  shift 2
fi

[[ $# -ge 2 && $(( $# % 2 )) -eq 0 ]] || usage

declare -a SRC_PATHS DEST_PATHS
while [[ $# -gt 0 ]]; do
  SRC_PATHS+=("${1%/}")
  DEST_PATHS+=("${2%/}")
  shift 2
done

# Preflight checks
if ! command -v git-filter-repo &>/dev/null && ! git filter-repo --version &>/dev/null 2>&1; then
  echo "error: git filter-repo not installed (pip install git-filter-repo)"
  exit 1
fi

GITHUB_USER="${GITHUB_USER:-$(gh api user --jq .login 2>/dev/null || git config github.user 2>/dev/null || echo "")}"
if [[ -z "${GITHUB_USER}" ]]; then
  echo "error: cannot determine GitHub username; set GITHUB_USER=<handle>"
  exit 1
fi
DEST_ORIGIN_URL="git@github.com:${GITHUB_USER}/${DEST_REPO_NAME}.git"

ensure_repo() {
  local dir="$1" upstream_url="$2" origin_url="${3:-}"
  if [[ ! -d "${dir}/.git" ]]; then
    git clone --origin "${UPSTREAM_REMOTE}" "${upstream_url}" "${dir}"
    [[ -n "${origin_url}" ]] && git -C "${dir}" remote add "${ORIGIN_REMOTE}" "${origin_url}"
  else
    git -C "${dir}" fetch "${UPSTREAM_REMOTE}"
    git -C "${dir}" checkout "${MAIN_BRANCH}"
    git -C "${dir}" merge --ff-only "${UPSTREAM_REMOTE}/${MAIN_BRANCH}"
  fi
}

ensure_repo "${SOURCE_DIR}" "${SOURCE_UPSTREAM_URL}"
ensure_repo "${DEST_DIR}"   "${DEST_UPSTREAM_URL}" "${DEST_ORIGIN_URL}"

# Validate --since ref exists in source
if [[ -n "${SINCE_REF}" ]]; then
  if ! git -C "${SOURCE_DIR}" rev-parse --verify "${SINCE_REF}^{commit}" &>/dev/null; then
    echo "error: '${SINCE_REF}' not found in source repo"
    exit 1
  fi
fi

SOURCE_SHA=$(git -C "${SOURCE_DIR}" rev-parse --short HEAD)
SOURCE_MODULE=$(grep -m1 '^module ' "${SOURCE_DIR}/go.mod" | awk '{print $2}')
DEST_MODULE=$(grep -m1 '^module ' "${DEST_DIR}/go.mod" | awk '{print $2}')

BRANCH_SLUG=$(IFS='-'; echo "${DEST_PATHS[*]}" | tr '/' '-')
BRANCH_NAME="migrate/${BRANCH_SLUG}"
[[ -n "${SINCE_REF}" ]] && BRANCH_NAME="migrate/since-${SINCE_REF}-${BRANCH_SLUG}"

echo "source:      ${SOURCE_DIR} (${SOURCE_MODULE})"
echo "destination: ${DEST_DIR} (${DEST_MODULE})"
[[ -n "${SINCE_REF}" ]] && echo "since:       ${SINCE_REF}"
for i in "${!SRC_PATHS[@]}"; do
  echo "migrating:   ${SRC_PATHS[$i]} -> ${DEST_PATHS[$i]}"
done
echo "branch:      ${BRANCH_NAME}"
echo

# Validate all pairs before touching anything
for i in "${!SRC_PATHS[@]}"; do
  [[ -e "${SOURCE_DIR}/${SRC_PATHS[$i]}" ]] || { echo "error: '${SRC_PATHS[$i]}' not found in source repo"; exit 1; }
  if [[ -z "${SINCE_REF}" ]]; then
    while IFS= read -r -d '' src_dir; do
      rel="${src_dir#"${SOURCE_DIR}/${SRC_PATHS[$i]}"}"
      dest_dir="${DEST_DIR}/${DEST_PATHS[$i]}${rel}"
      if [[ -d "${dest_dir}" ]] && \
           find "${src_dir}" -maxdepth 1 -name "*.go" -print -quit 2>/dev/null | grep -q . && \
           find "${dest_dir}" -maxdepth 1 -name "*.go" -print -quit 2>/dev/null | grep -q .; then
        echo "error: Go package conflict at '${DEST_PATHS[$i]}${rel}': both source and destination contain Go files"
        exit 1
      fi
    done < <(find "${SOURCE_DIR}/${SRC_PATHS[$i]}" -type d -print0)
  fi
done

if git -C "${DEST_DIR}" rev-parse --verify "${BRANCH_NAME}" &>/dev/null; then
  echo "error: branch '${BRANCH_NAME}' already exists; delete it first:"
  echo "  git -C ${DEST_DIR} branch -D ${BRANCH_NAME}"
  exit 1
fi

if ! git -C "${DEST_DIR}" diff --quiet || ! git -C "${DEST_DIR}" diff --cached --quiet; then
  echo "error: destination repo has uncommitted changes"
  exit 1
fi

# Build filter-repo args for all pairs
FILTER_ARGS=()
for i in "${!SRC_PATHS[@]}"; do
  FILTER_ARGS+=(--path "${SRC_PATHS[$i]}")
  [[ "${SRC_PATHS[$i]}" != "${DEST_PATHS[$i]}" ]] && FILTER_ARGS+=(--path-rename "${SRC_PATHS[$i]}:${DEST_PATHS[$i]}")
done

# Clone source, filter to target paths, and rewrite bare #NNN issue references to
# SOURCE_ORG/SOURCE_REPO_NAME#NNN so they link to the original repo rather than the
# destination. Matches only #NNN preceded by start-of-line, space, '(' or ',' to
# avoid hex literals, code-block references, and already-qualified org/repo#NNN refs.
MSG_CALLBACK="import re
return re.sub(rb'(?m)(^|[ (,])#(\\d+)', lambda m: m.group(1) + b'${SOURCE_ORG}/${SOURCE_REPO_NAME}#' + m.group(2), message)"

rm -rf "${FILTER_WORK_DIR}"
git clone "file://${SOURCE_DIR}" "${FILTER_WORK_DIR}"
git -C "${FILTER_WORK_DIR}" filter-repo "${FILTER_ARGS[@]}" \
  --message-callback "${MSG_CALLBACK}" \
  --force

if [[ -n "${SINCE_REF}" ]]; then
  # Use filter-repo's commit-map to translate SINCE_REF into the filtered history.
  # Find the latest source commit at or before SINCE_REF that touched any of the paths.
  SINCE_SRC_SHA=$(git -C "${SOURCE_DIR}" rev-list -1 "${SINCE_REF}" -- "${SRC_PATHS[@]}")
  if [[ -z "${SINCE_SRC_SHA}" ]]; then
    echo "no commits in source before ${SINCE_REF} touching the specified paths; nothing to sync"
    exit 0
  fi
  COMMIT_MAP="${FILTER_WORK_DIR}/.git/filter-repo/commit-map"
  SINCE_FILTERED=$(awk -v sha="${SINCE_SRC_SHA}" 'NR>1 && $1==sha {print $2; exit}' "${COMMIT_MAP}")
  if [[ -z "${SINCE_FILTERED}" || "${SINCE_FILTERED}" == "0000000000000000000000000000000000000000" ]]; then
    echo "error: could not map ${SINCE_SRC_SHA} to filter-work; the commit may not touch the specified paths"
    exit 1
  fi
  COMMITS=()
  while IFS= read -r line; do COMMITS+=("${line}"); done \
    < <(git -C "${FILTER_WORK_DIR}" rev-list --reverse "${SINCE_FILTERED}..HEAD" 2>/dev/null || true)
else
  COMMITS=()
  while IFS= read -r line; do COMMITS+=("${line}"); done \
    < <(git -C "${FILTER_WORK_DIR}" rev-list --reverse HEAD 2>/dev/null || true)
fi

if [[ ${#COMMITS[@]} -eq 0 ]]; then
  [[ -n "${SINCE_REF}" ]] && { echo "no new commits since ${SINCE_REF} affecting the specified paths"; exit 0; }
  echo "error: no commits found for given paths; check the paths"
  exit 1
fi
echo "${#COMMITS[@]} commits found"

# Apply filtered commits to a new branch in destination
git -C "${DEST_DIR}" checkout -b "${BRANCH_NAME}"
trap 'git -C "${DEST_DIR}" remote remove _migration 2>/dev/null || true' EXIT
git -C "${DEST_DIR}" remote add _migration "${FILTER_WORK_DIR}"
git -C "${DEST_DIR}" fetch _migration

if [[ -n "${SINCE_REF}" ]]; then
  if ! git -C "${DEST_DIR}" -c merge.directoryRenames=false cherry-pick --signoff -S "${COMMITS[@]}"; then
    echo
    echo "error: cherry-pick stopped due to conflicts"
    echo "  resolve, then: git -C ${DEST_DIR} cherry-pick --continue"
    exit 1
  fi
else
  MERGE_MSG="migrate: import from ${SOURCE_MODULE}"$'\n'
  for i in "${!SRC_PATHS[@]}"; do
    MERGE_MSG+=$'\n'"  ${SRC_PATHS[$i]} -> ${DEST_PATHS[$i]}"
  done
  git -C "${DEST_DIR}" merge --allow-unrelated-histories --signoff -S "_migration/${MAIN_BRANCH}" \
    --no-edit -m "${MERGE_MSG}"
fi

# Rewrite imports - single pass, all pairs applied per file
declare -a OLD_IMPORTS NEW_IMPORTS
for i in "${!SRC_PATHS[@]}"; do
  OLD_IMPORTS+=("${SOURCE_MODULE}/${SRC_PATHS[$i]}")
  NEW_IMPORTS+=("${DEST_MODULE}/${DEST_PATHS[$i]}")
  echo "rewriting imports: ${SOURCE_MODULE}/${SRC_PATHS[$i]} -> ${DEST_MODULE}/${DEST_PATHS[$i]}"
done

CHANGED=0
while IFS= read -r -d '' f; do
  for i in "${!OLD_IMPORTS[@]}"; do
    if grep -qF "\"${OLD_IMPORTS[$i]}" "${f}"; then
      SED_ARGS=()
      for j in "${!OLD_IMPORTS[@]}"; do
        SED_ARGS+=(-e "s|\"${OLD_IMPORTS[$j]}/|\"${NEW_IMPORTS[$j]}/|g")
        SED_ARGS+=(-e "s|\"${OLD_IMPORTS[$j]}\"|\"${NEW_IMPORTS[$j]}\"|g")
      done
      sed -i "${SED_ARGS[@]}" "${f}"
      echo "  ${f#"${DEST_DIR}/"}"
      CHANGED=$((CHANGED + 1))
      break
    fi
  done
done < <(find "${DEST_DIR}" -name "*.go" -not -path "*/.git/*" -print0)
echo "${CHANGED} file(s) updated"

cd "${DEST_DIR}"
go mod tidy  || echo "warning: go mod tidy has errors (expected for partial migrations with unresolved cross-package deps)"
go build ./... || echo "warning: go build has errors (expected for partial migrations with unresolved cross-package deps)"

if ! git diff --quiet || ! git diff --cached --quiet; then
  IMPORT_MSG="chore: rewrite imports from ${SOURCE_MODULE}"$'\n'
  for i in "${!SRC_PATHS[@]}"; do
    IMPORT_MSG+=$'\n'"  ${SOURCE_MODULE}/${SRC_PATHS[$i]} -> ${DEST_MODULE}/${DEST_PATHS[$i]}"
  done
  git add -A
  git commit --signoff -S -m "${IMPORT_MSG}"
fi

# Open PR
if [[ -n "${SINCE_REF}" ]]; then
  PR_TITLE="sync: ${SOURCE_MODULE} since ${SINCE_REF} @ ${SOURCE_SHA}"
  PR_INTRO="Picks up ${#COMMITS[@]} commit(s) to \`${SOURCE_MODULE}\` since \`${SINCE_REF}\` (now @ ${SOURCE_SHA}):"
else
  PR_TITLE="migrate: ${SOURCE_MODULE} @ ${SOURCE_SHA}"
  PR_INTRO="Migrates the following paths from \`${SOURCE_MODULE}\` (@ ${SOURCE_SHA}) with full git history:"
fi

PR_BODY="${PR_INTRO}"$'\n'
for i in "${!SRC_PATHS[@]}"; do
  PR_BODY+=$'\n'"- \`${SRC_PATHS[$i]}\` → \`${DEST_PATHS[$i]}\`"
done
PR_BODY+=$'\n\n'"To pick up future upstream changes: \`migrate-gaie-paths.sh --since ${SOURCE_SHA}"
for i in "${!SRC_PATHS[@]}"; do
  PR_BODY+=" ${SRC_PATHS[$i]} ${DEST_PATHS[$i]}"
done
PR_BODY+="\`"
[[ -z "${SINCE_REF}" ]] && PR_BODY+=$'\n\n'"**Merge via \"Create a merge commit\"** — squash or rebase discards the migrated history."

if [[ "${NO_PUSH:-0}" == "1" ]]; then
  echo
  echo "NO_PUSH=1 — skipping git push and PR creation."
  echo "PR title would be: ${PR_TITLE}"
  echo "To push manually: git -C ${DEST_DIR} push ${ORIGIN_REMOTE} ${BRANCH_NAME}"
else
  git push "${ORIGIN_REMOTE}" "${BRANCH_NAME}"
  gh pr create \
    --repo "${DEST_ORG}/${DEST_REPO_NAME}" \
    --head "${GITHUB_USER}:${BRANCH_NAME}" \
    --base "${MAIN_BRANCH}" \
    --title "${PR_TITLE}" \
    --body "${PR_BODY}"
fi
