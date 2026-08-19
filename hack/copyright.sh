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

# Manage copyright notices across .go and .sh files.
#
# Files migrated from sigs.k8s.io/gateway-api-inference-extension (GAIE) carry
# "The Kubernetes Authors"; files that originated in llm-d org repos carry
# "The llm-d Authors"; GAIE-derived files later modified here carry both.
#
# Usage:
#   hack/copyright.sh verify     Fail if any in-scope file has no recognized
#                                notice. Does not check which notice; that is
#                                a provenance question, not a syntax one. This
#                                is the default and the make presubmit gate.
#   hack/copyright.sh fix        Insert the llm-d notice into files that have
#                                no notice at all. Never touches a file that
#                                already carries any notice - see 'classify'
#                                for files whose existing notice is wrong.
#   hack/copyright.sh classify   Report provenance per file, derived from
#                                local git history, to guide manually
#                                correcting misattributed notices.
#
# Generated files ("Code generated ... DO NOT EDIT." in the first 5 lines)
# are skipped in every mode.

set -o errexit
set -o nounset
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-verify}"

# GAIE commits are self-identifying: scripts/migrate-gaie-paths.sh rewrites bare
# #NNN issue refs in migrated commit messages to this qualified form.
GAIE_MARKER='kubernetes-sigs/gateway-api-inference-extension#'
# Commits that only rewrite import paths after a migration; their bodies record
# the exact src -> dest map and introduce no new code.
MECHANICAL_GREP=(--grep='^chore: rewrite imports from' --grep='^chore: rename go model')

LLMD_RE='Copyright [0-9]{4}(, [0-9]{4})* The llm-d Authors\.'
K8S_RE='Copyright [0-9]{4}(, [0-9]{4})* The Kubernetes Authors\.'
GENERATED_RE='Code generated .* DO NOT EDIT\.'

LLMD_BLOCK_GO='/*
Copyright %s The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

'

LLMD_BLOCK_SH='# Copyright %s The llm-d Authors.
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

'

in_scope_files() {
  git ls-files '*.go' '*.sh'
}

is_generated() {
  head -5 "$1" | grep -qE "${GENERATED_RE}"
}

has_llmd_notice() { grep -qE "${LLMD_RE}" "$1"; }
has_k8s_notice()  { grep -qE "${K8S_RE}" "$1"; }
has_any_notice()  { has_llmd_notice "$1" || has_k8s_notice "$1"; }

# Year of a file's first commit, following renames; falls back to the current
# year for files git has never seen (new, uncommitted files).
creation_year() {
  local year
  year=$(git log --follow --diff-filter=A --format=%ad --date=format:%Y -- "$1" | tail -1)
  echo "${year:-$(date +%Y)}"
}

cmd_verify() {
  local missing=()
  local f
  while IFS= read -r f; do
    is_generated "${f}" && continue
    has_any_notice "${f}" || missing+=("${f}")
  done < <(in_scope_files)

  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "All in-scope files carry a recognized copyright notice."
    return 0
  fi

  echo "ERROR: the following files have no recognized copyright notice:"
  printf '  %s\n' "${missing[@]}"
  return 1
}

cmd_fix() {
  local fixed=0
  local f
  while IFS= read -r f; do
    is_generated "${f}" && continue
    has_any_notice "${f}" && continue

    local year block tmp
    year=$(creation_year "${f}")
    tmp=$(mktemp)
    case "${f}" in
      *.go)
        # A leading //go:build (or legacy // +build) constraint must stay the
        # very first thing in the file, ahead of the license block, or
        # gofmt moves it there itself.
        local build_lines=0
        if [[ "$(head -1 "${f}")" == '//go:build'* || "$(head -1 "${f}")" == '// +build'* ]]; then
          build_lines=$(awk '/^\/\/go:build|^\/\/ \+build|^$/{n++; next} {exit} END{print n+0}' "${f}")
        fi
        if [[ "${build_lines}" -gt 0 ]]; then
          head -n "${build_lines}" "${f}" > "${tmp}"
          printf "${LLMD_BLOCK_GO}" "${year}" >> "${tmp}"
          tail -n +"$((build_lines + 1))" "${f}" >> "${tmp}"
        else
          printf "${LLMD_BLOCK_GO}" "${year}" > "${tmp}"
          cat "${f}" >> "${tmp}"
        fi
        ;;
      *.sh)
        if [[ "$(head -1 "${f}")" == '#!'* ]]; then
          head -1 "${f}" > "${tmp}"
          echo >> "${tmp}"
          printf "${LLMD_BLOCK_SH}" "${year}" >> "${tmp}"
          tail -n +2 "${f}" >> "${tmp}"
        else
          printf "${LLMD_BLOCK_SH}" "${year}" > "${tmp}"
          cat "${f}" >> "${tmp}"
        fi
        ;;
      *)
        rm -f "${tmp}"
        continue
        ;;
    esac
    cat "${tmp}" > "${f}" && rm -f "${tmp}"
    echo "fixed: ${f}"
    fixed=$((fixed + 1))
  done < <(in_scope_files)
  echo "${fixed} file(s) updated"
}

cmd_classify() {
  echo "Building GAIE-commit marker set..." >&2
  local gaie_file mech_file notlocal_file
  gaie_file=$(mktemp); mech_file=$(mktemp); notlocal_file=$(mktemp)
  trap 'rm -f "${gaie_file}" "${mech_file}" "${notlocal_file}"' RETURN

  git log --format=%H --grep="${GAIE_MARKER}" -F | sort -u > "${gaie_file}"
  git log --format=%H "${MECHANICAL_GREP[@]}" | sort -u > "${mech_file}"
  sort -mu "${gaie_file}" "${mech_file}" > "${notlocal_file}"

  local gaie_count mech_count total_count
  gaie_count=$(wc -l < "${gaie_file}")
  mech_count=$(wc -l < "${mech_file}")
  total_count=$(git rev-list --count HEAD)
  echo "GAIE-marked commits: ${gaie_count} / ${total_count} total"
  echo "Mechanical import-rewrite commits: ${mech_count}"
  echo

  local llmd_only=0 k8s_only=0 both=0
  local f origin creator
  while IFS= read -r f; do
    creator=$(git log --follow --diff-filter=A --format=%H -- "${f}" | tail -1)
    if [[ -z "${creator}" ]]; then
      creator=$(git log --format=%H -- "${f}" | tail -1)
    fi
    if grep -qxF "${creator}" "${gaie_file}"; then
      origin="GAIE"
    else
      origin="LLMD"
    fi

    local has_l has_k
    has_llmd_notice "${f}" && has_l=1 || has_l=0
    has_k8s_notice "${f}" && has_k=1 || has_k=0

    if [[ "${origin}" == "LLMD" ]]; then
      llmd_only=$((llmd_only + 1))
      [[ "${has_k}" == 1 ]] && echo "MISATTRIBUTED(llm-d origin, k8s notice, needs human review): ${f}"
    else
      # GAIE origin: k8s-only vs both depends on whether local history added
      # real changes beyond the mechanical import rewrite.
      local touches
      touches=$(comm -23 <(git log --follow --format=%H -- "${f}" | sort -u) "${notlocal_file}" | wc -l)
      if [[ "${touches}" -eq 0 ]]; then
        k8s_only=$((k8s_only + 1))
        [[ "${has_l}" == 1 && "${has_k}" == 0 ]] && \
          echo "NEEDS-K8S-NOTICE(unmodified GAIE import, llm-d-only notice): ${f}"
        [[ "${has_l}" == 0 && "${has_k}" == 0 ]] && \
          echo "NEEDS-K8S-NOTICE(unmodified GAIE import, no notice): ${f}"
      else
        both=$((both + 1))
        [[ "${has_l}" == 0 ]] && echo "NEEDS-LLMD-NOTICE(modified GAIE import): ${f}"
        [[ "${has_l}" == 1 && "${has_k}" == 0 ]] && \
          echo "NEEDS-K8S-NOTICE(modified GAIE import, llm-d-only notice): ${f}"
      fi
    fi
  done < <(in_scope_files | grep '\.go$')

  echo
  echo "Summary (Go files):"
  echo "  llm-d origin:        ${llmd_only}"
  echo "  Kubernetes only:     ${k8s_only}"
  echo "  Both (GAIE + local): ${both}"
}

case "${MODE}" in
  verify)   cmd_verify ;;
  fix)      cmd_fix ;;
  classify) cmd_classify ;;
  *)
    echo "usage: $0 [verify|fix|classify]" >&2
    exit 2
    ;;
esac
