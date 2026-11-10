---
type: Playbook
title: Prepare a pull request
description: "Pre-submission author workflow for self-checking diffs via full-repo search, targeted non-code review rules, and make presubmit."
tags: [playbook, authoring, presubmit]
status: stable
---

# Procedure

1. Confirm non-trivial changes link a public GitHub issue (`Part of #` for multi-slice work, `Fixes #` only for complete issue resolution).
2. Read sibling implementations and callers across the repository, and grep any newly declared string or numeric constant before adding a duplicate.
3. Route touched paths through [Review a pull request](review_a_pull_request.md) and inspect any matching `0` to `2` Stage-1 rule files in `rules/` whose `When:` condition applies.
4. Run `make presubmit` (and `python3 hack/render_playbook.py --check` when editing `docs/review/`) before opening or updating the pull request.
