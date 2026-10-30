---
type: Playbook
title: Prepare a pull request
description: "Pre-submission author workflow for self-checking diffs against targeted review rules and running make presubmit."
tags: [playbook, authoring, presubmit]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Procedure

1. Confirm non-trivial changes link a public GitHub issue (`Part of #` for multi-slice work, `Fixes #` only for complete issue resolution).
2. Map touched paths through [Review a pull request](review_a_pull_request.md) and inspect each matching rule section in `rules/`.
3. Grep any newly declared string or numeric constant across the repository before adding a duplicate.
4. Run `make presubmit` (and `python3 hack/render_playbook.py --check` when editing `docs/review/`) before opening or updating the pull request.
