---
type: Decision
title: Pull request template distinguishes Fixes from Part of
description: "The pull request template distinguishes auto-closing Fixes references for complete resolutions from non-closing Part of references for multi-PR work."
tags: ["scope-and-process", "pull-requests", github]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
verified: { by: "human:conliu", at: "2026-10-29T00:00:00Z" }
---

# Context

Contributors frequently used `Fixes #N` on partial slices of multi-PR tracking issues, prematurely auto-closing public design and epic issues upon merge.

# Tradeoff

A bare `Fixes #` prompt in `.github/PULL_REQUEST_TEMPLATE.md` nudges authors toward auto-closing keywords even on incremental pull requests.

# Outcome

`.github/PULL_REQUEST_TEMPLATE.md` explicitly separates `Fixes #` (used only when the pull request resolves the entire issue) from `Part of #` (used for partial slices that keep the tracking issue open).

Related rule: [See linked review rule](../rules/scope_and_process.md#settle-design-tradeoffs-on-public-issues-and-scope-auto-close-keywords).
