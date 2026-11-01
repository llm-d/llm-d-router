---
type: Decision
title: GitHub CI checks are the source of truth for commit requirements
description: "Enforced GitHub CI workflows define commit sign-off and formatting requirements rather than duplicate prose rules in contributor docs."
tags: ["scope-and-process", ci, commits]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
verified: { by: "human:conliu", at: "2026-10-29T00:00:00Z" }
---

# Context

Contributor documentation and CI scripts (`scripts/check-commits.sh` and GitHub status checks) risked drifting when commit trailers or formatting rules changed.

# Tradeoff

Maintaining separate prose lists of commit rules across `AGENTS.md`, `CONTRIBUTING.md`, and `DEVELOPMENT.md` creates contradictions whenever workflow checks evolve.

# Outcome

GitHub CI checks are the canonical source of truth for commit requirements (including DCO sign-off via `git commit -s`); contributor docs point to enforced checks rather than restating separate rules.

Related rule: [See linked review rule](../rules/scope_and_process.md#iterate-in-place-from-personal-fork-branches-with-incremental-commits).
