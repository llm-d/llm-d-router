---
type: Decision
title: make presubmit runs unit tests
description: "The top-level make presubmit target includes unit test execution so local pre-merge verification covers behavioral regressions alongside static lints."
tags: ["testing-and-ci", presubmit, "nvc-01"]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
verified: { by: "human:conliu", at: "2026-10-29T00:00:00Z" }
---

# Context

Conflict audit `NVC-01` surfaced that `AGENTS.md` designates `make presubmit` as the pre-merge gate while `Makefile` ran only static checks and left `test-unit` to separate CI jobs.

# Tradeoff

Keeping `make presubmit` lint-only stays fast but lets broken unit tests pass local pre-merge checks; including unit tests aligns local `make presubmit` with CI expectations.

# Outcome

`make presubmit` runs unit tests in addition to static formatting, linting, security, vulnerability, image-tag, and boilerplate checks.

Related rule: [See linked review rule](../rules/testing_and_ci.md#wire-new-checks-into-make-presubmit-and-pin-third-party-container-image-tags).
