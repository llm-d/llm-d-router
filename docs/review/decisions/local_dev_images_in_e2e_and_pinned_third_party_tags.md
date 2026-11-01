---
type: Decision
title: "E2E suites test locally built :dev images and third-party images pin explicit versions"
description: "Pull request E2E workflows build and run local :dev images rather than floating :main tags, and third-party container images pin immutable version tags."
tags: ["maintainer-vs-maintainer", "mvm-19"]
status: stable
sources:
  - { id: "pr-1579-r3409481145", resource: "https://github.com/llm-d/llm-d-router/pull/1579#discussion_r3409481145", author: "human:ahg-g", last_modified: "2026-06-14T11:48:00Z" }
  - { id: "pr-1579-r3409523851", resource: "https://github.com/llm-d/llm-d-router/pull/1579#discussion_r3409523851", author: "human:elevran", last_modified: "2026-06-14T12:24:09Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers weighed tracking floating `:latest` / `:main` tags on `main` as cross-repo canaries against preventing unrelated upstream breakages from blocking pull requests. (Conflict audit ID: `MVM-19`.)

# Positions considered

- **Position A (human:ahg-g)**: Proposed in #1579.[^pr-1579-r3409481145]
- **Position B (human:elevran)**: Counter-position evaluated in #1579.[^pr-1579-r3409523851]

# Outcome

Reverted E2E defaults to locally built `:dev` images and enforced `check-latest-tags-strict` on third-party images while allowing documented `llm-d` namespace exceptions on `main`.

[^pr-1579-r3409481145]: [#1579](https://github.com/llm-d/llm-d-router/pull/1579#discussion_r3409481145) (human:ahg-g)
[^pr-1579-r3409523851]: [#1579](https://github.com/llm-d/llm-d-router/pull/1579#discussion_r3409523851) (human:elevran)
