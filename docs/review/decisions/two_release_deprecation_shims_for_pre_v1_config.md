---
type: Decision
title: "Two-release deprecation shims required even during pre-1.0 config refactors"
description: "Shipped user-facing configuration formats preserve backward-compatible deprecation shims for two releases rather than breaking operators across API graduations."
tags: ["maintainer-vs-maintainer", "mvm-11"]
status: stable
sources:
  - { id: "pr-1247", resource: "https://github.com/llm-d/llm-d-router/pull/1247", author: "human:ahg-g", last_modified: "2026-05-21T08:51:43Z" }
  - { id: "pr-1247-ic-4529235883", resource: "https://github.com/llm-d/llm-d-router/pull/1247#issuecomment-4529235883", author: "human:elevran", last_modified: "2026-05-24T15:54:07Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

During API graduation refactors, maintainers debated making clean breaking config changes versus accepting both legacy and replacement YAML fields across a deprecation window. (Conflict audit ID: `MVM-11`.)

# Positions considered

- **Position A (human:ahg-g)**: Proposed in #1247.[^pr-1247]
- **Position B (human:elevran)**: Counter-position evaluated in #1247.[^pr-1247-ic-4529235883]

# Outcome

Enforced backward-compatible deprecation shims on shipped user-facing config fields and removed the shims two releases later.

Related rule: [Provide a two-release deprecation window for shipped user-facing surfaces](../rules/configuration_helm_and_compatibility.md#provide-a-two-release-deprecation-window-for-shipped-user-facing-surfaces).

[^pr-1247]: [#1247](https://github.com/llm-d/llm-d-router/pull/1247) (human:ahg-g)
[^pr-1247-ic-4529235883]: [#1247](https://github.com/llm-d/llm-d-router/pull/1247#issuecomment-4529235883) (human:elevran)
