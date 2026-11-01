---
type: Decision
title: "Typed data-layer Slot handles over generic Extractor interfaces"
description: "Cross-plugin data-layer typing uses typed Slot[T] and DataKey handles while keeping extractor interfaces non-generic."
tags: ["maintainer-vs-maintainer", "mvm-03"]
status: stable
sources:
  - { id: "pr-883", resource: "https://github.com/llm-d/llm-d-router/pull/883", author: "human:hexfusion", last_modified: "2026-04-24T20:08:42Z" }
  - { id: "pr-883-ic-4321888471", resource: "https://github.com/llm-d/llm-d-router/pull/883#issuecomment-4321888471", author: "human:shmuelk", last_modified: "2026-04-26T11:11:52Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers evaluated reshaping the `Extractor` plugin interface around compile-time Go generics (`Extractor[T]`) versus keeping configuration loading decoupled from concrete source types. (Conflict audit ID: `MVM-03`.)

# Positions considered

- **Position A (human:hexfusion)**: Proposed in #883.[^pr-883]
- **Position B (human:shmuelk)**: Counter-position evaluated in #883.[^pr-883-ic-4321888471]

# Outcome

Closed the generic-extractor refactor and standardized type-safe cross-plugin access around `datalayer.Slot[T]` and `DataKey` handles.

Related rule: [Exchange cross-plugin state through typed data-layer slots and keys](../rules/plugin_framework_contracts.md#exchange-cross-plugin-state-through-typed-data-layer-slots-and-keys).

[^pr-883]: [#883](https://github.com/llm-d/llm-d-router/pull/883) (human:hexfusion)
[^pr-883-ic-4321888471]: [#883](https://github.com/llm-d/llm-d-router/pull/883#issuecomment-4321888471) (human:shmuelk)
