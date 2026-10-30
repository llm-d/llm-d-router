---
type: Decision
title: Delete stranded deprecation helpers in the removal pull request
description: Unreferenced registration and compatibility helpers are deleted in the pull request that removes their last caller rather than retained for hypothetical reuse.
tags: ["maintainer-vs-maintainer", "mvm-12"]
status: stable
sources:
  - id: "pr-2509-ic-5398059091"
    resource: "https://github.com/llm-d/llm-d-router/pull/2509#issuecomment-5398059091"
    author: "human:roytman"
    last_modified: "2026-08-24T16:16:22Z"
  - id: "pr-2509-ic-5398602893"
    resource: "https://github.com/llm-d/llm-d-router/pull/2509#issuecomment-5398602893"
    author: "human:elevran"
    last_modified: "2026-08-24T17:03:39Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

When removing expired deprecated plugin aliases, authors offered to keep `RegisterDeprecatedWithPluginDependencies` for future deprecations versus deleting it immediately. (Conflict audit ID: `MVM-12`.)

# Positions considered

- **Position A (human:roytman)**: Proposed in #2509.[^pr-2509-ic-5398059091]
- **Position B (human:elevran)**: Counter-position evaluated in #2509.[^pr-2509-ic-5398602893]

# Outcome

Deleted unreferenced deprecation helpers in the stranding change, relying on version control history if needed later.

Related rule: [Share common helpers and delete dead code in the stranding change](../rules/plugin_framework_contracts.md#share-common-helpers-and-delete-dead-code-in-the-stranding-change).

[^pr-2509-ic-5398059091]: [#2509](https://github.com/llm-d/llm-d-router/pull/2509#issuecomment-5398059091) (human:roytman)
[^pr-2509-ic-5398602893]: [#2509](https://github.com/llm-d/llm-d-router/pull/2509#issuecomment-5398602893) (human:elevran)
