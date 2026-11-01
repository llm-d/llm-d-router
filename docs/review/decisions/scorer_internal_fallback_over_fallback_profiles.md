---
type: Decision
title: "Scorer-internal fallback over scheduler fallback profiles"
description: "Missing telemetry fallbacks are handled inside individual scorers rather than adding multi-profile fallback chains to the core scheduler."
tags: ["maintainer-vs-maintainer", "mvm-07"]
status: stable
sources:
  - { id: "issue-1139", resource: "https://github.com/llm-d/llm-d-router/issues/1139", author: "human:kaushikmitr", last_modified: "2026-05-14T19:10:53Z" }
  - { id: "issue-1139-ic-4601790080", resource: "https://github.com/llm-d/llm-d-router/issues/1139#issuecomment-4601790080", author: "human:elevran", last_modified: "2026-06-02T11:11:13Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Contributors proposed scheduler-level fallback profiles when primary scorers lack telemetry, while maintainers questioned adding profile-chaining machinery for a single latency-scorer case. (Conflict audit ID: `MVM-07`.)

# Positions considered

- **Position A (human:kaushikmitr)**: Proposed in #1139.[^issue-1139]
- **Position B (human:elevran)**: Counter-position evaluated in #1139.[^issue-1139-ic-4601790080]

# Outcome

Closed scheduler fallback profiles and kept single weighted-linear-combination profile execution with graceful degradation inside individual scorers.

Related rule: [Keep core runtimes built in and policy plugins optional](../rules/project_direction.md#keep-core-runtimes-built-in-and-policy-plugins-optional).

[^issue-1139]: [#1139](https://github.com/llm-d/llm-d-router/issues/1139) (human:kaushikmitr)
[^issue-1139-ic-4601790080]: [#1139](https://github.com/llm-d/llm-d-router/issues/1139#issuecomment-4601790080) (human:elevran)
