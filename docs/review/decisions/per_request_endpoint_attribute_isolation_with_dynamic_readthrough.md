---
type: Decision
title: "Per-request endpoint attribute isolation with dynamic metric read-through"
description: "Scheduling snapshots isolate per-request endpoint attributes such as prefix-match scores while reading live shared metrics dynamically."
tags: ["maintainer-vs-maintainer", "mvm-17"]
status: stable
sources:
  - id: "pr-1478-r3355562756"
    resource: "https://github.com/llm-d/llm-d-router/pull/1478#discussion_r3355562756"
    author: "human:shmuelk"
    last_modified: "2026-06-04T11:16:44Z"
  - id: "pr-1478-r3356766432"
    resource: "https://github.com/llm-d/llm-d-router/pull/1478#discussion_r3356766432"
    author: "human:ahg-g"
    last_modified: "2026-06-04T14:35:21Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated replacing endpoint snapshot cloning with shallow slice copies versus preserving per-request attribute isolation across concurrent scheduling cycles. (Conflict audit ID: `MVM-17`.)

# Positions considered

- **Position A (human:shmuelk)**: Proposed in #1478.[^pr-1478-r3355562756]
- **Position B (human:ahg-g)**: Counter-position evaluated in #1478.[^pr-1478-r3356766432]

# Outcome

Merged dynamic attribute read-through for shared telemetry while isolating per-request computed attributes per scheduling cycle.

Related rule: [Document lock ordering, shallow-copy safety, and engine protocol quirks in code comments](../rules/documentation_and_comments.md#document-lock-ordering-shallow-copy-safety-and-engine-protocol-quirks-in-code-comments).

[^pr-1478-r3355562756]: [#1478](https://github.com/llm-d/llm-d-router/pull/1478#discussion_r3355562756) (human:shmuelk)
[^pr-1478-r3356766432]: [#1478](https://github.com/llm-d/llm-d-router/pull/1478#discussion_r3356766432) (human:ahg-g)
