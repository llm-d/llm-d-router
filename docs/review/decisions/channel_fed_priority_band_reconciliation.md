---
type: Decision
title: "Channel-fed maintenance loop with lock protection for dynamic priority bands"
description: "Dynamic InferenceObjective priority bands provision off the hot path via edge-triggered controller reconciliation feeding a lock-protected processor loop."
tags: ["maintainer-vs-maintainer", "mvm-10"]
status: stable
sources:
  - id: "issue-1184"
    resource: "https://github.com/llm-d/llm-d-router/issues/1184"
    author: "human:LukeAVanDrie"
    last_modified: "2026-05-18T06:13:38Z"
  - id: "issue-1184-ic-4477880575"
    resource: "https://github.com/llm-d/llm-d-router/issues/1184#issuecomment-4477880575"
    author: "human:shmuelk"
    last_modified: "2026-05-18T12:55:40Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers compared reconciling dynamic flow-control priority bands directly under registry locks against driving band provisioning through a single-owner processor channel. (Conflict audit ID: `MVM-10`.)

# Positions considered

- **Position A (human:LukeAVanDrie)**: Proposed in #1184.[^issue-1184]
- **Position B (human:shmuelk)**: Counter-position evaluated in #1184.[^issue-1184-ic-4477880575]

# Outcome

Merged edge-triggered controller reconciliation feeding the processor loop with read/write lock protection and idle drain before removal.

Related rule: [Reconcile dynamic priority bands on controller edges and drain before removal](../rules/concurrency_and_shared_state.md#reconcile-dynamic-priority-bands-on-controller-edges-and-drain-before-removal).

[^issue-1184]: [#1184](https://github.com/llm-d/llm-d-router/issues/1184) (human:LukeAVanDrie)
[^issue-1184-ic-4477880575]: [#1184](https://github.com/llm-d/llm-d-router/issues/1184#issuecomment-4477880575) (human:shmuelk)
