---
type: Open Question
title: "Sub-agent wire identity for concurrent KV session accounting"
description: "How parallel sub-agents supply distinct wire-level identifiers so stateful session ledgers do not overwrite parent-session KV token counts."
tags: ["project-direction", "session-state", "unr-10"]
status: draft
sources:
  - { id: "pr-3451-ic-6229313371", resource: "https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6229313371", author: "human:kaushikmitr", last_modified: "2026-10-17T03:54:53Z" }
  - { id: "pr-3451-ic-6463413681", resource: "https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6463413681", author: "human:sudoalok", last_modified: "2026-10-28T07:36:41Z" }
  - { id: "pr-3939-rev-5792817818", resource: "https://github.com/llm-d/llm-d-router/pull/3939#pullrequestreview-5792817818", author: "human:LukeAVanDrie", last_modified: "2026-11-05T22:06:17Z" }
  - { id: "issue-4021-ic-6745998614", resource: "https://github.com/llm-d/llm-d-router/issues/4021#issuecomment-6745998614", author: "human:D-Sai-Venkatesh", last_modified: "2026-11-09T16:54:44Z" }
---

# Question

How parallel sub-agents identify themselves on the wire so the shared session manager and `thunder-agent` track concurrent turns without overwriting root-session KV accounting.[^pr-3451-ic-6229313371][^pr-3451-ic-6463413681][^pr-3939-rev-5792817818][^issue-4021-ic-6745998614]

# Context

Surfaced in conflict audit `UNR-10`. Related rule: [External conversation history vs bounded routing state](../rules/settled_architectural_boundaries.md#external-conversation-history-vs-bounded-routing-state). Updated in #3939 (`agent-lineage` derived root) and #4021 (decoupling logical lineage from physical thread tracking).

# Positions

- **Position A**: Key the shared session manager by a composite of root session ID and sub-agent ID rather than landing single-plugin token-folding heuristics.[^pr-3451-ic-6229313371]
- **Position B**: Define a concrete wire header or payload field for sub-agent identity first because existing `agent-identity` headers all resolve to the root session.[^pr-3451-ic-6463413681]
- **Position C**: Decouple logical agent lineage (`TenantID` -> `ProgramID`) for flow-control fairness from physical thread and leaf session state (`ThreadID`/`SessionID`) for KV cache footprint and session affinity.[^pr-3939-rev-5792817818][^issue-4021-ic-6745998614]

[^pr-3451-ic-6229313371]: [#3451](https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6229313371) (human:kaushikmitr)
[^pr-3451-ic-6463413681]: [#3451](https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6463413681) (human:sudoalok)
[^pr-3939-rev-5792817818]: [#3939](https://github.com/llm-d/llm-d-router/pull/3939#pullrequestreview-5792817818) (human:LukeAVanDrie)
[^issue-4021-ic-6745998614]: [#4021](https://github.com/llm-d/llm-d-router/issues/4021#issuecomment-6745998614) (human:D-Sai-Venkatesh)
