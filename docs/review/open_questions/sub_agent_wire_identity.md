---
type: Open Question
title: "Sub-agent wire identity for concurrent KV session accounting"
description: "How parallel sub-agents supply distinct wire-level identifiers so stateful session ledgers do not overwrite parent-session KV token counts."
tags: ["project-direction", "session-state", "unr-10"]
status: draft
sources:
  - { id: "pr-3451-ic-6229313371", resource: "https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6229313371", author: "human:kaushikmitr", last_modified: "2026-10-17T03:54:53Z" }
  - { id: "pr-3451-ic-6463413681", resource: "https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6463413681", author: "human:sudoalok", last_modified: "2026-10-28T07:36:41Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

How parallel sub-agents identify themselves on the wire so the shared session manager and `thunder-agent` track concurrent turns without overwriting root-session KV accounting.[^pr-3451-ic-6229313371][^pr-3451-ic-6463413681]

# Context

Surfaced in conflict audit `UNR-10`. Related rule: [See linked review rule](../rules/project_direction.md#keep-client-conversation-state-out-of-the-router-and-orchestrate-multi-stage-pipelines-in-the-coordinator).

# Positions

- **Position A**: Key the shared session manager by a composite of root session ID and sub-agent ID rather than landing single-plugin token-folding heuristics.[^pr-3451-ic-6229313371]
- **Position B**: Define a concrete wire header or payload field for sub-agent identity first because existing `agent-identity` headers all resolve to the root session.[^pr-3451-ic-6463413681]

[^pr-3451-ic-6229313371]: [#3451](https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6229313371) (human:kaushikmitr)
[^pr-3451-ic-6463413681]: [#3451](https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6463413681) (human:sudoalok)
