---
type: Decision
title: "Sub-agent wire identity for concurrent KV session accounting"
description: "Parallel sub-agents identify themselves with the x-llm-d-thread-id and x-llm-d-parent-thread-id pair or their harness thread headers, which agent-identity publishes as the thread-id and parent-thread-id request attributes."
tags: ["project-direction", "session-state", "unr-10"]
status: stable
sources:
  - { id: "pr-3451-ic-6229313371", resource: "https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6229313371", author: "human:kaushikmitr", last_modified: "2026-10-17T03:54:53Z" }
  - { id: "pr-3451-ic-6463413681", resource: "https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6463413681", author: "human:sudoalok", last_modified: "2026-10-28T07:36:41Z" }
  - { id: "pr-4159", resource: "https://github.com/llm-d/llm-d-router/pull/4159", author: "human:D-Sai-Venkatesh", last_modified: "2026-11-26T17:37:22Z" }
---

# Context

How parallel sub-agents identify themselves on the wire so the shared session manager and `thunder-agent` track concurrent turns without overwriting root-session KV accounting.[^pr-3451-ic-6229313371][^pr-3451-ic-6463413681]

Surfaced in conflict audit `UNR-10`. Related rule: [External conversation history vs bounded routing state](../rules/settled_architectural_boundaries.md#external-conversation-history-vs-bounded-routing-state).

# Positions considered

- **Position A**: Key the shared session manager by a composite of root session ID and sub-agent ID rather than landing single-plugin token-folding heuristics.[^pr-3451-ic-6229313371]
- **Position B**: Define a concrete wire header or payload field for sub-agent identity first because existing `agent-identity` headers all resolve to the root session.[^pr-3451-ic-6463413681]

# Outcome

Position B. A sub-agent identifies itself with the standard `x-llm-d-thread-id` and `x-llm-d-parent-thread-id` pair, or with its harness thread headers. The `agent-identity` plugin publishes `thread-id`, which always names one conversation (the main thread or a sub-agent), and `parent-thread-id`, which names the thread that spawned it and is absent on a root; the `agent-identity` attribute is unchanged.[^pr-4159] The program root stays `derived-root` from agent-lineage, falling back to `agent-identity`. The header lists and their precedence are in the [agent-identity README](../../../pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity/README.md). A consumer that accounts per sub-agent can key on `thread-id` within the program root, which is the composite key Position A describes; each consumer chooses its own keying.

[^pr-3451-ic-6229313371]: [#3451](https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6229313371) (human:kaushikmitr)
[^pr-3451-ic-6463413681]: [#3451](https://github.com/llm-d/llm-d-router/pull/3451#issuecomment-6463413681) (human:sudoalok)
[^pr-4159]: [#4159](https://github.com/llm-d/llm-d-router/pull/4159) (human:D-Sai-Venkatesh)
