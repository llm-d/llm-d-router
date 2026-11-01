---
type: Decision
title: Session affinity as a shared stateless helper package
description: "Lightweight session-affinity header parsing lives in a shared helper package under pkg/epp/framework/plugins/scheduling/util/sessionaffinity rather than a mandatory DataProducer plugin."
tags: ["maintainer-vs-maintainer", "mvm-02", "plugin-framework-contracts", "session-affinity"]
status: stable
sources:
  - { id: "issue-815-ic-4535092791", resource: "https://github.com/llm-d/llm-d-router/issues/815#issuecomment-4535092791", author: "human:ahg-g", last_modified: "2026-05-25T14:41:03Z" }
  - { id: "issue-815-ic-4603503350", resource: "https://github.com/llm-d/llm-d-router/issues/815#issuecomment-4603503350", author: "human:liu-cong", last_modified: "2026-06-02T14:38:03Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers weighed modeling session-affinity token decoding as a pre-scheduling `DataProducer` plugin (matching stateful prefix-cache producers) against sharing a plain Go helper package between the session-affinity filter and scorer. (Conflict audit ID: `MVM-02`.)

# Positions considered

- **DataProducer plugin**: Model session-affinity token decoding as a pre-scheduling `DataProducer` plugin so downstream gating filters and scorers consume produced state uniformly.[^issue-815-ic-4535092791]
- **Shared stateless helper**: Share a lightweight Go package (`pkg/epp/framework/plugins/scheduling/util/sessionaffinity`) directly between the session-affinity filter and scorer, avoiding producer DAG dependencies and configuration ceremony for pure header parsing.[^issue-815-ic-4603503350]

# Outcome

Merged as the shared stateless helper package `pkg/epp/framework/plugins/scheduling/util/sessionaffinity`, avoiding producer DAG registration and config wiring overhead when extraction performs pure header parsing with no external I/O or cross-plugin caching.

Related rule: [Separate data producers from scheduling filters and scorers](../rules/plugin_framework_contracts.md#separate-data-producers-from-scheduling-filters-and-scorers).

[^issue-815-ic-4535092791]: [#815](https://github.com/llm-d/llm-d-router/issues/815#issuecomment-4535092791) (human:ahg-g)
[^issue-815-ic-4603503350]: [#815](https://github.com/llm-d/llm-d-router/issues/815#issuecomment-4603503350) (human:liu-cong)
