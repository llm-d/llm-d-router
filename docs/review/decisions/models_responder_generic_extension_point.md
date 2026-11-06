---
type: Decision
title: "Pool-wide /v1/models aggregation via generic Responder plugin interface"
description: "EPP answers GET /v1/models through the optional models-responder plugin behind a generic Responder interface without coupling core runtime packages to concrete plugins."
tags: ["maintainer-vs-maintainer", "mvm-22"]
status: stable
sources:
  - { id: "issue-1878-ic-4864624197", resource: "https://github.com/llm-d/llm-d-router/issues/1878#issuecomment-4864624197", author: "human:shmuelk", last_modified: "2026-07-02T10:16:39Z" }
  - { id: "pr-2308-rev-5453802061", resource: "https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5453802061", author: "human:elevran", last_modified: "2026-10-08T09:12:13Z" }
---

# Context

Maintainers initially closed router-side `/v1/models` aggregation under homogeneous-pool assumptions before re-evaluating multi-LoRA adapter visibility across endpoints. (Conflict audit ID: `MVM-22`.)

# Positions considered

- **Position A (human:shmuelk)**: Proposed in #1878.[^issue-1878-ic-4864624197]
- **Position B (human:elevran)**: Counter-position evaluated in #2308.[^pr-2308-rev-5453802061]

# Outcome

Introduced a generic `Responder` extension point and merged `models-responder` so endpoints with distinct loaded LoRA adapters union cleanly without core-to-plugin imports.

Related rule: [Homogeneous base model and role per InferencePool](../rules/settled_architectural_boundaries.md#homogeneous-base-model-and-role-per-inferencepool).

[^issue-1878-ic-4864624197]: [#1878](https://github.com/llm-d/llm-d-router/issues/1878#issuecomment-4864624197) (human:shmuelk)
[^pr-2308-rev-5453802061]: [#2308](https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5453802061) (human:elevran)
