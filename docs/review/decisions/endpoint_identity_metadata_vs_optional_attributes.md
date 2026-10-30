---
type: Decision
title: Core endpoint identity on EndpointMetadata and optional topology in AttributeMap
description: "Core identity fields live on EndpointMetadata while optional per-endpoint properties such as worker rank flow through datalayer AttributeMap."
tags: ["maintainer-vs-maintainer", "mvm-06"]
status: stable
sources:
  - id: "pr-1106-ic-4478669785"
    resource: "https://github.com/llm-d/llm-d-router/pull/1106#issuecomment-4478669785"
    author: "human:elevran"
    last_modified: "2026-05-18T14:27:01Z"
  - id: "pr-1106-r3271256891"
    resource: "https://github.com/llm-d/llm-d-router/pull/1106#discussion_r3271256891"
    author: "human:ahg-g"
    last_modified: "2026-05-20T04:41:17Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated placing endpoint rank and type directly on `EndpointMetadata` versus keeping `EndpointMetadata` minimal and storing optional properties in `Attributes`. (Conflict audit ID: `MVM-06`.)

# Positions considered

- **Position A (human:elevran)**: Proposed in #1106.[^pr-1106-ic-4478669785]
- **Position B (human:ahg-g)**: Counter-position evaluated in #1106.[^pr-1106-r3271256891]

# Outcome

`EndpointMetadata` carries core identity (`ID`, `Name`, `Type`), while worker rank and optional topology signals flow through data-layer attributes.

Related rule: [Model endpoints with orchestrator-neutral domain types](../rules/project_direction.md#model-endpoints-with-orchestrator-neutral-domain-types).

[^pr-1106-ic-4478669785]: [#1106](https://github.com/llm-d/llm-d-router/pull/1106#issuecomment-4478669785) (human:elevran)
[^pr-1106-r3271256891]: [#1106](https://github.com/llm-d/llm-d-router/pull/1106#discussion_r3271256891) (human:ahg-g)
