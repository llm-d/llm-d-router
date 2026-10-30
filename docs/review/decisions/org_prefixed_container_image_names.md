---
type: Decision
title: "Organization-prefixed container image names"
description: "Published container images retain the llm-d-router-* prefix under ghcr.io/llm-d for organization-wide consistency."
tags: ["maintainer-vs-maintainer", "mvm-05"]
status: stable
sources:
  - id: "pr-1098-r3242055232"
    resource: "https://github.com/llm-d/llm-d-router/pull/1098#discussion_r3242055232"
    author: "human:hexfusion"
    last_modified: "2026-05-14T14:31:10Z"
  - id: "pr-1098-r3242552188"
    resource: "https://github.com/llm-d/llm-d-router/pull/1098#discussion_r3242552188"
    author: "human:elevran"
    last_modified: "2026-05-14T15:49:16Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers weighed dropping the repeated `llm-d-` image prefix under `ghcr.io/llm-d/` against following the `llm-d` organization naming convention. (Conflict audit ID: `MVM-05`.)

# Positions considered

- **Position A (human:hexfusion)**: Proposed in #1098.[^pr-1098-r3242055232]
- **Position B (human:elevran)**: Counter-position evaluated in #1098.[^pr-1098-r3242552188]

# Outcome

Standardized published image names on `llm-d-router-endpoint-picker` and `llm-d-router-disagg-sidecar`.

[^pr-1098-r3242055232]: [#1098](https://github.com/llm-d/llm-d-router/pull/1098#discussion_r3242055232) (human:hexfusion)
[^pr-1098-r3242552188]: [#1098](https://github.com/llm-d/llm-d-router/pull/1098#discussion_r3242552188) (human:elevran)
