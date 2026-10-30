---
type: Decision
title: "Native engine render endpoints preferred over router-side API translation"
description: "New inference API integrations target native engine /render routes directly, allowing contained interim translation only while native engine endpoints land."
tags: ["maintainer-vs-maintainer", "mvm-14"]
status: stable
sources:
  - id: "pr-1896-rev-4657668118"
    resource: "https://github.com/llm-d/llm-d-router/pull/1896#pullrequestreview-4657668118"
    author: "human:sagearc"
    last_modified: "2026-07-08T20:25:23Z"
  - id: "pr-1896-ic-4924616643"
    resource: "https://github.com/llm-d/llm-d-router/pull/1896#issuecomment-4924616643"
    author: "human:vMaroon"
    last_modified: "2026-07-09T11:33:43Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers weighed maintaining router-side `/v1/messages` to chat-completions translation layers against waiting for or forwarding to native vLLM render endpoints. (Conflict audit ID: `MVM-14`.)

# Positions considered

- **Position A (human:sagearc)**: Proposed in #1896.[^pr-1896-rev-4657668118]
- **Position B (human:vMaroon)**: Counter-position evaluated in #1896.[^pr-1896-ic-4924616643]

# Outcome

Accepted a contained interim adapter for `/v1/messages` while establishing native engine `/render` pass-through as the required path for `/v1/responses` and subsequent APIs.

[^pr-1896-rev-4657668118]: [#1896](https://github.com/llm-d/llm-d-router/pull/1896#pullrequestreview-4657668118) (human:sagearc)
[^pr-1896-ic-4924616643]: [#1896](https://github.com/llm-d/llm-d-router/pull/1896#issuecomment-4924616643) (human:vMaroon)
