---
type: Decision
title: "Token count estimation encapsulated inside token-producer"
description: "Fast byte-length and tokenized prompt estimates live inside token-producer rather than as methods on core InferenceRequest types."
tags: ["maintainer-vs-maintainer", "mvm-15"]
status: stable
sources:
  - { id: "pr-1438-r3340589325", resource: "https://github.com/llm-d/llm-d-router/pull/1438#discussion_r3340589325", author: "human:ahg-g", last_modified: "2026-06-02T10:47:20Z" }
  - { id: "pr-1438-r3340290186", resource: "https://github.com/llm-d/llm-d-router/pull/1438#discussion_r3340290186", author: "human:vMaroon", last_modified: "2026-06-02T09:56:40Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated exposing token estimation directly on `InferenceRequest` to skip tokenization versus encapsulating all token logic inside `token-producer`. (Conflict audit ID: `MVM-15`.)

# Positions considered

- **Position A (human:ahg-g)**: Proposed in #1438.[^pr-1438-r3340589325]
- **Position B (human:vMaroon)**: Counter-position evaluated in #1438.[^pr-1438-r3340290186]

# Outcome

Unified tokenization and fast token estimation inside `token-producer` so downstream scorers consume one attribute contract.

[^pr-1438-r3340589325]: [#1438](https://github.com/llm-d/llm-d-router/pull/1438#discussion_r3340589325) (human:ahg-g)
[^pr-1438-r3340290186]: [#1438](https://github.com/llm-d/llm-d-router/pull/1438#discussion_r3340290186) (human:vMaroon)
