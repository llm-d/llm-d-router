---
type: Decision
title: "Upstream producer signal extraction without deep cross-API AST translation"
description: "token-producer extracts scheduling signals into request attributes while forwarding raw client payloads directly to engine render endpoints."
tags: ["maintainer-vs-maintainer", "mvm-13"]
status: stable
sources:
  - id: "issue-1364-ic-4560679710"
    resource: "https://github.com/llm-d/llm-d-router/issues/1364#issuecomment-4560679710"
    author: "human:zetxqx"
    last_modified: "2026-05-28T03:46:58Z"
  - id: "issue-1364-ic-4587275492"
    resource: "https://github.com/llm-d/llm-d-router/issues/1364#issuecomment-4587275492"
    author: "human:elevran"
    last_modified: "2026-05-31T16:17:22Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated standardizing all inference APIs into a deep router-side `UnifiedPrompt` intermediate representation versus keeping EPP a thin router with lazy bounded body access. (Conflict audit ID: `MVM-13`.)

# Positions considered

- **Position A (human:zetxqx)**: Proposed in #1364.[^issue-1364-ic-4560679710]
- **Position B (human:elevran)**: Counter-position evaluated in #1364.[^issue-1364-ic-4587275492]

# Outcome

Encapsulated token and prompt signal extraction inside `token-producer` while passing native client payloads directly to engine `/render` endpoints.

[^issue-1364-ic-4560679710]: [#1364](https://github.com/llm-d/llm-d-router/issues/1364#issuecomment-4560679710) (human:zetxqx)
[^issue-1364-ic-4587275492]: [#1364](https://github.com/llm-d/llm-d-router/issues/1364#issuecomment-4587275492) (human:elevran)
