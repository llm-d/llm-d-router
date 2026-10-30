---
type: Decision
title: "Component metrics reference in llm-d-router paired with user guides in llm-d/llm-d"
description: "llm-d-router maintains code-synchronized component metrics docs and dashboards while user-facing multi-component guides update llm-d/llm-d."
tags: ["maintainer-vs-maintainer", "mvm-18"]
status: stable
sources:
  - id: "pr-1636-ic-4696185999"
    resource: "https://github.com/llm-d/llm-d-router/pull/1636#issuecomment-4696185999"
    author: "human:gyliu513"
    last_modified: "2026-06-12T22:58:57Z"
  - id: "pr-1636-ic-4701069127"
    resource: "https://github.com/llm-d/llm-d-router/pull/1636#issuecomment-4701069127"
    author: "human:ahg-g"
    last_modified: "2026-06-14T07:38:02Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated keeping a single metrics documentation source inside `llm-d-router` versus updating `llm-d/llm-d` which publishes `llm-d.ai`. (Conflict audit ID: `MVM-18`.)

# Positions considered

- **Position A (human:gyliu513)**: Proposed in #1636.[^pr-1636-ic-4696185999]
- **Position B (human:ahg-g)**: Counter-position evaluated in #1636.[^pr-1636-ic-4701069127]

# Outcome

Retained component-level `docs/metrics.md` and Grafana dashboards in `llm-d-router` while publishing end-user operations guides in `llm-d/llm-d`.

Related rule: [Colocate plugin READMEs and link to canonical upstream guides](../rules/documentation_and_comments.md#colocate-plugin-readmes-and-link-to-canonical-upstream-guides).

[^pr-1636-ic-4696185999]: [#1636](https://github.com/llm-d/llm-d-router/pull/1636#issuecomment-4696185999) (human:gyliu513)
[^pr-1636-ic-4701069127]: [#1636](https://github.com/llm-d/llm-d-router/pull/1636#issuecomment-4701069127) (human:ahg-g)
