---
type: Decision
title: "Unified llm-d-router repository housing EPP, Coordinator, and disagg sidecar"
description: "Endpoint Picker, Coordinator, and disaggregation sidecar ship from the single llm-d-router repository rather than split per-binary repositories."
tags: ["maintainer-vs-maintainer", "mvm-04"]
status: stable
sources:
  - { id: "issue-929-ic-4421203934", resource: "https://github.com/llm-d/llm-d-router/issues/929#issuecomment-4421203934", author: "human:ahg-g", last_modified: "2026-05-11T13:35:44Z" }
  - { id: "issue-929-ic-4428488249", resource: "https://github.com/llm-d/llm-d-router/issues/929#issuecomment-4428488249", author: "human:vMaroon", last_modified: "2026-05-12T08:00:51Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated naming and scoping the repository as `llm-d-endpoint-picker` versus consolidating router binaries into one monorepo. (Conflict audit ID: `MVM-04`.)

# Positions considered

- **Position A (human:ahg-g)**: Proposed in #929.[^issue-929-ic-4421203934]
- **Position B (human:vMaroon)**: Counter-position evaluated in #929.[^issue-929-ic-4428488249]

# Outcome

Consolidated EPP, Coordinator, and the disaggregation sidecar into the unified `llm-d-router` repository.

[^issue-929-ic-4421203934]: [#929](https://github.com/llm-d/llm-d-router/issues/929#issuecomment-4421203934) (human:ahg-g)
[^issue-929-ic-4428488249]: [#929](https://github.com/llm-d/llm-d-router/issues/929#issuecomment-4428488249) (human:vMaroon)
