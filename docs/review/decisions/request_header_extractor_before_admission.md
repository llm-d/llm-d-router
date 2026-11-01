---
type: Decision
title: "Pre-admission header attribute extraction via dedicated plugin extension point"
description: "Caller identity headers are extracted into request attributes before flow-control admission via RequestHeaderExtractor plugins rather than PreAdmission hooks."
tags: ["maintainer-vs-maintainer", "mvm-16"]
status: stable
sources:
  - { id: "issue-1450-ic-4604976814", resource: "https://github.com/llm-d/llm-d-router/issues/1450#issuecomment-4604976814", author: "human:liu-cong", last_modified: "2026-06-02T17:14:08Z" }
  - { id: "issue-1450-ic-4701681953", resource: "https://github.com/llm-d/llm-d-router/issues/1450#issuecomment-4701681953", author: "human:elevran", last_modified: "2026-06-14T12:03:03Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers compared a built-in `headerPopulator` list, overloading `PreAdmission`, and introducing a pre-admission header extraction plugin point for `agent-identity`. (Conflict audit ID: `MVM-16`.)

# Positions considered

- **Position A (human:liu-cong)**: Proposed in #1450.[^issue-1450-ic-4604976814]
- **Position B (human:elevran)**: Counter-position evaluated in #1450.[^issue-1450-ic-4701681953]

# Outcome

Merged the pre-admission header extraction extension point (`agent-identity`) so fairness and admission policies read typed attributes.

Related rule: [Keep core runtimes built in and policy plugins optional](../rules/project_direction.md#keep-core-runtimes-built-in-and-policy-plugins-optional).

[^issue-1450-ic-4604976814]: [#1450](https://github.com/llm-d/llm-d-router/issues/1450#issuecomment-4604976814) (human:liu-cong)
[^issue-1450-ic-4701681953]: [#1450](https://github.com/llm-d/llm-d-router/issues/1450#issuecomment-4701681953) (human:elevran)
