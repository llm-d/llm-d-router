---
type: Decision
title: "Flow control remains opt-in by default until standalone default-on graduation"
description: "Flow control ships with GA code maturity while remaining gated off by default so baseline deployments incur zero queueing change without explicit opt-in."
tags: ["maintainer-vs-maintainer", "mvm-23"]
status: stable
sources:
  - { id: "pr-2180-ic-5119639333", resource: "https://github.com/llm-d/llm-d-router/pull/2180#issuecomment-5119639333", author: "human:ahg-g", last_modified: "2026-07-29T15:04:01Z" }
  - { id: "pr-2180-r3692815708", resource: "https://github.com/llm-d/llm-d-router/pull/2180#discussion_r3692815708", author: "human:liu-cong", last_modified: "2026-07-31T19:14:37Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated enabling flow control by default upon GA code maturity versus keeping the feature opt-in until default-on operational readiness is verified separately. (Conflict audit ID: `MVM-23`.)

# Positions considered

- **Position A (human:ahg-g)**: Proposed in #2180.[^pr-2180-ic-5119639333]
- **Position B (human:liu-cong)**: Counter-position evaluated in #2180.[^pr-2180-r3692815708]

# Outcome

Merged GA stability improvements with the `flowControl` gate off by default, deferring default-on activation to a dedicated graduation issue.

Related rule: [Ship safe out-of-the-box defaults and gate workload-sensitive features behind opt-in config](../rules/configuration_helm_and_compatibility.md#ship-safe-out-of-the-box-defaults-and-gate-workload-sensitive-features-behind-opt-in-config).

[^pr-2180-ic-5119639333]: [#2180](https://github.com/llm-d/llm-d-router/pull/2180#issuecomment-5119639333) (human:ahg-g)
[^pr-2180-r3692815708]: [#2180](https://github.com/llm-d/llm-d-router/pull/2180#discussion_r3692815708) (human:liu-cong)
