---
type: Decision
title: "Ratio-sampled per-plugin OpenTelemetry spans with high-overhead KV-event spans gated off"
description: "Request-path filter and scorer OpenTelemetry spans run under ratio sampling while high-overhead per-KV-event spans remain off by default."
tags: ["maintainer-vs-maintainer", "mvm-21"]
status: stable
sources:
  - id: "pr-1834-ic-5092454943"
    resource: "https://github.com/llm-d/llm-d-router/pull/1834#issuecomment-5092454943"
    author: "human:elevran"
    last_modified: "2026-07-27T14:14:52Z"
  - id: "pr-1834-ic-5096359788"
    resource: "https://github.com/llm-d/llm-d-router/pull/1834#issuecomment-5096359788"
    author: "human:gyliu513"
    last_modified: "2026-07-27T20:18:31Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers weighed trace export volume from per-filter and per-scorer spans against latency diagnosis value under ratio-based trace sampling. (Conflict audit ID: `MVM-21`.)

# Positions considered

- **Position A (human:elevran)**: Proposed in #1834.[^pr-1834-ic-5092454943]
- **Position B (human:gyliu513)**: Counter-position evaluated in #1834.[^pr-1834-ic-5096359788]

# Outcome

Kept per-plugin request spans under `ParentBased(TraceIDRatioBased)` sampling (default `0.1`) and gated per-KV-event tracing off by default after measuring 12-13% CPU overhead.

Related rule: [Acquire package-scoped OTel tracers and propagate spans end to end](../rules/observability.md#acquire-package-scoped-otel-tracers-and-propagate-spans-end-to-end).

[^pr-1834-ic-5092454943]: [#1834](https://github.com/llm-d/llm-d-router/pull/1834#issuecomment-5092454943) (human:elevran)
[^pr-1834-ic-5096359788]: [#1834](https://github.com/llm-d/llm-d-router/pull/1834#issuecomment-5096359788) (human:gyliu513)
