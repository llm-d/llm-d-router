---
type: Decision
title: Disaggregation sidecar serves /metrics on its existing HTTP listener mux
description: "The disaggregation sidecar exposes Prometheus /metrics on its existing HTTP mux rather than opening a second metrics port, while gRPC-based EPP keeps a dedicated metrics port."
tags: ["maintainer-vs-maintainer", "mvm-20"]
status: stable
sources:
  - id: "pr-1677-ic-4743512723"
    resource: "https://github.com/llm-d/llm-d-router/pull/1677#issuecomment-4743512723"
    author: "human:gyliu513"
    last_modified: "2026-06-18T15:26:22Z"
  - id: "pr-1677-ic-4742031723"
    resource: "https://github.com/llm-d/llm-d-router/pull/1677#issuecomment-4742031723"
    author: "human:shmuelk"
    last_modified: "2026-06-18T12:42:12Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated adding a dedicated `--metrics-port` (`9090`) to the disaggregation sidecar to mirror EPP versus registering `/metrics` on the sidecar's existing HTTP listener. (Conflict audit ID: `MVM-20`.)

# Positions considered

- **Position A (human:gyliu513)**: Proposed in #1677.[^pr-1677-ic-4743512723]
- **Position B (human:shmuelk)**: Counter-position evaluated in #1677.[^pr-1677-ic-4742031723]

# Outcome

Served `/metrics` on the existing sidecar HTTP mux and dropped `--metrics-port`, keeping a separate metrics port only on EPP where the data-plane port speaks gRPC.

Related rule: [Register metrics under canonical component subsystems with bounded label cardinality](../rules/observability.md#register-metrics-under-canonical-component-subsystems-with-bounded-label-cardinality).

[^pr-1677-ic-4743512723]: [#1677](https://github.com/llm-d/llm-d-router/pull/1677#issuecomment-4743512723) (human:gyliu513)
[^pr-1677-ic-4742031723]: [#1677](https://github.com/llm-d/llm-d-router/pull/1677#issuecomment-4742031723) (human:shmuelk)
