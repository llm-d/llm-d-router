---
type: Open Question
title: "Coordinator SGLang P/D route scoping, bootstrap port, and stream abort handling"
description: "How the Coordinator scopes SGLang disaggregated /generate routing, Python vs Rust bootstrap_port behavior, and mid-stream prefill error termination."
tags: ["project-direction", coordinator, sglang, "unr-13"]
status: draft
sources:
  - { id: "pr-3511", resource: "https://github.com/llm-d/llm-d-router/pull/3511", author: "human:sagearc", last_modified: "2026-10-19T14:33:13Z" }
  - { id: "pr-3511-r4377231436", resource: "https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4377231436", author: "human:roytman", last_modified: "2026-10-26T12:32:59Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

How the Coordinator scopes SGLang `/generate` interception without breaking non-disaggregated passthrough, handles Python vs Rust `bootstrap_port` differences, and aborts failed streams without `panic(http.ErrAbortHandler)` dropping error telemetry.[^pr-3511][^pr-3511-r4377231436]

# Context

Surfaced in conflict audit `UNR-13`. Related rule: [See linked review rule](../rules/project_direction.md#configure-backend-engine-identity-explicitly-and-isolate-engine-schemas).

# Positions

- **Position A**: Introduce SGLang prefill/decode streaming support in the Coordinator for single-sample DP1 `/generate` requests.[^pr-3511]
- **Position B**: Gate `/generate` interception on active P/D configuration, preserve Python SGLang `bootstrap_port` compatibility, and replace `panic(http.ErrAbortHandler)` so prefill failures are logged and counted.[^pr-3511-r4377231436]

[^pr-3511]: [#3511](https://github.com/llm-d/llm-d-router/pull/3511) (human:sagearc)
[^pr-3511-r4377231436]: [#3511](https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4377231436) (human:roytman)
