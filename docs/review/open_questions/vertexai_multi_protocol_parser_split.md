---
type: Open Question
title: "Single multi-protocol Vertex AI parser vs separate REST and gRPC parsers"
description: "Whether platforms serving both REST and gRPC inference routes use one parser with segment wildcards or split into protocol-specific parser plugins."
tags: ["protocol-parsing-and-wire-fidelity", parsers, "unr-11"]
status: draft
sources:
  - id: "pr-3479-r4301544118"
    resource: "https://github.com/llm-d/llm-d-router/pull/3479#discussion_r4301544118"
    author: "human:kaushikmitr"
    last_modified: "2026-10-17T05:24:39Z"
  - id: "pr-3479-rev-5574188230"
    resource: "https://github.com/llm-d/llm-d-router/pull/3479#pullrequestreview-5574188230"
    author: "human:zetxqx"
    last_modified: "2026-10-19T16:21:55Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

Whether Vertex AI REST and gRPC routes belong in one `vertexai-parser` using declarative path-segment wildcards or in separate `vertexai-parser` (gRPC/H2C) and `vertexai-rest-parser` (HTTP/1.1) plugins.[^pr-3479-r4301544118][^pr-3479-rev-5574188230]

# Context

Surfaced in conflict audit `UNR-11`. Related rule: [See linked review rule](../rules/protocol_parsing_and_wire_fidelity.md#claim-inference-routes-declaratively-with-unmatched-route-fallthrough).

# Positions

- **Position A**: Avoid adding single-caller wildcard syntax to the shared parser registry when `Claims` must apply cleanly across all parsers.[^pr-3479-r4301544118]
- **Position B**: Split REST and gRPC into separate parser plugins because gRPC requires `H2C` backends while REST uses `HTTP/1.1`, keeping transport health checks unambiguous.[^pr-3479-rev-5574188230]

[^pr-3479-r4301544118]: [#3479](https://github.com/llm-d/llm-d-router/pull/3479#discussion_r4301544118) (human:kaushikmitr)
[^pr-3479-rev-5574188230]: [#3479](https://github.com/llm-d/llm-d-router/pull/3479#pullrequestreview-5574188230) (human:zetxqx)
