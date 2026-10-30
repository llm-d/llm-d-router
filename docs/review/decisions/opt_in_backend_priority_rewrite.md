---
type: Decision
title: "Opt-in backend request priority rewrite via PriorityRewriter interface"
description: "Overwriting or stripping client-supplied priority fields on forwarded backend payloads requires explicit opt-in configuration through PriorityRewriter."
tags: ["maintainer-vs-maintainer", "mvm-24"]
status: stable
sources:
  - id: "pr-2265-r3825571633"
    resource: "https://github.com/llm-d/llm-d-router/pull/2265#discussion_r3825571633"
    author: "human:zetxqx"
    last_modified: "2026-08-20T21:40:45Z"
  - id: "pr-2265-r3906042639"
    resource: "https://github.com/llm-d/llm-d-router/pull/2265#discussion_r3906042639"
    author: "human:ahg-g"
    last_modified: "2026-09-01T16:11:13Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated stripping client-supplied `priority` fields on all requests in favor of `InferenceObjective` priority versus preserving wire compatibility by default. (Conflict audit ID: `MVM-24`.)

# Positions considered

- **Position A (human:zetxqx)**: Proposed in #2265.[^pr-2265-r3825571633]
- **Position B (human:ahg-g)**: Counter-position evaluated in #2265.[^pr-2265-r3906042639]

# Outcome

Implemented backend priority stamping behind an opt-in parser `PriorityRewriter` capability so default passthrough preserves client fields.

Related rule: [Preserve unrecognized client request fields and wire fidelity](../rules/protocol_parsing_and_wire_fidelity.md#preserve-unrecognized-client-request-fields-and-wire-fidelity).

[^pr-2265-r3825571633]: [#2265](https://github.com/llm-d/llm-d-router/pull/2265#discussion_r3825571633) (human:zetxqx)
[^pr-2265-r3906042639]: [#2265](https://github.com/llm-d/llm-d-router/pull/2265#discussion_r3906042639) (human:ahg-g)
