---
type: Review Rule Set
title: Protocol parsing and wire fidelity
description: "Passthrough JSON fidelity, declarative route claims, complete SSE frame buffering, multi-API multimodal parity, host-port-rank endpoint keys, and disaggregated stage ordering."
tags: ["protocol-parsing-and-wire-fidelity", "failure-class", "mvm-24", "apw-04", "unr-11", "apw-03"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - id: "issue-1045"
    resource: "https://github.com/llm-d/llm-d-router/issues/1045"
    author: "human:yankay"
    last_modified: "2026-05-11T04:56:45Z"
  - id: "issue-1072"
    resource: "https://github.com/llm-d/llm-d-router/issues/1072"
    author: "human:sagearc"
    last_modified: "2026-05-12T20:27:30Z"
  - id: "issue-1092"
    resource: "https://github.com/llm-d/llm-d-router/issues/1092"
    author: "human:ahg-g"
    last_modified: "2026-05-13T18:36:03Z"
  - id: "pr-1350-r3310261653"
    resource: "https://github.com/llm-d/llm-d-router/pull/1350#discussion_r3310261653"
    author: "human:roytman"
    last_modified: "2026-05-27T10:37:56Z"
  - id: "pr-1475-r3358215465"
    resource: "https://github.com/llm-d/llm-d-router/pull/1475#discussion_r3358215465"
    author: "human:zetxqx"
    last_modified: "2026-06-04T18:58:31Z"
  - id: "issue-1516"
    resource: "https://github.com/llm-d/llm-d-router/issues/1516"
    author: "human:hexfusion"
    last_modified: "2026-06-07T03:15:10Z"
  - id: "issue-1524"
    resource: "https://github.com/llm-d/llm-d-router/issues/1524"
    author: "human:hexfusion"
    last_modified: "2026-06-07T15:20:31Z"
  - id: "pr-3479-r4316182503"
    resource: "https://github.com/llm-d/llm-d-router/pull/3479#discussion_r4316182503"
    author: "human:zetxqx"
    last_modified: "2026-10-19T16:20:17Z"
  - id: "pr-1475-r3358239412"
    resource: "https://github.com/llm-d/llm-d-router/pull/1475#discussion_r3358239412"
    author: "human:zetxqx"
    last_modified: "2026-06-04T19:02:55Z"
  - id: "pr-2987-r4090885940"
    resource: "https://github.com/llm-d/llm-d-router/pull/2987#discussion_r4090885940"
    author: "human:roytman"
    last_modified: "2026-09-24T07:10:35Z"
  - id: "pr-822-r3182740743"
    resource: "https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740743"
    author: "human:elevran"
    last_modified: "2026-05-04T15:46:55Z"
  - id: "pr-916-r3161138182"
    resource: "https://github.com/llm-d/llm-d-router/pull/916#discussion_r3161138182"
    author: "human:hexfusion"
    last_modified: "2026-04-29T12:57:35Z"
  - id: "pr-2391-r3798712574"
    resource: "https://github.com/llm-d/llm-d-router/pull/2391#discussion_r3798712574"
    author: "human:sagearc"
    last_modified: "2026-08-17T20:51:12Z"
  - id: "pr-2432-rev-4976441002"
    resource: "https://github.com/llm-d/llm-d-router/pull/2432#pullrequestreview-4976441002"
    author: "human:nilig"
    last_modified: "2026-08-19T20:25:48Z"
  - id: "issue-848"
    resource: "https://github.com/llm-d/llm-d-router/issues/848"
    author: "human:elevran"
    last_modified: "2026-04-20T09:27:44Z"
  - id: "pr-860-r3273772782"
    resource: "https://github.com/llm-d/llm-d-router/pull/860#discussion_r3273772782"
    author: "human:elevran"
    last_modified: "2026-05-20T12:08:52Z"
  - id: "pr-932-r3178050269"
    resource: "https://github.com/llm-d/llm-d-router/pull/932#discussion_r3178050269"
    author: "human:ahg-g"
    last_modified: "2026-05-03T11:37:43Z"
  - id: "issue-1019"
    resource: "https://github.com/llm-d/llm-d-router/issues/1019"
    author: "human:liu-cong"
    last_modified: "2026-05-07T18:14:18Z"
  - id: "pr-2723-r4165902195"
    resource: "https://github.com/llm-d/llm-d-router/pull/2723#discussion_r4165902195"
    author: "human:roytman"
    last_modified: "2026-10-02T12:56:05Z"
  - id: "issue-1009-ic-4399431503"
    resource: "https://github.com/llm-d/llm-d-router/issues/1009#issuecomment-4399431503"
    author: "human:liu-cong"
    last_modified: "2026-05-07T17:22:57Z"
  - id: "issue-1009-ic-4400172236"
    resource: "https://github.com/llm-d/llm-d-router/issues/1009#issuecomment-4400172236"
    author: "human:liu-cong"
    last_modified: "2026-05-07T18:57:52Z"
  - id: "issue-841"
    resource: "https://github.com/llm-d/llm-d-router/issues/841"
    author: "human:zetxqx"
    last_modified: "2026-04-18T01:21:08Z"
  - id: "pr-1112-r3242332547"
    resource: "https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3242332547"
    author: "human:bongwoobak"
    last_modified: "2026-05-14T15:15:07Z"
  - id: "pr-1738-r3474223470"
    resource: "https://github.com/llm-d/llm-d-router/pull/1738#discussion_r3474223470"
    author: "human:bongwoobak"
    last_modified: "2026-06-25T12:08:00Z"
  - id: "pr-2831-r4033597903"
    resource: "https://github.com/llm-d/llm-d-router/pull/2831#discussion_r4033597903"
    author: "human:nilig"
    last_modified: "2026-09-17T05:49:10Z"
  - id: "pr-1444-r3346561765"
    resource: "https://github.com/llm-d/llm-d-router/pull/1444#discussion_r3346561765"
    author: "human:shmuelk"
    last_modified: "2026-06-03T07:05:32Z"
  - id: "pr-1653-r3412845630"
    resource: "https://github.com/llm-d/llm-d-router/pull/1653#discussion_r3412845630"
    author: "human:roytman"
    last_modified: "2026-06-15T10:44:51Z"
  - id: "pr-2094-r3730268179"
    resource: "https://github.com/llm-d/llm-d-router/pull/2094#discussion_r3730268179"
    author: "human:nilig"
    last_modified: "2026-08-06T15:50:18Z"
  - id: "pr-2299-r3731458268"
    resource: "https://github.com/llm-d/llm-d-router/pull/2299#discussion_r3731458268"
    author: "human:ahg-g"
    last_modified: "2026-08-06T19:35:33Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Protocol parsing and wire fidelity

Passthrough JSON fidelity, declarative route claims, complete SSE frame buffering, multi-API multimodal parity, host-port-rank endpoint keys, and disaggregated stage ordering.

## Preserve unrecognized client request fields and wire fidelity

### Rule

Decode only router-inspected fields into typed request structs while preserving unknown client fields, 64-bit integer precision, JSON key ordering, and explicit boolean flags on forwarded payloads. Normalize headers and paths once at ingress through read-only accessors without mutating forwarded headers.

### Scope and consequence

- **Applies to**: `pkg/epp/framework/plugins/requesthandling/**`, `pkg/common/request/**`, `pkg/coordinator/**`, `pkg/sidecar/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Re-serializing whole request structs, splitting SSE frames mid-event, or keying workers by IP alone drops vendor fields, corrupts streaming responses, and collides multi-worker pods.
- **Related decisions**: [Opt-in backend request priority rewrite via PriorityRewriter interface](../decisions/opt_in_backend_priority_rewrite.md).

### Evidence

Observed across 51 pull requests and 11 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-1045][^issue-1072][^issue-1092][^pr-1350-r3310261653]

## Claim inference routes declaratively with unmatched-route fallthrough

### Rule

Register plugin and endpoint route claims declaratively by HTTP method and path so unmatched routes fall through cleanly to downstream handlers without hardcoding endpoint paths in core dispatchers.

### Scope and consequence

- **Applies to**: `pkg/epp/framework/plugins/requesthandling/**`, `pkg/common/request/**`, `pkg/coordinator/**`, `pkg/sidecar/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Re-serializing whole request structs, splitting SSE frames mid-event, or keying workers by IP alone drops vendor fields, corrupts streaming responses, and collides multi-worker pods.
- **Conflict audit references**: `APW-04`, `UNR-11`.

### Reviewers accept

Representing plugin routing claims as separate path and protocol slices when claimed paths apply uniformly, and forwarding native client payloads directly to disaggregated workers when all stages run the same engine.[^pr-1475-r3358239412][^pr-2987-r4090885940]

### Evidence

Observed across 14 pull requests and 8 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1475-r3358215465][^issue-1516][^issue-1524][^pr-3479-r4316182503]

## Buffer streaming responses to complete SSE event boundaries and count terminal usage once

### Rule

Buffer chunked streaming responses to full SSE event delimiters before inspecting frames so network splits mid-event never drop `data:` prefixes. Preserve `http.Flusher` via `Unwrap()` on response writer wrappers, record token usage only from the terminal stream chunk, and treat premature EOF or missing terminal events as stream errors.

### Scope and consequence

- **Applies to**: `pkg/epp/framework/plugins/requesthandling/**`, `pkg/common/request/**`, `pkg/coordinator/**`, `pkg/sidecar/**` (referenced symbols: `http.Flusher`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Re-serializing whole request structs, splitting SSE frames mid-event, or keying workers by IP alone drops vendor fields, corrupts streaming responses, and collides multi-worker pods.

### Evidence

Distilled from 6 review clusters (15 pull requests / 9 maintainers, 167 pull requests / 17 maintainers, 150 pull requests / 14 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-822-r3182740743][^pr-916-r3161138182][^pr-2391-r3798712574][^pr-2432-rev-4976441002]

## Maintain feature parity across inference APIs and validate multimodal items individually

### Rule

Support tokenization, tool-call rendering, usage extraction, and length classification symmetrically across chat completions, legacy completions, responses, messages, and generate endpoints. Group per-item attributes into element structs instead of parallel slices, and validate multimodal metadata per item and modality rather than with request-wide booleans.

### Scope and consequence

- **Applies to**: `pkg/epp/framework/plugins/requesthandling/**`, `pkg/common/request/**`, `pkg/coordinator/**`, `pkg/sidecar/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Re-serializing whole request structs, splitting SSE frames mid-event, or keying workers by IP alone drops vendor fields, corrupts streaming responses, and collides multi-worker pods.

### Reviewers accept

Hashing multimodal items as whole units rather than chunking like text when the upstream inference engine hashes media items atomically.[^issue-1009-ic-4399431503][^issue-1009-ic-4400172236]

### Evidence

Distilled from 2 review clusters (13 pull requests / 9 maintainers, 51 pull requests / 11 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-848][^pr-860-r3273772782][^pr-932-r3178050269][^issue-1019][^pr-2723-r4165902195]

## Key endpoint state by host, port, and worker rank

### Rule

Key per-endpoint state, cache indexes, eviction hooks, and event-socket ports by host, port, and worker rank rather than pod IP alone so multi-rank workers on one host never collide.

### Scope and consequence

- **Applies to**: `pkg/epp/framework/plugins/requesthandling/**`, `pkg/common/request/**`, `pkg/coordinator/**`, `pkg/sidecar/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Re-serializing whole request structs, splitting SSE frames mid-event, or keying workers by IP alone drops vendor fields, corrupts streaming responses, and collides multi-worker pods.
- **Conflict audit references**: `APW-03`.

### Evidence

Observed across 13 pull requests and 8 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-841][^pr-1112-r3242332547][^pr-1738-r3474223470][^pr-2831-r4033597903]

## Order disaggregated stages and isolate stage-specific payloads

### Rule

In disaggregated pipelines, run upstream encode and prefill stages before dependent decode stages, isolate prefill-only payload mutations from decode requests, and cancel downstream stages immediately when an upstream stage fails.

### Scope and consequence

- **Applies to**: `pkg/epp/framework/plugins/requesthandling/**`, `pkg/common/request/**`, `pkg/coordinator/**`, `pkg/sidecar/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Re-serializing whole request structs, splitting SSE frames mid-event, or keying workers by IP alone drops vendor fields, corrupts streaming responses, and collides multi-worker pods.

### Evidence

Observed across 29 pull requests and 8 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1444-r3346561765][^pr-1653-r3412845630][^pr-2094-r3730268179][^pr-2299-r3731458268]

[^issue-1045]: [#1045](https://github.com/llm-d/llm-d-router/issues/1045) (human:yankay)
[^issue-1072]: [#1072](https://github.com/llm-d/llm-d-router/issues/1072) (human:sagearc)
[^issue-1092]: [#1092](https://github.com/llm-d/llm-d-router/issues/1092) (human:ahg-g)
[^pr-1350-r3310261653]: [#1350](https://github.com/llm-d/llm-d-router/pull/1350#discussion_r3310261653) (human:roytman)
[^pr-1475-r3358215465]: [#1475](https://github.com/llm-d/llm-d-router/pull/1475#discussion_r3358215465) (human:zetxqx)
[^issue-1516]: [#1516](https://github.com/llm-d/llm-d-router/issues/1516) (human:hexfusion)
[^issue-1524]: [#1524](https://github.com/llm-d/llm-d-router/issues/1524) (human:hexfusion)
[^pr-3479-r4316182503]: [#3479](https://github.com/llm-d/llm-d-router/pull/3479#discussion_r4316182503) (human:zetxqx)
[^pr-1475-r3358239412]: [#1475](https://github.com/llm-d/llm-d-router/pull/1475#discussion_r3358239412) (human:zetxqx)
[^pr-2987-r4090885940]: [#2987](https://github.com/llm-d/llm-d-router/pull/2987#discussion_r4090885940) (human:roytman)
[^pr-822-r3182740743]: [#822](https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740743) (human:elevran)
[^pr-916-r3161138182]: [#916](https://github.com/llm-d/llm-d-router/pull/916#discussion_r3161138182) (human:hexfusion)
[^pr-2391-r3798712574]: [#2391](https://github.com/llm-d/llm-d-router/pull/2391#discussion_r3798712574) (human:sagearc)
[^pr-2432-rev-4976441002]: [#2432](https://github.com/llm-d/llm-d-router/pull/2432#pullrequestreview-4976441002) (human:nilig)
[^issue-848]: [#848](https://github.com/llm-d/llm-d-router/issues/848) (human:elevran)
[^pr-860-r3273772782]: [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3273772782) (human:elevran)
[^pr-932-r3178050269]: [#932](https://github.com/llm-d/llm-d-router/pull/932#discussion_r3178050269) (human:ahg-g)
[^issue-1019]: [#1019](https://github.com/llm-d/llm-d-router/issues/1019) (human:liu-cong)
[^pr-2723-r4165902195]: [#2723](https://github.com/llm-d/llm-d-router/pull/2723#discussion_r4165902195) (human:roytman)
[^issue-1009-ic-4399431503]: [#1009](https://github.com/llm-d/llm-d-router/issues/1009#issuecomment-4399431503) (human:liu-cong)
[^issue-1009-ic-4400172236]: [#1009](https://github.com/llm-d/llm-d-router/issues/1009#issuecomment-4400172236) (human:liu-cong)
[^issue-841]: [#841](https://github.com/llm-d/llm-d-router/issues/841) (human:zetxqx)
[^pr-1112-r3242332547]: [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3242332547) (human:bongwoobak)
[^pr-1738-r3474223470]: [#1738](https://github.com/llm-d/llm-d-router/pull/1738#discussion_r3474223470) (human:bongwoobak)
[^pr-2831-r4033597903]: [#2831](https://github.com/llm-d/llm-d-router/pull/2831#discussion_r4033597903) (human:nilig)
[^pr-1444-r3346561765]: [#1444](https://github.com/llm-d/llm-d-router/pull/1444#discussion_r3346561765) (human:shmuelk)
[^pr-1653-r3412845630]: [#1653](https://github.com/llm-d/llm-d-router/pull/1653#discussion_r3412845630) (human:roytman)
[^pr-2094-r3730268179]: [#2094](https://github.com/llm-d/llm-d-router/pull/2094#discussion_r3730268179) (human:nilig)
[^pr-2299-r3731458268]: [#2299](https://github.com/llm-d/llm-d-router/pull/2299#discussion_r3731458268) (human:ahg-g)
