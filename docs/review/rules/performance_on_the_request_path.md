---
type: Review Rule Set
title: Performance on the request path
description: "Zero avoidable hot-path allocations, pooled transports and multiplexed event sockets, orthogonal load signals, contiguous prefix-cache scoring, and reproducible benchmarks."
tags: ["performance-on-the-request-path", "failure-class", "apw-09", "unr-09", "apw-14", "apw-15"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - { id: "pr-683-r3143174078", resource: "https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143174078", author: "human:elevran", last_modified: "2026-04-26T07:53:10Z" }
  - { id: "pr-860-r3367176286", resource: "https://github.com/llm-d/llm-d-router/pull/860#discussion_r3367176286", author: "human:ahg-g", last_modified: "2026-06-06T11:00:30Z" }
  - { id: "pr-880-r3149083076", resource: "https://github.com/llm-d/llm-d-router/pull/880#discussion_r3149083076", author: "human:shmuelk", last_modified: "2026-04-27T17:27:09Z" }
  - { id: "pr-890-r3224041877", resource: "https://github.com/llm-d/llm-d-router/pull/890#discussion_r3224041877", author: "human:shmuelk", last_modified: "2026-05-12T05:58:56Z" }
  - { id: "issue-2045", resource: "https://github.com/llm-d/llm-d-router/issues/2045", author: "human:hexfusion", last_modified: "2026-07-16T11:27:56Z" }
  - { id: "pr-2716-r4095311906", resource: "https://github.com/llm-d/llm-d-router/pull/2716#discussion_r4095311906", author: "human:hyeongyun0916", last_modified: "2026-09-24T15:22:46Z" }
  - { id: "pr-1534-r3401170547", resource: "https://github.com/llm-d/llm-d-router/pull/1534#discussion_r3401170547", author: "human:elevran", last_modified: "2026-06-12T06:23:54Z" }
  - { id: "pr-1580-r3391563214", resource: "https://github.com/llm-d/llm-d-router/pull/1580#discussion_r3391563214", author: "human:LukeAVanDrie", last_modified: "2026-06-10T21:16:12Z" }
  - { id: "pr-2032-r3643867571", resource: "https://github.com/llm-d/llm-d-router/pull/2032#discussion_r3643867571", author: "human:roytman", last_modified: "2026-07-24T08:03:05Z" }
  - { id: "pr-1438-r3338979029", resource: "https://github.com/llm-d/llm-d-router/pull/1438#discussion_r3338979029", author: "human:ahg-g", last_modified: "2026-06-02T05:56:31Z" }
  - { id: "pr-2968-r4214852975", resource: "https://github.com/llm-d/llm-d-router/pull/2968#discussion_r4214852975", author: "human:zetxqx", last_modified: "2026-10-08T04:51:37Z" }
  - { id: "pr-860-r3354232991", resource: "https://github.com/llm-d/llm-d-router/pull/860#discussion_r3354232991", author: "human:elevran", last_modified: "2026-06-04T07:25:24Z" }
  - { id: "issue-2031", resource: "https://github.com/llm-d/llm-d-router/issues/2031", author: "human:nilig", last_modified: "2026-07-15T14:23:36Z" }
  - { id: "pr-2032-r3610603777", resource: "https://github.com/llm-d/llm-d-router/pull/2032#discussion_r3610603777", author: "human:shmuelk", last_modified: "2026-07-19T13:05:58Z" }
  - { id: "issue-655", resource: "https://github.com/llm-d/llm-d-router/issues/655", author: "human:vMaroon", last_modified: "2026-02-25T18:08:46Z" }
  - { id: "issue-1006", resource: "https://github.com/llm-d/llm-d-router/issues/1006", author: "human:liu-cong", last_modified: "2026-05-06T18:25:22Z" }
  - { id: "pr-3068-ic-5930096809", resource: "https://github.com/llm-d/llm-d-router/pull/3068#issuecomment-5930096809", author: "human:ahg-g", last_modified: "2026-10-01T11:08:41Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Performance on the request path

Zero avoidable hot-path allocations, pooled transports and multiplexed event sockets, orthogonal load signals, contiguous prefix-cache scoring, and reproducible benchmarks.

## Avoid heap allocations and unhoisted work on the request path

### Rule

Keep per-request routing, scoring, and event-decoding paths free of avoidable byte-to-string conversions, slice or map clones, untyped map unmarshaling, and unmutated payload re-serialization. Hoist header normalization, extractor lookup maps, and type checks to startup, pre-allocate slices of known length, and fold consecutive read-locked lookups into one pass.

### Scope and consequence

- **Applies to**: `pkg/epp/scheduling/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/datalayer/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Avoidable per-request allocations, unpooled connections, or unbenchmarked scoring changes inflate router tail latency and socket churn under load.
- **Conflict audit references**: `APW-09`.

### Evidence

Observed across 85 pull requests and 15 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-683-r3143174078][^pr-860-r3367176286][^pr-880-r3149083076]

## Reuse pooled HTTP transports and multiplex per-endpoint event subscriptions

### Rule

Configure backend and metrics-scraper HTTP clients with keep-alive connection pooling, swapping the transport only when TLS certificates rotate rather than per scrape. Multiplex per-endpoint ZMQ or event subscriptions across in-process consumers instead of opening duplicate sockets per plugin, and batch periodic background network sweeps off the request path.

### Scope and consequence

- **Applies to**: `pkg/epp/scheduling/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/datalayer/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Avoidable per-request allocations, unpooled connections, or unbenchmarked scoring changes inflate router tail latency and socket churn under load.

### Evidence

Observed across 4 pull requests and 4 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-890-r3224041877][^issue-2045][^pr-2716-r4095311906]

## Evaluate orthogonal load signals over filtered candidate subsets

### Rule

Keep queue depth, in-flight requests, and KV-cache utilization as separate signals evaluated over the filtered candidate subset, and release every in-flight increment on cancellation or error.

### Scope and consequence

- **Applies to**: `pkg/epp/scheduling/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/datalayer/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Avoidable per-request allocations, unpooled connections, or unbenchmarked scoring changes inflate router tail latency and socket churn under load.

### Reviewers accept

Constant-time byte-length token estimation and single-endpoint soft-pinning session affinity when backed by engine contracts or multi-pod benchmarks.[^pr-1438-r3338979029][^pr-2968-r4214852975]

### Evidence

Observed across 32 pull requests and 9 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1534-r3401170547][^pr-1580-r3391563214][^pr-2032-r3643867571]

## Score prefix caches contiguously and break near-ties by load

### Rule

Score prefix caches contiguously per sequence by halting block credit at the first missing block, isolate cache keys by tenant salt, and break near-ties with load-weighted sampling so hot prefixes do not overload a single worker.

### Scope and consequence

- **Applies to**: `pkg/epp/scheduling/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/datalayer/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Avoidable per-request allocations, unpooled connections, or unbenchmarked scoring changes inflate router tail latency and socket churn under load.
- **Conflict audit references**: `UNR-09`.

### Evidence

Distilled from 2 review clusters (18 pull requests / 10 maintainers, 15 pull requests / 9 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-860-r3354232991][^issue-2031][^pr-2032-r3610603777]

## Gate routing policies and hot-path changes on reproducible before-and-after benchmarks

### Rule

Validate new scoring formulas, cache indexes, hot-path locks, and default threshold changes with reproducible microbenchmarks covering both hit and miss paths alongside cluster evaluations against active routing baselines. Commit reproducible benchmark scripts and summary reports while keeping raw execution logs out of version control.

### Scope and consequence

- **Applies to**: `pkg/epp/scheduling/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/datalayer/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Avoidable per-request allocations, unpooled connections, or unbenchmarked scoring changes inflate router tail latency and socket churn under load.
- **Conflict audit references**: `APW-14`, `APW-15`.

### Evidence

Distilled from 2 review clusters (42 pull requests / 14 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-655][^issue-1006][^pr-3068-ic-5930096809]

[^pr-683-r3143174078]: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143174078) (human:elevran)
[^pr-860-r3367176286]: [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3367176286) (human:ahg-g)
[^pr-880-r3149083076]: [#880](https://github.com/llm-d/llm-d-router/pull/880#discussion_r3149083076) (human:shmuelk)
[^pr-890-r3224041877]: [#890](https://github.com/llm-d/llm-d-router/pull/890#discussion_r3224041877) (human:shmuelk)
[^issue-2045]: [#2045](https://github.com/llm-d/llm-d-router/issues/2045) (human:hexfusion)
[^pr-2716-r4095311906]: [#2716](https://github.com/llm-d/llm-d-router/pull/2716#discussion_r4095311906) (human:hyeongyun0916)
[^pr-1534-r3401170547]: [#1534](https://github.com/llm-d/llm-d-router/pull/1534#discussion_r3401170547) (human:elevran)
[^pr-1580-r3391563214]: [#1580](https://github.com/llm-d/llm-d-router/pull/1580#discussion_r3391563214) (human:LukeAVanDrie)
[^pr-2032-r3643867571]: [#2032](https://github.com/llm-d/llm-d-router/pull/2032#discussion_r3643867571) (human:roytman)
[^pr-1438-r3338979029]: [#1438](https://github.com/llm-d/llm-d-router/pull/1438#discussion_r3338979029) (human:ahg-g)
[^pr-2968-r4214852975]: [#2968](https://github.com/llm-d/llm-d-router/pull/2968#discussion_r4214852975) (human:zetxqx)
[^pr-860-r3354232991]: [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3354232991) (human:elevran)
[^issue-2031]: [#2031](https://github.com/llm-d/llm-d-router/issues/2031) (human:nilig)
[^pr-2032-r3610603777]: [#2032](https://github.com/llm-d/llm-d-router/pull/2032#discussion_r3610603777) (human:shmuelk)
[^issue-655]: [#655](https://github.com/llm-d/llm-d-router/issues/655) (human:vMaroon)
[^issue-1006]: [#1006](https://github.com/llm-d/llm-d-router/issues/1006) (human:liu-cong)
[^pr-3068-ic-5930096809]: [#3068](https://github.com/llm-d/llm-d-router/pull/3068#issuecomment-5930096809) (human:ahg-g)
