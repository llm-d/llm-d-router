---
type: Review Rule Set
title: Concurrency and shared state
description: "Mutex critical sections, bounded state with endpoint deletion cleanup, edge-triggered priority bands, drain-scoped contexts, per-replica state sync, and KV-event sequence recovery."
tags: ["concurrency-and-shared-state", "failure-class", "apw-07", "apw-10", "apw-13", "mvm-10", "apw-05", "apw-06"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - id: "pr-707-r3190593630"
    resource: "https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190593630"
    author: "human:roytman"
    last_modified: "2026-05-05T18:11:33Z"
  - id: "issue-889"
    resource: "https://github.com/llm-d/llm-d-router/issues/889"
    author: "human:shmuelk"
    last_modified: "2026-04-26T12:30:18Z"
  - id: "pr-1128-r3275924435"
    resource: "https://github.com/llm-d/llm-d-router/pull/1128#discussion_r3275924435"
    author: "human:LukeAVanDrie"
    last_modified: "2026-05-20T17:38:55Z"
  - id: "pr-1678-r3448540954"
    resource: "https://github.com/llm-d/llm-d-router/pull/1678#discussion_r3448540954"
    author: "human:vMaroon"
    last_modified: "2026-06-21T14:07:14Z"
  - id: "pr-942-r3178202254"
    resource: "https://github.com/llm-d/llm-d-router/pull/942#discussion_r3178202254"
    author: "human:elevran"
    last_modified: "2026-05-03T13:48:31Z"
  - id: "pr-1354-r3314150441"
    resource: "https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314150441"
    author: "human:LukeAVanDrie"
    last_modified: "2026-05-27T22:17:05Z"
  - id: "pr-1354-r3314202507"
    resource: "https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314202507"
    author: "human:LukeAVanDrie"
    last_modified: "2026-05-27T22:32:05Z"
  - id: "pr-1690-r3455137825"
    resource: "https://github.com/llm-d/llm-d-router/pull/1690#discussion_r3455137825"
    author: "human:LukeAVanDrie"
    last_modified: "2026-06-22T20:10:07Z"
  - id: "pr-707-r3190233226"
    resource: "https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190233226"
    author: "human:roytman"
    last_modified: "2026-05-05T17:07:35Z"
  - id: "pr-1004-r3202130559"
    resource: "https://github.com/llm-d/llm-d-router/pull/1004#discussion_r3202130559"
    author: "human:shmuelk"
    last_modified: "2026-05-07T14:20:00Z"
  - id: "pr-1727-r3513605187"
    resource: "https://github.com/llm-d/llm-d-router/pull/1727#discussion_r3513605187"
    author: "human:elevran"
    last_modified: "2026-07-02T13:56:37Z"
  - id: "issue-1736"
    resource: "https://github.com/llm-d/llm-d-router/issues/1736"
    author: "human:RishabhSaini"
    last_modified: "2026-06-24T15:43:53Z"
  - id: "pr-2531-r4086142755"
    resource: "https://github.com/llm-d/llm-d-router/pull/2531#discussion_r4086142755"
    author: "human:ahg-g"
    last_modified: "2026-09-23T18:57:36Z"
  - id: "pr-901-r3269839644"
    resource: "https://github.com/llm-d/llm-d-router/pull/901#discussion_r3269839644"
    author: "human:liu-cong"
    last_modified: "2026-05-19T21:44:54Z"
  - id: "pr-1354-r3314143560"
    resource: "https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314143560"
    author: "human:LukeAVanDrie"
    last_modified: "2026-05-27T22:15:19Z"
  - id: "pr-1354-r3391664697"
    resource: "https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3391664697"
    author: "human:LukeAVanDrie"
    last_modified: "2026-06-10T21:36:00Z"
  - id: "issue-1606"
    resource: "https://github.com/llm-d/llm-d-router/issues/1606"
    author: "human:LukeAVanDrie"
    last_modified: "2026-06-11T18:02:08Z"
  - id: "pr-1608-ic-4688020707"
    resource: "https://github.com/llm-d/llm-d-router/pull/1608#issuecomment-4688020707"
    author: "human:LukeAVanDrie"
    last_modified: "2026-06-12T06:17:09Z"
  - id: "pr-822-r3182740756"
    resource: "https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740756"
    author: "human:elevran"
    last_modified: "2026-05-04T15:46:55Z"
  - id: "pr-890-r3149928582"
    resource: "https://github.com/llm-d/llm-d-router/pull/890#discussion_r3149928582"
    author: "human:hexfusion"
    last_modified: "2026-04-27T20:16:05Z"
  - id: "pr-2713-r3996027352"
    resource: "https://github.com/llm-d/llm-d-router/pull/2713#discussion_r3996027352"
    author: "human:vMaroon"
    last_modified: "2026-09-12T11:07:32Z"
  - id: "issue-3140"
    resource: "https://github.com/llm-d/llm-d-router/issues/3140"
    author: "human:vMaroon"
    last_modified: "2026-10-03T09:10:20Z"
  - id: "issue-1892"
    resource: "https://github.com/llm-d/llm-d-router/issues/1892"
    author: "human:vMaroon"
    last_modified: "2026-07-04T21:06:11Z"
  - id: "issue-1946"
    resource: "https://github.com/llm-d/llm-d-router/issues/1946"
    author: "human:elevran"
    last_modified: "2026-07-09T16:07:41Z"
  - id: "pr-2716-r3955475084"
    resource: "https://github.com/llm-d/llm-d-router/pull/2716#discussion_r3955475084"
    author: "human:bongwoobak"
    last_modified: "2026-09-08T07:41:10Z"
  - id: "pr-2837-r4214243393"
    resource: "https://github.com/llm-d/llm-d-router/pull/2837#discussion_r4214243393"
    author: "human:ahg-g"
    last_modified: "2026-10-08T03:20:14Z"
  - id: "issue-1737"
    resource: "https://github.com/llm-d/llm-d-router/issues/1737"
    author: "human:RishabhSaini"
    last_modified: "2026-06-24T15:48:04Z"
  - id: "pr-1887-r3684757376"
    resource: "https://github.com/llm-d/llm-d-router/pull/1887#discussion_r3684757376"
    author: "human:bongwoobak"
    last_modified: "2026-07-30T17:10:47Z"
  - id: "pr-2946-rev-5438730630"
    resource: "https://github.com/llm-d/llm-d-router/pull/2946#pullrequestreview-5438730630"
    author: "human:ahg-g"
    last_modified: "2026-10-07T06:59:11Z"
  - id: "issue-3399"
    resource: "https://github.com/llm-d/llm-d-router/issues/3399"
    author: "human:vMaroon"
    last_modified: "2026-10-13T21:16:04Z"
  - id: "issue-3400"
    resource: "https://github.com/llm-d/llm-d-router/issues/3400"
    author: "human:vMaroon"
    last_modified: "2026-10-13T21:16:06Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Concurrency and shared state

Mutex critical sections, bounded state with endpoint deletion cleanup, edge-triggered priority bands, drain-scoped contexts, per-replica state sync, and KV-event sequence recovery.

## Keep mutex critical sections tight and release locks before network or blocking calls

### Rule

Encapsulate synchronization inside methods on the state-owning type, complete struct initialization before registering callbacks, and never hold a mutex across network RPCs, channel sends, or socket teardown. Combine read-modify-write transitions and coupled counter reads under a single lock acquisition, and copy caller-owned maps before asynchronous handoff.

### Scope and consequence

- **Applies to**: `pkg/epp/datastore/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/scheduling/**`, `pkg/epp/statesync/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Holding locks across I/O, retaining state for deleted endpoints, or ignoring `ctx.Err()` causes request-path contention, unbounded memory growth, and goroutine leaks during pod churn.
- **Conflict audit references**: `APW-07`.

### Reviewers accept

Defensively copying caller-supplied slices or maps at public goroutine boundaries, coalescing full state snapshots in capacity-one channels, holding uncontended startup or shutdown locks when unlocking risks handle races, and keeping an in-memory handle invalidation loop under the lock when releasing it early would race with concurrent handle removal.[^pr-942-r3178202254][^pr-1354-r3314150441][^pr-1354-r3314202507][^pr-1690-r3455137825]

### Evidence

Observed across 61 pull requests and 12 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-707-r3190593630][^issue-889][^pr-1128-r3275924435][^pr-1678-r3448540954]

## Bound in-memory state with capacity limits and endpoint deletion cleanup

### Rule

Guard in-memory maps and queues keyed by requests, sessions, prompts, or endpoints with hard capacity caps and periodic TTL sweeps so long-running processes do not leak memory. Subscribe stateful plugins to endpoint lifecycle events to purge per-endpoint state on pod removal or empty controller reconciles.

### Scope and consequence

- **Applies to**: `pkg/epp/datastore/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/scheduling/**`, `pkg/epp/statesync/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Holding locks across I/O, retaining state for deleted endpoints, or ignoring `ctx.Err()` causes request-path contention, unbounded memory growth, and goroutine leaks during pod churn.
- **Conflict audit references**: `APW-10`, `APW-13`.

### Reviewers accept

A dedicated bounded registry in place of framework `PluginState` when the state needs a no-eviction mode or pinning of in-flight entries,[^pr-2531-r4086142755] and a periodic cleanup loop in place of lifecycle event subscription when a missed event would leave stale entries.[^pr-901-r3269839644]

### Evidence

Observed across 27 pull requests and 12 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-707-r3190233226][^pr-1004-r3202130559][^pr-1727-r3513605187][^issue-1736]

## Reconcile dynamic priority bands on controller edges and drain before removal

### Rule

Provision and remove dynamic `InferenceObjective` priority bands through edge-triggered controller reconciliation rather than background data-plane idleness sweeps, preserving statically configured default bands and draining in-flight queues before deleting removed bands.

### Scope and consequence

- **Applies to**: `pkg/epp/datastore/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/scheduling/**`, `pkg/epp/statesync/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Holding locks across I/O, retaining state for deleted endpoints, or ignoring `ctx.Err()` causes request-path contention, unbounded memory growth, and goroutine leaks during pod churn.
- **Related decisions**: [Channel-fed maintenance loop with lock protection for dynamic priority bands](../decisions/channel_fed_priority_band_reconciliation.md).

### Evidence

Distilled from 4 review clusters (27 pull requests / 12 maintainers, 150 pull requests / 14 maintainers, 4 pull requests / 5 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1354-r3314143560][^pr-1354-r3391664697][^issue-1606][^pr-1608-ic-4688020707]

## Propagate caller contexts and bound goroutine lifetimes to server drain

### Rule

Root background collectors and serving loops on a drain-scoped context, and pass caller request contexts to all outbound RPCs. Check `ctx.Err()` on loop iterations and multi-channel `select` wakeups so client disconnects halt downstream work and release in-flight reservations, and run graceful readiness drain on leader lease loss.

### Scope and consequence

- **Applies to**: `pkg/epp/datastore/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/scheduling/**`, `pkg/epp/statesync/**`, `pkg/kvcache/**`, `pkg/kvevents/**` (referenced symbols: `ctx.Err()`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Holding locks across I/O, retaining state for deleted endpoints, or ignoring `ctx.Err()` causes request-path contention, unbounded memory growth, and goroutine leaks during pod churn.

### Evidence

Distilled from 2 review clusters (16 pull requests / 6 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-822-r3182740756][^pr-890-r3149928582][^pr-2713-r3996027352][^issue-3140]

## Scope cross-replica state to per-replica keys with short remote-store TTLs

### Rule

Publish shared load and cache state under collision-free replica IDs and combine live local counters with cached peer levels at read time so one replica reconnecting or resetting never wipes another replica's state. Expire transient peer state with short remote-store TTLs scaled to the publish interval rather than comparing wall clocks across hosts, and do not block startup when optional peer stores are unreachable.

### Scope and consequence

- **Applies to**: `pkg/epp/datastore/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/scheduling/**`, `pkg/epp/statesync/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Holding locks across I/O, retaining state for deleted endpoints, or ignoring `ctx.Err()` causes request-path contention, unbounded memory growth, and goroutine leaks during pod churn.

### Evidence

Observed across 5 pull requests and 5 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-1892][^issue-1946][^pr-2716-r3955475084][^pr-2837-r4214243393]

## Recover KV-event sequence gaps and epoch resets from base snapshots

### Rule

Track per-endpoint sequence numbers on KV-cache event streams to detect missed batches or container restarts (`seq < lastSeq`), resetting stale index state and loading a base snapshot before applying subsequent events.

### Scope and consequence

- **Applies to**: `pkg/epp/datastore/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/scheduling/**`, `pkg/epp/statesync/**`, `pkg/kvcache/**`, `pkg/kvevents/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Holding locks across I/O, retaining state for deleted endpoints, or ignoring `ctx.Err()` causes request-path contention, unbounded memory growth, and goroutine leaks during pod churn.

### Evidence

Distilled from 4 review clusters (27 pull requests / 12 maintainers, 44 pull requests / 12 maintainers, 18 pull requests / 10 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-1737][^pr-1887-r3684757376][^pr-2946-rev-5438730630][^issue-3399][^issue-3400]

[^pr-707-r3190593630]: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190593630) (human:roytman)
[^issue-889]: [#889](https://github.com/llm-d/llm-d-router/issues/889) (human:shmuelk)
[^pr-1128-r3275924435]: [#1128](https://github.com/llm-d/llm-d-router/pull/1128#discussion_r3275924435) (human:LukeAVanDrie)
[^pr-1678-r3448540954]: [#1678](https://github.com/llm-d/llm-d-router/pull/1678#discussion_r3448540954) (human:vMaroon)
[^pr-942-r3178202254]: [#942](https://github.com/llm-d/llm-d-router/pull/942#discussion_r3178202254) (human:elevran)
[^pr-1354-r3314150441]: [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314150441) (human:LukeAVanDrie)
[^pr-1354-r3314202507]: [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314202507) (human:LukeAVanDrie)
[^pr-1690-r3455137825]: [#1690](https://github.com/llm-d/llm-d-router/pull/1690#discussion_r3455137825) (human:LukeAVanDrie)
[^pr-707-r3190233226]: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190233226) (human:roytman)
[^pr-1004-r3202130559]: [#1004](https://github.com/llm-d/llm-d-router/pull/1004#discussion_r3202130559) (human:shmuelk)
[^pr-1727-r3513605187]: [#1727](https://github.com/llm-d/llm-d-router/pull/1727#discussion_r3513605187) (human:elevran)
[^issue-1736]: [#1736](https://github.com/llm-d/llm-d-router/issues/1736) (human:RishabhSaini)
[^pr-2531-r4086142755]: [#2531](https://github.com/llm-d/llm-d-router/pull/2531#discussion_r4086142755) (human:ahg-g)
[^pr-901-r3269839644]: [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3269839644) (human:liu-cong)
[^pr-1354-r3314143560]: [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314143560) (human:LukeAVanDrie)
[^pr-1354-r3391664697]: [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3391664697) (human:LukeAVanDrie)
[^issue-1606]: [#1606](https://github.com/llm-d/llm-d-router/issues/1606) (human:LukeAVanDrie)
[^pr-1608-ic-4688020707]: [#1608](https://github.com/llm-d/llm-d-router/pull/1608#issuecomment-4688020707) (human:LukeAVanDrie)
[^pr-822-r3182740756]: [#822](https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740756) (human:elevran)
[^pr-890-r3149928582]: [#890](https://github.com/llm-d/llm-d-router/pull/890#discussion_r3149928582) (human:hexfusion)
[^pr-2713-r3996027352]: [#2713](https://github.com/llm-d/llm-d-router/pull/2713#discussion_r3996027352) (human:vMaroon)
[^issue-3140]: [#3140](https://github.com/llm-d/llm-d-router/issues/3140) (human:vMaroon)
[^issue-1892]: [#1892](https://github.com/llm-d/llm-d-router/issues/1892) (human:vMaroon)
[^issue-1946]: [#1946](https://github.com/llm-d/llm-d-router/issues/1946) (human:elevran)
[^pr-2716-r3955475084]: [#2716](https://github.com/llm-d/llm-d-router/pull/2716#discussion_r3955475084) (human:bongwoobak)
[^pr-2837-r4214243393]: [#2837](https://github.com/llm-d/llm-d-router/pull/2837#discussion_r4214243393) (human:ahg-g)
[^issue-1737]: [#1737](https://github.com/llm-d/llm-d-router/issues/1737) (human:RishabhSaini)
[^pr-1887-r3684757376]: [#1887](https://github.com/llm-d/llm-d-router/pull/1887#discussion_r3684757376) (human:bongwoobak)
[^pr-2946-rev-5438730630]: [#2946](https://github.com/llm-d/llm-d-router/pull/2946#pullrequestreview-5438730630) (human:ahg-g)
[^issue-3399]: [#3399](https://github.com/llm-d/llm-d-router/issues/3399) (human:vMaroon)
[^issue-3400]: [#3400](https://github.com/llm-d/llm-d-router/issues/3400) (human:vMaroon)
