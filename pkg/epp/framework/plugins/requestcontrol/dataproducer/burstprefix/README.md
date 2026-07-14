# Burst Prefix Cache Producer

**Type:** `burst-prefix-cache-producer`

A request-level data producer that co-locates bursts of prompt-sharing requests
so a shared prefix is prefilled once instead of scattered across replicas on a
cold cache.

## Problem

When many requests that share a prompt arrive at the same instant (for example
the `n` group samples of an RL rollout step), every replica's prefix cache is
still cold, so a cache-state scorer scores them all zero and load balancing
spreads them. The shared prompt is then prefilled redundantly on several
replicas and the prefix-cache benefit is lost.

## What it does

Requests arriving within a configurable window are assigned jointly:

1. Each request's prompt is hashed into prefix blocks (shared `prefixhash`).
2. Requests with an identical prompt prefix are grouped. An identical group of
   more than one member is always a placement unit; a single request becomes a
   unit only when `minColocateBlocks > 0` and it shares at least that many leading
   blocks with some other request in the batch.
3. Units are steered onto a replica (or a bounded set of replicas), filling one
   replica up to `maxPerReplica` before spilling to the next least-loaded replica.
   Identical groups are placed first and kept whole, so the proven same-prompt
   co-location is the firm structure; prefix-sharing units then attach to it. A
   unit prefers a replica that already holds a unit sharing at least
   `minColocateBlocks` leading blocks and is still under its fair share of the
   batch (prefix co-location bounded by balance, so a shared prefix is prefilled
   once without stampeding prefix-sharing units onto one replica); otherwise units
   are balanced across replicas. Longer-prefix units are placed first so shorter
   units match against the richest set of already-placed prefixes.
4. The producer emits `PrefixCacheMatchInfo` with a full match on the assigned
   replica and zero elsewhere.

A request that overlaps no other in the batch, and any prefix-less request,
receives no affinity (scored zero everywhere), leaving it to other scorers.
With `minColocateBlocks == 0` only identical groups are placed.

## Scoring

This producer emits `PrefixCacheMatchInfo`; it does not score. Reuse the
`prefix-cache-scorer` and point it at this producer:

```yaml
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: burst-prefix-cache-producer
```

## Configuration

| field | default | meaning |
|---|---|---|
| `windowDurationMs` | 100 | batch window T in milliseconds; must be in `1..10000`. Every request waits up to this long, so keep it small relative to other request processing times. |
| `maxPerReplica` | -1 | max samples of one group per replica (k); -1 = unlimited (whole group to one replica) |
| `blockSizeTokens` | 64 | token block size for prefix hashing |
| `maxPrefixTokensToMatch` | 0 | cap on matched prefix tokens; 0 uses the default block cap. A positive value must be >= `blockSizeTokens`, otherwise it yields zero prefix blocks. |
| `minColocateBlocks` | 0 | min shared leading blocks for inter-unit co-location and for a single request to gain an affinity; 0 disables both (only identical groups are placed, purely load-balanced) |
| `maxBatchSize` | 1000 | max requests one window may accumulate; Produce returns an error once reached. -1 = unlimited. |
| `balanceBy` | `requests` | quantity balanced across replicas within a window: `requests` (by request count) or `tokens` (by prefix-block load, charging each unit's prefix once per replica and discounting the leading blocks already held there). Token balancing spreads long-prompt units so no replica saturates on stacked prefixes while request counts stay even. |

## Operational notes

- The producer adds up to `windowDurationMs` of latency per request while a
  window fills; its producer timeout is extended to cover the window.
- Co-location is decided within a single window from the request prompts. Identity
  (the samples of one prompt) is matched exactly; partial overlap between distinct
  prompts is matched to `minColocateBlocks` leading blocks. Cross-step warm-cache
  reuse is left to the persistent (approximate or precise) prefix cache producer,
  which can run alongside this one.
