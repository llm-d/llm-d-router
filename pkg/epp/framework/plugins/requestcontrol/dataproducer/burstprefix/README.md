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
2. Requests with an identical prompt prefix are grouped.
3. Each group with more than one member is steered onto a replica (or a bounded
   set of replicas), filling one replica up to `maxPerReplica` before spilling
   to the next least-loaded replica. When `minColocateBlocks > 0`, a group
   prefers a replica that already holds a placed group sharing at least that many
   leading blocks and is still under its fair share of the batch (inter-group
   prefix co-location bounded by balance, so a long shared prefix is prefilled
   once without stampeding prefix-sharing groups onto one replica); otherwise
   groups are balanced across replicas. Longer-prefix groups are placed first so
   shorter groups match against the richest set of already-placed prefixes.
4. The producer emits `PrefixCacheMatchInfo` with a full match on the assigned
   replica and zero elsewhere.

Singletons and prefix-less requests receive no affinity (scored zero
everywhere), leaving them to other scorers.

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
| `windowDurationMs` | 100 | batch window T in milliseconds |
| `maxPerReplica` | -1 | max samples of one group per replica (k); -1 = unlimited (whole group to one replica) |
| `blockSizeTokens` | 64 | token block size for prefix hashing |
| `maxPrefixTokensToMatch` | 0 | cap on matched prefix tokens; 0 uses the default block cap |
| `minColocateBlocks` | 0 | min shared leading blocks to co-locate two distinct groups; 0 disables inter-group co-location (placement is purely load-balanced) |

## Operational notes

- The producer adds up to `windowDurationMs` of latency per request while a
  window fills; its producer timeout is extended to cover the window.
- Grouping is by exact prompt-prefix match. Cross-step warm-cache reuse is left
  to the persistent (approximate or precise) prefix cache producer, which can
  run alongside this one.
