# KV-Cache Indexer

Scores model-serving pods by KV-cache locality: given a request's tokens, it
determines which pods already hold the corresponding KV blocks and ranks them by
longest shared prefix, so the scheduler can route to a pod that maximizes cache
reuse.

## What It Does

The `Indexer` is the read side of the KV-cache subsystem. It turns a tokenized
prompt into KV-block keys, looks those keys up in the block index (kept current
by the [`kvevents`](../kvevents/README.md) subscriber), and produces a per-pod
score. The precise-prefix-cache scheduling scorer consumes these scores.

Tokenization happens externally: callers pass tokens in via `ScoreTokens`. The
indexer owns block-key computation, index lookup, and scoring.

## How It Works

- **Block-key computation.** `ComputeBlockKeysFromTokens` runs the injected
  [`kvblock.TokenProcessor`](kvblock/README.md) to chunk tokens into
  fixed-size blocks and hash each block (chaining the previous block's hash so a
  key encodes its whole prefix). `extraFeatures` taints the hash with per-block
  multimodal metadata when present.
- **Lookup.** `ScoreTokens` queries the [`kvblock.Index`](kvblock/README.md)
  for the pods that hold each block key, optionally restricted to a caller-
  supplied pod set.
- **Scoring.** A `KVBlockScorer` reduces the lookup result to per-pod scores.
  The default `LongestPrefixScorer` credits each pod for its longest run of
  consecutive block hits starting from block 0, weighted per device tier
  (`BackendConfigs`), so a pod that holds a longer contiguous prefix ranks
  higher.
- **Shared tiers.** Index entries keyed by a pseudo-pod identifier
  (`node:<nodeName>` for a node-local tier shared by all pods on a node,
  `pool:<name>` for a fleet-wide tier) are resolved to the candidate endpoints
  they cover before scoring (`kvblock.ResolvePseudoPods`). `pool:` entries
  pass every `Lookup` filter; `node:` entries must be named in the filter.

## Device Tier Weights

`kvCacheBackendConfigs` maps a `medium` string carried by KV events to a
scoring weight. Defaults: `gpu` 1.0, `cpu` 0.8, `lmcache-l1` 0.8. Tiers absent
from the list score at `kvCacheDefaultBackendWeight` (default 1.0). Slower
shared tiers such as `lmcache-l2-<backend>` have no default entry; configure
them explicitly, for example:

```yaml
indexerConfig:
  kvCacheDefaultBackendWeight: 0.5
  kvCacheBackendConfigs:
    - name: gpu
      weight: 1.0
    - name: lmcache-l1
      weight: 0.8
    - name: lmcache-l2-fs
      weight: 0.4
```
- **Tracing.** The index and scorer are wrapped with OpenTelemetry
  instrumentation that is a no-op when tracing is not configured.

## Key Types

| Symbol | Role |
|--------|------|
| `Indexer` | Entry point; constructed with `NewKVCacheIndexer(ctx, config, tokenProcessor)`. |
| `ScoreTokens` | Tokens-in scoring: tokens -> block keys -> lookup -> per-pod scores. |
| `ComputeBlockKeysFromTokens` | Tokens -> block keys, without scoring. |
| `KVBlockIndex` | Accessor for the underlying `kvblock.Index`. |
| `KVBlockScorer` / `LongestPrefixScorer` | Scoring strategy over block-hit results. |
| `Config` | Wires the block-index backend, scorer, and per-tier backend weights. |

## Related Documentation

- [KV-Block Index](kvblock/README.md) -- block index backends and token processing
- [KV-Events](../kvevents/README.md) -- keeps the index current from engine events
- [Metrics](metrics/README.md) -- index and event metrics
