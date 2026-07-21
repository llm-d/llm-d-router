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
