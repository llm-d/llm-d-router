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
indexer owns block-key computation, index lookup, and prefix matching.

## How It Works

- **Block-key computation.** `ComputeBlockKeysFromTokens` runs the injected
  [`kvblock.TokenProcessor`](kvblock/README.md) to chunk tokens into
  fixed-size blocks and hash each block (chaining the previous block's hash so a
  key encodes its whole prefix). `extraFeatures` taints the hash with per-block
  multimodal metadata when present.
- **Matching.** `MatchBlockKeys` reads the pods that hold each block key
  from the [`kvblock.Index`](kvblock/README.md), optionally restricted to a
  caller-supplied pod set, and folds them into one `PodMatch` per pod holding
  the first key: the pod's longest run of consecutive block hits starting
  from block 0, its weighted score (per device tier, `BackendConfigs`), and
  the run length per tier. It walks the index in key order when the backend
  is a `kvblock.KeyWalker` and materializes `Lookup` otherwise; both feed the
  one accumulator that holds the matching rules, so every caller and backend
  sees the same semantics. The contiguous-chain hit metrics are recorded
  here, under the index's `enableMetrics` option.
- **Scoring.** `ScoreTokens` reports each pod's weighted score, so a pod that
  holds a longer contiguous prefix ranks higher.
- **Tracing.** Index operations and the matcher emit OpenTelemetry spans,
  no-ops when tracing is not configured. The `match_block_keys` span emits
  `key_count`, `pod_filter_count`, `walked`, `pods_matched`, and `longest_chain`
  under the `llm_d.kv_cache.prefix_match` attribute prefix.

## Key Types

| Symbol | Role |
|--------|------|
| `Indexer` | Entry point; constructed with `NewKVCacheIndexer(ctx, config, tokenProcessor)`. |
| `ScoreTokens` | Tokens-in scoring: tokens -> block keys -> match -> per-pod scores. |
| `MatchBlockKeys` / `PodMatch` | Block keys -> per-pod prefix match: weighted score, matched blocks, blocks per tier. |
| `ComputeBlockKeysFromTokens` | Tokens -> block keys, without matching. |
| `KVBlockIndex` | Accessor for the underlying `kvblock.Index`. |
| `LongestPrefixScorer` | Deprecated `KVBlockScorer`; projects the matcher over a materialized lookup result. |
| `Config` | Wires the block-index backend, scoring strategy, and per-tier backend weights. |

## Related Documentation

- [KV-Block Index](kvblock/README.md) -- block index backends and token processing
- [KV-Events](../kvevents/README.md) -- keeps the index current from engine events
- [Metrics](metrics/README.md) -- index and event metrics
