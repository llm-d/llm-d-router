# KV-Block Index

The block-level store that maps KV-block keys to the pods holding them, plus the
token-to-block-key processing that feeds it. This is the shared state the
[`kvcache.Indexer`](../README.md) reads and the [`kvevents`](../../kvevents/README.md)
subscriber writes.

## What It Does

- **Token processing.** `TokenProcessor` (backed by `ChunkedTokenDatabase`)
  turns a token sequence into KV-block keys: it chunks tokens into fixed-size
  blocks and hashes each block, chaining the previous block's hash so a key
  encodes its full prefix. `ComputeBlockExtraFeatures` folds multimodal
  metadata into per-block `BlockExtraFeatures` that taint the hash. The first
  block's features can also carry the request's cache salt.
- **Block index.** `Index` maps each block key to the set of pods (`PodEntry`,
  carrying a pod identifier and device tier) that currently hold it. Producers
  `Add` and `Evict` entries; the indexer's `Lookup` reads them.
- **Ordered walk.** `KeyWalker` is an optional capability for callers that
  fold entries in key order: it visits the requested keys in sequence,
  reporting misses, and lends each key's entries as `WalkEntry` values that
  carry the index's pod and tier ordinals (one per pod and tier pair, group
  duplicates folded). Entries are borrowed for the visit call only. The
  in-memory and dense backends have the capability and the decorators keep
  it; the [`kvcache`](../README.md) prefix matcher walks when it can and
  falls back to `Lookup`.
- **Residency stats.** `StatsProvider` is an optional capability a memory
  backend can expose: resident keys, estimated resident bytes, engine
  mappings, group rows, and truncated entries. The dense backend implements
  it; decorators do not forward it.

## Backends

`NewIndex` selects a backend from `IndexConfig` (first configured wins):

| Backend | Constructor | Use |
|---------|-------------|-----|
| In-memory | `NewInMemoryIndex` | Single-replica, LRU-bounded local index. |
| Dense memory | `NewDenseMemoryIndex` | Single-replica local index with a byte budget: packed 32-bit entry words, O(1) `Clear` through a dead-pod bitmap, and sampled eviction under the budget. Holds at most 15 pods per key (configurable down) and 8 device tiers. |
| Cost-aware memory | `NewCostAwareMemoryIndex` | Local index with cost-aware eviction. |
| Redis | `NewRedisIndex` | Shared index across EPP replicas (Redis-protocol server). |

Two decorators wrap any backend: `NewTracedIndex` adds OpenTelemetry spans
(no-op when tracing is off) and `NewInstrumentedIndex` records the
[metrics](../metrics/README.md).

## Grouped (HMA) Entries

`GroupCatalog` tracks group identity for heterogeneous-memory-aware (HMA)
models, where a block may be announced by a group of engines; eviction is
reference-counted at the group level so a block is removed only once no group
still references it.

## Key Types

| Symbol | Role |
|--------|------|
| `Index` | Block-key -> pods mapping (`Add` / `Lookup` / `Evict`). |
| `KeyWalker` / `WalkEntry` | Optional ordered walk over requested keys; borrowed entries with pod and tier ordinals. |
| `OrdinalResolver` | Name-to-ordinal assignment for pod identifiers and device tiers; every `KeyWalker` also implements it. |
| `StatsProvider` / `MemoryIndexStats` | Optional residency snapshot of a memory backend. |
| `TokenProcessor` | Tokens -> block keys; exposes `BlockSize`. |
| `BlockHash` | A single block key. |
| `PodEntry` | A pod holding a block, with its device tier. |
| `BlockExtraFeatures` / `ComputeBlockExtraFeatures` | Per-block multimodal metadata and cache salt folded into the hash. |
| `PlaceholderRange` | Placeholder-token range for a multimodal item. |

## Related Documentation

- [KV-Cache Indexer](../README.md) -- reads this index to score pods
- [KV-Events](../../kvevents/README.md) -- writes to this index from engine events
