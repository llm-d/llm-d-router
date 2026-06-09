# Precise Prefix Cache Producer

**Type:** `precise-prefix-cache-producer`

DataProducer that owns the precise KV-block index and publishes
per-endpoint `PrefixCacheMatchInfo`. Pairs with the generic
[`prefix-cache-scorer`](../../../scheduling/scorer/prefix/); the scorer
must reference this producer by name:

```yaml
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
```

Without the `prefixMatchInfoProducerName` field, the scorer falls back
to the auto-spawned approx producer.

Pipeline per request:
- Consume `TokenizedPrompt` from `token-producer`.
- Hash tokens → KV-block keys → `kvblock.Index.Lookup`.
- Write `PrefixCacheMatchInfo(matchBlocks, totalBlocks, blockSizeTokens)` per endpoint.
- (`PreRequest`) Speculative-index the selected endpoint(s) with TTL eviction.
- (`EndpointExtractor`) Per-pod ZMQ subscriber lifecycle on add/delete.

Requires `TokenizedPrompt` on the request — set by a `token-producer`
upstream. No-op otherwise.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenProcessorConfig` | object | `kvblock.DefaultTokenProcessorConfig()` | KV-block hashing for the EPP-recomputed keys (block size, hash seed). |
| `indexerConfig` | object | `kvcache.NewDefaultConfig()` | `kvcache.Indexer` config. |
| `kvEventsConfig` | object | `kvevents.DefaultConfig()` | KV-events pool config. |
| `speculativeIndexing` | bool | `false` | Seed predicted entries on routing decisions. |
| `speculativeTTL` | duration | `2s` | TTL for speculative entries. |

See [llm-d-kv-cache/docs/configuration.md](https://github.com/llm-d/llm-d-kv-cache/blob/main/docs/configuration.md)
for nested parameter details.

## Engine compatibility

Block keys are recomputed by the EPP from `TokenizedPrompt` (tokens, model,
multimodal features, cache salt) on both the lookup path and the KV-event
ingestion path, using this plugin's `tokenProcessorConfig`. The engine's own
block hashes serve only as opaque keys for the engine-to-request mapping, so
`blockSize`/`hashSeed` need not match the engine.

The cross-engine requirement is that the engine emits, in its KV-events, the
hash-affecting inputs the EPP hashes: `token_ids`, and `extra_keys` carrying
multimodal identifiers and `cache_salt`. An input the engine omits from
`extra_keys` is absent on the event side, so requests carrying it do not
correlate.

| Engine | `extra_keys` in KV-events | `cache_salt` |
|--------|---------------------------|--------------|
| vLLM | emitted | in block-0 `extra_keys`; salted prefixes isolated and precise-routed |
| SGLang | not emitted | baked into engine block hashes but not surfaced; salted requests are precise-cache misses until SGLang emits `extra_keys` |

Salt isolation is enforced by the engine regardless; the above affects only
routing accuracy for salted requests.
