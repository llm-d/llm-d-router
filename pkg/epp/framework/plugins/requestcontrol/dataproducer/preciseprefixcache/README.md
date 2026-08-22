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
- Write `PrefixCacheMatchInfo(matchBlocks, totalBlocks, blockSizeTokens)` per endpoint, including the unweighted cached-block count and its per-device-tier breakdown.
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
| `fullReportRepair` | object | disabled | Lets the producer ask vLLM for a full report of cached blocks reused by selected requests. |
| `fullReportRepair.fullReportThreshold` | number | `0.80` | Request a full report when confirmed coverage is below this fraction. |
| `fullReportRepair.minMissingBlocks` | integer | `32` | Minimum number of missing blocks required before requesting a full report. |
| `fullReportRepair.cooldown` | duration | `10s` | Minimum interval between full-report requests per endpoint. |

When an endpoint picker (EPP) starts after vLLM, it can miss earlier cache events
and undercount warm prefixes. `fullReportRepair` asks vLLM to report the cached
blocks reused by selected requests so the precise index can recover.

Confirmed coverage is the selected endpoint's contiguous, non-speculative match
divided by the prompt's total blocks. The producer requests a full report when at
least `minMissingBlocks` are missing, coverage is below `fullReportThreshold`,
and no report was requested for the endpoint within `cooldown`; the cooldown
exists because a report only lands after the flagged request completes, so every
request in that window would otherwise re-request the same report. A
missing-parent error forces one report for the next eligible request. The
endpoint stays eligible until its subscriber is removed; one full report repairs
only that request's prefix.

This option requires vLLM pod discovery without replay. Global ZMQ and replay
configurations are rejected; replay re-delivers only the engine's bounded
recent-event buffer, so it cannot recover state that predates the buffer and is
not validated in combination with reports. The producer sets
`vllm_xargs.kv_cache_report_mode: full` on JSON request bodies (proto and raw
bodies are forwarded unchanged); the engine must emit the report as standard KV
events on the endpoint's stream, and an engine that ignores the argument leaves
repair ineffective at one attempt per endpoint per `cooldown`. A disaggregated
decoder receiving the same body also builds a report, whether or not its pods
are subscribed. A report can
re-announce blocks the index already holds, which defers eviction of those rows
to the index's capacity eviction. `kv_cache_full_report_requests_total` counts
requested reports by reason; measure report cost before enabling this option in
production. Use `kvEventsConfig.podDiscoveryConfig.podLabelSelector` to
subscribe only to prefiller pods when the precise index represents prefill
cache state.

Set `kvEventsConfig.engineType` to `sglang` for SGLang KV-events. It defaults
to `vllm` when omitted.

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
