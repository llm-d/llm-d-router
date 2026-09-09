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
| `fullReportRepair.minMissingBlocks` | integer | `32` | Minimum missing-block count for the coverage threshold path; integrity faults bypass this floor. |
| `fullReportRepair.cooldown` | duration | `10s` | Minimum interval between full-report requests per endpoint. |
| `fullReportRepair.prefillProfile` | string | `prefill` | Prefill profile whose selected endpoint owns the cache being repaired; match the disaggregation handler's `profiles.prefill`. |

When an endpoint picker (EPP) starts after vLLM, it can miss earlier cache events
and undercount warm prefixes. `fullReportRepair` asks vLLM to report the cached
blocks reused by selected requests so the precise index can recover.

Confirmed coverage is the selected endpoint's contiguous, non-speculative match
divided by the prompt's total blocks. The producer requests a full report when
at least `minMissingBlocks` are missing and coverage is below
`fullReportThreshold`. A missing-parent fault bypasses both conditions for
requests containing at least one complete block. The fault remains armed until
the dropped blocks are indexed, removed in the same cache tier and group, or
cleared by a cache reset. Unrelated reports cannot resolve it. Both paths share
the per-endpoint `cooldown` to bound report frequency.

This option requires vLLM pod discovery without replay. Global ZMQ and replay
configurations are rejected. It also requires vLLM to distinguish newly cached
blocks from reused-block reports using `BlockStored.origin` (`NEW` or `REUSED`),
as proposed in [vLLM #51699](https://github.com/vllm-project/vllm/pull/51699).
Repair stays inactive until the subscriber receives an event with either origin.
Reused-block reports restore residency without incrementing physical reference
counts, so the final removal can evict the block from the index.

The producer sets `vllm_xargs.kv_cache_report_mode: full` on JSON request bodies;
native Generate requests use `sampling_params.vllm_xargs`. Proto and raw bodies
are forwarded unchanged. Anthropic requests require a vLLM version containing
[vLLM #53308](https://github.com/vllm-project/vllm/pull/53308). An engine that
ignores the argument leaves repair ineffective. A disaggregated decoder receiving
the same body also builds a report. `kv_cache_full_report_requests_total` counts
requested reports by reason. Use
`kvEventsConfig.podDiscoveryConfig.podLabelSelector` to subscribe only to
prefiller pods when the precise index represents prefill cache state.

Set `kvEventsConfig.engineType` to `sglang` for SGLang KV-events. It defaults
to `vllm` when omitted.

Set `kvEventsConfig.tracing` to `true` to emit OpenTelemetry spans for the
KV-event pipeline (`events_receive`, `events_process`, `events_decode`). It
defaults to `false`: KV events arrive at many times the inference request rate,
so with a shared head sampler always-on event spans crowd request traces out of
the exported volume. The EPP `--tracing` flag gates tracing as a whole, so this
field has no effect while that is off.

See [llm-d-kv-cache/docs/configuration.md](https://github.com/llm-d/llm-d-kv-cache/blob/main/docs/configuration.md)
for nested parameter details.

## Engine compatibility

Block keys are recomputed by the EPP from `TokenizedPrompt` (tokens, model,
multimodal features, cache salt) on both the lookup path and the KV-event
ingestion path, using this plugin's `tokenProcessorConfig`. The engine's own
block hashes serve only as opaque keys for the engine-to-request mapping, so
`blockSizeTokens`/`hashSeed` need not match the engine.

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
