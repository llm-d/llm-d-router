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
| `kvEventsSourceRef` | string | _(none)_ | Name of a `kv-events-source` whose decoded batches this producer indexes. The producer then opens no subscribers of its own, and `kvEventsConfig` is rejected. |
| `speculativeIndexing` | bool | `false` | Seed predicted entries on routing decisions. |
| `speculativeTTL` | duration | `2s` | TTL for speculative entries. |
| `seedDecodeWhenDisaggregated` | bool | `false` | Also seed a disaggregated request's speculative entries on its decode endpoint. By default only the prefill endpoint is seeded: the decode endpoint holds no KV until the P/D transfer lands, so a same-prefix request inside the TTL would see a full hit there, `prefix-based-pd-decider` would skip prefill, and the decode endpoint would recompute the whole prompt. Decode-side affinity for those requests comes from confirmed KV events. |

Set `kvEventsConfig.engineType` to `sglang` for SGLang KV-events. It defaults
to `vllm` when omitted.

The producer subscribes to each discovered endpoint's event socket and
attributes events to the endpoint it dialed. `kvEventsConfig.zmqEndpoint` is a
deprecated alternative that binds one socket every engine publishes to. A
bound socket cannot tell publishers apart, so it takes the pod identity from
the topic, and any peer that reaches the socket can publish events under any
pod's identity. The EPP refuses to bind a `zmqEndpoint` on a non-loopback
address unless `kvEventsConfig.allowUntrustedPublishers` is `true`. Set it only
when a network policy limits who can reach the port, and prefer per-pod
subscriptions.

SGLang EAGLE/MTP events describe cached positions as overlapping token pairs.
The index preserves their trailing lookahead and learns each pod's page size.
Request matching hashes complete bigram pages using the same format.
A page of `P` cached positions requires `P+1` matching request tokens.
Bigram endpoints report `PrefixCacheMatchInfo` in token-sized blocks.
This preserves cache-tier weights and exact cached-token counts across page sizes.
Plain-token workers use `tokenProcessorConfig.blockSizeTokens`.

Set `kvEventsConfig.tracing` to `true` to emit OpenTelemetry spans for the
KV-event pipeline (`events_receive`, `events_process`, `events_decode`). It
defaults to `false`: KV events arrive at many times the inference request rate,
so with a shared head sampler always-on event spans crowd request traces out of
the exported volume. The EPP `--tracing` flag gates tracing as a whole, so this
field has no effect while that is off.

Set `kvEventsConfig.podDiscoveryConfig.snapshotSocketPort` to the port where
vLLM serves KV-event snapshots (offset by data-parallel rank like `socketPort`)
to warm a pod's index from its current cache state. This is experimental: it
requires a vLLM build with
[vllm-project/vllm#57789](https://github.com/vllm-project/vllm/pull/57789),
whose wire format can still change.

A subscriber that joins mid-stream, or sees the engine restart, loads the
snapshot instead of replaying buffered events, then follows live events after
the snapshot's sequence number. A restart shows as a sequence that moves
backwards or as a new publisher identity, which vLLM sends with every batch
and snapshot. Without `replaySocketPort`, a gap in the event sequence reloads
the snapshot as well. While the snapshot fails, is unavailable or comes from
another publisher identity, the pod is indexed from live events alone, as it
is with neither port set, and the load is retried after the replay cooldown.
A snapshot that loads replaces that state.

`kvEventsConfig.podDiscoveryConfig.maxConcurrentReplays` (default 8) bounds
the snapshot loads and replays in flight across all pods. Pods beyond it wait
for a slot, and a snapshot's timeout starts once it holds one.

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

### Shared event input

Set `kvEventsSourceRef` to the name of a
[`kv-events-source`](../../../datalayer/source/kvevents/README.md) to index the
decoded batches it shares with other consumers. The source owns the subscriptions
and their recovery. Startup fails when the name matches no plugin, when the named
plugin is of another type, or when `kvEventsConfig` is set alongside the reference.
