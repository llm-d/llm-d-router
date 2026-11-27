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
- (`PreRequest`) Hint the request's block keys as held by the selected endpoint(s) for `speculativeTTL`.
- (`EndpointExtractor`) Per-pod ZMQ subscriber lifecycle on add/delete.

Requires `TokenizedPrompt` on the request from a `token-producer` that emits exact
engine tokens (`exact_tokens: "true"`). Configure `backend: vllm` for text pools or
`backend: passthrough` for pre-tokenized pools. EPP rejects `backend: estimate` at
startup so byte-packed pseudo-tokens never enter the KV-block index. A request with
no `TokenizedPrompt` is a no-op.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenProcessorConfig` | object | `kvblock.DefaultTokenProcessorConfig()` | KV-block hashing for the EPP-recomputed keys (block size, hash seed). |
| `indexerConfig` | object | `kvcache.NewDefaultConfig()` | `kvcache.Indexer` config. |
| `kvEventsConfig` | object | `kvevents.DefaultConfig()` | KV-events pool config. |
| `kvEventsSourceRef` | string | _(none)_ | Name of a `kv-events-source` whose decoded batches this producer indexes. The producer then opens no subscribers of its own, and `kvEventsConfig` is rejected. |
| `speculativeIndexing` | bool | `false` | Hint the selected endpoint(s) as holding the request's blocks on routing decisions, so a same-prefix request matches them before the engine's KV-events arrive. A hint scores as the `speculative` tier and covers only whole engine blocks of the block size the endpoint reports in its metrics. An endpoint that reports none gets every router block. Hints are held in EPP memory next to the KV-block index, so they are not shared between EPP replicas. |
| `speculativeTTL` | duration | `2s` | Lifetime of a hint. A request routed to the same endpoint with the same blocks renews it. |
| `seedDecodeWhenDisaggregated` | bool | `false` | Also hint a disaggregated request's decode endpoint. By default only the prefill endpoint is hinted: the decode endpoint holds no KV until the P/D transfer lands, so a same-prefix request inside the TTL would see a full hit there, `prefix-based-pd-decider` would skip prefill, and the decode endpoint would recompute the whole prompt. Decode-side affinity for those requests comes from confirmed KV events. |

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

When a pod serves several data-parallel ranks behind its port (see
[Data-parallel ranks behind one port](../../../../../../../docs/data-parallel-ranks.md)), each rank of the pod
is its own endpoint on the pod's address and port. With pod discovery, each
rank's events reach the index under `<address>:<port>/rank-<RankIndex>`, so the
ranks of one pod are scored separately. Without pod discovery, the index learns
a pod's identity from the event topic, and all ranks of a pod share one score.

A rank endpoint's subscriber dials `socketPort` plus the endpoint's
data-parallel rank, as vLLM offsets each engine's publisher, replay and snapshot
ports.

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

### Cache tiers

Each KV event names the medium that holds its blocks. The index stores the
medium lowercased as the block's device tier, and
`indexerConfig.kvCacheBackendConfigs` sets each tier's scoring weight. A tier
with no configured weight scores 0. A configured list replaces the defaults in
`kvcache.DefaultKVCacheBackendConfig()`.

SGLang publishes these media:

| SGLang medium | Device tier | Blocks |
|---------------|-------------|--------|
| `GPU` | `gpu` | device memory |
| `CPU_PINNED` | `cpu_pinned` | HiCache host memory |
| `EXTERNAL` | `external` | HiCache storage, for every `--hicache-storage-backend`; requires an SGLang build with [sgl-project/sglang#50154](https://github.com/sgl-project/sglang/pull/50154) |
| `DISK` | `disk` | defined by SGLang but not published |

A `kvCacheBackendConfigs` for SGLang pods lists `cpu_pinned` to score
host-memory blocks; a `cpu` entry does not match them.

### Shared event input

Set `kvEventsSourceRef` to the name of a
[`kv-events-source`](../../../datalayer/source/kvevents/README.md) to index the
decoded batches it shares with other consumers. The source owns the subscriptions
and their recovery. Startup fails when the name matches no plugin, when the named
plugin is of another type, or when `kvEventsConfig` is set alongside the reference.
