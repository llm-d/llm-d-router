# Session prefix cache producer

`session-prefix-cache-producer` resolves externally supplied engine-block paths
against current GPU residency and publishes `PrefixCacheMatchInfo` for the existing
prefix scorer, affinity filter, and load producer. It performs no rendering or
tokenization. A token producer can independently supply real tokens or estimates.

The external session producer owns identity, request correlation, ordered block
paths, concurrent branches, and confidence. The cache producer owns residency.
Eviction removes residency without deleting the external producer's lineage.

## Request contract

Publish `attribute/session.SessionCacheRequest` on the request under the configured
`sessionCacheRequestProducerName`:

- `SessionID`: the stamp sent in the engine request's `session_id`. Use a unique
  request stamp when concurrent requests share a logical session.
- `FullReport`: request reused blocks as well as newly stored blocks through
  `vllm_xargs.kv_cache_report_mode`.
- `TotalTokens`: full prompt length in engine-token units, measured or estimated.
  Zero means unavailable: stamping remains active, cache matches are omitted, and
  downstream load accounting retains its own estimate.
- `Prefixes`: separate ordered paths, each with a compatible `CacheNamespace`,
  engine `BlockHashes`, `BlockSizeTokens`, and an `Exact` assertion.

`CacheNamespace` must identify compatible model, engine hash algorithm, hash seed,
and cache salt settings across workers. Configure a separate producer and event
source for incompatible caches. Each producer uses its own in-memory index.

Only leading resident blocks contribute to a match. Paths, ranks, and cache groups
are queried separately. Exact paths contribute cached-token coverage; uncertain
paths contribute affinity only. Block count times block size determines covered
tokens, while `TotalTokens` bounds coverage and accounts for the remaining prompt.
The supplied full length can itself be an estimate.

An absent request attribute or empty stamp skips both lookup and body mutation.
An explicit stamp replaces a client-supplied body `session_id`; other body fields
are preserved. The producer does not generate identities or infer continuations.

## Event input

Bind the cache producer to a [`kv-events-source`](../../../datalayer/source/kvevents/README.md).
An external manager implementing `kvevents.EventConsumer` can bind to the same
source to learn stamp-to-block associations from the same decoded batches.
`Reset` invalidates observed residency; the manager can retain lineage while
marking its observations incomplete.

Local GPU full-attention and MLA stores establish residency. CPU, remote,
offloaded, and partial-context cache reports are ignored. Removals, cache clears,
and source resets invalidate residency. Logical paths can match any compatible
worker; endpoint, rank, and cache group identify each physical copy.

## Configuration

Configure the external request producer as `sessions`. If it also implements
`EventConsumer`, add `sessions` to the source's extractor list.

```yaml
plugins:
- type: kv-events-source
  name: engine-events
  parameters:
    podDiscoveryConfig:
      podLabelSelector: app=vllm
      socketPort: 5557
      replaySocketPort: 5558
- type: session-prefix-cache-producer
  name: session-cache
  parameters:
    cacheNamespace: qwen-hash-v1-seed0-unsalted
    sessionCacheRequestProducerName: sessions
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: session-cache
dataLayer:
  sources:
  - pluginRef: engine-events
    extractors:
    - pluginRef: session-cache
```

Point every cache-aware consumer's `prefixMatchInfoProducerName` at `session-cache`.
Otherwise a consumer can select a default token-based cache producer. The session
cache producer is Alpha and requires experimental plugins to be enabled. Engines
must support request `session_id` stamps and KV-event session reporting.
