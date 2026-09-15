# Session Prefix Cache Producer

**Type:** `session-prefix-cache-producer`

DataProducer that resolves a session producer's engine-block prefixes against
block residency observed from vLLM KV events, and publishes per-endpoint
`PrefixCacheMatchInfo`. It serves requests whose tokens are estimated rather
than rendered: the prefixes are the engine's own block hashes, learned from
events, so neither the lookup path nor the ingestion path hashes tokens.
Pairs with the generic [`prefix-cache-scorer`](../../../scheduling/scorer/prefix/),
which must reference this producer by name.

Pipeline per request:
- Consume `SessionCacheRequest` from the configured session producer.
- Resolve each candidate prefix against every observed cache (endpoint,
  data-parallel rank, KV-cache group) of the candidate endpoints and keep the
  best match per endpoint.
- Write `PrefixCacheMatchInfo` per endpoint with unit-size blocks, so match and
  coverage are token counts whatever block size the engine uses.
- (`PreRequest`) Write the request's session identity into the body and set
  the engine's report mode.
- (`EndpointExtractor`) Per-pod ZMQ subscriber lifecycle on add/delete.

A request without a `SessionCacheRequest` attribute, or with an empty
`SessionID`, gets no match data and no body change.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sessionCacheRequestProducerName` | string | required | Instance name of the plugin that publishes `SessionCacheRequest`. |
| `indexConfig` | object | `kvblock.DefaultIndexConfig()` | In-memory residency index. `redisConfig` is rejected. |
| `kvEventsConfig` | object | `kvevents.DefaultConfig()` | KV-events pool config. Per-endpoint pod discovery of vLLM engines is required; a global `zmqEndpoint` and `engineType: sglang` are rejected. |

```yaml
- type: session-prefix-cache-producer
  name: session-cache
  parameters:
    sessionCacheRequestProducerName: sessions
    kvEventsConfig:
      podDiscoveryConfig:
        socketPort: 5557
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: session-cache
```

The `sessions` plugin is a session producer configured by the integration.
The repository ships no session producer; configuration loading fails when the
named plugin is absent.

## Session producer contract

The session producer owns session identity, request-to-session association,
continuation matching, and retention. It publishes
[`SessionCacheRequest`](../../../datalayer/attribute/session/data_types.go)
under its instance name. This producer declares that attribute as a required
dependency, so the session producer runs first. The attribute is the only
interface between the two plugins.

`SessionCacheRequest` carries:

- `SessionID`: the identity written to the request body's `session_id`. vLLM
  echoes it in the KV events for the blocks the request stores. A session
  producer that must tell concurrent requests of one logical session apart
  uses a per-request value, for example the request ID, and keeps its own
  binding to the logical session.
- `FullReport`: whether the engine should report reused blocks as well as
  newly stored ones.
- `TotalTokens`: the full prompt length in engine-token units, measured or
  estimated. It bounds coverage and is the prompt length the inflight-load
  producer accounts for this request.
- `Prefixes`: candidate prompt prefixes as ordered engine block hashes with
  their block size. Each is resolved on its own; candidates are never
  concatenated. `Exact` marks a prefix the prompt is known to begin with.

A session producer learns block hashes from the engine's KV events. It can
subscribe to them on its own with `kvevents.NewConsumerPool`, keeping one
subscriber per discovered engine through `kvevents.EndpointSubscriptions`; the
pool delivers decoded batches with the reporting endpoint, data-parallel rank,
cache group, and the `session_id` each stored block was reported under. The hashes reported for a session's request, cut at
the prompt boundary from the response usage, are that session's next candidate
prefix. The session producer scopes candidates by tenant, model, and cache
salt, and bounds its own storage.

## Residency

Residency is recorded per physical cache: endpoint, data-parallel rank, and
KV-cache group. A block hash reported by two caches has two entries. A prefix
learned on one worker matches blocks reported by another worker, including
reports that carry no session identity.

Only local GPU blocks of full-attention or MLA cache groups establish
residency; an engine that reports no group is treated as one full-attention
cache. Offloaded, remote, sliding-window, and lower-tier reports are ignored.
A GPU removal event drops the reported hashes in that cache. The engine does
not distinguish a new allocation from a reused-block report, so store reports
are idempotent and a removal marks the hash unavailable in that cache until
the next store report.

`AllBlocksCleared`, a stream gap without replay, a failed replay, a reconnect,
subscriber attachment, and endpoint removal each reset the affected endpoint's
residency. The subscriber then replays from the start of the engine's retained
history, and a history that no longer reaches the requested sequence rebuilds
residency from the retained suffix.

The index is in memory and private to the producer instance. Each EPP replica
subscribes to the engines and builds its own residency; one replica's reset
does not affect another. Residency does not persist across EPP restarts.
Cross-replica sharing of session associations, where needed, belongs to the
session producer.

## Match semantics

A request is treated as one prompt. For each candidate prefix and each
observed cache of a candidate endpoint that reports the prefix's block size,
the producer counts the leading hashes resident in that cache. The best result
per endpoint, by matched tokens and then by cached tokens, is published.

`MatchBlocks` and `TotalBlocks` are token counts (block size 1). With `Exact`
unset, `CachedBlockCount` and `CachedBlocksByTier` are zero: the scorer
receives an affinity score and prefill-versus-decode decisions see no cached
tokens. With `Exact` set, hashes past the prompt's own length are not queried
and the matched tokens are reported as cached GPU tokens. `TotalTokens` on the
result carries the session producer's prompt length to the inflight-load
producer.

## Request stamping

`PreRequest` replaces the body's `session_id` with `SessionID` and sets
`vllm_xargs.kv_cache_report_mode` to `full` or `incremental`. Other fields of
`vllm_xargs` and the rest of the body are preserved. vLLM resolves the body
field ahead of the session header and `vllm_xargs.session_id`, so a client's
own identity in those inputs no longer reaches the engine; a session producer
that honors client identity reads it before publishing `SessionID`. The
resolution order is defined in
[vLLM's session-ID resolver](https://github.com/vllm-project/vllm/blob/f4eccdadefc6501fafeb1a0bf7f171ff24f984b0/vllm/entrypoints/generate/base/serving.py#L261).

The engine side of this contract, the `session_id`, `locality`, and
`ownership` fields on KV events and the `kv_cache_report_mode` argument, is
[vllm-project/vllm#51381](https://github.com/vllm-project/vllm/pull/51381).
