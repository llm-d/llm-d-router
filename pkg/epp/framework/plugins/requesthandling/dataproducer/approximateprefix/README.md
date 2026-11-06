# Approximate Prefix Cache Producer Plugin

**Type:** `approx-prefix-cache-producer`

Prepares per-endpoint prefix cache match data consumed by the `prefix-cache-affinity-filter`, `prefix-cache-scorer`, and `disaggregatedset-rollout-screener` (with `revisionGating.affinity.prefixCache`).

**When it runs.** By default the producer runs after screeners, on the screened endpoints, before admission and scheduling. When a screener consumes `PrefixCacheMatchInfo`, or the producer is listed under `requestHandler.dataProducers`, it runs during request handling instead, on every located endpoint before screeners, and its `token-producer` moves with it. It runs once per request either way: in request handling, filters and scorers read the match info it wrote there. Running during request handling matches more endpoints per request, so per-endpoint work grows with the candidates the screeners remove. See [Request Control](../../../../../../../docs/architecture.md#request-control).

For each request, the plugin consumes `request.Body.TokenizedPrompt` (token IDs), hashes the token IDs into fixed-size blocks, and looks up predicted prefix matches for each endpoint. It writes a `PrefixCacheMatchInfo` attribute onto each candidate endpoint, then records the selected endpoint(s) in the index after scheduling completes (via `PreRequest`).

Conditional decode requests (`Prefer: if-available`) do not update the predicted cache state. This prevents rejected requests from creating false cache hits for later requests.

`TokenizedPrompt` is produced by a `token-producer`. When none is configured, the framework auto-creates one with the tokenizer-free `estimate` backend, so prefix caching works without extra setup; configure a `token-producer` explicitly to select the vLLM `/render` backend.

**Parameters:**

- `autoTune` (bool, optional, default: `true`): Infer `blockSizeTokens` and `lruCapacityPerServer` from endpoint metrics when available.
- `blockSizeTokens` (int, optional, default: `64`): Prefix block size in tokens. Used when `autoTune` is false or endpoint metrics are unavailable; ignored when auto-tuned from metrics. Values below the minimum of `64` are clamped up at request time (see #1158).
- `maxPrefixBlocksToMatch` (int, optional, default: `2048`): Maximum number of prefix blocks hashed and matched per request. Not auto-tuned. Applies only when `maxPrefixTokensToMatch` is `0`.
- `maxPrefixTokensToMatch` (int, optional, default: `131072`): Cap expressed in tokens instead of blocks (`maxBlocks = maxPrefixTokensToMatch / blockSizeTokens`). Not auto-tuned. Takes precedence over `maxPrefixBlocksToMatch` when set (> 0); set to `0` to fall back to the block-based cap. The `131072` default (128K, the context window of large production models such as gpt-oss 120b) is a reasonable upper bound that covers the long-prompt use cases seen in production.

Setting both caps to `0` matches the whole prompt. Prompt length is bounded by the model server's context window, so this caps matching at one context window without needing to name a token count. It raises EPP CPU on long prompts (see [operations](../../../../../../../docs/operations.md)); to turn prefix matching off, remove this producer from the configuration rather than zeroing the caps.
- `lruCapacityPerServer` (int, optional, default: `31250`): Per-pod LRU index capacity. Used when `autoTune` is false or endpoint metrics are unavailable; ignored when auto-tuned from metrics. With `autoTune`, a pod first indexed before its metrics are scraped starts at this capacity and is resized to the metrics-derived capacity once metrics arrive; a later scrape that lacks the metrics does not shrink it back.
- `syncCrossReplicaState` (bool, optional, default: `true`): When a `CrossReplicaSyncer` (for example `redis-syncer`) is configured, publish a capped recent window of per-prompt prefix block hashes per decode endpoint and merge peer-only prompts into the local indexer before matching. Local recent-window prompts are not re-applied on every request. This shares recent placements across active-active EPP replicas; it does not snapshot the full local LRU and is not a restart restore. Set `false` to keep the producer local-only. See also [operations](../../../../../../../docs/operations.md#scaling-modes-active-active-vs-active-passive).
- `maxSyncedTokens` (int, optional, default: `2000000`): Cap on total tokens kept in each replica's per-endpoint recent window published through the syncer (`hash budget = maxSyncedTokens / blockSizeTokens`). Default matches the default per-pod index size (`lruCapacityPerServer` × `blockSizeTokens`). Aggregate unions replica windows without re-applying this cap, so N replicas can contribute up to N windows of peer state.
- `excludeDecodeWhenDisaggregated` (bool, optional, default: `false`): Record a disaggregated request, one that also picked a prefill endpoint, on the prefill endpoint only. Set it for hybrid Mamba/GDN models served with P/D: their decode endpoints cannot serve a prefix hit from KV received over the P/D transfer, so without it `prefix-based-pd-decider` skips prefill on a repeat and the decode endpoint recomputes the whole prompt. It also removes decode-side prefix affinity for disaggregated requests: the decode profile's prefix scorer no longer prefers the decode endpoint that served the prompt, so a repeat may land on a different decode endpoint. Decode-only requests still record the decode endpoint.
- `blockSize` (int, optional): Deprecated — character-based block size. Use `blockSizeTokens` instead.

**Configuration Examples:**

Standard single instance:
```yaml
plugins:
  - type: approx-prefix-cache-producer
    parameters:
      autoTune: true
      lruCapacityPerServer: 1000
```

Configuring multiple named instances (e.g., for tiered caching with different parameters):
```yaml
plugins:
  - name: gpuPrefixProducer
    type: approx-prefix-cache-producer
    parameters:
      autoTune: false
      blockSizeTokens: 64
  - name: cpuPrefixProducer
    type: approx-prefix-cache-producer
    parameters:
      autoTune: false
      blockSizeTokens: 128
  - name: gpuPrefixScorer
    type: prefix-cache-scorer
    parameters:
      prefixMatchInfoProducerName: gpuPrefixProducer
  - name: cpuPrefixScorer
    type: prefix-cache-scorer
    parameters:
      prefixMatchInfoProducerName: cpuPrefixProducer
```

---

## Related Documentation
- [Prefix Cache Scorer](../../../scheduling/scorer/prefix/README.md)
- [Prefix Cache Affinity Filter](../../../scheduling/filter/prefixcacheaffinity/README.md)
- [DisaggregatedSet Rollout Screener: Prefix Cache Affinity](../../screener/disaggregatedsetrollout/README.md#prefix-cache-affinity)

## Multi-cluster support

`multicluster-approx-prefix-cache-producer` is the cluster-scoped variant. Its indexer keys on endpoint identity, so over cluster-endpoints it tracks (cluster, block) and routes a repeated prefix back to its cluster. It hashes 64-token blocks, so a shared prefix must be at least about 64 tokens to match.
