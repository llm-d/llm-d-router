# Approximate Prefix Cache Producer Plugin

**Type:** `approx-prefix-cache-producer`

Prepares per-endpoint prefix cache match data consumed by the `prefix-cache-affinity-filter` and `prefix-cache-scorer`. Runs in the request handling's `DataProducer` phase before scheduling.

For each request, the plugin consumes `request.Body.TokenizedPrompt` (token IDs), hashes the token IDs into fixed-size blocks, and looks up which endpoints have recently served requests with a matching prefix. It writes a `PrefixCacheMatchInfo` attribute onto each candidate endpoint, then records the selected endpoint(s) in the index after scheduling completes (via `PreRequest`).

`TokenizedPrompt` is produced by a `token-producer`. When none is configured, the framework auto-creates one with the tokenizer-free `estimate` backend, so prefix caching works without extra setup; configure a `token-producer` explicitly to select the vLLM `/render` backend.

**Parameters:**

- `autoTune` (bool, optional, default: `true`): Infer `blockSizeTokens` and `lruCapacityPerServer` from endpoint metrics when available.
- `blockSizeTokens` (int, optional, default: `16`): Prefix block size in tokens. Used when `autoTune` is false or endpoint metrics are unavailable; ignored when auto-tuned from metrics. Values below the minimum of `64` are clamped up at request time (see #1158), so the `16` default is effectively `64` absent a larger metric/configured value.
- `maxPrefixBlocksToMatch` (int, optional, default: `2048`): Maximum number of prefix blocks hashed and matched per request. Not auto-tuned. `0` disables matching (zero blocks hashed).
- `maxPrefixTokensToMatch` (int, optional, default: `131072`): Cap expressed in tokens instead of blocks (`maxBlocks = maxPrefixTokensToMatch / blockSizeTokens`). Not auto-tuned. Takes precedence over `maxPrefixBlocksToMatch` when set (> 0); set to `0` to fall back to the block-based cap. The `131072` default (128K, the context window of large production models such as gpt-oss 120b) is a reasonable upper bound that covers the long-prompt use cases seen in production.
- `lruCapacityPerServer` (int, optional, default: `31250`): Per-pod LRU index capacity. Used when `autoTune` is false or endpoint metrics are unavailable; ignored when auto-tuned from metrics.
- `prefixHashSource` (string, optional, default: `body`): Where block hashes come from. `body` tokenizes and hashes the request body via the `token-producer` (the default path above). `metadata` instead reads a precomputed block-hash chain that an upstream ext_proc (for example the IPP) placed in the request's `llm-d.routing` Envoy dynamic metadata, and skips the `token-producer` dependency; the body is never re-parsed. The upstream stage must hash with the same block-hash scheme, otherwise the hashes will not match any indexed prefix.
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
      blockSizeTokens: 16
  - name: cpuPrefixProducer
    type: approx-prefix-cache-producer
    parameters:
      blockSizeTokens: 64
  - name: gpuPrefixScorer
    type: prefix-cache-scorer
    parameters:
      prefixMatchInfoProducerName: gpuPrefixProducer
  - name: cpuPrefixScorer
    type: prefix-cache-scorer
    parameters:
      prefixMatchInfoProducerName: cpuPrefixProducer
```

Sourcing prefix hashes from request metadata (an upstream ext_proc supplies the chain, no tokenizer needed):
```yaml
plugins:
  - type: approx-prefix-cache-producer
    parameters:
      prefixHashSource: metadata
  - type: prefix-cache-scorer
```

---

## Related Documentation
- [Prefix Cache Scorer](../../../scheduling/scorer/prefix/README.md)
- [Prefix Cache Affinity Filter](../../../scheduling/filter/prefixcacheaffinity/README.md)
