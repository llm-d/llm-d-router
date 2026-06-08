# Approximate Prefix Cache Producer Plugin

**Type:** `approx-prefix-cache-producer`

Prepares per-endpoint prefix cache match data consumed by the `prefix-cache-affinity-filter` and `prefix-cache-scorer`. Runs in the request handling's `DataProducer` phase before scheduling.

For each request, the plugin consumes `request.Body.TokenizedPrompt` (token IDs), hashes the token IDs into fixed-size blocks, and looks up which endpoints have recently served requests with a matching prefix. It writes a `PrefixCacheMatchInfo` attribute onto each candidate endpoint, then records the selected endpoint(s) in the index after scheduling completes (via `PreRequest`).

`TokenizedPrompt` is produced by a `token-producer`. When none is configured, the framework auto-creates one with the tokenizer-free `estimate` backend, so prefix caching works without extra setup; configure a `token-producer` explicitly to select the vLLM `/render` backend.

**Parameters:**
- `autoTune` (bool, optional, default: `true`): Infer block size and LRU capacity from endpoint metrics when available.
- `blockSizeTokens` (int, optional, default: `0`): Prefix block size in tokens. Used when `autoTune` is false or metrics are unavailable.
- `maxPrefixBlocksToMatch` (int, optional, default: `0`): Maximum number of prefix blocks considered per request. `0` means unlimited.
- `maxPrefixTokensToMatch` (int, optional, default: `0`): Alternative cap expressed in tokens instead of blocks. Takes precedence over `maxPrefixBlocksToMatch` when set.
- `lruCapacityPerServer` (int, optional, default: `0`): Default per-pod LRU index capacity when endpoint metrics are unavailable.
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

---

## Related Documentation
- [Prefix Cache Scorer](../../../scheduling/scorer/prefix/README.md)
- [Prefix Cache Affinity Filter](../../../scheduling/filter/prefixcacheaffinity/README.md)
