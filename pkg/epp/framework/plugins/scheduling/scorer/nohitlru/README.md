# No-Hit LRU Scorer

**Type:** `no-hit-lru-scorer`

Scores pods based on least-recently-used (LRU) ordering for cold requests (requests with no KV cache hits). Helps evenly distribute cache growth across pods, since cold requests result in new KV blocks being created.

Integrates with a prefix cache plugin to determine whether a request has cache hits:
- **Cold requests (no cache hits):** Ranks pods by LRU order — never-used or least-recently-used pods score higher (up to `1.0`); most-recently-used pods score lower (approaching `0.0`).
- **Warm requests (cache hits):** Returns neutral scores (`0.5`) for all pods to avoid interfering with cache locality optimization.

LRU tracking is specific to cold requests only — a pod is added to the LRU when it serves a cold request, not when it serves a cached one.

> **Note:** Designed to work alongside a prefix cache scorer (such as `prefix-cache-scorer`). If no prefix cache state is available, all requests are treated as cold. The prefix-cache scorer should be defined first in the scheduling profile.

**Parameters:**
- `prefixPluginType` (string, optional, default: `"prefix-cache-scorer"`): Type of the prefix-cache scorer to read for cache-hit detection.
- `prefixPluginName` (string, optional, default: `"prefix-cache-scorer"`): Name of the prefix-cache scorer to read for cache-hit detection.
- `lruSize` (int, optional, default: `1024`): Maximum number of pods tracked in the LRU.

**Configuration Example:**
```yaml
plugins:
  - type: precise-prefix-cache-producer

  - type: prefix-cache-scorer
    name: cache-scorer
    parameters:
      prefixMatchInfoProducerName: precise-prefix-cache-producer

  - type: no-hit-lru-scorer
    name: lru-scorer
    parameters:
      prefixPluginType: "prefix-cache-scorer"
      prefixPluginName: "cache-scorer"
      lruSize: 1024

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: cache-scorer
        weight: 10
      - pluginRef: lru-scorer
        weight: 5
```

---

## Related Documentation
- [Precise Prefix Cache Scorer](../preciseprefixcache/)
- [Prefix Cache Scorer](../prefix/)
