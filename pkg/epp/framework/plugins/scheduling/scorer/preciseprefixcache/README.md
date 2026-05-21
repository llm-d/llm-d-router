# Precise Prefix Cache Scorer

**Type:** `precise-prefix-cache-scorer`

The `precise-prefix-cache-scorer` scores a request based on KV-cache localities.
Similarly to the `prefix-cache-scorer`, it provides a score based on the number of
 matching KV-cache blocks between the request's prompt and the KV-cache contents of each pod.
 However, unlike the `prefix-cache-scorer`, which relies on estimations based on scheduling history,
 the `precise-prefix-cache-scorer` tracks the real-time KV-cache states across the vLLM instances to
 provide more accurate scoring.

When enabled, the scorer will use the [`llm-d-kv-cache`](https://github.com/llm-d/llm-d-kv-cache.git) to track the KV-cache states
 across the vLLM instances. It will use the `kvcache.Indexer` to score the pods based on the
 number of matching blocks in the KV-cache. It will also use the `kvevents.Pool` to subscribe
 to the KV-Events emitted by the vLLM instances and update the KV-cache states in near-real-time.

**Parameters:**
- `tokenProcessorConfig`: Configuration for the `kvblock.TokenProcessor`.
- `indexerConfig`: Configuration for the `kvcache.Indexer`.
- `kvEventsConfig`: Configuration for the `kvevents.Pool`.

See the full parameter reference at [llm-d-kv-cache/docs/configuration.md](https://github.com/llm-d/llm-d-kv-cache/blob/main/docs/configuration.md).

Tokens are sourced from the `token-producer` DataProducer plugin
(`request.Body.TokenizedPrompt`).

**Example configuration with the above parameters set**
```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: hf-repo/model-name
      vllm:
        http: http://localhost:8000
  - type: precise-prefix-cache-scorer
    parameters:
      tokenProcessorConfig:
        blockSize: 64
      indexerConfig:
        kvBlockIndexConfig:
          enableMetrics: true
```

**Example configuration for automatic pod discovery in active-active multi-replica scheduler deployments**
```yaml
plugins:
  - type: precise-prefix-cache-scorer
    parameters:
      tokenProcessorConfig:
        blockSize: 64
      indexerConfig:
        kvBlockIndexConfig:
          enableMetrics: true
      kvEventsConfig:
        topicFilter: "kv@"
        concurrency: 4
        discoverPods: true   # enables automatic pod discovery for active-active HA
        podDiscoveryConfig:
          socketPort: 5556
```

vLLM engines configured to emit KV-events on port `5556`:
```yaml
--kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:5556\",\"topic\":\"kv@${POD_IP}@Qwen/Qwen3-32B\"}"
```

**Configuration Example — all parameters:**
```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: hf-repo/model-name
      vllm:
        http: http://localhost:8000
  - type: precise-prefix-cache-scorer
    parameters:
      tokenProcessorConfig:
        blockSize: 16
        hashSeed: "12345"
      kvEventsConfig:
        topicFilter: "kv@"
        concurrency: 4
        discoverPods: true   # enables automatic pod discovery for active-active HA
        podDiscoveryConfig:
          socketPort: 5556
      indexerConfig:
        kvBlockIndexConfig:
          inMemoryConfig:
            size: 100000000
            podCacheSize: 10
          enableMetrics: true
```

##### Pod discovery via the data layer (recommended)

When `discoverPods: true`, the scorer needs to know when pods come and go so
it can install (and tear down) per-pod ZMQ subscribers. Two mechanisms are
supported:

- **Legacy (default, backwards-compatible).** The scorer opportunistically
  installs subscribers for every endpoint it sees during scoring. No extra
  YAML required. Subscribers for pods that disappear are not actively
  removed — this is the historical behavior.
- **Data layer endpoint-notification-source (recommended).** The scorer
  implements `EndpointExtractor` and reacts to add/delete events from the
  data layer. This gives clean subscriber teardown when pods leave the pool
  and avoids opportunistic subscribe-on-Score traffic.

The two paths are mutually exclusive at runtime: the first time the scorer
receives an `ExtractEndpoint` call from the data layer, the legacy in-Score
path turns itself off, leaving the data layer as the sole authority over
per-pod subscriber lifecycle. No config flag is required.

To enable the data layer path, declare the source plugin and wire it under
`dataLayer.sources`:

```yaml
plugins:
  - type: endpoint-notification-source
  - type: metrics-data-source
  - type: core-metrics-extractor
  - type: precise-prefix-cache-scorer
    # ...same parameters as above
dataLayer:
  sources:
    - pluginRef: metrics-data-source
      extractors:
        - pluginRef: core-metrics-extractor
    - pluginRef: endpoint-notification-source
      extractors:
        - pluginRef: precise-prefix-cache-scorer
```

The same scorer instance serves both roles (Scorer and EndpointExtractor),
no second factory is needed.

##### Speculative Indexing

Speculative indexing closes the gap between a routing decision and KV event arrival by immediately writing a predicted cache entry to the prefix-cache index. This lets the next request with the same prefix hit the cache without waiting for engine confirmation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `speculativeIndexing` | `bool` | No | `false` | Enable speculative index entries on routing decisions. |
| `speculativeTTL` | `string` | No | `"2s"` | TTL for speculative entries. Accepts Go duration strings (e.g. `"500ms"`, `"2s"`). Only used when `speculativeIndexing` is `true`. |

```yaml
plugins:
  - type: precise-prefix-cache-scorer
    parameters:
      speculativeIndexing: true
      speculativeTTL: "2s"
```