# Core Metrics Extractor

**Type:** `core-metrics-extractor`

> [!NOTE]
> This plugin is enabled by default together with `metrics-data-source`. You do not need to explicitly declare it in your configuration, but it can be disabled if metrics collection is unnecessary.

The Core Metrics Extractor is a data layer plugin responsible for extracting model server metrics from a data source and storing them as endpoint attributes. It supports multiple inference engines and can be configured to map engine-specific metric names to a standard set of internal keys.

## What it does

1.  Receives a `PrometheusMetricMap` from a metrics data source (e.g., `metrics-data-source`).
2.  Identifies the inference engine type of the endpoint (e.g., vLLM, SGLang, Triton) using a Pod label.
3.  Looks up the metric specifications for that engine.
4.  Extracts values for standard metrics:
    -   **Waiting Queue Size**: Number of requests waiting in the engine's queue.
    -   **Running Requests Size**: Number of requests currently being processed.
    -   **KV Cache Usage**: Percentage of KV cache currently utilized.
    -   **LoRA Adapters**: Information about active and waiting LoRA adapters.
    -   **Cache Configuration**: Block size and total number of GPU blocks.
    -   **Tiered Offloading** (optional): vLLM KV cache tiering metrics (`kv_offload_tiering_*`, vLLM 0.28.0 or later) when vLLM is configured with `TieringOffloadingSpec`. Skipped without an extraction error on non-tiering deployments.
5.  Stores these values as attributes on the endpoint, making them available to scheduling plugins.

### Families with several series

A model server that runs several engines in one Pod exposes one series per engine, distinguished by a label such as `engine`. vLLM data parallelism with internal load balancing is one example. The pod-level attributes cover every series that matches the metric spec:

-   `WaitingQueueSize` and `RunningRequestsSize` are the sum over the matching series, so both count requests for the whole Pod. The utilization detector's scrape-lag credit compares their sum with the Pod's in-flight request count, and queue-depth thresholds apply to the Pod total.
-   `KVCacheUsagePercent` is the maximum over the matching series: the engine closest to its limit describes the Pod, and the value stays a fraction.
-   The tiered offloading attributes are the sum over the matching series, so they cover every engine rank of the Pod. Series of the primary tier (`tier="0:primary"`) are never included.

A family with a single matching series yields that series' value. A NaN or infinite value on any matching series fails the extraction of that attribute, which keeps its previous value. LoRA, cache configuration, and `customMetrics` entries without `aggregation` read one series. A `customMetrics` entry with `aggregation: sum` adds every matching series.

## Attributes produced

The plugin populates several standard fields on the endpoint's `Metrics` struct:

-   `WaitingQueueSize` (int)
-   `RunningRequestsSize` (int)
-   `KVCacheUsagePercent` (float64)
-   `MaxActiveModels` (int)
-   `ActiveModels` (map[string]int)
-   `WaitingModels` (map[string]int)
-   `UpdateTime` (time.Time)

The three scalar load and utilization metrics (`WaitingQueueSize`, `RunningRequestsSize`, and `KVCacheUsagePercent`) are also published to the endpoint's attribute map (`ep.GetAttributes()`) as `ScalarMetricValue` under `ScalarMetricDataKey`, so `endpoint-attribute-scorer` and `endpoint-attribute-filter` can consume them directly without duplicating them in `customMetrics`. When a `customMetrics` entry uses one of those built-in keys as its `attributeKey`, that entry exclusively owns the attribute on `ep.GetAttributes()` while the built-in spec continues to populate `Metrics`.

The built-in `vllm` config also stores the NIXL KV transfer failure counters as scalar attributes. vLLM reports them only when it runs with the NixlConnector, and an endpoint that has never reported them carries none of these attributes.

| Attribute | vLLM metric | Counted on |
|---|---|---|
| `NixlFailedTransfers` | `vllm:nixl_num_failed_transfers_total` | The endpoint reading the KV cache (decode). |
| `NixlFailedNotifications` | `vllm:nixl_num_failed_notifications_total` | The endpoint reading the KV cache (decode). |
| `NixlKVExpiredRequests` | `vllm:nixl_num_kv_expired_reqs_total` | The endpoint that produced the KV cache (prefill). |

The EPP exposes the same values per endpoint, see [Inference pool metrics](../../../../../../../docs/metrics.md#inference-pool).

When tiered offloading metrics are detected, the following scalar metric attributes are also produced (stored via `ScalarMetricDataKey`):

-   `tiering_chunk_hits` (float64)
-   `tiering_chunk_queries` (float64)
-   `tiering_read_bytes` (float64)
-   `tiering_read_time` (float64)
-   `tiering_promotion_failures` (float64)
-   `tiering_allocation_failures` (float64)

vLLM 0.28.0 or later reports these counters, and only when it runs with `TieringOffloadingSpec`. vLLM creates each series on its first event, so an endpoint that offloads can still lack some of them, for example the failure counters until the first failure. An attribute without a series stays unset, or keeps its last value if its series was reported before. A plugin sees an unset attribute as missing: `endpoint-attribute-scorer` scores the endpoint `0.0`, and `endpoint-attribute-filter` keeps or drops it according to `onMissing`.

No built-in plugin reads these attributes by default. `endpoint-attribute-scorer` reads them through `attributeKey`; `endpoint-attribute-filter` reads them through `attribute`. The values are cumulative counters.

## Configuration

The plugin config supports:

-   `engineLabelKey`: The Pod label key used to identify the engine type. Defaults to `llm-d.ai/engine-type`.
-   `defaultEngine`: The engine type to use if the label is missing. Defaults to `vllm`.
-   `engineConfigs`: A list of engine-specific metric specifications.
    Each engineConfig can include `customMetrics` entries.
    Each entry maps a scalar metric selector to an endpoint attribute key.
    A source metric missing from the scrape is an extraction error unless its
    `customMetrics` entry sets `optional: true`, which skips that entry for
    the scrape.
    A selector supports exact label matching (`label="value"`) and excluding one label value (`label!="value"`).
    Without `aggregation`, the extractor reads the latest matching series.
    With `aggregation: sum`, the extractor adds the values of every Prometheus series that matches metricSpec.
    An engineConfig named after a built-in engine replaces that built-in config, including its `customMetrics`.
    Only the built-in vLLM config includes the tiering counters. Other built-in engine configs do not extract them. An engine that exposes equivalent metrics can configure them with `customMetrics` entries.

### Built-in Engine Configurations

The plugin comes with built-in support for the following engines:
-   `vllm`
-   `sglang`
-   `atom`
-   `trtllm-serve`
-   `triton-tensorrt-llm`

To correctly establish the mapping, model server Pods should be labeled using the `engineLabelKey` with the engine type as follows:

```yaml
metadata:
  labels:
    llm-d.ai/engine-type: vllm # other options: sglang, atom, trtllm-serve, triton-tensorrt-llm, triton 

```

### Custom Engine Configuration Example

```yaml
type: core-metrics-extractor
parameters:
  engineConfigs:
    - name: "my-custom-engine"
      queuedRequestsSpec: "custom_queue_size{status=waiting}"
      runningRequestsSpec: "custom_running_size"
      kvUsageSpec: "custom_cache_utilization"
      customMetrics:
        - attributeKey: "custom.queue_depth"
          metricSpec: "custom_queue_depth{tier=gold}"
```

and the model server deployment Pods should have the label:

```yaml
metadata:
  labels:
    llm-d.ai/engine-type: my-custom-engine

```

### vLLM Tiered Offloading Configuration

The built-in vLLM selectors for the five tier-labeled counters exclude `tier="0:primary"` and use `aggregation: sum`.
Each attribute is the sum over all secondary tiers and engine ranks.
The allocation-failure counter has no `tier` label and also uses `aggregation: sum`.
Deployments that need these totals need no override.

To read the tiers separately, define a vLLM entry in `engineConfigs` with one `customMetrics` item for each tier,
and name the tier in each selector. To read the primary tier's lookup counters, use a `customMetrics` selector that names
`tier="0:primary"`.

vLLM does not label `vllm:kv_offload_tiering_promotion_allocation_failures_total`
with a tier, so its selector must not name one. A selector that names a tier
never matches it, and `tiering_allocation_failures` stays unset.

vLLM builds the label as `{position}:{type}`, where `position` is the entry's
index in `secondary_tiers` plus one and `type` is that entry's `type` string.
The types built into vLLM 0.28.0 are `fs`, `p2p`, `obj` (S3-compatible stores)
and `example`; later vLLM versions also build in `kvcr`. An out-of-tree tier
reports the `type` it declares. Ordering decides the number, so
`secondary_tiers: [p2p, fs]` reports `1:p2p` and `2:fs`, and the specs here must
mirror the deployed order.

> [!NOTE]
> Overriding `engineConfigs` for `"vllm"` replaces the entire built-in
> config, so all standard metric specs must be re-specified alongside the
> tiering entries.

```yaml
type: core-metrics-extractor
parameters:
  engineConfigs:
    - name: "vllm"
      # Standard specs (required, since overriding replaces the built-in config)
      queuedRequestsSpec: "vllm:num_requests_waiting"
      runningRequestsSpec: "vllm:num_requests_running"
      kvUsageSpec: "vllm:kv_cache_usage_perc"
      loraSpec: "vllm:lora_requests_info"
      cacheInfoSpec: "vllm:cache_config_info"
      # Per-tier offloading specs
      customMetrics:
        - attributeKey: "tiering_chunk_hits_fs"
          metricSpec: "vllm:kv_offload_tiering_chunk_hits_total{tier=\"1:fs\"}"
          aggregation: sum
          optional: true
        - attributeKey: "tiering_chunk_hits_p2p"
          metricSpec: "vllm:kv_offload_tiering_chunk_hits_total{tier=\"2:p2p\"}"
          aggregation: sum
          optional: true
        - attributeKey: "tiering_chunk_hits_obj"
          metricSpec: "vllm:kv_offload_tiering_chunk_hits_total{tier=\"3:obj\"}"
          aggregation: sum
          optional: true
```

### vLLM CPU-Only Offloading

The metrics of `CPUOffloadingSpec` (
`vllm:kv_offload_cpu_cache_usage_perc`,
`vllm:kv_offload_cpu_cache_write_usage_perc`,
`vllm:kv_offload_cpu_cache_read_usage_perc`,
`vllm:kv_offload_store_bytes_total`,
`vllm:kv_offload_load_bytes_total`
) are not extracted by default. Add the ones you need as `customMetrics` entries:

```yaml
type: core-metrics-extractor
parameters:
  engineConfigs:
    - name: "vllm"
      # Standard specs (required, overriding replaces the built-in config)
      queuedRequestsSpec: "vllm:num_requests_waiting"
      runningRequestsSpec: "vllm:num_requests_running"
      kvUsageSpec: "vllm:kv_cache_usage_perc"
      loraSpec: "vllm:lora_requests_info"
      cacheInfoSpec: "vllm:cache_config_info"
      customMetrics:
        - attributeKey: "cpu_offload_usage"
          metricSpec: "vllm:kv_offload_cpu_cache_usage_perc"
```

## Multi-cluster support

`multicluster-metrics-extractor` is the cluster-scoped variant. It reads only a pool's aggregate metrics (`llm_d_epp_average_kv_cache_utilization`, `llm_d_epp_average_queue_size`) into the `llm-d.ai/multicluster-*` attributes the multicluster scorers read, rather than running per-pod engine extraction. Pair it with `multicluster-metrics-data-source`. See the [wiring example](../../../README.md#example).
