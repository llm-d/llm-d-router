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
    -   **Tiered Offloading** (optional): KV cache tiering metrics (`kv_offload_tiering_*`) when vLLM is configured with `TieringOffloadingSpec`. Silently skipped on non-tiering deployments.
5.  Stores these values as attributes on the endpoint, making them available to scheduling plugins.

## Attributes produced

The plugin populates several standard keys on the endpoint:

-   `WaitingQueueSize` (int)
-   `RunningRequestsSize` (int)
-   `KVCacheUsagePercent` (float64)
-   `MaxActiveModels` (int)
-   `ActiveModels` (int)
-   `WaitingModels` (int)
-   `UpdateTime` (time.Time)

When tiered offloading metrics are detected, the following scalar metric attributes are also produced (stored via `ScalarMetricDataKey`):

-   `tiering_block_hits` (float64)
-   `tiering_block_queries` (float64)
-   `tiering_read_bytes` (float64)
-   `tiering_read_time` (float64)
-   `tiering_promotion_failures` (float64)
-   `tiering_allocation_failures` (float64)

## Configuration

The plugin config supports:

-   `engineLabelKey`: The Pod label key used to identify the engine type. Defaults to `llm-d.ai/engine-type`. 
    The deprecated GAIE key `inference.networking.k8s.io/engine-type` is also supported as a fallback, 
    but will be removed in a future release.
-   `defaultEngine`: The engine type to use if the label is missing. Defaults to `vllm`.
-   `engineConfigs`: A list of engine-specific metric specifications.
    Each engine config can include `tieredOffloadingSpecs` and `customMetrics`
    entries. Each entry maps a scalar metric selector to an endpoint attribute key.
    The built-in vLLM config includes tiered offloading specs for the default
    single-tier filesystem configuration (`tier="1:fs"`). Multi-tier deployments
    should override with per-tier entries (see example below).

### Built-in Engine Configurations

The plugin comes with built-in support for the following engines:
-   `vllm`
-   `sglang`
-   `trtllm-serve`
-   `triton-tensorrt-llm`

To correctly establish the mapping, model server Pods should be labeled using the `engineLabelKey` with the engine type as follows:

```yaml
metadata:
  labels:
    llm-d.ai/engine-type: vllm # other options: sglang, trtllm-serve, triton-tensorrt-llm, triton 

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

### Multi-Tier Offloading Configuration

The built-in vLLM config extracts tiered offloading metrics for tier `1:fs`
(the default single filesystem tier). Single-tier deployments need no
configuration.

Multi-tier deployments must override the vLLM engine config to add entries
for each tier. The tier label format is `{index}:{type}`, where `index` is
the position in the `secondary_tiers` list (starting at 1) and `type` is the
tier type (`fs`, `p2p`, `obj` for S3-compatible stores, or a custom type).
Check your vLLM `TieringOffloadingSpec` config or scrape output to confirm
the exact labels.

> [!NOTE]
> Overriding `engineConfigs` for `"vllm"` replaces the entire built-in
> config, so all standard metric specs must be re-specified alongside the
> tiering entries.

```yaml
type: core-metrics-extractor
parameters:
  engineConfigs:
    - name: "vllm"
      # Standard specs (required — overriding replaces the built-in config)
      queuedRequestsSpec: "vllm:num_requests_waiting"
      runningRequestsSpec: "vllm:num_requests_running"
      kvUsageSpec: "vllm:kv_cache_usage_perc"
      loraSpec: "vllm:lora_requests_info"
      cacheInfoSpec: "vllm:cache_config_info"
      # Per-tier offloading specs
      tieredOffloadingSpecs:
        - attributeKey: "tiering_block_hits_fs"
          metricSpec: "vllm:kv_offload_tiering_block_hits_total{tier=\"1:fs\"}"
        - attributeKey: "tiering_block_hits_p2p"
          metricSpec: "vllm:kv_offload_tiering_block_hits_total{tier=\"2:p2p\"}"
        - attributeKey: "tiering_block_hits_obj"
          metricSpec: "vllm:kv_offload_tiering_block_hits_total{tier=\"3:obj\"}"
```

## Multi-cluster support

`multicluster-metrics-extractor` is the cluster-scoped variant. It reads only a pool's aggregate metrics (`llm_d_epp_average_kv_cache_utilization`, `llm_d_epp_average_queue_size`) into the `llm-d.ai/multicluster-*` attributes the multicluster scorers read, rather than running per-pod engine extraction. Pair it with `multicluster-metrics-data-source`. See the [wiring example](../../../README.md#example).
