# Metrics

The `llm-d-router` Endpoint Picker (EPP) exposes Prometheus metrics to monitor its behavior and
performance. These are in addition to the Inference Gateway metrics; for how to view metrics, see the
Gateway API Inference Extension [metrics and observability guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/site-src/guides/metrics-and-observability.md).

## Subsystems and naming

A metric's full Prometheus name is `<subsystem>_<name>`. The EPP uses two current subsystems:

| Prefix | Scope |
|---|---|
| `llm_d_epp_` | Canonical, EPP-wide: request/latency, pool, scheduler, plugin, data layer, flow control, disaggregation, ext_proc, prefix indexer, multimodal, program-aware fairness, and predicted-latency metrics. |
| `llm_d_router_epp_` | The embedded llm-d-kv-cache metrics only (see [Embedded llm-d-kv-cache metrics](#embedded-llm-d-kv-cache-metrics)). |

Earlier releases emitted metrics under `llm_d_inference_scheduler_`, `inference_objective_`,
`inference_pool_`, `inference_extension_`, and `kvcache_`. Those prefixes are **deprecated** but
still emitted: each recorder that has a deprecated predecessor writes both the legacy series and its
current twin (dual emission), so existing dashboards keep working during migration. See [Deprecated series](#deprecated-series).

## Scrape topology

### EPP metrics endpoint

The EPP serves its metrics from the controller-runtime metrics registry at `/metrics` on the metrics
port (default `9090`, configurable with `--metrics-port`). Every metric on this page is exposed on
this single endpoint. Metric authentication and TLS are configurable via `--metrics-endpoint-auth`.

### Model-server metrics (data layer)

The `metrics-data-source` plugin scrapes each model-server pod's own `/metrics` endpoint (path
configurable) and feeds the results into the data layer for scorers. These are the model server's
metrics, distinct from the EPP metrics above; scrape the pods directly to collect them.

### Embedded llm-d-kv-cache metrics

When the precise prefix cache is enabled (`precise-prefix-cache-producer` /
`precise-prefix-cache-scorer`) with `indexerConfig.kvBlockIndexConfig.enableMetrics: true`, the
embedded llm-d-kv-cache index registers its `llm_d_router_epp_kv_cache_*` metrics on the **same**
controller-runtime registry the EPP `/metrics` endpoint already serves. No separate kv-cache HTTP
endpoint or scrape target is required.

`enableMetrics` defaults to `false`, and shipped sample configs (e.g.
`deploy/config/sim-epp-kvcache-config.yaml`) leave it off, so these series are absent until an
operator opts in.

## Deprecated series

All deprecated series are still emitted as back-compat aliases alongside their current twins. Prefer
the current `llm_d_epp_*` (or `llm_d_router_epp_*`) names in new dashboards and alerts.

| Deprecated prefix | Current replacement |
|---|---|
| `llm_d_inference_scheduler_*` (disagg, data-layer errors) | `llm_d_epp_*` |
| `inference_objective_*` (request/latency, predicted latency) | `llm_d_epp_*` |
| `inference_pool_*` (pool averages, queue size) | `llm_d_epp_*` |
| `inference_extension_*` (scheduler, plugin, info, flow control, prefix indexer) | `llm_d_epp_*` |
| `kvcache_index_*`, `kvcache_kvevents_*` | `llm_d_router_epp_kv_cache_*` |

## Metrics catalog

Names below omit the subsystem prefix. Unless a section states otherwise, the prefix is `llm_d_epp_`
and the release stage is ALPHA. Request and latency metrics share the label set
`{model_name, target_model_name, fairness_id, priority}`.

### Request and latency

| Name | Type | Notes |
|---|---|---|
| `request_total` | Counter | Total requests. |
| `request_error_total` | Counter | Errored requests; adds label `error_code`. |
| `request_duration_seconds` | Histogram | End-to-end request latency. |
| `request_size_bytes` | Histogram | Request body size. |
| `response_size_bytes` | Histogram | Response body size. |
| `request_input_tokens` | Histogram | Input token count. |
| `request_output_tokens` | Histogram | Output token count. |
| `request_cached_tokens` | Histogram | Prompt tokens served from cache. |
| `request_running` | Gauge | Requests currently in flight. |
| `request_ntpot_seconds` | Histogram | Normalized time per output token. |
| `request_ttft_seconds` | Histogram | Time to first token; adds label `streaming`. |
| `request_streaming_tpot_seconds` | Histogram | Time per output token (streaming). |
| `request_streaming_itl_seconds` | Histogram | Inter-token latency (streaming). |

### Inference pool

Label `{name}` (the pool name).

| Name | Type | Notes |
|---|---|---|
| `average_kv_cache_utilization` | Gauge | Mean KV-cache utilization across the pool. |
| `average_queue_size` | Gauge | Mean queue depth. |
| `average_running_requests` | Gauge | Mean in-flight requests. |
| `std_dev_kv_cache_utilization` | Gauge | Spread of KV-cache utilization. |
| `std_dev_queue_size` | Gauge | Spread of queue depth. |
| `std_dev_running_requests` | Gauge | Spread of in-flight requests. |
| `ready_endpoints` | Gauge | Ready endpoints in the pool. |
| `per_endpoint_queue_size` | Gauge | Per-endpoint queue depth; labels `{name, model_server_endpoint}`. |

### Scheduler

| Name | Type | Notes |
|---|---|---|
| `scheduler_e2e_duration_seconds` | Histogram | End-to-end scheduling latency. |
| `scheduler_attempts_total` | Counter | Scheduling attempts; labels `{status, target_model_name, endpoint_name, namespace, port}`. |

### Plugin, info, and model rewrite

| Name | Type | Notes |
|---|---|---|
| `plugin_duration_seconds` | Histogram | Per-plugin execution time; labels `{extension_point, plugin_type, plugin_name}`. |
| `info` | Gauge | Build info; labels `{commit, build_ref}`. |
| `model_rewrite_decisions_total` | Counter | Model-rewrite decisions; labels `{model_rewrite_name, model_name, target_model}`. |

### Data layer errors

| Name | Type | Notes |
|---|---|---|
| `datalayer_poll_errors_total` | Counter | Data-source poll failures; label `{source_type}`. |
| `datalayer_extract_errors_total` | Counter | Extractor failures; labels `{source_type, extractor_type}`. |

### Prefix cache indexer (approximate)

Labels `{plugin_name, plugin_type}`.

| Name | Type | Notes |
|---|---|---|
| `prefix_indexer_size` | Gauge | Entries in the approximate prefix index. |
| `prefix_indexer_hit_ratio` | Histogram | Prefix-match hit ratio. |
| `prefix_indexer_hit_bytes` | Histogram | Bytes matched per lookup. |

### Multimodal encoder cache

| Name | Type | Notes |
|---|---|---|
| `encoder_cache_queries_total` | Counter | Encoder-cache lookups; labels `{plugin_type, plugin_name, modality}`. |
| `encoder_cache_hits_total` | Counter | Encoder-cache hits; labels `{plugin_type, plugin_name, pod, modality}`. |
| `encoder_cache_hit_ratio` | Histogram | Hit ratio; labels `{plugin_type, plugin_name}`. |

### Program-aware fairness

| Name | Type | Notes |
|---|---|---|
| `program_aware_jains_fairness_index` | Gauge | Jain's fairness index across programs. |
| `program_aware_avg_wait_time_milliseconds` | Gauge | Mean wait time; label `{program_id}`. |
| `program_aware_attained_service_tokens` | Gauge | Attained service; label `{program_id}`. |

### Predicted latency and SLO

Labels `{plugin_name, plugin_type, model_name, target_model_name}` (some add `type`).

| Name | Type | Notes |
|---|---|---|
| `inference_request_metric` | Gauge | Observed request metric; adds label `type`. |
| `request_predicted_ttft_seconds` | Histogram | Predicted time to first token. |
| `request_ttft_prediction_duration_seconds` | Histogram | Time spent computing the TTFT prediction. |
| `request_predicted_tpot_seconds` | Histogram | Predicted time per output token. |
| `request_tpot_prediction_duration_seconds` | Histogram | Time spent computing the TPOT prediction. |
| `request_slo_violation_total` | Counter | SLO violations; adds label `type`. |

### Disaggregation

Both metrics carry labels `{plugin_name, plugin_type, model_name, decision_type}`. The current names
are under `llm_d_epp_`; each has a deprecated `llm_d_inference_scheduler_*` twin.

#### `disagg_decision_total`

*   **Type:** Counter
*   **Labels:**
    *   `model_name`: the target model name, or "unknown" if empty
    *   `decision_type`: one of
        *   `decode-only` - decode-only path (no disaggregation)
        *   `prefill-decode` - split into prefill and decode stages (P/D or EP/D)
        *   `encode-decode` - encode disaggregation with local prefill+decode (E/PD)
        *   `encode-prefill-decode` - full three-stage pipeline (E/P/D)
*   **Description:** Counts requests processed, broken down by the disaggregation routing decision.
*   **Actionability:** Monitor the distribution across decision types to understand engagement per
    disaggregation mode. Sudden ratio changes may indicate configuration issues, workload shifts, or
    problems in the decision logic.

#### `pd_decision_total` (deprecated handler)

> Prefer `disagg_decision_total`. This metric is maintained for the deprecated `pd-profile-handler`
> and covers only P/D disaggregation, not encode disaggregation.

*   **Type:** Counter
*   **Labels:** `model_name`; `decision_type` (`decode-only` or `prefill-decode`).
*   **Description:** Counts requests by the Prefill/Decode disaggregation decision.

### Flow control

Exposed when the `flowControl` feature gate is enabled.

#### `flow_control_request_queue_duration_seconds`

*   **Type:** Histogram
*   **Labels:** `fairness_id`, `priority`, `outcome` (`Dispatched`, `RejectedCapacity`,
    `RejectedOther`, `EvictedTTL`, `EvictedContextCancelled`, `EvictedOther`), `inference_pool`,
    `model_name`, `target_model_name`
*   **Description:** Total time a request spends in the Flow Control layer, from enqueue to final
    outcome.
*   **Usage:** Primary latency signal for flow control. Rising p99 indicates backends are saturated
    or capacity limits are too tight.

#### `flow_control_dispatch_cycle_duration_seconds`

*   **Type:** Histogram
*   **Description:** Time taken for each internal dispatch cycle.
*   **Usage:** Measures the overhead of the dispatch loop itself. Rising values indicate increasing
    cost per cycle from saturation detection, priority band iteration, or fairness evaluation.

#### `flow_control_request_enqueue_duration_seconds`

*   **Type:** Histogram
*   **Labels:** `fairness_id`, `priority`, `outcome`
*   **Description:** Time taken to enqueue a request into the Flow Control layer.
*   **Usage:** Measures the time spent in capacity checks and queue insertion within the processor.

#### `flow_control_queue_size`

*   **Type:** Gauge
*   **Labels:** `fairness_id`, `priority`, `inference_pool`, `model_name`, `target_model_name`
*   **Description:** Current number of requests actively held in the Flow Control queue.
*   **Usage:** Tracks queue depth per priority band and tenant. A steadily growing value indicates
    the dispatch rate is lower than the arrival rate.

#### `flow_control_queue_bytes`

*   **Type:** Gauge
*   **Labels:** `fairness_id`, `priority`, `inference_pool`, `model_name`, `target_model_name`
*   **Description:** Current total size in bytes of requests actively held in the Flow Control queue.
*   **Usage:** Tracks memory pressure from queued requests. Compare against the configured `maxBytes`
    capacity to gauge how close a band is to rejecting new requests.

#### `flow_control_pool_saturation`

*   **Type:** Gauge
*   **Labels:** `inference_pool`
*   **Description:** Current saturation level of the inference pool (0.0 = empty, 1.0 = fully
    saturated).
*   **Usage:** When saturation reaches the usage limit threshold, the dispatch cycle skips
    dispatching and requests remain queued. Sustained 1.0 indicates all backends are at capacity.

#### `flow_control_requests_total`

*   **Type:** Counter
*   **Labels:**
    *   `outcome`: terminal outcome, one of `Dispatched`, `RejectedCapacity`, `RejectedNoEndpoints`
        (candidate pool had no endpoints at the capacity boundary; surfaces as HTTP 503 rather than
        429), `RejectedOther`, `EvictedTTL`, `EvictedContextCancelled`, `EvictedOther`
    *   `priority`, `inference_pool`
*   **Description:** Total requests processed by the Flow Control layer, incremented once per request
    after its terminal outcome is determined.
*   **Usage:** Direct signal for rejection and eviction rates without log parsing. Unlike
    `flow_control_request_queue_duration_seconds_count`, this counter also captures controller-level
    early rejections where no queue item is created (e.g. rejection during controller shutdown).
*   **Actionability:**
    *   Rising `outcome="RejectedCapacity"`: queue limits too tight or backends persistently
        saturated - tune `maxBytes`/`maxRequests` or scale backends.
    *   Rising `outcome="RejectedNoEndpoints"`: the pool scaled to zero or all endpoints unregistered
        - investigate pool health and scaling.
    *   Rising `outcome="EvictedTTL"`: requests waiting longer than their TTL - investigate backend
        throughput or tighten admission.
    *   `outcome="Dispatched"` is the healthy baseline; compare against total request rate for the
        acceptance ratio.

### ext_proc streams

Three metrics covering the ext_proc gRPC stream lifecycle. Disabled by default; enable with
`--enable-grpc-stream-metrics`.

#### `extproc_streams_inflight`

*   **Type:** Gauge
*   **Description:** Number of ext_proc gRPC streams currently open.
*   **Usage:** Sized at one stream per Envoy worker per EPP backend. A persistent increase under
    steady load indicates streams are being opened faster than they close.

#### `extproc_stream_duration_seconds`

*   **Type:** Histogram
*   **Description:** Duration an ext_proc gRPC stream stays open, in seconds.
*   **Usage:** Long-lived streams are normal; the histogram surfaces the distribution. A sudden shift
    toward short durations can indicate Envoy reconnecting due to handler errors.

#### `extproc_streams_total`

*   **Type:** Counter
*   **Labels:** `code` - the gRPC status code at stream close (`OK`, `Canceled`, `DeadlineExceeded`,
    `Internal`, ...). Bare `context.Canceled` and `context.DeadlineExceeded` are classified to their
    canonical codes rather than collapsing into `Unknown`.
*   **Description:** Total ext_proc gRPC streams completed, by gRPC status code.
*   **Usage:** Rate of `code="OK"` is the healthy completion rate. Rising `code="Internal"` or
    `code="Unknown"` indicates handler errors. `code="Canceled"` is expected on Envoy restarts and
    rolling EPP updates.

### KV-cache index

Prefix `llm_d_router_epp_`. Registered only when the embedded llm-d-kv-cache metrics are enabled (see
[Embedded llm-d-kv-cache metrics](#embedded-llm-d-kv-cache-metrics)). Unlabeled.

| Name | Type | Notes |
|---|---|---|
| `kv_cache_index_admissions_total` | Counter | Blocks admitted to the index. |
| `kv_cache_index_evictions_total` | Counter | Blocks evicted from the index. |
| `kv_cache_index_lookup_requests_total` | Counter | Index lookups performed. |
| `kv_cache_index_lookup_hits_total` | Counter | Lookups that matched at least one block. |
| `kv_cache_index_max_pod_hit_count_total` | Counter | Best per-pod hit count observed per lookup. |
| `kv_cache_index_lookup_latency_seconds` | Histogram | Index lookup latency. |
| `kv_cache_events_dedup_removed_hashes_suppressed_total` | Counter | Deduplicated removal hashes suppressed. |
| `kv_cache_events_dedup_removed_hashes_forwarded_total` | Counter | Deduplicated removal hashes forwarded. |

## Related work

Broader observability work tracked separately (not part of this document):

- Metrics naming / plugin labels: [#1243](https://github.com/llm-d/llm-d-router/issues/1243)
- Deprecate/remove legacy metrics: [#1070](https://github.com/llm-d/llm-d-router/issues/1070), [#962](https://github.com/llm-d/llm-d-router/issues/962)
- EPP operations guide: [#1291](https://github.com/llm-d/llm-d-router/issues/1291)
- E2E metrics stability: [#1192](https://github.com/llm-d/llm-d-router/issues/1192)
