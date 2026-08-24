# Metrics

The `llm-d-router` Endpoint Picker (EPP) exposes Prometheus metrics to monitor its behavior and
performance. These are in addition to the Inference Gateway metrics. For how to reach the EPP's
metrics endpoint and what it serves, see [Scrape topology](#scrape-topology) below.

## Subsystems and naming

A metric's full Prometheus name is `<subsystem>_<name>`. The EPP uses the canonical subsystem:

| Prefix | Scope |
|---|---|
| `llm_d_epp_` | Canonical, EPP-wide: request/latency, in-flight load, pool, scheduler, plugin, data layer, flow control, disaggregation, ext_proc, prefix indexer, multimodal, program-aware fairness, predicted latency, and embedded KV-cache metrics. |

Earlier releases emitted metrics under `llm_d_inference_scheduler_`, `inference_objective_`,
`inference_pool_`, `inference_extension_`, `llm_d_router_epp_`, and `kvcache_`. Those prefixes are **deprecated** but
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
embedded llm-d-kv-cache index registers its `llm_d_epp_kv_cache_*` metrics on the **same**
controller-runtime registry the EPP `/metrics` endpoint already serves. No separate kv-cache HTTP
endpoint or scrape target is required.

`enableMetrics` defaults to `false`, and shipped sample configs (e.g.
`deploy/config/sim-epp-kvcache-config.yaml`) leave it off, so these series are absent until an
operator opts in.

## Deprecated series

All deprecated series are still emitted as back-compat aliases alongside their current twins. Prefer
the current `llm_d_epp_*` names in new dashboards and alerts.

| Deprecated prefix | Current replacement |
|---|---|
| `llm_d_inference_scheduler_*` (disagg, data-layer errors) | `llm_d_epp_*` |
| `inference_objective_*` (request/latency, predicted latency) | `llm_d_epp_*` |
| `inference_pool_*` (pool averages, queue size) | `llm_d_epp_*` |
| `inference_extension_*` (scheduler, plugin, info, flow control, prefix indexer) | `llm_d_epp_*` |
| `kvcache_index_*`, `kvcache_kvevents_*` | `llm_d_epp_kv_cache_*` |

## Metrics catalog

Names below omit the subsystem prefix. Unless a section states otherwise, the prefix is `llm_d_epp_`
and the release stage is ALPHA. Request and latency metrics share the label set
`{model_name, target_model_name, fairness_id, priority}`.

Client-derived label values are cardinality-bounded: `model_name` and `target_model_name` (from the
request body) share a cap of 1000 distinct values, and `fairness_id` (from the
`x-llm-d-inference-fairness-id` header) is capped at 1000 distinct values. Caps apply over the
lifetime of the process; once a cap is reached, new values are reported as `other`. Model names
configured through InferenceModelRewrite rules never fold to `other`. Flow control series for a
`fairness_id` are removed when its flow is garbage collected.

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

### EPP processing overhead

Unlabelled.

| Name | Type | Notes |
|---|---|---|
| `request_processing_duration_seconds` | Histogram | Time from request receipt until the request body has been handled. Includes admission control, so with flow control enabled this covers queue wait; `flow_control_request_queue_duration_seconds` separates it out. |
| `response_processing_duration_seconds` | Histogram | Sum of the per-chunk handler slices for a streamed response, so model-server generation time between chunks is excluded. For a non-streaming response, the interval from response headers to completion. |

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

#### DisaggregatedSet rollout

These metrics are exposed by the Alpha `disaggregatedset-rollout-screener` plugin.

| Name | Type | Labels | Notes |
|---|---|---|---|
| `disaggregatedset_strict_header_no_match_total` | Counter | `plugin_type`, `plugin_name`, `selector` | Strict header selections that matched no endpoint and failed closed. |
| `disaggregatedset_revision_gating_share` | Gauge | `plugin_type`, `plugin_name`, `mode`, `revision` | Current weighted share from `0` to `1`. Incomplete revisions report `0`; a revision's series is removed when it disappears from the observed Pod set. |

### Flow control

Exposed when the `flowControl` feature gate is enabled.

#### `flow_control_request_queue_duration_seconds`

*   **Type:** Histogram
*   **Labels:** `fairness_id`, `priority`, `outcome` (`Dispatched`, `RejectedCapacity`,
    `RejectedOther`, `EvictedTTL`, `EvictedNoEndpoints`, `EvictedContextCancelled`, `EvictedOther`),
    `inference_pool`, `model_name`, `target_model_name`
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
*   **Labels:** `inference_pool`, `stage`
*   **Description:** Pool saturation signal gating dispatch. The `stage` label partitions by
    pipeline role: `prefill` and `decode` are per-stage signals, `effective` is
    `max(prefill, decode)` and is the value used for gating. In monolithic deployments (no role
    labels) all endpoints land in the decode stage. 1.0 is the gating set point; values above 1.0
    indicate the magnitude of oversubscription past it (deliberately not clamped). An empty pool
    reads as 1.0, and with the default utilization detector, endpoints with missing or stale
    metrics score as fully saturated (fail-closed).
*   **Usage:** When the effective saturation reaches the usage limit threshold, the dispatch cycle
    skips dispatching and requests remain queued. A reading pinned at exactly 1.0 can be
    fail-closed stale-metrics or an empty pool rather than genuine overload (which typically reads
    above 1.0); check `flow_control_stale_endpoints` to disambiguate.

#### `flow_control_stale_endpoints`

*   **Type:** Gauge
*   **Labels:** `detector`
*   **Description:** Number of candidate endpoints whose metrics are missing or older than the
    staleness threshold, as of the most recent saturation evaluation. Recorded by the utilization
    saturation detector; emitted under the `llm_d_epp` prefix only (no deprecated
    `inference_extension_*` twin). This gauge carries no `stage` label and is written on every
    detector call, so it reflects the most recently evaluated stage. A reading of 0 does not rule
    out stale metrics in another stage; per-stage stale accounting is tracked in
    [#2475](https://github.com/llm-d/llm-d-router/issues/2475).
*   **Usage:** A nonzero value during a dispatch stall indicates a model-server metrics collection
    problem (scrape path, port, TLS, auth) rather than genuine overload.

#### `flow_control_capacity_utilization_requests`

*   **Type:** Gauge
*   **Labels:** `priority`, `inference_pool`
*   **Description:** Fraction of a priority band's **effective** request-count capacity currently
    occupied (0.0-1.0), aggregated over every flow in the band. This is not a per-flow-queue metric:
    `priority` identifies the band. A band that does not configure `maxRequests` falls back to a
    default denominator, so every configured band reports a series. The all-bands rollup is a
    separate metric, `flow_control_global_capacity_utilization_requests`.
*   **Usage:** Lets operators alert on "the band is at N% of its request limit" without joining
    configured `maxRequests` values into the query. Sustained values near 1.0 precede
    `flow_control_requests_total{outcome="RejectedCapacity"}` rising. Note the denominator is always
    the band's own capacity: when a global cap sits below the sum of the band caps, admission is
    bounded by the global cap first, so the per-band ratio understates real pressure — read it
    alongside `flow_control_global_capacity_utilization_requests`. (Making a band's denominator
    `min(band, global)` would make one band's ratio depend on other bands' configuration, which is
    worse.)

#### `flow_control_capacity_utilization_bytes`

*   **Type:** Gauge
*   **Labels:** `priority`, `inference_pool`
*   **Description:** Byte-size counterpart of `flow_control_capacity_utilization_requests`: the
    fraction of a priority band's effective byte-size capacity currently occupied (0.0-1.0),
    aggregated over every flow in the band, with the same default-denominator fallback.
*   **Usage:** Memory-pressure equivalent of the request-count ratio; a band can hit its `maxBytes`
    ceiling long before its `maxRequests` one when payloads are large. The same global-cap caveat
    applies — compare against `flow_control_global_capacity_utilization_bytes`.

#### `flow_control_global_capacity_utilization_requests`

*   **Type:** Gauge
*   **Labels:** `inference_pool`
*   **Description:** Fraction of the global request-count capacity currently occupied across all
    priority bands (0.0-1.0). Global capacity is optional and unset by default, so this series is
    emitted only when it is configured — absent, not 0.
*   **Usage:** The rollup companion to the per-band ratio. It lives in its own metric family so that
    aggregations over the per-band family (`sum`, `max`, `topk`) do not double count or rank against
    the rollup.

#### `flow_control_global_capacity_utilization_bytes`

*   **Type:** Gauge
*   **Labels:** `inference_pool`
*   **Description:** Byte-size counterpart of `flow_control_global_capacity_utilization_requests`,
    with the same optional-and-omitted-when-unset behaviour.
*   **Usage:** Shows whether the global byte ceiling, rather than any single band, is what is
    bounding admission.

#### `flow_control_requests_total`

*   **Type:** Counter
*   **Labels:**
    *   `outcome`: terminal outcome, one of `Dispatched`, `RejectedCapacity`, `RejectedNoEndpoints`
        (candidate pool had no endpoints at the capacity boundary; surfaces as HTTP 503 rather than
        429), `RejectedOther`, `EvictedTTL`, `EvictedNoEndpoints` (queue-wait budget expired while the
        candidate pool had no endpoints; surfaces as HTTP 503 rather than 429), `EvictedContextCancelled`,
        `EvictedOther`
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
    *   Rising `outcome="EvictedNoEndpoints"`: requests waited out `noEndpointRequestTTL` without an
        endpoint appearing - investigate scaling latency, or raise the budget above pod startup.
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

### In-flight load

In-flight load, emitted under the `llm_d_epp_` prefix. Present only when an `InFlightLoadProducer` is
configured: the producer owns these metrics and registers them through the plugin metrics recorder. The
per-endpoint gauges are updated as requests are admitted and released.

#### `inflight_requests`

*   **Type:** Gauge
*   **Labels:**
    *   `endpoint_name`: string — the target endpoint (pod) name.
    *   `namespace`: string — the endpoint's namespace.
    *   `producer_name`: string — the configured `InFlightLoadProducer` instance name, so multiple producers emit distinct series.
    *   `fairness_id`: string — the flow-control fairness queue identity.
    *   `priority`: string — the request priority.
*   **Release Stage:** ALPHA
*   **Description:** Requests currently in flight on each endpoint (scheduled, not yet completed), as tracked by the in-flight load producer.
*   **Usage:** Per-replica queue depth for load-aware routing and capacity analysis.

#### `inflight_tokens`

*   **Type:** Gauge
*   **Labels:**
    *   `endpoint_name`: string — the target endpoint (pod) name.
    *   `namespace`: string — the endpoint's namespace.
    *   `producer_name`: string — the configured `InFlightLoadProducer` instance name.
    *   `fairness_id`: string — the flow-control fairness queue identity.
    *   `priority`: string — the request priority.
*   **Release Stage:** ALPHA
*   **Description:** Tokens currently in flight on each endpoint — uncached prompt tokens, optionally plus estimated output tokens when the producer's `addEstimatedOutputTokens` is set.
*   **Usage:** Per-replica token pressure, a finer load signal than request count when request sizes vary widely.

### KV-cache index

Prefix `llm_d_epp_`. Registered only when the embedded llm-d-kv-cache metrics are enabled (see
[Embedded llm-d-kv-cache metrics](#embedded-llm-d-kv-cache-metrics)). Unlabeled.

| Name | Type | Notes |
|---|---|---|
| `kv_cache_index_admissions_total` | Counter | Blocks admitted to the index. |
| `kv_cache_index_evictions_total` | Counter | Blocks evicted from the index. |
| `kv_cache_index_lookup_requests_total` | Counter | Index lookups performed. |
| `kv_cache_index_lookup_hits_total` | Counter | Contiguous prefix blocks matched by the best pod per lookup. |
| `kv_cache_index_max_pod_hit_count_total` | Counter | Longest contiguous per-pod prefix chain observed per lookup. |
| `kv_cache_index_lookup_latency_seconds` | Histogram | Index lookup latency. |
| `kv_cache_events_dedup_removed_hashes_suppressed_total` | Counter | Deduplicated removal hashes suppressed. |
| `kv_cache_events_dedup_removed_hashes_forwarded_total` | Counter | Deduplicated removal hashes forwarded. |

### MoRI-IO DNS re-resolution

Prefix `moriio_dns_`. Emitted by the sidecar proxy when MoRI-IO peer host specs
are DNS names that are re-resolved on the request path (see the
[MoRI-IO feature guide](../pkg/sidecar/proxy/MORIIO_README.md)). Registered on
the same controller-runtime registry as the other metrics on this page.
Unlabeled.

Unlike the EPP `/metrics` endpoint, the sidecar does not serve metrics by
default. Pass `--metrics-port` (e.g. `--metrics-port=9090`) to the sidecar to
expose these counters at `/metrics` on that port; `0` (the default) disables it.
The `MORIIO_METRICS_ADDR` env var (e.g. `:9090`) is a backward-compatible
fallback, consulted only when `--metrics-port` is unset.

| Name | Type | Notes |
|---|---|---|
| `moriio_dns_reresolve_total` | Counter | Successful request-path re-resolutions of a peer DNS name (counted per actual lookup; concurrent lookups coalesced by singleflight count once). |
| `moriio_dns_ip_changed_total` | Counter | Re-resolutions where the peer resolved to a different IP than the cached value (peer pod likely restarted at a new IP). |
| `moriio_dns_lookup_failures_total` | Counter | Failed peer DNS lookups on the request path; the resolver then serves the last-known-good IP (or, on cold start, the raw spec). |

## Related work

Broader observability work tracked separately (not part of this document):

- Metrics naming / plugin labels: [#1243](https://github.com/llm-d/llm-d-router/issues/1243)
- Deprecate/remove legacy metrics: [#1070](https://github.com/llm-d/llm-d-router/issues/1070), [#962](https://github.com/llm-d/llm-d-router/issues/962)
- EPP operations guide: [#1291](https://github.com/llm-d/llm-d-router/issues/1291)
- E2E metrics stability: [#1192](https://github.com/llm-d/llm-d-router/issues/1192)
