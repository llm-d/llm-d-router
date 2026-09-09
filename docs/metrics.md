# Metrics

The router deployment has three metric sources:

- The router pod runs the EPP. Its registry contains EPP metrics, plugin metrics, and KV-cache
  metrics.
- Model server pods expose engine metrics at `/metrics`. The EPP's `metrics-data-source` plugin
  fetches and parses each endpoint's metrics directly. The `core-metrics-extractor` selects the
  values used by routing plugins.
- The P/D sidecar coordinates disaggregated inference.

## Component overview

| Component | Metrics | Endpoint |
|---|---|---|
| EPP / router pod | EPP metrics, plugin metrics, and KV-cache metrics | EPP `/metrics`, default port `9090` |
| Model server / engine pods | vLLM or other engine metrics | Each model server's `/metrics` |
| P/D sidecar | MoRI-IO metrics only | HTTP `/metrics`, disabled by default |

The EPP sends HTTP or HTTPS requests directly to each model server endpoint, per the
`metrics-data-source` plugin's `scheme` parameter (default `http`).

## Metric naming

The tables in this document show complete metric names.

| Component | Metric prefix | Scope |
|---|---|---|
| EPP / router pod | `llm_d_epp_` | EPP request and latency, in-flight load, pool, scheduler, plugin, data layer, flow control, disaggregation, ext_proc, prefix indexer, multimodal, fairness, predicted latency, model rewrite, and KV-cache metrics. |
| Model server / engine | Engine-specific | Metrics exposed by the model server implementation. |
| P/D sidecar | `moriio_dns_` | MoRI-IO peer DNS re-resolution metrics. |

The model server and sidecar metrics use separate registries and prefixes. See [Deprecated series](#deprecated-series)
for the legacy EPP names that remain available.

## Metric endpoints and flow

### Router pod / EPP

The EPP serves its controller-runtime registry at `/metrics` on the metrics port (default `9090`,
configurable with `--metrics-port`). This endpoint contains the EPP metrics and all collectors
registered by EPP plugins, including the embedded KV-cache collectors. Metric authentication is
configurable via `--metrics-endpoint-auth` (default `true`). TLS is a separate setting, configurable
via `--metrics-cert-dir`; mutual TLS additionally requires `--metrics-client-ca-file`.

### Model server / engine

The `metrics-data-source` plugin sends an HTTP or HTTPS request (`scheme`, default `http`; TLS
options include `caCertPath` and `clientCertPath`/`clientKeyPath` for mTLS) to each model server
endpoint's `/metrics` path (configurable) and parses the response. The `core-metrics-extractor`
selects configured values and stores them on the endpoint for scorers. These are model server
metrics, not EPP metrics. The raw engine metrics remain available at the model server endpoint.

### P/D sidecar: MoRI-IO metrics

The P/D sidecar currently exposes only the `moriio_dns_*` MoRI-IO metrics through an HTTP `/metrics`
endpoint. It exposes them only when `--metrics-port` or the backward-compatible
`MORIIO_METRICS_ADDR` environment variable is set. The P/D sidecar metrics server does not configure
TLS. `SecureServing` applies to the sidecar data-plane listener.

This endpoint belongs to the sidecar process;
its controller-runtime registry is separate from the router pod's EPP registry.

## Metrics catalog

Names below include the full subsystem prefix. EPP metrics use `llm_d_epp_` and the release stage is
ALPHA. Labels are listed in each metric table.

Client-derived label values are cardinality-bounded on EPP metrics that use them. `model_name` and
`target_model_name` from the request body share a cap of 1000 distinct values. Model names configured
through InferenceModelRewrite rules never fold to `other`.

The `fairness_id` label, populated from the `x-llm-d-inference-fairness-id` header or an agent
identity, has a cap that defaults to 1000 distinct values. Configure it with
`--fairness-id-metric-label-limit`; setting it to 0 collapses every `fairness_id` to `other`. Caps
apply over the lifetime of the process. Once a cap is reached, new values are reported as `other`.
These bounds apply whether or not the `flowControl` feature gate is enabled. When Flow Control is
enabled, its series for a `fairness_id` are removed when the corresponding flow is garbage collected.

### Request and latency

These metrics are owned by EPP request and response instrumentation. They record the request
lifecycle handled by the router.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_request_total` | Counter | `model_name`, `target_model_name`, `fairness_id`, `priority` | Total requests. |
| `llm_d_epp_request_error_total` | Counter | `model_name`, `target_model_name`, `fairness_id`, `priority`, `error_code` | Errored requests. |
| `llm_d_epp_request_duration_seconds` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority` | End-to-end request latency. |
| `llm_d_epp_request_size_bytes` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority` | Request body size. |
| `llm_d_epp_response_size_bytes` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority` | Response body size. |
| `llm_d_epp_request_input_tokens` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority` | Input token count. |
| `llm_d_epp_request_output_tokens` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority` | Output token count. |
| `llm_d_epp_request_cached_tokens` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority` | Prompt tokens served from cache. |
| `llm_d_epp_request_running` | Gauge | `model_name`, `target_model_name`, `fairness_id`, `priority` | Requests currently in flight. |
| `llm_d_epp_request_ntpot_seconds` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority` | Normalized time per output token. |
| `llm_d_epp_request_ttft_seconds` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority`, `streaming` | Time to first token. |
| `llm_d_epp_request_streaming_tpot_seconds` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority` | Time per output token for streaming. |
| `llm_d_epp_request_streaming_itl_seconds` | Histogram | `model_name`, `target_model_name`, `fairness_id`, `priority` | Inter-token latency for streaming. |

### Inference pool

These metrics are owned by EPP pool aggregation. They summarize model-server endpoint metrics for
an `InferencePool`.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_average_kv_cache_utilization` | Gauge | `name` (InferencePool name) | Mean KV-cache utilization across the pool. |
| `llm_d_epp_average_queue_size` | Gauge | `name` (InferencePool name) | Mean queue depth. |
| `llm_d_epp_average_running_requests` | Gauge | `name` (InferencePool name) | Mean in-flight requests. |
| `llm_d_epp_std_dev_kv_cache_utilization` | Gauge | `name` (InferencePool name) | Spread of KV-cache utilization. |
| `llm_d_epp_std_dev_queue_size` | Gauge | `name` (InferencePool name) | Spread of queue depth. |
| `llm_d_epp_std_dev_running_requests` | Gauge | `name` (InferencePool name) | Spread of in-flight requests. |
| `llm_d_epp_ready_endpoints` | Gauge | `name` (InferencePool name) | Ready endpoints in the pool. |
| `llm_d_epp_per_endpoint_queue_size` | Gauge | `name` (InferencePool name), `model_server_endpoint` | Per-endpoint queue depth. |

### Scheduler

These metrics are owned by EPP scheduler instrumentation.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_scheduler_e2e_duration_seconds` | Histogram | - | End-to-end scheduling latency. |
| `llm_d_epp_scheduler_attempts_total` | Counter | `status`, `target_model_name`, `endpoint_name`, `namespace`, `port` | Scheduling attempts. |

### EPP processing overhead

These metrics are owned by EPP request and response processing instrumentation.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_request_processing_duration_seconds` | Histogram | - | Time from request receipt until the request body has been handled. Includes admission control, so with flow control enabled this covers queue wait; `llm_d_epp_flow_control_request_queue_duration_seconds` separates it out. |
| `llm_d_epp_response_processing_duration_seconds` | Histogram | - | Sum of the per-chunk handler slices for a streamed response, so model server generation time between chunks is excluded. For a non-streaming response, the interval from response headers to completion. |

### Plugin, info, and model rewrite

These metrics come from EPP plugin instrumentation, EPP build information, and model-rewrite
decisions.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_plugin_duration_seconds` | Histogram | `extension_point`, `plugin_type`, `plugin_name` | Per-plugin execution time. |
| `llm_d_epp_info` | Gauge | `commit`, `build_ref` | Build info. |
| `llm_d_epp_model_rewrite_decisions_total` | Counter | `model_rewrite_name`, `model_name`, `target_model` | Model-rewrite decisions. |

### Data layer errors

These metrics are owned by the EPP data-layer runtime.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_datalayer_poll_errors_total` | Counter | `source_type` | Data-source poll failures. |
| `llm_d_epp_datalayer_extract_errors_total` | Counter | `source_type`, `extractor_type` | Extractor failures. |

### Prefix cache indexer (approximate)

These metrics belong to the `approx-prefix-cache-producer`. They describe its approximate prefix
index and are separate from the precise KV-cache index metrics. The metric families are registered
when the producer is created; observations require prefix lookups.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_prefix_indexer_size` | Gauge | `plugin_name`, `plugin_type` | Entries in the approximate prefix index. |
| `llm_d_epp_prefix_indexer_hit_ratio` | Histogram | `plugin_name`, `plugin_type` | Prefix-match hit ratio. |
| `llm_d_epp_prefix_indexer_hit_bytes` | Histogram | `plugin_name`, `plugin_type` | Bytes matched per lookup. |

### Multimodal encoder cache

These metrics belong to the `mm-embeddings-cache-producer`, not Flow Control. The producer keeps an
EPP-side LRU of multimodal item hashes per endpoint to estimate model-server encoder-cache locality.
It does not report raw encoder metrics from the model server. The metric families are registered
when the producer is created; observations require multimodal cache lookups.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_encoder_cache_queries_total` | Counter | `plugin_type`, `plugin_name`, `modality` | Encoder-cache lookups. |
| `llm_d_epp_encoder_cache_hits_total` | Counter | `plugin_type`, `plugin_name`, `pod`, `modality` | Encoder-cache hits. |
| `llm_d_epp_encoder_cache_hit_ratio` | Histogram | `plugin_type`, `plugin_name` | Hit ratio. |

### Program-aware fairness

These metrics are owned by the `program-aware-fairness` Flow Control policy. They require the
`flowControl` feature gate and a Flow Control configuration that selects this policy. Enabling Flow
Control alone does not emit them.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_program_aware_jains_fairness_index` | Gauge | - | Jain's fairness index across programs. |
| `llm_d_epp_program_aware_avg_wait_time_milliseconds` | Gauge | `program_id` | Mean wait time. |
| `llm_d_epp_program_aware_attained_service_tokens` | Gauge | `program_id` | Attained service. |

### Predicted latency and SLO

These metrics are registered and recorded by the `predicted-latency-producer`. They are present
only when that plugin is configured and records the related prediction, observation, or SLO event.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_inference_request_metric` | Gauge | `plugin_name`, `plugin_type`, `model_name`, `target_model_name`, `type` | Observed request metric. |
| `llm_d_epp_request_predicted_ttft_seconds` | Histogram | `plugin_name`, `plugin_type`, `model_name`, `target_model_name` | Predicted time to first token. |
| `llm_d_epp_request_ttft_prediction_duration_seconds` | Histogram | `plugin_name`, `plugin_type`, `model_name`, `target_model_name` | Time spent computing the TTFT prediction. |
| `llm_d_epp_request_predicted_tpot_seconds` | Histogram | `plugin_name`, `plugin_type`, `model_name`, `target_model_name` | Predicted time per output token. |
| `llm_d_epp_request_tpot_prediction_duration_seconds` | Histogram | `plugin_name`, `plugin_type`, `model_name`, `target_model_name` | Time spent computing the TPOT prediction. |
| `llm_d_epp_request_slo_violation_total` | Counter | `plugin_name`, `plugin_type`, `model_name`, `target_model_name`, `type` | SLO violations. |

### Disaggregation

The EPP disaggregation metrics come from two different plugins.

#### `disagg-profile-handler`

This plugin records the routing decision for each request.

#### `llm_d_epp_disagg_decision_total`

*   **Type:** Counter
*   **Labels:**
    *   `plugin_name`, `plugin_type`: the `disagg-profile-handler` plugin instance recording the
        decision
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

#### `disaggregatedset-rollout-screener`

This Alpha plugin records revision-match failures and revision-gating state.

#### `llm_d_epp_disaggregatedset_strict_revision_no_match_total`

*   **Type:** Counter
*   **Labels:**
    *   `plugin_type`, `plugin_name`: the `disaggregatedset-rollout-screener` plugin instance
        recording the failure
*   **Description:** Strict revision selections that matched no endpoint and failed closed. A
    request carrying the revision header (`x-llm-d-disagg-revision` by default) filters candidates
    to that revision; when none match, the request has no eligible endpoint, which surfaces as
    HTTP 503 rather than falling back to another endpoint.
*   **Actionability:** A nonzero rate means requests are arriving with a revision header value that
    has no Ready endpoint. Check rollout progress and Pod readiness for the revision the header
    identifies.

#### `llm_d_epp_disaggregatedset_revision_gating_share`

*   **Type:** Gauge
*   **Labels:**
    *   `plugin_type`, `plugin_name`: the `disaggregatedset-rollout-screener` plugin instance
    *   `mode`: `sum` or `max-role` when revision gating is active; empty when
        `revisionGating.mode` is `disabled` (gating is inactive, so the configured mode is not
        recorded)
    *   `revision`: the observed `DisaggregatedSet` revision label
*   **Description:** Current weighted share, from `0` to `1`, of traffic the revision-gating
    algorithm assigns to this revision. A revision missing a required role reports `0`. A
    revision's series is removed once it disappears from the observed Pod set.
*   **Actionability:** Track this alongside rollout progress to confirm traffic shifts toward the
    new revision at the rate expected for the configured gating mode.

### Flow control

Flow Control is disabled by default. These metrics are exposed only when the `flowControl` feature
gate is enabled with `featureGates: ["flowControl"]`.

These metrics are owned by the EPP Flow Control layer.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_flow_control_request_queue_duration_seconds` | Histogram | `fairness_id`, `priority`, `outcome`, `inference_pool`, `model_name`, `target_model_name` | Time from queue entry to terminal outcome. |
| `llm_d_epp_flow_control_dispatch_cycle_duration_seconds` | Histogram | - | Cost of one dispatch cycle. |
| `llm_d_epp_flow_control_request_enqueue_duration_seconds` | Histogram | `fairness_id`, `priority`, `outcome` | Time spent admitting a request to the queue. |
| `llm_d_epp_flow_control_queue_size` | Gauge | `fairness_id`, `priority`, `inference_pool`, `model_name`, `target_model_name` | Requests currently held in the queue. |
| `llm_d_epp_flow_control_queue_bytes` | Gauge | `fairness_id`, `priority`, `inference_pool`, `model_name`, `target_model_name` | Bytes currently held in the queue. |
| `llm_d_epp_flow_control_pool_saturation` | Gauge | `inference_pool`, `stage` | Saturation signal used to gate dispatch. |
| `llm_d_epp_flow_control_stale_endpoints` | Gauge | `detector` | Candidate endpoints with missing or stale metrics. |
| `llm_d_epp_flow_control_capacity_utilization_requests` | Gauge | `priority`, `inference_pool` | Per-priority-band request capacity use. |
| `llm_d_epp_flow_control_capacity_utilization_bytes` | Gauge | `priority`, `inference_pool` | Per-priority-band byte capacity use. |
| `llm_d_epp_flow_control_global_capacity_utilization_requests` | Gauge | `inference_pool` | Global request capacity use. |
| `llm_d_epp_flow_control_global_capacity_utilization_bytes` | Gauge | `inference_pool` | Global byte capacity use. |
| `llm_d_epp_flow_control_requests_total` | Counter | `outcome`, `priority`, `inference_pool` | Requests by terminal outcome. |
| `llm_d_epp_flow_control_revocations_issued_total` | Counter | `priority`, `inference_pool` | In-flight revocations issued. |
| `llm_d_epp_flow_control_revocations_total` | Counter | `outcome`, `inference_pool` | In-flight revocations by terminal outcome. |
| `llm_d_epp_flow_control_reclaim_target` | Gauge | `inference_pool` | Last computed reclamation deficit. |
| `llm_d_epp_flow_control_pending_reclaim` | Gauge | `inference_pool` | Outstanding and cooling reclamation debit. |
| `llm_d_epp_flow_control_revocation_confirmation_seconds` | Histogram | `inference_pool` | Time from revocation to confirmed stream termination. |

#### `llm_d_epp_flow_control_request_queue_duration_seconds`

*   **Type:** Histogram
*   **Labels:** `fairness_id`, `priority`, `outcome` (`Dispatched`, `RejectedCapacity`,
    `RejectedOther`, `EvictedTTL`, `EvictedNoEndpoints`, `EvictedContextCancelled`, `EvictedOther`),
    `inference_pool`, `model_name`, `target_model_name`
*   **Description:** Total time a request spends in the Flow Control layer, from enqueue to final
    outcome.
*   **Usage:** Primary latency signal for flow control. Rising p99 indicates backends are saturated
    or capacity limits are too tight.

#### `llm_d_epp_flow_control_dispatch_cycle_duration_seconds`

*   **Type:** Histogram
*   **Description:** Time taken for each internal dispatch cycle.
*   **Usage:** Measures the overhead of the dispatch loop itself. Rising values indicate increasing
    cost per cycle from saturation detection, priority band iteration, or fairness evaluation.

#### `llm_d_epp_flow_control_request_enqueue_duration_seconds`

*   **Type:** Histogram
*   **Labels:** `fairness_id`, `priority`, `outcome`
*   **Description:** Time taken to enqueue a request into the Flow Control layer.
*   **Usage:** Measures the time spent in capacity checks and queue insertion within the processor.

#### `llm_d_epp_flow_control_queue_size`

*   **Type:** Gauge
*   **Labels:** `fairness_id`, `priority`, `inference_pool`, `model_name`, `target_model_name`
*   **Description:** Current number of requests actively held in the Flow Control queue.
*   **Usage:** Tracks queue depth per priority band and tenant. A steadily growing value indicates
    the dispatch rate is lower than the arrival rate.

#### `llm_d_epp_flow_control_queue_bytes`

*   **Type:** Gauge
*   **Labels:** `fairness_id`, `priority`, `inference_pool`, `model_name`, `target_model_name`
*   **Description:** Current total size in bytes of requests actively held in the Flow Control queue.
*   **Usage:** Tracks memory pressure from queued requests. Compare against the configured `maxBytes`
    capacity to gauge how close a band is to rejecting new requests.

#### `llm_d_epp_flow_control_pool_saturation`

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
    above 1.0); check `llm_d_epp_flow_control_stale_endpoints` to disambiguate.

#### `llm_d_epp_flow_control_stale_endpoints`

*   **Type:** Gauge
*   **Labels:** `detector`
*   **Description:** Number of candidate endpoints whose metrics are missing or older than the
    staleness threshold, as of the most recent saturation evaluation. Recorded by the utilization
    saturation detector; emitted under the `llm_d_epp` prefix only (no deprecated
    `inference_extension_*` twin). This gauge carries no `stage` label and is written on every
    detector call, so it reflects the most recently evaluated stage. A reading of 0 does not rule
    out stale metrics in another stage; per-stage stale accounting is tracked in
    [#2475](https://github.com/llm-d/llm-d-router/issues/2475).
*   **Usage:** A nonzero value during a dispatch stall indicates a model server metrics collection
    problem (endpoint path, port, TLS, or authentication) rather than genuine overload.

#### `llm_d_epp_flow_control_capacity_utilization_requests`

*   **Type:** Gauge
*   **Labels:** `priority`, `inference_pool`
*   **Description:** Fraction of a priority band's **effective** request-count capacity currently
    occupied (0.0-1.0), aggregated over every flow in the band. This is not a per-flow-queue metric:
    `priority` identifies the band. A band that does not configure `maxRequests` falls back to a
    default denominator, so every configured band reports a series. The all-bands rollup is a
    separate metric, `llm_d_epp_flow_control_global_capacity_utilization_requests`.
*   **Usage:** Lets operators alert on "the band is at N% of its request limit" without joining
    configured `maxRequests` values into the query. Sustained values near 1.0 precede
    `llm_d_epp_flow_control_requests_total{outcome="RejectedCapacity"}` rising. Note the denominator is always
    the band's own capacity: when a global cap sits below the sum of the band caps, admission is
    bounded by the global cap first, so the per-band ratio understates real pressure — read it
    alongside `llm_d_epp_flow_control_global_capacity_utilization_requests`. (Making a band's denominator
    `min(band, global)` would make one band's ratio depend on other bands' configuration, which is
    worse.)

#### `llm_d_epp_flow_control_capacity_utilization_bytes`

*   **Type:** Gauge
*   **Labels:** `priority`, `inference_pool`
*   **Description:** Byte-size counterpart of `llm_d_epp_flow_control_capacity_utilization_requests`: the
    fraction of a priority band's effective byte-size capacity currently occupied (0.0-1.0),
    aggregated over every flow in the band, with the same default-denominator fallback.
*   **Usage:** Memory-pressure equivalent of the request-count ratio; a band can hit its `maxBytes`
    ceiling long before its `maxRequests` one when payloads are large. The same global-cap caveat
    applies — compare against `llm_d_epp_flow_control_global_capacity_utilization_bytes`.

#### `llm_d_epp_flow_control_global_capacity_utilization_requests`

*   **Type:** Gauge
*   **Labels:** `inference_pool`
*   **Description:** Fraction of the global request-count capacity currently occupied across all
    priority bands (0.0-1.0). Global capacity is optional and unset by default, so this series is
    emitted only when it is configured — absent, not 0.
*   **Usage:** The rollup companion to the per-band ratio. It lives in its own metric family so that
    aggregations over the per-band family (`sum`, `max`, `topk`) do not double count or rank against
    the rollup.

#### `llm_d_epp_flow_control_global_capacity_utilization_bytes`

*   **Type:** Gauge
*   **Labels:** `inference_pool`
*   **Description:** Byte-size counterpart of `llm_d_epp_flow_control_global_capacity_utilization_requests`,
    with the same optional-and-omitted-when-unset behaviour.
*   **Usage:** Shows whether the global byte ceiling, rather than any single band, is what is
    bounding admission.

#### `llm_d_epp_flow_control_requests_total`

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
    `llm_d_epp_flow_control_request_queue_duration_seconds_count`, this counter also captures controller-level
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

### EPP ext_proc streams

Three metrics covering the ext_proc gRPC stream lifecycle. Disabled by default; enable with
`--enable-grpc-stream-metrics`.

These metrics are owned by the EPP ext_proc server.

#### `llm_d_epp_extproc_streams_inflight`

*   **Type:** Gauge
*   **Description:** Number of ext_proc gRPC streams currently open.
*   **Usage:** Sized at one stream per Envoy worker per EPP backend. A persistent increase under
    steady load indicates streams are being opened faster than they close.

#### `llm_d_epp_extproc_stream_duration_seconds`

*   **Type:** Histogram
*   **Description:** Duration an ext_proc gRPC stream stays open, in seconds.
*   **Usage:** Long-lived streams are normal; the histogram surfaces the distribution. A sudden shift
    toward short durations can indicate Envoy reconnecting due to handler errors.

#### `llm_d_epp_extproc_streams_total`

*   **Type:** Counter
*   **Labels:** `code` - the gRPC status code at stream close (`OK`, `Canceled`, `DeadlineExceeded`,
    `Internal`, ...). Bare `context.Canceled` and `context.DeadlineExceeded` are classified to their
    canonical codes rather than collapsing into `Unknown`.
*   **Description:** Total ext_proc gRPC streams completed, by gRPC status code.
*   **Usage:** Rate of `code="OK"` is the healthy completion rate. Rising `code="Internal"` or
    `code="Unknown"` indicates handler errors. `code="Canceled"` is expected on Envoy restarts and
    rolling EPP updates.

### In-flight load

These metrics belong to `inflight-load-producer`. They are registered only when the producer is
configured. The producer updates the per-endpoint gauges when requests are admitted and released.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_inflight_requests` | Gauge | `endpoint_name`, `namespace`, `producer_name`, `fairness_id`, `priority` | Current requests in flight on each endpoint. |
| `llm_d_epp_inflight_tokens` | Gauge | `endpoint_name`, `namespace`, `producer_name`, `fairness_id`, `priority` | Current tokens in flight on each endpoint. |

#### `llm_d_epp_inflight_requests`

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

#### `llm_d_epp_inflight_tokens`

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

These metrics belong to the precise-prefix-cache pipeline. Set
`indexerConfig.kvBlockIndexConfig.enableMetrics` to `true` on the
`precise-prefix-cache-producer` parameters. The option defaults to `false`, and shipped sample
configs (e.g. [`deploy/config/sim-epp-kvcache-config.yaml`](../deploy/config/sim-epp-kvcache-config.yaml))
leave it off, so these series are absent until an operator opts in. The embedded llm-d-kv-cache
index and KV-event pool register these metrics in the router pod's EPP registry, so they are
available from the existing EPP `/metrics` endpoint. Index metrics are unlabeled. KV-event metrics
use the labels shown below.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `llm_d_epp_kv_cache_index_admissions_total` | Counter | - | Blocks admitted to the index. |
| `llm_d_epp_kv_cache_index_evictions_total` | Counter | - | Blocks evicted from the index. |
| `llm_d_epp_kv_cache_index_lookup_requests_total` | Counter | - | Index lookups performed. |
| `llm_d_epp_kv_cache_index_lookup_hits_total` | Counter | - | Contiguous prefix blocks matched by the best pod per lookup. |
| `llm_d_epp_kv_cache_index_max_pod_hit_count_total` | Counter | - | Longest contiguous per-pod prefix chain observed per lookup. |
| `llm_d_epp_kv_cache_index_lookup_latency_seconds` | Histogram | - | Index lookup latency. |
| `llm_d_epp_kv_cache_events_dedup_removed_hashes_suppressed_total` | Counter | - | Deduplicated removal hashes suppressed. |
| `llm_d_epp_kv_cache_events_dedup_removed_hashes_forwarded_total` | Counter | - | Deduplicated removal hashes forwarded. |
| `llm_d_epp_kv_cache_events_stores_skipped_total` | Counter | `cache_kind`, `reason` | KV store events skipped before prefix indexing. |
| `llm_d_epp_kv_cache_events_removals_skipped_total` | Counter | `cache_kind`, `reason` | KV removal events skipped before prefix indexing. |
| `llm_d_epp_kv_cache_events_active_subscribers` | Gauge | - | ZMQ subscribers currently managed. |
| `llm_d_epp_kv_cache_events_subscriber_reconnections_total` | Counter | `pod_identifier` | ZMQ subscriber reconnection attempts. |
| `llm_d_epp_kv_cache_events_messages_received_total` | Counter | `pod_identifier` | Messages received from ZMQ subscribers. |
| `llm_d_epp_kv_cache_events_zmq_errors_total` | Counter | `pod_identifier`, `operation` | ZMQ subscriber errors. |
| `llm_d_epp_kv_cache_events_pool_queue_depth` | Gauge | - | Messages queued across event-pool workers. |
| `llm_d_epp_kv_cache_events_pool_capacity` | Gauge | - | Event-pool worker capacity. |

### MoRI-IO DNS re-resolution

Emitted by the sidecar proxy when MoRI-IO peer host specs are DNS names that are re-resolved on the
request path (see the
[MoRI-IO feature guide](../pkg/sidecar/proxy/MORIIO_README.md)). The collectors are registered on
the sidecar process's controller-runtime registry. That registry is separate from the router pod's
EPP registry. These metrics are unlabeled.

Unlike the EPP `/metrics` endpoint, the sidecar does not serve metrics by
default. Pass `--metrics-port` (e.g. `--metrics-port=9090`) to the sidecar to
expose these counters at `/metrics` on that port; `0` (the default) disables it.
The `MORIIO_METRICS_ADDR` env var (e.g. `:9090`) is a backward-compatible
fallback, consulted only when `--metrics-port` is unset.

| Full metric name | Type | Labels | Notes |
|---|---|---|---|
| `moriio_dns_reresolve_total` | Counter | - | Successful request-path re-resolutions of a peer DNS name (counted per actual lookup; concurrent lookups coalesced by singleflight count once). |
| `moriio_dns_ip_changed_total` | Counter | - | Re-resolutions where the peer resolved to a different IP than the cached value (peer pod likely restarted at a new IP). |
| `moriio_dns_lookup_failures_total` | Counter | - | Failed peer DNS lookups on the request path; the resolver then serves the last-known-good IP (or, on cold start, the raw spec). |

## Deprecated series

Selected legacy series remain available as aliases alongside current `llm_d_epp_*` series. Prefer
the current names in new dashboards and alerts. The aliases do not cover every current metric.

| Legacy series | Current replacement | Notes |
|---|---|---|
| `llm_d_inference_scheduler_disagg_decision_total` | `llm_d_epp_disagg_decision_total` | Dual emission. |
| `llm_d_inference_scheduler_datalayer_poll_errors_total`, `llm_d_inference_scheduler_datalayer_extract_errors_total` | `llm_d_epp_datalayer_poll_errors_total`, `llm_d_epp_datalayer_extract_errors_total` | Dual emission. |
| `inference_objective_*` request, latency, predicted-latency, and SLO series | Corresponding `llm_d_epp_*` series | Dual emission. Some predicted-latency labels differ between legacy and current series. |
| `inference_pool_average_*`, `inference_pool_ready_pods` | Corresponding `llm_d_epp_*` series | Dual emission. Current standard-deviation series have no legacy twin. |
| `inference_pool_per_pod_queue_size` | `llm_d_epp_per_endpoint_queue_size` | Dual emission with different metric names and labels. |
| `inference_extension_scheduler_e2e_duration_seconds` | `llm_d_epp_scheduler_e2e_duration_seconds` | Dual emission. |
| `inference_extension_scheduler_attempts_total` | `llm_d_epp_scheduler_attempts_total` | Dual emission. |
| `inference_extension_plugin_duration_seconds` | `llm_d_epp_plugin_duration_seconds` | Dual emission. |
| `inference_extension_info` | `llm_d_epp_info` | Dual emission. |
| `inference_extension_model_rewrite_decisions_total` | `llm_d_epp_model_rewrite_decisions_total` | Dual emission. |
| `inference_extension_flow_control_request_queue_duration_seconds` | `llm_d_epp_flow_control_request_queue_duration_seconds` | Dual emission. |
| `inference_extension_flow_control_dispatch_cycle_duration_seconds` | `llm_d_epp_flow_control_dispatch_cycle_duration_seconds` | Dual emission. |
| `inference_extension_flow_control_request_enqueue_duration_seconds` | `llm_d_epp_flow_control_request_enqueue_duration_seconds` | Dual emission. |
| `inference_extension_flow_control_queue_size` | `llm_d_epp_flow_control_queue_size` | Dual emission. |
| `inference_extension_flow_control_queue_bytes` | `llm_d_epp_flow_control_queue_bytes` | Dual emission. |
| `inference_extension_flow_control_pool_saturation` | `llm_d_epp_flow_control_pool_saturation` | Dual emission. |
| `inference_extension_prefix_indexer_size` | `llm_d_epp_prefix_indexer_size` | Dual emission. |
| `inference_extension_prefix_indexer_hit_ratio` | `llm_d_epp_prefix_indexer_hit_ratio` | Dual emission. |
| `inference_extension_prefix_indexer_hit_bytes` | `llm_d_epp_prefix_indexer_hit_bytes` | Dual emission. |
| `kvcache_index_*` index series | `llm_d_epp_kv_cache_index_*` | Six index series are dual-emitted. |
| `kvcache_kvevents_dedup_removed_hashes_suppressed_total`, `kvcache_kvevents_dedup_removed_hashes_forwarded_total` | Corresponding `llm_d_epp_kv_cache_events_*` series | These two KV-event series are dual-emitted. Other KV-event series are current-only. |

The historical `llm_d_router_epp_*` prefix is not emitted by the current Go code.

Current-only examples include `llm_d_epp_flow_control_stale_endpoints`, flow-control capacity and
revocation metrics, `llm_d_epp_plugin_data_scope_violations_total`, EPP processing-overhead metrics,
`llm_d_epp_prefix_cache_affinity_filter_decisions_total`, and most KV-event metrics.

## Related work

Broader observability work tracked separately (not part of this document):

- Metrics naming / plugin labels: [#1243](https://github.com/llm-d/llm-d-router/issues/1243)
- Deprecate/remove legacy metrics: [#1070](https://github.com/llm-d/llm-d-router/issues/1070), [#962](https://github.com/llm-d/llm-d-router/issues/962)
- EPP operations guide: [#1291](https://github.com/llm-d/llm-d-router/issues/1291)
- E2E metrics stability: [#1192](https://github.com/llm-d/llm-d-router/issues/1192)
