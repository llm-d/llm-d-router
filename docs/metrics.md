# Metrics

The `llm-d-router` exposes the following Prometheus metrics to monitor its behavior and performance, particularly concerning Encode/Prefill/Decode disaggregation.

All metrics are in the `llm_d_inference_scheduler` subsystem.

## Scrape and see the metric

Metrics defined by llm-d Router are in addition to Inference Gateway metrics. For more details of seeing metrics, see the [metrics and observability section](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/site-src/guides/metrics-and-observability.md).

## Metrics Details

### `disagg_decision_total`

*   **Type:** Counter
*   **Labels:**
    *   `model_name`: string (the target model name, or "unknown" if empty)
    *   `decision_type`: string - one of:
        *   `decode-only` - the request used the decode-only path (no disaggregation)
        *   `prefill-decode` - the request was split into prefill and decode stages (P/D or EP/D)
        *   `encode-decode` - the request used encode disaggregation with local prefill+decode (E/PD)
        *   `encode-prefill-decode` - the request used the full three-stage pipeline (E/P/D)
*   **Release Stage:** ALPHA
*   **Description:** Counts the number of requests processed, broken down by the disaggregation routing decision.
*   **Usage:** Provides a high-level view of how many requests are utilizing each disaggregation topology.
*   **Actionability:**
    *   Monitor the distribution across decision types to understand engagement rates for each disaggregation mode.
    *   Sudden changes in ratios might indicate configuration issues, changes in workload patterns, or problems with the decision logic.

### `pd_decision_total` (deprecated)

> **Deprecated:** Use `disagg_decision_total` instead.

*   **Type:** Counter
*   **Labels:**
    *   `model_name`: string (the target model name, or "unknown" if empty)
    *   `decision_type`: string ("decode-only" or "prefill-decode")
*   **Release Stage:** ALPHA
*   **Description:** Counts the number of requests processed, broken down by the Prefill/Decode disaggregation decision. This metric only covers P/D disaggregation and does not account for encode disaggregation.

> [!NOTE]
> This metric is maintained for backward compatibility with the deprecated
> `pd-profile-handler`. New deployments should use `disagg_decision_total`.

## Opt-in ext_proc Stream Metrics

Three metrics covering ext_proc gRPC stream lifecycle. Disabled by default; enable with `--enable-grpc-stream-metrics`. These metrics are emitted under the `llm_d_epp_` prefix (separate from `llm_d_inference_scheduler_*`).

### `extproc_streams_inflight`

*   **Type:** Gauge
*   **Release Stage:** ALPHA
*   **Description:** Number of ext_proc gRPC streams currently open.
*   **Usage:** Sized at one stream per Envoy worker per EPP backend. A persistent increase under steady load indicates streams are being opened faster than they close.

### `extproc_stream_duration_seconds`

*   **Type:** Histogram
*   **Release Stage:** ALPHA
*   **Description:** Duration an ext_proc gRPC stream stays open, in seconds.
*   **Usage:** Long-lived streams are normal; the histogram surfaces the distribution. A sudden shift toward short durations can indicate Envoy reconnecting due to handler errors.

### `extproc_streams_total`

*   **Type:** Counter
*   **Labels:**
    *   `code`: string — the gRPC status code at stream close (`OK`, `Canceled`, `DeadlineExceeded`, `Internal`, ...). Bare `context.Canceled` and `context.DeadlineExceeded` are classified to their canonical codes rather than collapsing into `Unknown`.
*   **Release Stage:** ALPHA
*   **Description:** Total ext_proc gRPC streams completed, by gRPC status code.
*   **Usage:** Rate of `code="OK"` is the healthy stream-completion rate. A rising rate of `code="Internal"` or `code="Unknown"` indicates handler errors. `code="Canceled"` is expected on Envoy restarts and rolling EPP updates.

## In-flight load metrics

In-flight load, emitted under the `llm_d_epp_` prefix. Present only when an `InFlightLoadProducer` is
configured: the producer owns these metrics and registers them through the plugin metrics recorder. The
per-endpoint gauges read the producer's live per-endpoint counters (the same source as the
`/debug/plugins/state` dump and the token-load scorer); the per-model `request_inflight` gauge is moved
by the producer as requests are admitted and completed.

### `inflight_requests`

*   **Type:** Gauge
*   **Labels:**
    *   `endpoint_name`: string — the target endpoint (pod) name.
    *   `namespace`: string — the endpoint's namespace.
    *   `producer_name`: string — the configured `InFlightLoadProducer` instance name, so multiple producers emit distinct series.
*   **Release Stage:** ALPHA
*   **Description:** Requests currently in flight on each endpoint (scheduled, not yet completed), as tracked by the in-flight load producer.
*   **Usage:** Per-replica queue depth for load-aware routing and capacity analysis. Unlike the per-model `request_inflight` gauge (admitted-but-not-completed, aggregated by model), this is broken down by endpoint so it shows which replica is loaded.

### `inflight_tokens`

*   **Type:** Gauge
*   **Labels:**
    *   `endpoint_name`: string — the target endpoint (pod) name.
    *   `namespace`: string — the endpoint's namespace.
    *   `producer_name`: string — the configured `InFlightLoadProducer` instance name.
*   **Release Stage:** ALPHA
*   **Description:** Tokens currently in flight on each endpoint — uncached prompt tokens, optionally plus estimated output tokens when the producer's `addEstimatedOutputTokens` is set.
*   **Usage:** Per-replica token pressure, a finer load signal than request count when request sizes vary widely.

### `request_inflight`

*   **Type:** Gauge
*   **Labels:**
    *   `model_name`: string — the model named in the request body.
    *   `target_model_name`: string — the target model after traffic split.
    *   `fairness_id`: string — the flow-control fairness queue identity.
    *   `priority`: string — the request priority.
*   **Release Stage:** ALPHA
*   **Description:** Requests admitted to the endpoint picker but not yet completed, aggregated by model.
*   **Usage:** Picker-wide concurrency by model. Unlike the per-endpoint `inflight_requests` gauge, this is aggregated across endpoints, so it answers "how much is in flight for this model" rather than "which replica is loaded".
