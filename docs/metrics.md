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

## Flow Control Metrics

Exposed when the `flowControl` feature gate is enabled. All carry the `llm_d_epp_` prefix.

### `flow_control_request_queue_duration_seconds`

*   **Type:** Histogram
*   **Labels:**
    *   `fairness_id`: string (the tenant or flow identifier for fairness rotation)
    *   `priority`: string (the priority band, e.g., "0", "10")
    *   `outcome`: string (`Dispatched`, `RejectedCapacity`, `RejectedOther`, `EvictedTTL`, `EvictedContextCancelled`, `EvictedOther`)
    *   `inference_pool`: string
    *   `model_name`: string
    *   `target_model_name`: string
*   **Release Stage:** ALPHA
*   **Description:** Total time a request spends in the Flow Control layer, from enqueue to final outcome.
*   **Usage:** Primary latency signal for flow control. Rising p99 indicates backends are saturated or capacity limits are too tight.

### `flow_control_dispatch_cycle_duration_seconds`

*   **Type:** Histogram
*   **Release Stage:** ALPHA
*   **Description:** Time taken for each internal dispatch cycle.
*   **Usage:** Measures the overhead of the dispatch loop itself. Rising values indicate increasing cost per cycle from saturation detection, priority band iteration, or fairness evaluation.

### `flow_control_request_enqueue_duration_seconds`

*   **Type:** Histogram
*   **Labels:**
    *   `fairness_id`: string (the tenant or flow identifier)
    *   `priority`: string (the priority band)
    *   `outcome`: string
*   **Release Stage:** ALPHA
*   **Description:** Time taken to enqueue a request into the Flow Control layer.
*   **Usage:** Measures the time spent in capacity checks and queue insertion within the processor.

### `flow_control_queue_size`

*   **Type:** Gauge
*   **Labels:**
    *   `fairness_id`: string (the tenant or flow identifier)
    *   `priority`: string (the priority band)
    *   `inference_pool`: string
    *   `model_name`: string
    *   `target_model_name`: string
*   **Release Stage:** ALPHA
*   **Description:** Current number of requests actively held in the Flow Control queue.
*   **Usage:** Tracks queue depth per priority band and tenant. A steadily growing value indicates the dispatch rate is lower than the arrival rate.

### `flow_control_queue_bytes`

*   **Type:** Gauge
*   **Labels:**
    *   `fairness_id`: string (the tenant or flow identifier)
    *   `priority`: string (the priority band)
    *   `inference_pool`: string
    *   `model_name`: string
    *   `target_model_name`: string
*   **Release Stage:** ALPHA
*   **Description:** Current total size in bytes of requests actively held in the Flow Control queue.
*   **Usage:** Tracks memory pressure from queued requests. Compare against the configured `maxBytes` capacity to gauge how close a band is to rejecting new requests.

### `flow_control_pool_saturation`

*   **Type:** Gauge
*   **Labels:**
    *   `inference_pool`: string
*   **Release Stage:** ALPHA
*   **Description:** Current saturation level of the inference pool (0.0 = empty, 1.0 = fully saturated).
*   **Usage:** When saturation reaches the usage limit threshold, the dispatch cycle skips dispatching and requests remain queued. Sustained 1.0 indicates all backends are at capacity.


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
