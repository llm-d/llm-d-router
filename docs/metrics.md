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
