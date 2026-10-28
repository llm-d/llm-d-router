# Latency Reporter

**Type:** `latency-reporter`
**Interfaces:** `requestcontrol.ResponseHeaderProcessor`, `requestcontrol.ResponseBodyProcessor`
**Stability:** Alpha

Records each completed request's outcome, time to first token (TTFT) and time per output token (TPOT), labeled by served priority, prompt-length bucket and prefill/decode path.

The plugin holds no latency targets. It records what happened; thresholds and attainment ratios are chosen at query time in PromQL, so they can change without redeploying the EPP.

## What It Does

At end of stream the plugin records:

- **Outcome**:
  - `success` is a 200 status with a natural end of stream.
  - `failed` is any other status, or a stream that ended without completing (client disconnect, eviction, error).
- **TTFT**: request receipt at the EPP to the first response byte, for successful requests. It includes Flow Control queue wait, prefill, KV transfer and the first decode step.
- **TPOT**: the per-request mean, `(complete - first token) / (output tokens - 1)`. It is recorded for successful streamed responses with at least two output tokens.

**Prompt bucket**: taken from the reported `usage.prompt_tokens`, falling back to the tokenized prompt.

**Path**:
- `remote`: the prefill profile picked an endpoint.
- `local`: the prefill profile ran or was declined without one.
- `aggregated`: no prefill profile exists.

Report latency per path as well as combined: a healthy aggregate can hide a slow remote path behind a fast local share.

Requests rejected before dispatch (`429`, `503`) never produce a response and are not recorded here. Count them from `llm_d_epp_request_error_total` and `llm_d_epp_flow_control_requests_total{outcome}`.

The existing `llm_d_epp_request_ttft_seconds` and `llm_d_epp_request_streaming_tpot_seconds` already carry `fairness_id` and `priority`. This plugin adds the `prompt_bucket` and `path` dimensions they lack.

## Configuration

```yaml
plugins:
- type: latency-reporter
  parameters:
    promptBuckets: [1024, 8192]      # labels: le1024, le8192, gt8192
    prefillProfileName: prefill
    priorities: [100]                # optional; empty records every priority
```

| Field | Default | Description |
|---|---|---|
| `promptBuckets` | none | Strictly increasing upper bounds, in prompt tokens. With none, every request is in bucket `all`. |
| `prefillProfileName` | `prefill` | Scheduling profile that marks a remote prefill. Match the `disagg-profile-handler` profile name. |
| `priorities` | all | Served priorities to record. Limiting it bounds label cardinality. |

## Observability

| Metric | Type | Labels |
|---|---|---|
| `llm_d_epp_latency_requests_total` | Counter | `target_model_name`, `fairness_id`, `priority`, `prompt_bucket`, `path`, `outcome` |
| `llm_d_epp_latency_ttft_seconds` | Histogram | `priority`, `prompt_bucket`, `path` |
| `llm_d_epp_latency_tpot_seconds` | Histogram | `priority`, `path` |

The histograms record successful requests only. Their bucket bounds include common thresholds (TTFT 0.5, 1, 2, 4 s; TPOT 20, 30, 50 ms), so a threshold on a bound is exact in PromQL.

### PromQL

p95 TTFT per prompt bucket and path:

```promql
histogram_quantile(0.95, sum by (le, prompt_bucket, path) (rate(llm_d_epp_latency_ttft_seconds_bucket{priority="100"}[5m])))
```

Fraction of priority-100 requests with TTFT at or under 2 s, over 30 days. The threshold is chosen here:

```promql
sum(increase(llm_d_epp_latency_ttft_seconds_bucket{priority="100",le="2"}[30d]))
/
sum(increase(llm_d_epp_latency_ttft_seconds_count{priority="100"}[30d]))
```

Failure ratio per path:

```promql
sum by (path) (rate(llm_d_epp_latency_requests_total{outcome="failed"}[5m]))
/
sum by (path) (rate(llm_d_epp_latency_requests_total[5m]))
```

A request counts toward the TTFT fraction only if it succeeded. To score latency and failures together, divide the TTFT-under-threshold count by `llm_d_epp_latency_requests_total` instead of the histogram count.

When the remote path is slow, split it by stage using the sidecar's `llm_d_disagg_sidecar_prefill_duration_seconds` and the KV-transfer metrics.

## Limits

- For a non-streamed response the first byte arrives with the whole answer, so its TTFT equals end-to-end latency.
- TPOT needs token counts in the response. A streaming client that doesn't set `stream_options.include_usage` reports no usage, so only TTFT is recorded.
