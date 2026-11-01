# Latency Reporter

**Type:** `latency-reporter`
**Interfaces:** `requestcontrol.ResponseHeaderProcessor`, `requestcontrol.ResponseBodyProcessor`
**Stability:** Alpha

Records each dispatched request's outcome, time to first token (TTFT) and time per output token (TPOT), labeled by served priority, prompt-length bucket and prefill/decode path.

The plugin holds no latency targets. It records what happened; thresholds and attainment ratios are chosen at query time in PromQL, so they can change without redeploying the EPP.

## What It Does

At end of stream the plugin records:

- **Outcome**, mirroring the values and precedence of `llm_d_epp_request_outcome_total`. The plugin classifies from the response status and how the stream ended, and sees only dispatched requests, so its counts are not expected to match that counter's:
  - `completed` is a 200 status with a natural end of stream.
  - `evicted` is a stream the EPP terminated to reclaim capacity. It wins over any status.
  - `failed` is a non-200 status, or a stream that ended with an error. A non-200 status wins over a disconnect.
  - `cancelled` is a stream torn down before the response finished, usually a client disconnect.
- **TTFT**: request receipt at the EPP to the first response byte, for completed requests. It includes Flow Control queue wait, prefill, KV transfer and the first decode step.
- **TPOT**: the per-request mean, `(complete - first token) / (output tokens - 1)`. It is recorded for completed streamed responses with at least two output tokens.

**Prompt bucket**: taken from the reported `usage.prompt_tokens`, falling back to the tokenized prompt.

**Path**:
- `remote`: the prefill profile picked an endpoint.
- `local`: the prefill profile ran or was declined without one.
- `aggregated`: no prefill profile exists.

Report latency per path as well as combined: a healthy aggregate can hide a slow remote path behind a fast local share.

Requests rejected before dispatch (`429`, `503`) never produce a response and are not recorded here. Count them from `llm_d_epp_request_outcome_total{outcome="rejected"}`.

The existing `llm_d_epp_request_ttft_seconds`, `llm_d_epp_request_streaming_tpot_seconds` and `llm_d_epp_request_outcome_total` already carry `fairness_id` and `priority`. This plugin adds the `prompt_bucket` and `path` dimensions they lack, and leaves per-tenant and per-model breakdowns to them. Its request counter carries the same labels as its TTFT histogram, because the counter is the denominator of attainment ratios over that histogram. A tenant or model label on the counter would have no matching latency series to divide into.

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
| `llm_d_epp_latency_requests_total` | Counter | `priority`, `prompt_bucket`, `path`, `streaming`, `outcome` |
| `llm_d_epp_latency_ttft_seconds` | Histogram | `priority`, `prompt_bucket`, `path`, `streaming` |
| `llm_d_epp_latency_tpot_seconds` | Histogram | `priority`, `path` |

The histograms record completed requests only. Their bucket bounds include common thresholds (TTFT 0.5, 1, 2, 4 s; TPOT 20, 30, 50 ms), so a threshold on a bound is exact in PromQL.

`streaming` is `true` when the model server answered with an event stream, and otherwise the client's `stream` flag, as on `llm_d_epp_request_ttft_seconds`. A `streaming="false"` TTFT equals end-to-end latency, so filter on `streaming="true"` for prefill latency. TPOT records streaming requests only and carries no `streaming` label.

### PromQL

p95 TTFT per prompt bucket and path, for streaming requests:

```promql
histogram_quantile(0.95, sum by (le, prompt_bucket, path) (rate(llm_d_epp_latency_ttft_seconds_bucket{priority="100",streaming="true"}[5m])))
```

Fraction of priority-100 streaming requests with TTFT at or under 2 s, over 30 days. The threshold is chosen here:

```promql
sum(increase(llm_d_epp_latency_ttft_seconds_bucket{priority="100",streaming="true",le="2"}[30d]))
/
sum(increase(llm_d_epp_latency_ttft_seconds_count{priority="100",streaming="true"}[30d]))
```

Failure ratio per path, counting evictions as failures:

```promql
sum by (path) (rate(llm_d_epp_latency_requests_total{outcome=~"failed|evicted"}[5m]))
/
sum by (path) (rate(llm_d_epp_latency_requests_total{outcome!="cancelled"}[5m]))
```

A request counts toward the TTFT fraction only if it completed. To score latency and failures together, divide the TTFT-under-threshold count by `llm_d_epp_latency_requests_total{streaming="true",outcome!="cancelled"}` instead of the histogram count. Whether a client disconnect counts against the objective is a policy choice; this query leaves it out.

When the remote path is slow, split it by stage using the sidecar's `llm_d_disagg_sidecar_prefill_duration_seconds` and the KV-transfer metrics.

## Limits

- For a non-streamed response the first byte arrives with the whole answer, so its TTFT equals end-to-end latency. It is recorded under `streaming="false"`.
- TPOT needs token counts in the response. A streaming client that doesn't set `stream_options.include_usage` reports no usage, so only TTFT is recorded.
