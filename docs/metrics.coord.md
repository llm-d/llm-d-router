# Coordinator Metrics

The coordinator exposes Prometheus metrics describing the requests it accepts and the pipeline it
runs to serve them (see [Coordinator Architecture](coordinator_architecture.md)). They are separate
from the Endpoint Picker (EPP) metrics documented in [Metrics](metrics.md): the two components
measure different points in the same request path.

## Subsystems and naming

Every series carries the subsystem prefix `llm_d_coordinator_`, so a metric's full Prometheus name is
`llm_d_coordinator_<name>`. Names below omit the prefix. All metrics are ALPHA stage.

Naming mirrors the EPP request family (`pkg/epp/metrics`) so PromQL expressions and dashboard panels
translate between the two components. Where a name matches an EPP metric, the
[Relationship to EPP metrics](#relationship-to-epp-metrics) section states what differs.

`model_name` is taken from the request body; an empty or absent model is recorded as `unknown`.

## Scrape topology

### Coordinator metrics endpoint

Every metric on this page is exposed on a single `/metrics` endpoint served by the coordinator
process, alongside the inference paths and `/healthz` and `/readyz` on the coordinator's chi router
(`pkg/coordinator/server/server.go`). The registry it serves, the port or flag that selects it, and
whether it is authenticated are open. EPP's Prometheus wiring is
separate: it lives in `cmd/epp/runner` and serves the controller-runtime registry rather than the
default one.

### Metrics served elsewhere

All coordinator metrics are self-instrumented: the coordinator counts and times its own work
in-process, and scrapes nothing. Two series record a signal that originates upstream but is still
measured by the coordinator: `upstream_request_duration_seconds` times each encode sub-request, and
`decode_cache_lookups_total` records the decode server's response to the conditional-decode probe.

One client request passes through the coordinator, the EPP behind the gateway, and the vLLM workers,
and each of the three serves its own `/metrics`. Measurements this page does not list are on one of
the other two endpoints: per-request token counts and pool aggregates such as KV-cache utilization
and queue depth on EPP's (see [Metrics](metrics.md)), multimodal cache counters on vLLM's. Scrape all
three to follow a request end to end.

## Label vocabularies

Two vocabularies use the words `encode`, `prefill`, and `decode`. They are not interchangeable.

**`step`** is a stage of the coordinator's internal pipeline, observed once per request per stage. A
step covers local work, orchestration, and any backend calls that stage makes.

| Value | Stage |
|---|---|
| `render` | Call the render service to tokenize and apply the chat template. |
| `replace-media-urls` | Rewrite media URLs in the request body (local work only). |
| `encode` | Encode multimodal inputs; fans out one call per image. |
| `prefill` | Prefill sub-request, producing KV transfer parameters. |
| `conditional-decode` | Probe the decode cache for the 412 fast path. |
| `decode` | Decode and generation, streamed back to the client. |

**`phase`** is a single outbound backend call, counted once per call. Its values are `encode`,
`prefill`, and `decode`, matching `gateway.EPPProfileHeader`.

A step may issue zero or many backend calls and includes local work; a phase is one call. Only
`render` and `replace-media-urls` have no phase counterpart.

`conditional-decode` issues a decode-phase call: `steps/decode_proxy.go` `newDecodeProxyRequest`
stamps `EPPProfileHeader=PhaseDecode` unconditionally, and both the probe and the decode step go
through it, so a cache-miss request sends two decode-phase calls. See the
[open decision](#upstream-phase-family) on the phase family.

## Error classes

`error_code` on `request_error_total` and `pipeline_step_errors_total` uses one coarse set of values:
`bad_request`, `upstream_4xx`, `upstream_5xx`, `internal`.

This classification is finer than `server/handlers.go` `classifyPipelineError`, which collapses
upstream 5xx faults and internal faults into HTTP 502. The metrics helper inspects the error type
directly: an `UpstreamError` carrying a 5xx `StatusCode` is `upstream_5xx`, and any other
non-`ErrBadRequest` error is `internal`.

The conditional-decode probe's 412 is not an error in any class. `steps/conditional_decode.go`
intercepts that status in `ModifyResponse` and converts it to a cache miss, so the step returns
`nil` and the pipeline continues; a hit returns `ErrPipelineDone`, a successful early exit. Neither
outcome reaches the classifier, which is why the probe has its own metric,
[`decode_cache_lookups_total`](#decode_cache_lookups_total). The same 412 does count as an error on
the EPP side, as `request_error_total{error_code="PreconditionFailed"}`.

`upstream_4xx` and `upstream_5xx` therefore describe the render, prefill, and encode steps, the only
ones that build an `UpstreamError` from a non-2xx response. The conditional-decode and decode steps
proxy the upstream response straight to the client without constructing an error, so an upstream
failure on the decode leg reaches the client but is absent from both error metrics.

## Metrics catalog

### Request family

Recorded in `server/handlers.go` `handleInference`. Requests that exit early because they are
malformed (body-read error, 413, invalid JSON) count as `bad_request` with `model_name=unknown`.
Unless otherwise noted, metrics in this family share the label `{model_name}`.

| Name | Type | Notes |
|---|---|---|
| `request_total` | Counter | Every inbound client request, including malformed ones. |
| `request_error_total` | Counter | Failed requests by coarse class; adds label `error_code`. |
| `request_duration_seconds` | Histogram | End-to-end request latency; `generalLatencyBuckets` (5ms to 1h). |
| `request_size_bytes` | Histogram | Request body length; powers-of-2 buckets. |
| `response_size_bytes` | Histogram | Bytes streamed to the client, measured by a counting `ResponseWriter` wrapper in the handler rather than by parsing the body. |
| `request_running` | Gauge | Requests in flight; unlabelled. |

`request_running` is fleet-wide while the rest of the family is labelled by `model_name`, so a
dashboard panel that groups the family by model has no series for it. EPP's
`llm_d_epp_request_running` is labelled by model, fairness ID, and priority.

### Pipeline step family

Recorded in `pipeline.go` `Execute`, one observation per step per request. This family measures each
stage's whole wall time, local work plus orchestration plus any backend calls the stage makes, not
the individual outbound calls. It answers where time goes inside the coordinator and which internal
stage failed. Metrics in this family share the label `{step}`.

| Name | Type | Notes |
|---|---|---|
| `pipeline_step_duration_seconds` | Histogram | Per-step latency, from the timings the pipeline already computes. |
| `pipeline_step_errors_total` | Counter | Step failures by class; adds label `error_code`. |

`pipeline_step_duration_seconds` uses a dedicated bucket set, not `generalLatencyBuckets`: steps span
microseconds (render, a skipped encode) to seconds (upstream-bound decode), and a 5ms floor collapses
every fast step into one bucket. `stepLatencyBuckets` is 100us, 250us, 500us, 1ms, 2.5ms, 5ms, 10ms,
25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s, 60s.

`pipeline.Execute` pre-sizes its timings slice for all steps (`make([]stepTiming, len(p.steps))`). On
an early exit (`ErrPipelineDone`, which every conditional-decode hit takes) or a step failure, the
trailing entries stay zero-valued, so iterating the slice emits a `step=""` series with a 0s
observation for every step that never ran. Record inside the loop, or skip entries with an empty
name.

Each step that ran expands to its own histogram block. For three decode steps taking 1.9s, 2.1s and
1.7s:

```
..._bucket{step="decode",le="1"}    0
..._bucket{step="decode",le="2.5"}  3
..._bucket{step="decode",le="+Inf"} 3
..._sum{step="decode"}   5.7
..._count{step="decode"} 3
```

p95 of the decode step:

```promql
histogram_quantile(0.95, sum by (le) (rate(
  llm_d_coordinator_pipeline_step_duration_seconds_bucket{step="decode"}[5m])))
```

### Upstream phase family

Recorded in `steps/encode.go`, `prefill.go` and `decode_proxy.go`. This family counts and times the
outbound sub-requests the coordinator sends to the gateway carrying the epp-profile header. Only
gateway phase calls count: the render-service call and media-URL fetches are outbound too but carry
no phase header, and their latency appears in
`pipeline_step_duration_seconds{step="render"|"replace-media-urls"}`. Metrics in this family share the label `{phase}`.

| Name | Type | Notes |
|---|---|---|
| `upstream_request_total` | Counter | Gateway sub-requests, one per call; encode contributes one per image. |
| `upstream_request_duration_seconds` | Histogram | Per-call encode latency (encode only). |

The family is deliberately narrower than the step family it sits beside, because for prefill and
decode one step is one gateway call:

- `upstream_request_total` covers all three phases: the backend fan-out ratio
  (`upstream_request_total / request_total`) is not derivable from the step family, and it costs
  three series.
- `upstream_request_duration_seconds` is encode-only: encode fans out N concurrent calls, so
  `pipeline_step_duration_seconds{step="encode"}` is the fan-out envelope (slowest call plus the EC
  merge) and hides per-image call latency. For prefill and decode the call latency already equals the
  step duration. Two loose ends remain: the `phase` label then carries exactly one value, which
  invites empty `phase="prefill"` queries, and `generalLatencyBuckets` runs to 1h, far past any
  single encode call. Either drop the label and name the metric for what it measures, or keep it
  explicitly for forward compatibility, and consider `stepLatencyBuckets`.
- There is no `upstream_request_error_total`. A phase-call failure aborts its step and is already
  counted by `pipeline_step_errors_total` under the same coarse class. Per-call encode fan-out
  failure granularity is the one gap, but the errgroup context cancels sibling calls on the first
  error, so a clean per-call failure count is not reliably available. This is a limitation, not a
  metric.

**Open decision:** `phase="decode"` counts two different calls. The conditional-decode probe carries
the same `PhaseDecode` header as the decode step, so a cache miss produces two `phase="decode"`
increments for one client request, which makes the fan-out reading wrong. Pick one before
implementing: add a `conditional-decode` value to the `phase` label (diverges from the header
vocabulary, but each call type stays countable, and this is the recommended option); skip the probe
in `upstream_request_total` (loses probe-call visibility, though `decode_cache_lookups_total` still
counts probes); or accept the double count and document it.

Inbound against outbound, for one multimodal chat request with two images in a pipeline without a
conditional-decode step:

```
request_total{model_name="llama"}       += 1
upstream_request_total{phase="encode"}  += 2
upstream_request_total{phase="prefill"} += 1
upstream_request_total{phase="decode"}  += 1
```

### Disaggregation decision and decode cache

#### `disagg_decision_total`

*   **Type:** Counter
*   **Labels:** `model_name`, `decision_type` (`decode-only`, `prefill-decode`, `encode-prefill-decode`)
*   **Description:** Which phases actually ran.

`disagg_decision_total`'s value set is narrower than EPP's, which also has `encode-decode`. In the
coordinator encode always implies prefill, because encode produces the EC transfer parameters prefill
consumes, so encode without prefill is not a reachable path.

#### `decode_cache_lookups_total`

*   **Type:** Counter
*   **Labels:** `result` (`hit` or `miss`)
*   **Description:** Conditional-decode probe outcome.

`decode_cache_lookups_total` is emitted by exactly one place, the conditional-decode step, once per
request that runs it. The step optimistically sends the request to the decode worker with
`Prefer: if-available`. On a `hit` the worker already held the prompt in its KV cache and streamed
the response directly, so the pipeline stops early and the request's disaggregation path is
decode-only. On a `miss` the worker returns 412 and the pipeline continues through the full
encode/prefill/decode path. If the pipeline is configured without a conditional-decode step, the
metric never moves.

`hit / (hit + miss)` is the fraction of probed requests served straight from the decode worker's
cache; `hit / request_total` is the share of all client requests. The name follows the
`*_cache_*_total{result}` convention used by EPP's `encoder_cache_queries_total` and vLLM's
`mm_cache_queries` rather than repeating the step name.

The two are not redundant: on a miss, `decode_cache_lookups_total` cannot tell whether the request
then went `prefill-decode` or `encode-prefill-decode`, which is what `disagg_decision_total` records.

## Relationship to EPP metrics

| Coordinator metric | EPP counterpart | Difference |
|---|---|---|
| `request_total`, `request_error_total`, `request_duration_seconds`, `request_size_bytes`, `response_size_bytes` | `llm_d_epp_*`, same names | The coordinator counts client requests at its own entry point; EPP counts what reaches the gateway, which for one client request includes each phase sub-request. |
| `request_running` | `llm_d_epp_request_running` | The coordinator series is unlabelled; EPP labels by model, fairness ID and priority. |
| `disagg_decision_total` | `llm_d_epp_disagg_decision_total` | Same name and same `model_name`/`decision_type` labels, and EPP adds plugin identity labels. EPP counts the routing decision its profile handler makes; the coordinator counts the phases it actually executed. |
| `pipeline_step_*`, `upstream_request_*`, `decode_cache_lookups_total` | none | Coordinator-only. |

EPP exposes many metrics the coordinator does not: token counts, TTFT/TPOT/ITL, `scheduler_*`,
`plugin_duration_seconds`, pool aggregates, `flow_control_*`, `extproc_*` and `datalayer_*`. These
measure responsibilities the coordinator does not have (endpoint scheduling, flow control, pool
observation, ext_proc streaming).

## Deliberate omissions

### Token counts

The coordinator emits no token-count metrics. EPP owns per-request token accounting and the
coordinator does not duplicate it.

Input and prompt tokens are available cheaply in-process as `len(reqCtx.TokenIDs)` after the render
step (`steps/render.go` sets `TokenIDs` from the render service or from client-supplied token IDs).
Output and cached tokens are not: the decode step is a pure byte proxy (`steps/decode.go` streams
`proxy.ServeHTTP` straight to the client), so extracting them means intercepting and parsing the
streamed SSE, exactly the invasiveness `response_size_bytes` avoids by counting bytes.

There is no coverage gap in coordinator-deployment mode. The decode phase call goes out through the
gateway, where EPP parses the vLLM `usage` block and emits `request_input_tokens`,
`request_output_tokens` and `request_cached_tokens`. The one coordinator-native cheap signal, prompt
tokens, overlaps EPP's; its only unique value is correlating prompt size with encode fan-out or
per-step latency without a cross-component join. That is a candidate, not a commitment:

| Candidate | Type | Source |
|---|---|---|
| `request_prompt_tokens` | Histogram | `len(reqCtx.TokenIDs)` after render; token-count buckets. Label: `model_name`. |

### Per-request image visibility

Neither the coordinator nor EPP exposes per-request image count or image size. Candidates, listed so
the gap is documented:

| Candidate | Type | Source |
|---|---|---|
| `request_images` | Histogram | `len(reqCtx.MultimodalEntries)`; small-integer buckets. Label: `model_name`. |
| `image_placeholder_tokens` | Histogram | `MultimodalEntry.Placeholder.Length`; token-count buckets. Label: `model_name`. |

Image count is visible only indirectly, as `upstream_request_total{phase="encode"}`, which is
aggregate rather than per-request. Image byte size is not reliably available at all: inline base64
images fall under `request_size_bytes`, but URL images are fetched by `replace-media-urls` and never
traverse the coordinator. Placeholder tokens are the coordinator's only uniform,
transport-independent measure of image size, and are already computed in render for the
`max_total_placeholder_tokens` check.

## Cardinality

`model_name` labels five of the six request-family metrics, three of them histograms, and
`server/handlers.go` takes it straight from the request body (`model, _ := parsed["model"].(string)`)
without validation. An arbitrary set of client-supplied strings multiplied by the latency buckets is
a cardinality-explosion vector against the coordinator's own memory, reachable by any client. The
mitigation has to be chosen before implementing: allowlist against configured models, cap the
distinct values as EPP does, or accept the risk explicitly.

## Related documentation

- [Coordinator Architecture](coordinator_architecture.md) - the pipeline and steps these metrics measure
- [Metrics](metrics.md) - EPP metrics
- [Disaggregation](disaggregation.md) - the encode, prefill and decode phases
