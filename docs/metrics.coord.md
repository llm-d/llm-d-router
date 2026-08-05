# Coordinator Metrics

The coordinator exposes Prometheus metrics describing the requests it accepts and the pipeline it
runs to serve them (see [Coordinator Architecture](coordinator_architecture.md)). They are separate
from the Endpoint Picker (EPP) metrics documented in [Metrics](metrics.md): the two components
measure different points in the same request path.

## Subsystem and naming

A metric's full Prometheus name is `<subsystem>_<name>`. The coordinator uses a single subsystem:

| Prefix | Scope |
|---|---|
| `llm_d_coordinator_` | Every coordinator metric: request, pipeline step, upstream phase, disaggregation decision, and decode cache. |

Naming mirrors the EPP request family so PromQL expressions and dashboard panels
translate between the two components. Where a name matches an EPP metric, the
[Relationship to EPP metrics](#relationship-to-epp-metrics) section states what differs.

`model_name` is taken from the request body; an empty or absent model is recorded as `unknown`.

## Scrape topology

### Coordinator metrics endpoint

Every metric on this page is exposed on a single `/metrics` endpoint served by the coordinator
process, alongside the inference paths and `/healthz` and `/readyz`. EPP's metrics endpoint is
separate, served from its own process on the controller-runtime registry.

TBD: the registry the endpoint serves, its address, and whether it is authenticated. EPP authenticates
its endpoint by default (`--metrics-endpoint-auth`), which requires RBAC for TokenReview and
SubjectAccessReview; matching that would extend the coordinator's ServiceAccount permissions.

### Other scrape targets

All coordinator metrics are self-instrumented: the coordinator counts and times its own work
in-process, and scrapes nothing. Two series record a signal that originates upstream but is still
measured by the coordinator: `upstream_request_duration_seconds` times each encode sub-request, and
`decode_cache_lookups_total` records the decode server's response to the conditional-decode probe.

One client request passes through the coordinator, the EPP behind the gateway, and the vLLM workers,
and each of the three serves its own `/metrics`. Measurements this page does not list are on one of
the other two endpoints: per-request token counts and pool aggregates such as KV-cache utilization
and queue depth on EPP's (see [Metrics](metrics.md)), multimodal cache counters on vLLM's. Scrape all
three to follow a request end to end.

## Labels

Two labels use the words `encode`, `prefill`, and `decode`. They are not interchangeable.

The **`step` label** names a stage of the coordinator's internal pipeline, observed once per request
per stage. A step covers local work, orchestration, and any backend calls that stage makes. The
values are the built-in step names; [Coordinator Architecture](coordinator_architecture.md) describes
what each one does and how a pipeline is assembled from them.

| `step` value | Stage |
|---|---|
| `render` | Call the render service to tokenize and apply the chat template. |
| `replace-media-urls` | Rewrite media URLs in the request body (local work only). |
| `encode` | Encode multimodal inputs; fans out one call per image. |
| `prefill` | Prefill sub-request, producing KV transfer parameters. |
| `conditional-decode` | Probe the decode cache for the 412 fast path. |
| `decode` | Decode and generation, streamed back to the client. |

The **`phase` label** names a single outbound backend call, counted once per call. Its values are
`encode`, `prefill`, `decode`, and `conditional-decode`.

A step may issue zero or many backend calls and includes local work; a phase is one call. Only the
`render` and `replace-media-urls` steps have no phase counterpart.

The first three values match the epp-profile header the coordinator sends. `conditional-decode` does
not: the probe carries the decode profile header, like the decode step, and is distinguished by the
`Prefer: if-available` header it adds. See the [Upstream phase family](#upstream-phase-family).

## Error classes

`error_code` on `request_error_total` and `pipeline_step_errors_total` uses one set of four values:
`bad_request`, `upstream_4xx`, `upstream_5xx`, `internal`.

This classification is finer than the handler's client-facing status mapping, which collapses upstream
5xx faults and internal faults into HTTP 502. Classification inspects the error type directly: an
upstream error carrying a 5xx status is `upstream_5xx`, and any other error that is not a bad request
is `internal`.

The conditional-decode probe's 412 is not an error in any class. The step intercepts that status and
converts it to a cache miss, so it returns no error and the pipeline continues; a hit ends the
pipeline early, which is also not an error. Neither outcome reaches the classifier, which is why the
probe has its own metric,
[`decode_cache_lookups_total`](#decode_cache_lookups_total). The same 412 does count as an error on
the EPP side, as `request_error_total{error_code="PreconditionFailed"}`.

`upstream_4xx` and `upstream_5xx` therefore describe the render, prefill, and encode steps, the only
ones that turn a non-2xx response into an error. The conditional-decode and decode steps proxy the
upstream response straight to the client without constructing an error, so an upstream failure on the
decode leg reaches the client but is absent from both error metrics. This covers the worst case: a
worker that dies mid-generation truncates the client's response while the request still counts as a
success. Both failure modes are already detected by the decode proxy, which writes 502 when the call
fails before the response starts and logs a truncation when the copy fails after it, so closing the
gap is a matter of recording those two points rather than adding detection.

## Metrics catalog

Names below omit the subsystem prefix, which is `llm_d_coordinator_` throughout, and every metric is
ALPHA stage. Each family states the label set its metrics share; a metric that carries an extra label
says so in its own row.

### Request family

Label set `{model_name}` (the request's model).

Recorded by the request handler. Requests that exit early because they are malformed (body-read
error, 413, invalid JSON) count as `bad_request` with `model_name=unknown`.

| Name | Type | Notes |
|---|---|---|
| `request_total` | Counter | Every inbound client request, including malformed ones. |
| `request_error_total` | Counter | Failed requests; adds label `error_code`. |
| `request_duration_seconds` | Histogram | End-to-end request latency; `generalLatencyBuckets` (5ms to 1h). |
| `request_size_bytes` | Histogram | Request body length; powers-of-2 buckets. |
| `response_size_bytes` | Histogram | Bytes streamed to the client, measured by a counting `ResponseWriter` wrapper in the handler rather than by parsing the body. |
| `request_running` | Gauge | Requests in flight. |

EPP's `llm_d_epp_request_running` adds `fairness_id` and `priority`. Both come from flow control,
which the coordinator does not implement, so neither has a coordinator counterpart.

### Pipeline step family

Label set `{step}` (a stage of the coordinator's internal pipeline, see [Labels](#labels)).

Recorded by the pipeline executor, one observation per step per request. This family measures each
stage's whole wall time, local work plus orchestration plus any backend calls the stage makes, not
the individual outbound calls. It answers where time goes inside the coordinator and which internal
stage failed.

| Name | Type | Notes |
|---|---|---|
| `pipeline_step_duration_seconds` | Histogram | Per-step latency, from the timings the pipeline already computes. |
| `pipeline_step_errors_total` | Counter | Step failures; adds label `error_code`. |

`pipeline_step_duration_seconds` uses a dedicated bucket set, not `generalLatencyBuckets`: steps span
microseconds (render, a skipped encode) to seconds (upstream-bound decode), and a 5ms floor collapses
every fast step into one bucket. `stepLatencyBuckets` is 100us, 250us, 500us, 1ms, 2.5ms, 5ms, 10ms,
25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s, 60s.

Only steps that ran are observed. The executor pre-sizes its timings for the full step list, so on an
early exit (every conditional-decode hit) or a step failure the trailing entries are unset; recording
them would emit a `step=""` series with a 0s observation for every step that never ran.

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

Label set `{phase}` (one outbound backend call, not a pipeline stage, see [Labels](#labels)).

Recorded by the encode, prefill, and decode steps. This family counts and times the outbound
sub-requests the coordinator sends to the gateway carrying the epp-profile header. Only gateway phase
calls count: the render-service call and media-URL fetches are outbound too but carry no phase
header, and their latency appears in
`pipeline_step_duration_seconds{step="render"|"replace-media-urls"}`.

| Name | Type | Notes |
|---|---|---|
| `upstream_request_total` | Counter | Gateway sub-requests, one per call; encode contributes one per image. |
| `upstream_request_duration_seconds` | Histogram | Latency of one encode call, so one observation per image (encode only). |

One multimodal chat request with two images, in a pipeline whose conditional-decode probe misses,
counts once inbound and five times outbound:

```
request_total{model_name="llama"}                  += 1
upstream_request_total{phase="conditional-decode"} += 1
upstream_request_total{phase="encode"}             += 2
upstream_request_total{phase="prefill"}            += 1
upstream_request_total{phase="decode"}             += 1
```

Had the probe hit, it would be the only outbound call: the pipeline stops there and the encode,
prefill, and decode series do not move.

Dividing outbound calls by inbound requests gives the fan-out. Over a live window both sides are
rates, so the per-second units cancel and the result reads as calls per request:

```promql
# mean backend calls per client request, all phases: 5 if every request matched the example
sum(rate(llm_d_coordinator_upstream_request_total[5m]))
  / sum(rate(llm_d_coordinator_request_total[5m]))

# mean encode calls per client request, so mean images per request: 2 for the example
sum(rate(llm_d_coordinator_upstream_request_total{phase="encode"}[5m]))
  / sum(rate(llm_d_coordinator_request_total[5m]))
```

`sum` is needed on both sides because the two counters carry different labels (`phase` against
`model_name`), so they have no label pairs to divide against directly.

The family is deliberately narrower than the step family it sits beside, because for prefill and
decode one step is one gateway call:

- `upstream_request_total` covers all three phases: the fan-out ratios above are not derivable from
  the step family, and it costs three series.
- `upstream_request_duration_seconds` is encode-only, and is the per-image latency: encode fans out
  one concurrent call per image, so `pipeline_step_duration_seconds{step="encode"}` observes the
  whole stage once per request (the fan-out envelope: slowest call plus the EC merge) and hides what
  a single image cost. A step p95 well above the call p95 means the width of the fan-out, not any one
  image, is the expense. For prefill and decode the call latency already equals the step duration. Two
  loose ends remain. The `phase` label carries exactly one value, which invites empty
  `phase="prefill"` queries: either drop it and name the metric for what it measures, or keep it
  explicitly for forward compatibility. And `generalLatencyBuckets` runs to 1h, far past any single
  encode call, so `stepLatencyBuckets` fits the range better; note that `generalLatencyBuckets` is
  EPP's own set, so switching gives up bucket-for-bucket comparison with
  `llm_d_epp_request_duration_seconds`.
- There is no `upstream_request_error_total`. A phase-call failure aborts its step and is already
  counted by `pipeline_step_errors_total` under the same `error_code`. Per-call encode fan-out
  failure granularity is the one gap, but the errgroup context cancels sibling calls on the first
  error, so a clean per-call failure count is not reliably available. This is a limitation, not a
  metric.

`phase="conditional-decode"` counts the probe, which is a real call to the decode worker and on a hit
is the only backend call the request makes. It gets its own value rather than counting under
`decode`: without it, the miss above would report two decode calls for one client request. Its volume
matches `decode_cache_lookups_total`, which breaks the same probes down by outcome; carrying the
probe here keeps the fan-out ratio complete and every outbound call attributable to one phase.

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
| `request_total`, `request_error_total`, `request_duration_seconds`, `request_size_bytes`, `response_size_bytes`, `request_running` | `llm_d_epp_*`, same names | The coordinator counts client requests at its own entry point; EPP counts what reaches the gateway, which for one client request includes each phase sub-request. EPP also labels by `fairness_id` and `priority`, which have no coordinator counterpart. |
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

Input and prompt tokens are available cheaply in-process after the render step, which holds the token
IDs from the render service or from client-supplied token IDs. Output and cached tokens are not: the
decode step is a pure byte proxy that streams the response straight to the client, so extracting them
means intercepting and parsing the streamed SSE, exactly the invasiveness `response_size_bytes` avoids
by counting bytes.

There is no coverage gap in coordinator-deployment mode. The decode phase call goes out through the
gateway, where EPP parses the vLLM `usage` block and emits `request_input_tokens`,
`request_output_tokens` and `request_cached_tokens`. The one coordinator-native cheap signal, prompt
tokens, overlaps EPP's; its only unique value is correlating prompt size with encode fan-out or
per-step latency without a cross-component join. That is a candidate, not a commitment:

| Candidate | Type | Source |
|---|---|---|
| `request_prompt_tokens` | Histogram | Token count after render; token-count buckets. Label: `model_name`. |

### Per-request image visibility

Neither the coordinator nor EPP exposes per-request image count or image size. Candidates, listed so
the gap is documented:

| Candidate | Type | Source |
|---|---|---|
| `request_images` | Histogram | Multimodal entries per request; small-integer buckets. Label: `model_name`. |
| `image_placeholder_tokens` | Histogram | Placeholder length per entry; token-count buckets. Label: `model_name`. |

Image count is visible only indirectly, as `upstream_request_total{phase="encode"}`, which is
aggregate rather than per-request. Image byte size is not reliably available at all: inline base64
images fall under `request_size_bytes`, but URL images are fetched by `replace-media-urls` and never
traverse the coordinator. Placeholder tokens are the coordinator's only uniform,
transport-independent measure of image size, and are already computed in render for the
`max_total_placeholder_tokens` check.

## Cardinality

`model_name` labels every request-family metric, three of them histograms, and the handler takes it
straight from the request body without validation. Each distinct value creates its own set of series,
and a histogram multiplies that by its bucket count, so a client looping over invented model names
grows the coordinator's memory without bound. This is reachable by any client, and it is a mitigation
the implementation has to carry.

Capping the distinct values is the approach that fits: EPP caps `model_name` at 1000 over the process
lifetime and reports further values as `other`, and matching that keeps the two components
consistent. An allowlist has no source of truth here, since the coordinator's config carries no model
list and the coordinator is otherwise model-agnostic. The overflow value must not be `unknown`, which
already means the request carried no model at all.

## Related documentation

- [Coordinator Architecture](coordinator_architecture.md) - the pipeline and steps these metrics measure
- [Metrics](metrics.md) - EPP metrics
- [Disaggregation](disaggregation.md) - the encode, prefill and decode phases
