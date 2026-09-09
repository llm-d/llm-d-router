# Coordinator Metrics

The coordinator exposes Prometheus metrics describing the requests it accepts and the pipeline it
runs to serve them (see [Coordinator Architecture](coordinator_architecture.md)). They are separate
from the Endpoint Picker (EPP) metrics documented in [Metrics](metrics.md): the two components
measure different points in the same request path.

## Subsystem and naming

A metric's full Prometheus name is `<subsystem>_<name>`. The coordinator uses a single subsystem:

| Prefix | Scope |
|---|---|
| `llm_d_coordinator_` | Canonical, coordinator-wide: request, pipeline step and latency, and the remaining coordinator metrics. |

Naming mirrors the EPP request family so PromQL expressions and dashboard panels
translate between the two components. Where a name matches an EPP metric, the
[Relationship to EPP metrics](#relationship-to-epp-metrics) section states what differs.

`model_name` is taken from the request body; an empty or absent model is recorded as `unknown`.

## Scrape topology

### Coordinator metrics endpoint

Every metric on this page is exposed on a single `/metrics` endpoint served by the coordinator
process on the metrics port (default 9090, configurable with `--metrics-port`), separate from the listener
carrying the inference paths and `/healthz` and `/readyz`. A non-positive port disables the
endpoint.

The endpoint serves HTTP when no certificate path is set. Set `--metrics-cert-path` or
`server.metrics_cert_path` to a directory containing `tls.crt` and `tls.key` to serve HTTPS.
If a certificate path is set, missing or invalid files stop the coordinator. The metrics listener does not fall back to HTTP.
Valid certificate changes take effect without restarting the coordinator.

The endpoint serves the shared controller-runtime registry, so controller-runtime's process
collectors appear alongside the coordinator metrics. It is unauthenticated. Authenticating it costs
RBAC for TokenReview and SubjectAccessReview on the coordinator's ServiceAccount.

### Other scrape targets

All coordinator metrics are self-instrumented: the coordinator counts and times its own work
in-process and scrapes no other component. Two of them describe upstream behavior but are still
measured locally, from the coordinator's side of the call: `upstream_request_duration_seconds` times
each outbound call, and `conditional_decode_probes_total` records how the decode worker answered the
conditional-decode probe.

One client request passes through the coordinator, the EPP behind the gateway, and the vLLM workers,
and each of the three serves its own `/metrics`. Following a request end to end means scraping all
three: what this page does not list is on EPP's endpoint (see [Metrics](metrics.md)) or vLLM's.

## Labels

The `step` and `upstream` labels carry the same values, but measure different boundaries: a step is a pipeline stage, while an upstream is a single outbound call. They diverge where a stage is not one call: the `encode` step fans out one concurrent sub-request per multimodal entry, so a request with six images records one `step="encode"` observation and six `upstream="encode"` observations.

- **`step`**: A stage of the internal pipeline, observed once per request per stage. Covers local work and all outbound calls made by that stage. The label value is the step's registered `Name()` (see `pipeline.Register`), so its cardinality is bounded by the set of registered pipeline steps. Values in the built-in registry: `render`, `replace-media-urls`, `encode`, `prefill`, `conditional-decode`, `decode`. See [Coordinator Architecture](coordinator_architecture.md).
- **`upstream`**: A single outbound call, whatever its destination. Values: `render`, `replace-media-urls`, `encode`, `prefill`, `conditional-decode`, `decode`. A step that gains an outbound call gains a value here.
- **`path`**: The sequence of disaggregation phases a request actually executed. Values: `decode-only`, `prefill-decode`, `encode-prefill-decode`. `encode-decode` is intentionally unreachable because encode implies prefill.

## Error classes

The `error_code` label on `request_error_total` and `step_errors_total` uses five values:

- **`bad_request`**: Client-side errors (e.g., malformed body).
- **`upstream_4xx`**: 4xx errors from upstream (e.g., render, prefill, encode).
- **`upstream_5xx`**: 5xx errors from upstream.
- **`upstream_transport`**: The round trip failed before a response arrived (connection refused, timeout, TCP reset), so no status code was received.
- **`internal`**: All other coordinator-internal faults. Not used for reachability failures, which are `upstream_transport`.

**Key behaviors:**
- The conditional-decode probe's HTTP 412 (the worker declined to serve it) is handled internally and is **not** an error. It is tracked separately by [`conditional_decode_probes_total`](#conditional_decode_probes_total-counter).
- `upstream_4xx`, `upstream_5xx` and `upstream_transport` are recorded for every step that calls out, including `decode` and `conditional-decode`: their reverse proxy captures the upstream status in `ModifyResponse` and transport failures in `ErrorHandler`, and both steps translate a 4xx/5xx or transport error into an `UpstreamStreamedError` before any bytes are forwarded. A transport failure carries no status code, so it classifies as `upstream_transport` instead of by status band. What stays uncountable is a failure that surfaces after streaming has begun, since the 200 and a partial body are already on the wire.

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
| `request_duration_seconds` | Histogram | End-to-end request latency; `GeneralLatencyBuckets` (5 ms to 1 h). |
| `request_size_bytes` | Histogram | Request body size; `RequestSizeBuckets` (64 B to 1 GiB, powers of two). |
| `request_input_tokens` | Histogram | Prompt token count, recorded after the render step; `TokenCountBuckets` (1 to ~1 M). |
| `request_running` | Gauge | Requests in flight. |

### Pipeline step family

Label set `{step}` (a stage of the coordinator's internal pipeline, see [Labels](#labels)).

Recorded by the pipeline executor, which brackets every step it runs. This family measures each stage's total wall time (local work, orchestration, and all backend calls made by the stage). It answers where time goes inside the coordinator and which internal stage failed.

| Name | Type | Notes |
|---|---|---|
| `step_duration_seconds` | Histogram | Per-step latency; `GeneralLatencyBuckets` (5 ms to 1 h). |
| `step_errors_total` | Counter | Step failures; adds label `error_code`. |
| `step_running` | Gauge | Requests currently executing the step. Saturation rather than latency: `decode` stays in flight for the whole stream, so its value is the number of active streams. |

**Key details:**
*   Only steps that actually executed are recorded. Trailing steps from early exits (e.g., conditional-decode hit) or failures are not emitted.

### Upstream call family

Label set `{upstream}` (one outbound call, not a pipeline stage, see [Labels](#labels)).

Recorded by every step that calls out: render to the renderer service, replace-media-urls to image URLs, and conditional-decode, encode, prefill, and decode to the gateway. The step family answers where time went in the pipeline; this family answers how many external calls a request made and how slow each one was.

| Name | Type | Notes |
|---|---|---|
| `upstream_request_total` | Counter | Outbound calls, one per call (encode contributes one per image, replace-media-urls one per URL). |
| `upstream_request_duration_seconds` | Histogram | Latency of one call; `GeneralLatencyBuckets` (5 ms to 1 h). |

**Key details:**
*   **Fan-out:** A single client request can result in multiple outbound calls (e.g., three image fetches, one conditional-decode probe, three encode calls, one prefill, one decode). For the fan-out upstreams this is the only per-call latency available, since the step duration covers the whole concurrent batch.
*   **No `upstream_request_error_total`:** A failed call aborts its step and is already counted by `step_errors_total`.
*   **Conditional-decode:** The probe gets its own value rather than counting as `decode`, keeping fan-out ratios accurate.

### Execution path and conditional decode

#### `execution_path_total` (Counter)
*   **Labels:** `model_name`, `path` (`decode-only`, `prefill-decode`, `encode-prefill-decode`)
*   **Description:** Records which set of disaggregation phases actually ran for a client request. `encode-decode` is intentionally unreachable because encode implies prefill in the coordinator.

#### `conditional_decode_probes_total` (Counter)
*   **Labels:** `result` (`served`, `deferred`, `error`, or `transport_error`)
*   **Description:** Counts conditional-decode probes by the worker's answer. The coordinator sees the status code, not the reason behind it.
    *   `served`: The worker handled the request itself, whether because the prompt was cached or too short to disaggregate. The pipeline stops early (`decode-only`).
    *   `deferred`: The worker returned HTTP 412. The pipeline continues to encode/prefill/decode.
    *   `error`: Any other 4xx/5xx status. The step returns an `UpstreamStreamedError` carrying that status, and the request is classified as `upstream_4xx` or `upstream_5xx`.
    *   `transport_error`: No response arrived (connection refused, timeout, TCP reset). The step returns an `UpstreamStreamedError` with no status code, and the request is classified as `upstream_transport`.
*   **Note:** Not redundant with `execution_path_total` since a deferred probe does not indicate whether the subsequent path was `prefill-decode` or `encode-prefill-decode`.

## Relationship to EPP metrics

| Coordinator metric | EPP counterpart | Difference |
|---|---|---|
| Request family (`llm_d_coordinator_request_total`, etc.) | `llm_d_epp_*` (same names) | Coordinator counts single client requests at entry; EPP counts every sub-request reaching the gateway. EPP adds flow-control labels (`fairness_id`, `priority`). |
| `llm_d_coordinator_execution_path_total` | `llm_d_epp_disagg_decision_total` | Coordinator observes the phases that actually ran on the client request; EPP records the routing decision that was made and adds plugin labels. |
| `llm_d_coordinator_step_*`, `llm_d_coordinator_upstream_request_*`, `llm_d_coordinator_conditional_decode_probes_total` | None | Unique to coordinator. |

**EPP-only metrics:** Scheduling, flow control, and pool aggregates have no coordinator counterpart. EPP's token counts and TTFT do, but they are per leg: the same prompt reaches EPP on more than one leg, and decode-leg TTFT starts after render, encode, and prefill have finished, so neither describes a client request. Only the coordinator sees a client request as one request. See [Deliberate omissions](#deliberate-omissions) for what it could report and why it does not today.

## Deliberate omissions

### Output and cached token counts

The coordinator emits no output or cached token-count metrics. Those values live in the vLLM `usage` block, and reading them means parsing the streamed SSE response, which the decode step does not do: it proxies bytes straight to the client. Input tokens are recorded (see [`request_input_tokens`](#request-family)): the renderer returns the token IDs, so the count is in hand after render, with no response parsing and no dependency on EPP.

### Time to first token

The coordinator emits no TTFT metric. Identifying the first token requires parsing the streamed
response, which the decode step does not do: it proxies bytes straight to the client. EPP measures
TTFT, but only per leg, since each phase reaches it as a separate request, so its decode-leg
`request_ttft_seconds` starts after render, encode, and prefill have already finished. Neither
component reports the client's full wait for output.

### Per-request image visibility

Neither the coordinator nor EPP tracks per-request image count or size. Image count is only visible in aggregate via `upstream_request_total{upstream="encode"}`. Size is available but unrecorded: `replace-media-urls` downloads each URL image and base64-encodes it, so every entry, inline or fetched, carries its payload through the coordinator.

*Future candidates:* 
*   `request_images`: Histogram of multimodal entries per request.
*   `image_bytes`: Histogram of decoded payload size per entry.
*   `image_placeholder_tokens`: Histogram of placeholder length per entry, the token-space counterpart to `image_bytes`.

## Cardinality

`model_name` labels every request-family metric and `execution_path_total`, which together are
every metric that carries it. The handler takes the value straight from the request body without
validation. Each distinct value creates its own set of series, and a histogram multiplies that by
its bucket count, so a client looping over invented model names grows the coordinator's memory
without bound. This is reachable by any client, and it is a mitigation the implementation has to
carry.

Capping the distinct values is the approach that fits: `model_name` is capped at 1000 over the
process lifetime and further values are reported as `other`. The bounded-label helper lives in
[`pkg/common/observability/metrics/cardinality.go`](../pkg/common/observability/metrics/cardinality.go)
(`BoundedLabel`, `OverflowValue = "other"`) and is instantiated with the same cap by both
[`pkg/coordinator/metrics/cardinality.go`](../pkg/coordinator/metrics/cardinality.go) and
[`pkg/epp/metrics/cardinality.go`](../pkg/epp/metrics/cardinality.go), so the two components share
one guard. An allowlist has no source of truth here, since the coordinator's config carries no model
list and the coordinator is otherwise model-agnostic. The overflow value must not be `unknown`,
which already means the request carried no model at all.

## Related documentation

- [Coordinator Architecture](coordinator_architecture.md) - the pipeline and steps these metrics measure
- [Metrics](metrics.md) - EPP metrics
- [Disaggregation](disaggregation.md) - the encode, prefill and decode phases
