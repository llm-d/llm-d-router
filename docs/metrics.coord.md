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

The `step` and `phase` labels share most of their values, but measure different boundaries: a step is a pipeline stage (which may make multiple calls), while a phase is a single backend call.

- **`step`**: A stage of the internal pipeline, observed once per request per stage. Covers local work and all backend calls made by that stage. Values: `render`, `replace-media-urls`, `encode`, `prefill`, `conditional-decode`, `decode`. See [Coordinator Architecture](coordinator_architecture.md).
- **`phase`**: A single outbound backend call. Values: `encode`, `prefill`, `decode`, `conditional-decode`.
- **`decision_type`**: The sequence of phases a request actually executed. Values: `decode-only`, `prefill-decode`, `encode-prefill-decode`.

## Error classes

The `error_code` label on `request_error_total` and `pipeline_step_errors_total` uses four values:

- **`bad_request`**: Client-side errors (e.g., malformed body).
- **`upstream_4xx`**: 4xx errors from upstream (e.g., render, prefill, encode).
- **`upstream_5xx`**: 5xx errors from upstream.
- **`internal`**: All other coordinator-internal faults.

**Key behaviors:**
- The conditional-decode probe's HTTP 412 (cache miss) is handled internally and is **not** an error. It is tracked separately by [`decode_cache_lookups_total`](#decode_cache_lookups_total).
- `upstream_4xx` and `upstream_5xx` only apply to the `render`, `prefill`, and `encode` steps. The `decode` and `conditional-decode` steps proxy responses directly to the client, so upstream failures during decode currently reach the client without incrementing these error metrics.

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

Recorded by the pipeline executor, one observation per step per request. This family measures each stage's total wall time (local work, orchestration, and all backend calls made by the stage). It answers where time goes inside the coordinator and which internal stage failed.

| Name | Type | Notes |
|---|---|---|
| `pipeline_step_duration_seconds` | Histogram | Per-step latency. Uses fine-grained `stepLatencyBuckets` (100us to 60s) to capture both microsecond steps (render) and seconds-long steps (decode). |
| `pipeline_step_errors_total` | Counter | Step failures; adds label `error_code`. |

**Key details:**
*   Only steps that actually executed are recorded. Trailing steps from early exits (e.g., conditional-decode hit) or failures are not emitted.

### Upstream phase family

Label set `{phase}` (one outbound backend call, not a pipeline stage, see [Labels](#labels)).

Recorded by the encode, prefill, and decode steps. This family counts and times the outbound sub-requests to the gateway. (Render and media fetches are not included, their latency is tracked in `pipeline_step_duration_seconds`).

| Name | Type | Notes |
|---|---|---|
| `upstream_request_total` | Counter | Gateway sub-requests, one per call. (e.g. encode contributes one per image). |
| `upstream_request_duration_seconds` | Histogram | Latency of one encode call (encode only). |

**Key details:**
*   **Fan-out:** A single client request can result in multiple backend calls (e.g., one conditional-decode probe, two encode calls for two images, one prefill, one decode).
*   **Narrower scope than steps:** 
    *   `upstream_request_duration_seconds` is encode-only because it tracks per-image latency. Prefill and decode already have a 1:1 mapping between call latency and step duration, so their timings are tracked solely in `pipeline_step_duration_seconds`.
    *   There is no `upstream_request_error_total`. A phase-call failure aborts its step and is already counted by `pipeline_step_errors_total`.
*   **Conditional-decode:** The `phase="conditional-decode"` counts the cache probe. It gets its own phase to distinguish it from the actual `decode` phase, maintaining accurate fan-out ratios.

### Disaggregation decision and decode cache

#### `disagg_decision_total` (Counter)
*   **Labels:** `model_name`, `decision_type` (`decode-only`, `prefill-decode`, `encode-prefill-decode`)
*   **Description:** Records which phases actually ran for a request. Unlike EPP, it lacks `encode-decode` because encode always implies prefill in the coordinator.

#### `decode_cache_lookups_total` (Counter)
*   **Labels:** `result` (`hit` or `miss`)
*   **Description:** Records the outcome of the conditional-decode probe. 
    *   `hit`: The decode worker already had the prompt in its KV cache and served the response directly. The pipeline stops early (`decode-only`).
    *   `miss`: The worker returned HTTP 412. The pipeline continues to encode/prefill/decode.
*   **Note:** Not redundant with `disagg_decision_total` since a cache miss does not indicate whether the subsequent path was `prefill-decode` or `encode-prefill-decode`.

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
