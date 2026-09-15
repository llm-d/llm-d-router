# Request Attribution Metrics — Coordinator Implementation

**Authors**: Sima Nadler (_IBM_)

**Related Issue**: [llm-d/llm-d-router#2741](https://github.com/llm-d/llm-d-router/issues/2741)

## Summary

llm-d tracks token usage and infrastructure cost per model today but has no mechanism to
attribute those costs to the **business entities** that generate them: tenants, users, and
workloads. Without this, multi-tenant platform teams cannot do chargeback, enforce
per-tenant budgets, or identify which workloads are cost-inefficient.

This proposal introduces request attribution implemented by the [coordinator](../../README.coord.md) — tagging every
inference request with a `tenant_id`, `user_id`, and `workload_id` carried in request
headers and propagating those identities through the coordinator's observability stack.

A [separate proposal](./open-cost-new-dimensions-plan-coordinator.md) describes the enhancements to OpenCost for generating inference costs per tenant, workload and user based on the new metrics described in this proposal.


## Motivation

The existing cost tracking stack ([OpenCost](https://github.com/opencost/opencost/blob/develop/docs/inference-cost-tracking.md) + vLLM metrics) answers "what does it cost to
serve model X?" It cannot answer:

- Which team or customer is responsible for that cost?
- Which application, agent, or pipeline is driving GPU spend?
- Is a specific user consuming a disproportionate share of shared infrastructure?

### Goals

The primary goal of this proposal is for llm-d to generate the metrics needed to calculate inference costs
attributed by tenant, workload, and user — enabling chargeback, per-team budget
enforcement, and cost anomaly detection without changes to vLLM, the EPP, or the
Inference Gateway.

Supporting goals:

- **Aggregate attribution** — Prometheus histograms carry `tenant_id`, `workload_id`, and
  `serving_model` labels (and optionally `user_id`) so dashboards, alerts, and OpenCost
  billing queries work without a log aggregation backend.
- **Per-request audit** — a structured JSON log record is emitted for every request,
  always including all three identity fields and token counts, as the authoritative
  billing and audit record for deployments where Prometheus `user_id` cardinality is
  disabled.
- **Zero footprint on the critical path** — attribution is opt-in, degrades gracefully
  when headers are absent, and adds no latency visible to the client.

### Non-Goals

- Replacing or deprecating `x-llm-d-inference-fairness-id`. That header governs
  flow-control scheduling in the EPP and is unchanged.
- Forwarding attribution headers to vLLM or the EPP beyond what they already carry
  through normal header forwarding.
- Implementing or specifying client authentication or authorization.
- Per-request payload inspection or body-based routing.
- Real-time budget enforcement or request admission based on spend.
- Log aggregation infrastructure. The structured log record is written to the
  coordinator's standard logger; shipping it to a log backend is the operator's
  responsibility.

### Example Use Cases

- **Workload cost comparison** — an engineering team compares two agent versions
  side-by-side to see which costs less per million tokens, directly in dollars.
- **Platform team chargeback** — a platform team bills internal product teams by querying
  per-tenant cost breakdowns at the end of each month.
- **Per-user billing with unbounded users** — a SaaS platform serves thousands of end
  users and produces per-user invoices by querying the coordinator's structured log stream
  when Prometheus label cardinality makes per-user metrics impractical.
- **Cost governance and anomaly detection** — a platform team configures alerts on
  per-tenant token consumption; when a tenant exceeds its expected monthly budget
  mid-cycle, an alert fires before the bill arrives.
- **A/B model rollout cost comparison** — during a canary rollout, querying OpenCost by
  `serving_model` and `workload_id` shows whether the quality improvement justifies the
  higher cost per million tokens.

## Why the Coordinator

The coordinator is the single entry point for every client request in the llm-d
coordinator deployment model. It owns the full request lifecycle — from the inbound
OpenAI request through each disaggregated phase (encode → prefill → decode) to streaming
the final response — giving it natural access to attribution headers, token counts,
serving model, and end-to-end timing without any additional instrumentation hooks. It
also already has `pkg/coordinator/metrics`, a structured logger, and a `RequestContext`
that threads per-request state through the pipeline, so attribution slots in as a
first-class extension.

For a discussion of alternative components that were considered (IPP, EPP, gateway
filter, sidecar), see [Alternatives Considered](#alternatives-considered).

## Proposal

### Two Output Tiers

Attribution data is produced at two granularities:

| Tier | Mechanism | Granularity | `user_id` | Primary use case |
|---|---|---|---|---|
| **Tier 1** | Prometheus metric labels | Aggregate | Configurable | Real-time dashboards, alerting, chargeback |
| **Tier 2** | structured JSON log | Per-request, 100% | Always | Complete billing audit, per-user attribution |

**Tier 1 — Prometheus metric labels** are the right tool for questions like "how many
tokens did `acme-corp` consume this hour?" They feed directly into Grafana dashboards
and OpenCost billing queries. Adding `user_id` as a label on high-cardinality deployments
causes Prometheus time-series explosion, so it is controlled by a configuration gate.

**Tier 2 — Structured JSON log** is the complete authoritative attribution record. It is
emitted for every request, always includes all three identity fields and token counts, and
is the sole required mechanism for accurate per-user billing and audit.

### Attribution Headers

Three request headers, all in the `x-llm-d-*` namespace:

| Header | Description | Example value |
|---|---|---|
| `x-llm-d-tenant-id` | Organizational unit responsible for the request | `acme-corp` |
| `x-llm-d-user-id` | Individual end user within the tenant | `user-7f3a` |
| `x-llm-d-workload-id` | Application, agent, or pipeline submitting the request | `coding-agent-v2` |

All three are optional. Omitted headers result in the literal value `unknown` in metric
labels and log fields.

> [!WARNING]
> **Trust Boundary**: In a production deployment, end-users must never be permitted to
> self-assert these values. These headers must be stripped from external requests and
> injected by a trusted upstream component (API gateway, identity proxy, or service mesh
> policy) based on verified credential data. See [Appendix A](#appendix-a-populating-attribution-headers-in-production).

### Coordinator Metric Labels (Aggregate Attribution)

Two new histogram metric families are added under `llm_d_coordinator`:

| Metric | Type | Labels (always) | Labels (conditional) | Description |
|---|---|---|---|---|
| `llm_d_coordinator_request_input_tokens_attributed` | Histogram | `model_name`, `tenant_id`, `workload_id`, `serving_model` | `user_id` | Prompt token count per request with attribution dimensions |
| `llm_d_coordinator_request_output_tokens_attributed` | Histogram | `model_name`, `tenant_id`, `workload_id`, `serving_model` | `user_id` | Completion token count per request with attribution dimensions |

`model_name` is the requested model from the request body (same as the existing
`requestInputTokens` histogram). `serving_model` is the model name returned by vLLM in
the decode response body — the authoritative OpenCost join key, which may differ from
`model_name` when a model-selector rewrite is in effect. When `serving_model` is
unavailable (error path, non-streaming response body not yet parsed), it defaults to
`model_name`.

`tenant_id` and `workload_id` are always present as labels (bounded sets — safe for
Prometheus). `user_id` is **conditionally** included, controlled by coordinator
configuration:

```yaml
requestAttribution:
  enabled: true
  prometheusUserIdLabel: true   # default: true
                                # set false for deployments with unbounded
                                # end-user populations to avoid cardinality explosion
```

> [!WARNING]
> **Cardinality**: Do not enable `prometheusUserIdLabel` for public-facing deployments
> with unbounded end-user populations without first assessing `user_id` cardinality.
> Bounded requestor populations (service accounts, teams, named integrations) are safe.
> Individual human end-users at scale are not. Per-user attribution remains fully
> available via the structured log regardless of this setting.

The existing `llm_d_coordinator_request_input_tokens` histogram (model-only label,
measured after render) is **not changed**. The new attributed histograms are additive.
OpenCost queries the `_sum` series of the new histograms using the `last_over_time`
delta pattern described in
[`open-cost-new-dimensions-plan-coordinator.md`](open-cost-new-dimensions-plan-coordinator.md).

### Structured Log Record (Per-Request Audit)

A structured JSON record is emitted by the coordinator at request completion for **every
request**, regardless of Prometheus label configuration. This is the authoritative billing
and audit record.

```json
{
  "level": "info",
  "ts": "2025-07-01T14:23:01.123Z",
  "logger": "coordinator.attribution",
  "msg": "request.complete",
  "tenant_id": "acme-corp",
  "user_id": "user-7f3a",
  "workload_id": "coding-agent-v2",
  "requested_model": "Qwen3-32B",
  "serving_model": "Qwen3-32B",
  "namespace": "llm-d",
  "prompt_tokens": 128,
  "completion_tokens": 512,
  "request_id": "req-abc123"
}
```

- **Always includes `user_id`**, even when `prometheusUserIdLabel: false`.
- **`requested_model`** is the value from the request body; **`serving_model`** is the
  authoritative value from the decode response body (defaults to `requested_model` on
  error paths).
- **`namespace`** is the Kubernetes namespace of the coordinator pod, populated from the
  `COORDINATOR_NAMESPACE` environment variable (defaults to `default` when absent).
- Is emitted at `INFO` level on the `coordinator.attribution` logger.
- Does **not** include request or response body content — metadata only.

## Design Details

### Architecture

```mermaid
flowchart TD
    classDef client   fill:#e8f0fe,stroke:#4a6cf7,color:#1a1a2e,font-weight:bold
    classDef upstream fill:#fff3e0,stroke:#fb8c00,color:#4e2b00
    classDef coord    fill:#e8f5e9,stroke:#43a047,color:#1b5e20,font-weight:bold
    classDef vllm     fill:#f3e5f5,stroke:#8e24aa,color:#4a148c
    classDef obs      fill:#fce4ec,stroke:#e53935,color:#7f0000
    classDef gw       fill:#e3f2fd,stroke:#1e88e5,color:#0b3c5d,font-weight:bold

    Client(["External Client\n(agent / pipeline / user)"]):::client
    Client -->|"HTTP request\n+ auth credential"| upstream

    subgraph upstream["Upstream component  (see Appendix A)"]
        Auth["Authenticate credential\n(API key / JWT)"]:::upstream
        Strip["Strip client-supplied\nx-llm-d-*-id headers"]:::upstream
        Inject["Inject verified headers\nx-llm-d-tenant-id: acme-corp\nx-llm-d-user-id: user-7f3a\nx-llm-d-workload-id: coding-agent-v2"]:::upstream
        Auth --> Strip --> Inject
    end

    upstream -->|"request + verified\nattribution headers"| GWIn

    subgraph gw["Inference Gateway"]
        GWIn["Default route → Coordinator"]:::gw
        GWPhase["Phase route\n(EPP-Profile header)"]:::gw
    end

    EPP(["EPP\n(ext_proc pod selection)"]):::gw
    GWPhase -->|"pick pod"| EPP
    EPP -->|"selected pod"| GWPhase

    GWIn -->|"forward to coordinator"| Handler

    subgraph coord["Coordinator"]
        Handler["handleInference\nRead x-llm-d-tenant/user/workload-id\nStore on RequestContext"]:::obs
        Pipeline["Pipeline steps\n(render, conditional-decode,\nencode, prefill, decode)\nModifyResponse hook captures token counts\nEmit Prometheus metrics + structured log"]:::coord
        Handler --> Pipeline
    end

    Pipeline -->|"one call per phase"| GWPhase
    GWPhase -->|"response"| Pipeline

    subgraph vllm["vLLM workers  (unchanged)"]
        VLLMInfer["Execute inference\nReturn response with token usage"]:::vllm
    end

    GWPhase -->|"forward to pod"| VLLMInfer
    VLLMInfer -->|"response + token usage"| GWPhase

    subgraph outputs["Observability outputs"]
        Prom[("Prometheus\nmetrics")]:::obs
        Log[("Structured log\n(Elasticsearch / Loki)")]:::obs
    end

    Pipeline --> Prom
    Pipeline --> Log
```

> The metric are generated synchronously **after the response is returned to the client**. The reasons for this approach are described in
> [Appendix B: Synchronous vs. Asynchronous Emission](#appendix-b-synchronous-vs-asynchronous-emission).

### Where Attribution Data Lives on `RequestContext`

Three fields are added to [`pkg/coordinator/pipeline/context.go`](../../pkg/coordinator/pipeline/context.go):

```go
// Attribution holds the three identity dimensions extracted from inbound
// x-llm-d-*-id headers. Values default to "unknown" when the header is absent.
// Set once by handleInference before the pipeline runs; read-only thereafter.
TenantID   string
UserID     string
WorkloadID string
```

Three additional fields are populated by the attribution hook inside the decode steps
after the response is received:

```go
// ServingModel is the authoritative model name from the decode response body
// (the vLLM "model" field). Populated by the attribution hook; defaults to
// RequestContext.Model (the requested model) when the response body is unavailable.
ServingModel string

// CompletionTokens is the completion token count from the decode response body.
// Populated by the attribution hook from the vLLM "usage.completion_tokens" field.
// Zero when the decode response body was unavailable or unparseable.
CompletionTokens int

// PromptTokensFromBody is the prompt token count read from the decode response body's
// usage.prompt_tokens field. Used when the render step is not in the pipeline and
// len(TokenIDs) is zero. Populated by the attribution hook.
PromptTokensFromBody int
```

### `handleInference`

#### Header extraction

In [`pkg/coordinator/server/handlers.go`](../../pkg/coordinator/server/handlers.go),
after the `RequestContext` is constructed, extract the three attribution headers from
`r.Header` and apply the `unknown` default:

```go
reqCtx.TenantID   = attributionHeader(r.Header, "x-llm-d-tenant-id")
reqCtx.UserID     = attributionHeader(r.Header, "x-llm-d-user-id")
reqCtx.WorkloadID = attributionHeader(r.Header, "x-llm-d-workload-id")
```

Where `attributionHeader` is a small helper that returns the header value or `"unknown"`:

```go
func attributionHeader(h http.Header, key string) string {
    if v := h.Get(key); v != "" {
        return v
    }
    return "unknown"
}
```

Attribution headers are **not** added to the `internalForwardingHeaders` block-list in
`ForwardedHeaders()` — they should be forwarded to upstream services so that any step
that calls the gateway carries them through (consistent with how other x-llm-d-* headers
are handled).

#### `stream_options.include_usage` injection

When `stream == true` and `requestAttribution.enabled == true`, inject
`stream_options.include_usage = true` into the parsed body before the pipeline runs.
This ensures the upstream model returns a usage object in the final SSE chunk, which
the attribution hook reads from the `ModifyResponse` callback:

```go
if stream && s.attributionCfg.Enabled {
    injectStreamOptions(parsed)
}
```

```go
func injectStreamOptions(body map[string]any) {
    opts, _ := body["stream_options"].(map[string]any)
    if opts == nil {
        opts = map[string]any{}
        body["stream_options"] = opts
    }
    if alreadySet, _ := opts["include_usage"].(bool); !alreadySet {
        opts["include_usage"] = true
    }
}
```

This is gated on `requestAttribution.enabled: true` so operators without attribution
are unaffected. It is idempotent: clients that already send `include_usage: true` are
not changed.

### The Attribution Hook

Attribution data is captured via a **`ModifyResponse` hook injected into `DecodeStep`
and `ConditionalDecodeStep` at construction time**. Both steps already accept an
optional `modifyResponse func(*http.Response) error` parameter through `newDecodeProxy`;
they pass `nil` today. When attribution is enabled, the builder passes a closure that
intercepts the upstream response inside the proxy, before `ServeHTTP` returns.

#### Decode response body capture

Because decode uses a streaming reverse proxy (`httputil.ReverseProxy`), the body
cannot be read directly inside `ModifyResponse` without consuming it and leaving nothing
for the proxy to forward to the client. Instead, `resp.Body` is replaced in place with
an `io.TeeReader` that copies bytes to a side buffer as the proxy reads them:

- For **non-streaming** responses: vLLM always includes a `usage` object in non-streaming
  responses. The `TeeReader` accumulates a full copy in a `bytes.Buffer`. After
  `ServeHTTP` returns, parse `usage.prompt_tokens`, `usage.completion_tokens`, and `model`
  from the buffer. Store on `RequestContext` as `CompletionTokens`, `ServingModel`, and
  `PromptTokensFromBody`.
- For **streaming** responses (SSE): the `TeeReader` copies bytes to a
  `lastChunkTracker` — a writer that keeps only the last non-empty `data:` line it has
  seen. The final `data:` chunk before `data: [DONE]` carries a `usage` object when
  `stream_options.include_usage: true` is set. After `ServeHTTP` returns, the tracker's
  captured line is parsed for `usage.completion_tokens` and `model`. The full stream is
  **never** buffered.

The hook fires for both cache-hit and cache-miss paths:

- **Cache hit** (`conditional-decode` returns `ErrPipelineDone`): `ModifyResponse` fires
  on the 200 response before `ServeHTTP` returns, which is before `ErrPipelineDone` is
  returned by `Execute`. Attribution is captured correctly.
- **Cache miss** (`conditional-decode` returns `nil` after a 412): `ModifyResponse` is
  not called by the proxy (the `ErrorHandler` swallows `errCacheMiss`). The
  `conditional-decode` hook therefore does nothing. The pipeline continues to `decode`,
  whose hook captures and emits attribution for the actual response.

#### Metric and log emission

After `CompletionTokens` and `ServingModel` are populated by the hook, the decode step
calls `hook.Emit(reqCtx)` immediately after `proxy.ServeHTTP` returns:

**Prometheus** (in [`pkg/coordinator/metrics/record.go`](../../pkg/coordinator/metrics/record.go)):

```go
RecordAttributedInputTokens(modelName, tenantID, workloadID, servingModel, userID, promptTokens)
RecordAttributedOutputTokens(modelName, tenantID, workloadID, servingModel, userID, completionTokens)
```

Where `userID` is passed only when `prometheusUserIdLabel: true`; otherwise the
label dimension is omitted (the metric is registered without it when the config gate is
`false`, avoiding any label cardinality from the start).

**Structured log record** (on `coordinator.attribution` logger):

```go
logger.Info("request.complete",
    "tenant_id",       reqCtx.TenantID,
    "user_id",         reqCtx.UserID,
    "workload_id",     reqCtx.WorkloadID,
    "requested_model", reqCtx.Model,
    "serving_model",   reqCtx.ServingModel,
    "namespace",       cfg.Namespace,
    "prompt_tokens",   promptTokens, // len(reqCtx.TokenIDs) if render ran, else PromptTokensFromBody
    "completion_tokens", reqCtx.CompletionTokens,
    "request_id",      reqCtx.RequestID,
)
```

### Token Count Sources

| Token type | Source | Notes |
|---|---|---|
| Prompt tokens | `len(reqCtx.TokenIDs)` after the render step, or `reqCtx.PromptTokensFromBody` parsed by the attribution hook | When `render` is not in the pipeline, the hook reads `usage.prompt_tokens` from the decode response body and stores it in `PromptTokensFromBody`. |
| Completion tokens | `usage.completion_tokens` from the decode response body | Parsed by the attribution hook; not available before decode. |
| Serving model | `model` field from the decode response body | Parsed by the attribution hook; falls back to `reqCtx.Model` on error or unavailability. |

For streaming responses, vLLM emits a final SSE chunk containing only `usage` when
`stream_options.include_usage: true` is set:

```
data: {"id":"...","object":"chat.completion.chunk","model":"Qwen3-32B","usage":{"prompt_tokens":128,"completion_tokens":512,"total_tokens":640},"choices":[]}

data: [DONE]
```

The `lastChunkTracker` in the `TeeReader` side-path watches for this pattern and retains
it for parsing after `ServeHTTP` returns.

### Prompt Token Count — Interaction with Existing Metric

The coordinator already records `llm_d_coordinator_request_input_tokens` (a histogram
with only the `model_name` label) from the render step. The new
`llm_d_coordinator_request_input_tokens_attributed` histogram is a separate metric that
adds attribution labels. Both are emitted independently. The existing metric is not
changed.

When the `render` step is not in the pipeline (text-only OpenAI-format requests where
tokenization happens on the worker), `len(reqCtx.TokenIDs)` is zero. In that case, the
attribution hook reads `usage.prompt_tokens` from the decode response body and stores it
in `reqCtx.PromptTokensFromBody`, which `Emit` uses for both the attributed metric and
the log record. The un-attributed `request_input_tokens` metric is also zero in this case
(it is populated only by the render step).

### Configuration

A new top-level `requestAttribution` block is added to the coordinator config
([`pkg/coordinator/config/config.go`](../../pkg/coordinator/config/config.go)):

```go
type RequestAttributionConfig struct {
    Enabled               bool `mapstructure:"enabled"`
    PrometheusUserIDLabel bool `mapstructure:"prometheus_user_id_label"`
}
```

In the YAML config:

```yaml
requestAttribution:
  enabled: false               # default: false (opt-in)
  prometheus_user_id_label: true  # default: true; set false for unbounded user populations
```

Environment overrides (via viper `AutomaticEnv`):
- `COORDINATOR_REQUESTATTRIBUTION_ENABLED`
- `COORDINATOR_REQUESTATTRIBUTION_PROMETHEUS_USER_ID_LABEL`

The coordinator namespace label is read from the `COORDINATOR_NAMESPACE` environment
variable; it is not in the YAML config because it is typically injected by Kubernetes
via the Downward API.

When `enabled: false`, no header extraction, no metric emission, no log record, and no
`stream_options` injection are performed. The attribution hook is `nil` and both decode
steps behave identically to their pre-attribution state.

### Output Tiers Summary

| Tier | Mechanism | Granularity | `user_id` | Use case |
|---|---|---|---|---|
| **Tier 1** | Prometheus metric labels | Aggregate | Configurable | Dashboards, alerting, chargeback |
| **Tier 2** | Structured JSON log | Per-request, 100% | Always | Billing audit, per-user attribution at scale |

### OpenCost Billing Queries

The attribution labels on `llm_d_coordinator_request_input_tokens_attributed_sum` and
`llm_d_coordinator_request_output_tokens_attributed_sum` are consumed by OpenCost to
produce per-`tenant_id`, per-`workload_id`, and per-`user_id` cost breakdowns alongside
the existing per-model cost tracking. See
[`open-cost-new-dimensions-plan-coordinator.md`](open-cost-new-dimensions-plan-coordinator.md) for the full
OpenCost implementation plan (the Prometheus query pattern is identical; only the metric
names change from `llm_d_epp_*` to `llm_d_coordinator_*`).

```promql
-- input tokens consumed by tenant in a billing window
  sum by (tenant_id, serving_model, namespace) (
    last_over_time(llm_d_coordinator_request_input_tokens_attributed_sum[<window>m] @ <end_unix>)
  )
- sum by (tenant_id, serving_model, namespace) (
    last_over_time(llm_d_coordinator_request_input_tokens_attributed_sum[2m] @ <start_unix>)
  )
```

### Fallback Behaviour

If an attribution header is absent (e.g. internal health-check traffic, or a deployment
that has not yet configured header injection):

- Metric labels default to `unknown`.
- Structured log fields default to `unknown`.
- No request is rejected; attribution is best-effort and degrades gracefully.

If the decode response body is unparseable or the decode step returns an error:

- `ServingModel` defaults to `reqCtx.Model` (the requested model).
- `CompletionTokens` defaults to `0`.
- The structured log record and metrics are still emitted (with the fallback values) so
  the request is not silently dropped from attribution.

### Security

The three headers follow the same trust model as the existing
`x-llm-d-inference-fairness-id` header. The coordinator reads them as trusted input; it
does not validate or re-derive them. Operators are responsible for ensuring that only
trusted upstream components can set these headers before requests reach the coordinator.
The trust boundary requirement and recommended mitigations are documented in Appendix A.

### Changes by Repository

| Repository | Change |
|---|---|
| `llm-d/llm-d-router` | `RequestContext`: add `TenantID`, `UserID`, `WorkloadID`, `ServingModel`, `CompletionTokens`, `PromptTokensFromBody` fields. `handleInference`: extract three attribution headers; inject `stream_options.include_usage` on streaming requests when attribution enabled. `config.go`: add `RequestAttributionConfig`. New file `pkg/coordinator/steps/attribution.go`: `AttributionHook` type, `ModifyResponse` closure, `Emit` function. Modify `pkg/coordinator/steps/decode.go` and `conditional_decode.go`: accept and invoke the hook. `pkg/coordinator/metrics/`: two new attributed histogram families + recording functions. `pkg/coordinator/pipeline/builder/builder.go`: inject hook into decode steps when enabled. |
| `llm-d/llm-d` | New doc `docs/operations/observability/attribution.md`; update `docs/api-reference/coordinator-http-headers.md` to list the three attribution headers; update metrics docs; add example PromQL queries. |
| `opencost/opencost` | Update metric names from `llm_d_epp_*` to `llm_d_coordinator_*` in `QueryInferenceDimensionTokens` — see [`open-cost-new-dimensions-plan-coordinator.md`](open-cost-new-dimensions-plan-coordinator.md) |

### Example User Stories — Implementation Approaches

#### Story 1: Workload cost comparison

An engineering team runs two coding agents (`coding-agent-v1` and `coding-agent-v2`) and
wants to compare their inference costs. The coordinator labels all token metrics with
`workload_id`; OpenCost aggregates these into per-workload costs. The team calls the
OpenCost inference cost API
(`GET /inferencecost?aggregate=workload_id&filter=workload_id:"coding-agent-v1",workload_id:"coding-agent-v2"&window=week`)
and sees that `coding-agent-v2` costs 30% less per million tokens.

#### Story 2: Platform team chargeback

A platform team runs a shared coordinator deployment serving three internal product
teams. An upstream API gateway injects `x-llm-d-tenant-id` for every request based on
the caller's credential. The coordinator emits attributed token histograms; OpenCost
aggregates into per-tenant allocation and usage costs. At the end of the month, the
platform team calls
(`GET /inferencecost?aggregate=tenant_id&window=month`) to retrieve a per-tenant cost
breakdown.

#### Story 3: Per-user billing audit with unbounded users

A SaaS platform serves thousands of individual end users. The operator has set
`prometheus_user_id_label: false`. For per-user invoice breakdown, the billing team
queries the coordinator's structured log stream (shipped to Elasticsearch) for all
records with `tenant_id = "acme-corp"`, groups by `user_id`, and sums
`prompt_tokens` and `completion_tokens`.

## Alternatives Considered

### IPP (Inference Pipeline Proxy) — deprecated

IPP was an earlier llm-d component that sat between the client and the Inference
Gateway, acting as a request proxy before the coordinator architecture existed.
Attribution was initially considered for IPP because it occupied the same logical
position as the coordinator: the single entry point that saw every client request before
it reached the gateway.

IPP was removed from the llm-d stack and replaced by the coordinator. No attribution
implementation was built in IPP. This plan targets the coordinator — IPP's successor —
which provides the same architectural position (full request/response lifecycle
visibility) along with the structured pipeline extension model (`Step` interface,
`RequestContext`) that makes attribution a clean addition rather than a standalone proxy
concern.

### EPP (Endpoint Picker) — considered and abandoned

In the EPP model, attribution metrics would be emitted from the EPP (the scheduling
component that selects vLLM pods per request via the Gateway's ext-proc protocol). The
EPP sees every scheduling call and has access to request headers, making it a plausible
attribution point.

**Why this was abandoned for the coordinator deployment model:**

The EPP participates only in the ext-proc side-channel to pod selection — one scheduling
call per disaggregation phase — and returns before the inference response is produced.
It does not hold the complete request/response cycle. This means:

- `completion_tokens` and `serving_model` — both required for accurate per-request
  billing — come from the vLLM response body. The EPP never sees the response body on
  the critical path; it would need to be wired into a separate response interception hook
  (`ResponseBodyProcessor`) that is architecturally awkward compared to the coordinator's
  natural ownership of the decode response.
- The EPP emits metrics under a different binary and Prometheus subsystem (`llm_d_epp_*`).
  OpenCost would need to reconcile two metric origins — unnecessary complexity when the
  coordinator already owns the full lifecycle.
- The EPP-based approach would source `prompt_tokens` from the ext-proc request headers
  (a tokenization estimate), whereas the coordinator can use the render step's authoritative
  token count or the response body's `usage.prompt_tokens` — more accurate in both cases.

The EPP-based design remains valid for the **sidecar deployment model** (llm-d without
the coordinator), where there is no coordinator entry point and the EPP is the most
natural attribution boundary. The two approaches are complementary. This plan targets
only the coordinator deployment model.

### Implement as a gateway-level filter (Envoy / Istio)

Attribution could be captured at the Inference Gateway level via an Envoy filter or
Lua plugin, before the request reaches the coordinator.

**Rejected**: The coordinator is the only component that accumulates `ServingModel` and
`CompletionTokens` from the decode response body. A gateway filter would need to parse
the vLLM response and correlate it with the inbound request identity — a stateful
cross-request correlation problem that the coordinator solves naturally by holding
`RequestContext` for the full request lifetime.

### Use a separate telemetry sidecar

A separate sidecar could intercept traffic between the client and the coordinator
to record attribution.

**Rejected**: The coordinator model explicitly removes sidecars. Adding a telemetry
sidecar re-introduces the sidecar complexity the coordinator model eliminates.

### Add attribution labels to the existing `request_input_tokens` metric

Rather than two new histograms, the three attribution dimensions could be added as
labels on the existing `llm_d_coordinator_request_input_tokens` histogram, and a
`request_output_tokens` histogram added alongside it with the same label set.

**Rejected** for three independent reasons:

1. **Different lifecycle, different values.** [`request_input_tokens`](../../pkg/coordinator/metrics/llm_d_coordinator_metrics.go:68)
   is recorded in the pipeline `defer` in [`pipeline.go`](../../pkg/coordinator/pipeline/pipeline.go:122)
   after the render step and before decode runs. `serving_model` and
   `completion_tokens` — two of the required attribution dimensions — come from the
   decode response body and do not exist at that point in the pipeline. They cannot be
   added to a metric that is recorded before decode has completed.

2. **`HistogramVec` label sets are fixed at registration time.** The `prometheusUserIdLabel`
   configuration gate requires the `user_id` label dimension to be present in some
   deployments and absent in others. In `client_golang`, a `HistogramVec` is registered
   once at process startup with a fixed label set; label dimensions cannot be added or
   removed at runtime. Two separate metrics — each registered with the appropriate fixed
   label set based on the startup config — is the only way to implement the gate cleanly.

3. **Adding labels to a deployed metric is a breaking change for existing consumers.**
   `request_input_tokens` is already scraped by dashboards, alert rules, and OpenCost
   queries that select it by name. Adding new label dimensions silently multiplies its
   cardinality and changes the meaning of any `sum()` or `rate()` expression that does
   not enumerate the new labels explicitly. Existing consumers break without any change
   on their end. Separate metrics preserve full backward compatibility: existing consumers
   continue to work unchanged, and new consumers opt in by querying the `_attributed`
   variants.

---

## Appendix A: Populating Attribution Headers in Production

This appendix is unchanged from [`request-attribution-metrics.md` §Appendix A](../../../request-attribution-metrics.md).
The same three options — JWT from an identity provider (recommended), Kong API key, and
LiteLLM virtual key — apply identically to the coordinator deployment; only the
destination changes from "the llm-d Inference Gateway → EPP" to "the coordinator
endpoint". The injection pattern, the trust boundary, and the Istio/Kong/LiteLLM
configuration examples all carry over verbatim.

### Common pattern

Regardless of the credential type or upstream component chosen, the injection pattern is:

```mermaid
flowchart LR
    classDef client   fill:#e8f0fe,stroke:#4a6cf7,color:#1a1a2e,font-weight:bold
    classDef cred     fill:#fff3e0,stroke:#fb8c00,color:#4e2b00
    classDef action   fill:#e8f5e9,stroke:#43a047,color:#1b5e20
    classDef header   fill:#fce4ec,stroke:#e53935,color:#7f0000,font-weight:bold
    classDef gw       fill:#e3f2fd,stroke:#1e88e5,color:#0b3c5d,font-weight:bold

    Client(["Client\n(agent / user / pipeline)"]):::client
    Cred(["Credential\n(API key or JWT)"]):::cred

    subgraph upstream["Upstream component  (Kong / LiteLLM / Istio / Envoy AuthZ)"]
        Validate["1. Validate credential"]:::action
        Lookup["2. Resolve identity\ntenant_id, user_id, workload_id"]:::action
        StripH["3. Strip client-supplied\nx-llm-d-*-id headers"]:::action
        InjectH["4. Inject verified headers"]:::action
        Validate --> Lookup --> StripH --> InjectH
    end

    Headers["x-llm-d-tenant-id: acme-corp\nx-llm-d-user-id: user-7f3a\nx-llm-d-workload-id: coding-agent-v2"]:::header

    Coord(["Coordinator\n(Inference Gateway → EPP → vLLM)"]):::gw

    Client -->|"request + credential"| upstream
    Cred -. "bound to credential store\nor JWT claims" .-> Lookup
    InjectH --> Headers
    Headers -->|"trusted request"| Coord
```

For Option 1 (JWT/Istio), Option 2 (Kong API key), and Option 3 (LiteLLM virtual key),
see [`request-attribution-metrics.md` §Appendix A](../../../request-attribution-metrics.md) for
the full configuration examples. The YAML and Lua snippets apply unchanged; replace
references to "llm-d EPP" with "Coordinator".

---

## Appendix B: Synchronous vs. Asynchronous Emission

### How the existing coordinator metrics work

Every existing metric emission in the coordinator is **synchronous and on the request
goroutine**. The evidence from the code:

- [`pipeline.go` `runStep`](../../pkg/coordinator/pipeline/pipeline.go) calls
  `coordmetrics.IncStepRunning`, `RecordStepDuration`, and `IncStepErrorTotal` directly
  in the defer — no goroutine, no channel.
- [`handlers.go` `handleInference`](../../pkg/coordinator/server/handlers.go)
  calls `IncRequestTotal`, `RecordRequestDuration`, `RecordRequestSize`, and
  `DecRequestRunning` in its defer — same pattern.
- The render step calls `coordmetrics.StartUpstreamCall(...).Done()` inline after the
  HTTP round-trip returns.

This is the universal pattern for Prometheus client_golang metrics: `histogram.Observe`
and `counter.Inc` are mutex-protected in-process writes to a float64 or atomic counter.
They complete in microseconds and carry no I/O. There is no meaningful latency benefit to
offloading them to a goroutine.

### Attribution emission is also synchronous — and correctly so

The attribution metric and log calls happen **on the request goroutine, after
`proxy.ServeHTTP` returns** (i.e., after the last response byte has been written to the
client). The sequence on the request goroutine is:

```
1. proxy.ServeHTTP(reqCtx.ResponseWriter, proxyReq)
   └── streams full response to client; returns when done
2. hook.Emit(reqCtx) is called by the decode step
   ├── histogram.Observe(promptTokens)      ← microseconds; in-process memory write
   ├── histogram.Observe(completionTokens)  ← microseconds; in-process memory write
   └── logger.Info("request.complete", ...)  ← synchronous write to the logger's sink
3. decode step Execute returns
4. handleInference defer runs (IncRequestTotal, RecordRequestDuration, etc.)
5. request goroutine exits
```

The client is fully served before steps 2–4 run. The **client-perceived latency is
zero** — it has already received its complete response. The request goroutine is held for
a few microseconds longer (steps 2–4) before being returned to the Go runtime. At any
reasonable request rate this is negligible.

### Asynchronous emission — considered and not recommended

An async approach would fire a goroutine (or write to a buffered channel) at step 1's
return, releasing the request goroutine immediately:

```go
// async option
go func() {
    histogram.Observe(promptTokens)
    histogram.Observe(completionTokens)
    logger.Info("request.complete", ...)
}()
```

**Why this is not the right choice here:**

| Concern | Synchronous (proposed) | Asynchronous |
|---|---|---|
| Client latency impact | None — client is already done | None — same timing from client's view |
| Request goroutine held | ~microseconds extra | Released immediately |
| Metric ordering guarantee | Metrics are recorded before the next scrape for this request | Possible (rare) scrape-before-record race under very high load |
| Log ordering | Completion record follows request log entries in order | Log record may appear out-of-order relative to other request logs |
| Goroutine accounting | One goroutine per request lifecycle, cleanly bounded | Spawns an unbounded background goroutine per request; can accumulate under load |
| Back-pressure | Natural — goroutine pool of chi's server bounds concurrency | None — goroutines pile up if the logger sink is slow (e.g. disk flush under load) |
| Shutdown correctness | All records written before server exits (pipeline drain is synchronous) | Requires a WaitGroup or drain channel at shutdown to avoid losing the last N records |

The practical cost of synchronous emission is negligible (microseconds, no I/O), and the
back-pressure and shutdown-correctness properties are free. Asynchronous emission is only
warranted when the emission path itself is slow — for example, if attribution were
writing directly to a remote sink (a Kafka topic, a remote logging agent over TCP). That
is not the case here: Prometheus metrics are in-process writes, and the logger writes to
its configured sink (typically `stderr` or a buffered file writer), which is fast.

> [!NOTE]
> If a future iteration adds direct write to a remote sink (e.g. a billing event stream),
> that path should be async with an explicit bounded buffer and a drop-or-block policy
> under backpressure — but that change should be scoped to that sink, not retrofitted
> onto the Prometheus and log paths.

---

## Appendix C: Wrapper Step — Considered and Rejected

An earlier iteration of this design implemented attribution as a **wrapper pipeline
step** (`AttributionStep`) that sat in the step slice and delegated to the decode step
as its inner step. This approach was rejected after review.

### What the wrapper step did

`AttributionStep.Execute` would:

1. Replace `reqCtx.ResponseWriter` with an intercepting writer that teed bytes through
   to the original while capturing data for attribution parsing.
2. Delegate to `innerStep.Execute` (the decode or conditional-decode step), which ran
   the proxy and streamed the response through the intercepting writer.
3. Read `ServingModel` and `CompletionTokens` from the captured data after the inner
   step returned.
4. Emit metrics and the log record.

### Why it was rejected

**Problem 1 — `execution_path_total` and step metrics break silently.**
`pipeline.Execute` tracks which steps ran in `started[step.Name()]` and
`executed[step.Name()]` maps. `classifyExecutionPath` checks for the literal strings
`"decode"` and `"conditional-decode"` to classify the request's execution path. A wrapper
step whose `Name()` returns `"attribution"` means `started["decode"]` and
`started["conditional-decode"]` are never set. `execution_path_total` stops recording
for all requests, and `step_running`/`step_duration_seconds` metrics are recorded under
`"attribution"` instead of the actual decode step name. This is a silent regression in
observability.

**Problem 2 — `conditional-decode` cache-miss path is broken.**
`ConditionalDecodeStep.Execute` returns `nil` on a 412 cache miss, allowing the pipeline
to continue to `decode`. A wrapper around only `conditional-decode` would fire attribution
on the cache-miss `nil` return — before the actual response has been produced. A wrapper
around only `decode` would miss cache-hit responses entirely (since on a cache hit the
pipeline exits via `ErrPipelineDone` before `decode` runs). A dual-wrapper solution —
wrapping both steps — requires coordination state on `RequestContext`, still has the
`Name()` problem, and doubles the complexity.

### Why the `ModifyResponse` hook is the right answer

The `ModifyResponse` hook pattern (see [The Attribution Hook](#the-attribution-hook))
avoids both problems:

- Neither decode step changes its `Name()`. All existing pipeline observability is
  unaffected.
- The hook fires inside the proxy response path for whichever step actually served the
  request. Cache-hit and cache-miss paths are both handled correctly with no coordination
  required.
- The `io.TeeReader` approach (wrapping `resp.Body` inside `ModifyResponse`) is cleaner
  than the `ResponseWriter` swap: it operates at the HTTP response layer rather than the
  writer layer, and the proxy already owns the body-copy loop.
