# OpenCost Inference Costs: New Dimensions Plan

**Authors**: Sima Nadler (_IBM_)

Add `tenant_id`, `user_id`, and `workload_id` tracking to the OpenCost inference cost capabilities based on attribution metrics emitted by the llm-d **coordinator**.

See the following related documents:

- **Proposal for adding new dimensions to llm-d:** [`coordinator-request-attribution-metrics.md`](coordinator-request-attribution-metrics.md) — covers the changes required to emit attribution labels on Prometheus metrics and structured logs from the coordinator. See the [Motivation](coordinator-request-attribution-metrics.md#motivation) section for background on why these dimensions are needed.
- **API usage examples:** [Appendix B](#appendix-b--api-usage-examples) — shows how to query inference costs by tenant, workload, and user once attribution is enabled.
---

## 1. Proposed Implementation

### 1.1 New types (`core/pkg/source/inference_results.go`)

Add three new types to support per-attribution-dimension token sums:

```go
// InferenceDimensionKey identifies a (user_id, tenant_id, workload_id, serving_model, namespace)
// combination. UserID is "unknown" when the x-llm-d-user-id header was absent from the request;
// it is never empty because the coordinator always emits a label value.
type InferenceDimensionKey struct {
    UserID       string
    TenantID     string
    WorkloadID   string
    ServingModel string
    Namespace    string
}

// InferenceDimensionTokens holds the prompt and completion token counts for one
// dimension key, sourced from llm_d_coordinator_request_input_tokens_attributed_sum and
// llm_d_coordinator_request_output_tokens_attributed_sum respectively.
type InferenceDimensionTokens struct {
    PromptTokens     float64
    CompletionTokens float64
}

// InferenceDimensionResult holds per-dimension token counts from attribution metrics.
type InferenceDimensionResult struct {
    Values map[InferenceDimensionKey]*InferenceDimensionTokens
}
```

### 1.2 `MetricsQuerier` interface extension (`core/pkg/source/datasource.go`)

One new query method and its constant:

```go
QueryInferenceDimensionTokens = "QueryInferenceDimensionTokens"

// QueryInferenceDimensionTokens returns prompt and completion token sums from
// llm_d_coordinator_request_input_tokens_attributed_sum and
// llm_d_coordinator_request_output_tokens_attributed_sum,
// broken down by user_id, tenant_id, workload_id, serving_model, and namespace.
QueryInferenceDimensionTokens(start, end time.Time) *Future[InferenceDimensionResult]
```

Stubs required in: `core/pkg/source/noop.go`, `record.go`, `mock.go`, and `modules/collector-source/pkg/collector/metricsquerier.go`.
Decoder required in: `core/pkg/source/decoders.go` (`DecodeInferenceDimensionResult`).

### 1.3 Prometheus queries (`modules/prometheus-source/pkg/prom/inference_queries.go`)

#### 1.3.1 `QueryInferenceDimensionTokens` — histogram `_sum` delta strategy

`llm_d_coordinator_request_input_tokens_attributed` and
`llm_d_coordinator_request_output_tokens_attributed` are **histograms**. Their `_sum`
series accumulates the total tokens observed across all requests and behaves as a
monotonically increasing counter. OpenCost queries the `_sum` series using the same
`last_over_time` delta pattern as `queryCounterDelta` — no separate `_total` counter
metrics are needed.

`user_id` is **always present** as a label on these series (the coordinator emits
`"unknown"` when the header is absent, never an empty or missing label). The PromQL
`sum by (…)` always includes `user_id`; the resulting `InferenceDimensionKey` is built
with `UserID = "unknown"` for unauthenticated requests. There is no need for a
separate query path to handle an absent `user_id` label.

`QueryInferenceDimensionTokens` issues **two queries per token type** (four total:
input + output, each at start + end of window):

- **End-of-window query** (input tokens):
  ```promql
  sum by (user_id, tenant_id, workload_id, serving_model, namespace) (
    last_over_time(llm_d_coordinator_request_input_tokens_attributed_sum[<window>m] @ <end_unix>)
  )
  ```
- **Start-of-window query** (narrow lookback to anchor the delta):
  ```promql
  sum by (user_id, tenant_id, workload_id, serving_model, namespace) (
    last_over_time(llm_d_coordinator_request_input_tokens_attributed_sum[2m] @ <start_unix>)
  )
  ```
  Delta = end value − start value per key. Negative delta (counter reset) → use end value.

Where `<window>m` = `windowDuration.Minutes()` (minimum 2), matching the
`queryCounterDelta` convention. Both queries are issued at `effectiveEnd`
(clamped to `time.Now()`) to avoid future-timestamp errors.

The same pattern applies to `llm_d_coordinator_request_output_tokens_attributed_sum`.
The helper `queryDimensionCounterDelta` encapsulates the start/end pair and is called
twice (once per metric). Results are merged into a single `InferenceDimensionResult`.

If `COORDINATOR_REQUESTATTRIBUTION_ENABLED` is false,
`QueryInferenceDimensionTokens` returns an empty result immediately without querying
Prometheus.

### 1.4 `InferenceCost` type extensions

Add three fields to each struct, and update `newInferenceCostResponse()` to copy them:

```go
// pkg/inferencecost/types.go — InferenceCostProperties
// Empty when not broken down by dimension.
// UserID is "unknown" (not empty) when the x-llm-d-user-id header was absent.
UserID     string
TenantID   string
WorkloadID string

// pkg/inferencecost/apitypes.go — InferenceCostAPIProperties
UserID     string `json:"userId,omitempty"`
TenantID   string `json:"tenantId,omitempty"`
WorkloadID string `json:"workloadId,omitempty"`
```

### 1.5 Collector changes (`pkg/inferencecost/collector.go`)

**Call ordering** — the new `buildDimensionCosts()` depends on per-million rates
computed by the calculator. Because the calculator currently runs in `runner.go` (not
inside `CollectMetrics`), `buildDimensionCosts()` must also be called from `runner.go`
after `calculator.CalculateCosts()`. The sequence becomes:

```
// in runner.go runOnce():
1. modelCosts, err := collector.CollectMetrics()  // returns model-level slice; also stores InferenceDimensionResult internally
2. calculator.CalculateCosts(modelCosts)           // populates InputCostPerMillionTokens, OutputCostPerMillionTokens
3. dimCosts := collector.BuildDimensionCosts(modelCosts) // reads stored InferenceDimensionResult; produces dimension-level []*InferenceCost
4. exporter.Export(modelCosts, dimCosts)           // exports both slices
```

The alternative of moving step 3 inside `CollectMetrics` would require the calculator
to run inside the collector, creating a circular dependency. Keeping `BuildDimensionCosts`
as a separate method on `Collector` and calling it from `runner.go` preserves the
existing separation.

**`buildDimensionCosts()` join formula**:

```go
// For each (user_id, tenant_id, workload_id, serving_model, namespace) in dimensionTokens:
//   look up model cost by serving_model:namespace
//   apply join formula independently per cost basis:

AllocationTotalCost = promptTokens × (InputCostPerMillionTokens[allocation] / 1_000_000)
                    + completionTokens × (OutputCostPerMillionTokens[allocation] / 1_000_000)

UsageTotalCost      = promptTokens × (InputCostPerMillionTokens[usage] / 1_000_000)
                    + completionTokens × (OutputCostPerMillionTokens[usage] / 1_000_000)
```

Token counts are set directly from `InferenceDimensionTokens` (not scaled fractions).
All other `Properties` fields (namespace, cluster, model name, etc.) are copied from the
matched model entry. `AllocationMethod` is inherited from the model entry.

**`CollectMetrics` return type** stays `([]*InferenceCost, error)` — it returns only
the model-level slice as today. The dimension token data (`InferenceDimensionResult`) is
stored as a field on `Collector` after the collect step, allowing `BuildDimensionCosts`
(called from `runner.go` after the calculator) to read it without changing
`CollectMetrics`' signature. `queryservice.go`'s `computeStep` calls
`BuildDimensionCosts` directly after calling `CollectMetrics` and the (local) calculator.

**Data quality check** — after building dimension costs, compare total attributed input
token sums against the coordinator's un-attributed prompt token totals per
`model_name:namespace`. The coordinator emits `llm_d_coordinator_request_input_tokens`
(no attribution labels) from the render step; this is the ground-truth prompt token
count against which attribution coverage is measured:

```go
// sum(llm_d_coordinator_request_input_tokens_attributed_sum[model:ns])
//   / llm_d_coordinator_request_input_tokens_sum[model:ns] < 0.9
// → log.Warnf("InferenceCost: attribution coverage low for model=%s ns=%s (%.0f%% of coordinator tokens attributed)")
```

> [!NOTE]
> When the `render` step is not in the coordinator pipeline (text-only OpenAI-format
> requests), `llm_d_coordinator_request_input_tokens` is zero for those requests. In
> that case the coverage ratio will undercount. The warning threshold is configurable;
> operators in render-less deployments should lower it or disable it to avoid false
> alerts. The attributed histogram always captures prompt tokens from the decode response
> body regardless of whether render ran, so attribution coverage is unaffected — only
> the denominator of the check is smaller.

### 1.6 Exporter (`pkg/inferencecost/exporter.go`)

Add `dimensionCost` gauge and update `Export()` to accept and iterate the new
dimension-level slice:

```go
dimensionCost = prometheus.NewGaugeVec(
    prometheus.GaugeOpts{
        Name: "llm_dimension_hourly_cost",
        Help: "...",
    },
    []string{"model_name", "namespace", "cost_basis", "tenant_id", "workload_id", "user_id"},
)
```

`user_id` is always a non-empty string (`"unknown"` for unauthenticated requests), so
the gauge is always emitted with a meaningful label value — no empty-label edge case.

### 1.7 Aggregation & API

**`pkg/inferencecost/aggregate.go`** — add three new dimensions to
`supportedAggregateProperties`, plus three new `case` arms in `aggKey()` and
`matchesFilter()`, and three new property-clear guards in `aggregate()`.

**`pkg/inferencecost/queryservice.go` — `computeStep()`** — call
`collector.BuildDimensionCosts(modelCosts, dimResult)` immediately after the local
`CalculateCosts` call, then include dimension-level entries in the `InferenceCostSet`.
Their properties carry all three new fields plus the full model/namespace/cluster
properties, so existing aggregation by `model_name` or `namespace` correctly sums
across users/tenants/workloads.

**`pkg/inferencecost/queryservice_helper.go`** — update validation error messages to
list the three new supported dimensions: `user_id`, `tenant_id`, `workload_id`.

### 1.8 File change summary

| File | Change | What |
|---|---|---|
| `core/pkg/source/inference_results.go` | Add | `InferenceDimensionKey`, `InferenceDimensionTokens`, `InferenceDimensionResult` |
| `core/pkg/source/datasource.go` | Extend | `QueryInferenceDimensionTokens` constant + interface method |
| `core/pkg/source/decoders.go` | Add | `DecodeInferenceDimensionResult` |
| `core/pkg/source/noop.go` | Add | No-op impl for `QueryInferenceDimensionTokens` |
| `core/pkg/source/record.go` | Add | Recording impl for `QueryInferenceDimensionTokens` |
| `core/pkg/source/mock.go` | Add | Mock impl with override injection for `QueryInferenceDimensionTokens` |
| `modules/prometheus-source/pkg/prom/inference_queries.go` | Add | `QueryInferenceDimensionTokens` (histogram `_sum` delta strategy), `queryDimensionCounterDelta`, decoder, `mergeDimensionDeltas` |
| `modules/collector-source/pkg/collector/metricsquerier.go` | Add | Stub impl (returns empty) for `QueryInferenceDimensionTokens` |
| `pkg/inferencecost/types.go` | Extend | `UserID/TenantID/WorkloadID` on `InferenceCostProperties`; `AttributionEnabled` on `Config` |
| `pkg/inferencecost/env.go` | Extend | Reader for `COORDINATOR_REQUESTATTRIBUTION_ENABLED` |
| `pkg/inferencecost/apitypes.go` | Extend | `UserID/TenantID/WorkloadID` on `InferenceCostAPIProperties`; update `newInferenceCostResponse` |
| `pkg/inferencecost/collector.go` | Extend | `QueryInferenceDimensionTokens` future; stores `InferenceDimensionResult` as field after collect; `BuildDimensionCosts(modelCosts)` public method; coverage warning against `llm_d_coordinator_request_input_tokens_sum` |
| `pkg/inferencecost/aggregate.go` | Extend | 3 new dimensions in map + switch statements |
| `pkg/inferencecost/exporter.go` | Extend | `llm_dimension_hourly_cost` gauge; updated `Export` signature |
| `pkg/inferencecost/runner.go` | Extend | Call `collector.BuildDimensionCosts()` after `calculator.CalculateCosts()`; pass both slices to `exporter.Export()` |
| `pkg/inferencecost/queryservice.go` | Extend | Call `collector.BuildDimensionCosts()` after local `CalculateCosts()`; add dimension entries to `InferenceCostSet` |
| `pkg/inferencecost/queryservice_helper.go` | Extend | Updated error messages listing `user_id`, `tenant_id`, `workload_id` |

### 1.9 Tests

| Test file | What to add |
|---|---|
| `core/pkg/source/decoders_test.go` | `TestDecodeInferenceDimensionResult` — including the `user_id = "unknown"` case (header absent, coordinator always emits the label) |
| `pkg/inferencecost/collector_test.go` | `buildDimensionCosts` unit tests: basic join formula, zero model cost (no panic), empty attribution result, `user_id = "unknown"` label, coverage warning when attributed token sums diverge from coordinator un-attributed total |
| `pkg/inferencecost/aggregate_test.go` | Aggregation by `user_id`/`tenant_id`/`workload_id`; filter by `tenant_id:"acme-corp"`; aggregation with `user_id = "unknown"` |
| `pkg/inferencecost/exporter_test.go` | `llm_dimension_hourly_cost` emitted correctly; `user_id = "unknown"` emitted with that string value (never empty) |
| `modules/prometheus-source/pkg/prom/inference_queries_test.go` | `queryDimensionCounterDelta` unit tests: both token types present; one absent (graceful degradation); `user_id = "unknown"` present in result |

---

### Appendix B — API Usage Examples

This appendix shows how a user queries the OpenCost inference cost API once the
new dimensions are enabled. Two endpoints are available:

| Endpoint | Description |
|---|---|
| `GET /inferenceCost/total` | Returns a single aggregated result covering the full `window` |
| `GET /inferenceCost/timeseries` | Returns one result set per `accumulate` step (requires `accumulate`) |

**Common query parameters:**

| Parameter | Required | Description |
|---|---|---|
| `window` | ✓ | Time range: named (`today`, `yesterday`, `week`, `lastweek`, `month`, `lastmonth`), duration (`1d`, `7d`, `24h`), or RFC3339 range `2025-01-01T00:00:00Z,2025-01-02T00:00:00Z` |
| `costBasis` | | `allocation` (default) or `usage`. Controls which cost basis is returned in `totalCost`, `inputCost`, `outputCost`. |
| `aggregate` | | Comma-separated list of properties to group by. Supported values after this change: `model_name`, `model_version`, `namespace`, `cluster`, `pod`, `controller`, `controller_kind`, `container`, `workload_type`, **`tenant_id`**, **`workload_id`**, **`user_id`** (bold = new). |
| `accumulate` | ✓ for timeseries | Step size: `hour`, `day`, `week`, `month`. |
| `filter` | | AND-filter: `prop:"value"` terms joined by `+`. Supported properties are the same as `aggregate`. |

---

#### B.1 Cost broken down by tenant

Returns one entry per `tenant_id` across all models and namespaces, summed over
the last day.

```
GET /inferenceCost/total?window=1d&aggregate=tenant_id
```

Example response (allocation basis, the default):

```json
{
  "data": {
    "inferenceCosts": {
      "acme-corp": {
        "properties": { "tenantId": "acme-corp" },
        "window": { "start": "2025-06-04T00:00:00Z", "end": "2025-06-05T00:00:00Z" },
        "costBasis": "allocation",
        "totalCost": 4.27,
        "inputCost": 2.56,
        "outputCost": 1.71,
        "promptTokens": 1200000,
        "generationTokens": 340000,
        "totalTokens": 1540000,
        "costPerMillionTokens": 2.77,
        "inputCostPerMillionTokens": 2.13,
        "outputCostPerMillionTokens": 5.03,
        "cacheSavingsFraction": 0.18,
        "allocationMethod": "compute_time"
      },
      "initech": {
        "properties": { "tenantId": "initech" },
        "window": { "start": "2025-06-04T00:00:00Z", "end": "2025-06-05T00:00:00Z" },
        "costBasis": "allocation",
        "totalCost": 1.84,
        "inputCost": 1.10,
        "outputCost": 0.74,
        "promptTokens": 520000,
        "generationTokens": 147000,
        "totalTokens": 667000,
        "costPerMillionTokens": 2.76,
        "inputCostPerMillionTokens": 2.12,
        "outputCostPerMillionTokens": 5.03,
        "cacheSavingsFraction": 0.12,
        "allocationMethod": "compute_time"
      }
    },
    "window": { "start": "2025-06-04T00:00:00Z", "end": "2025-06-05T00:00:00Z" }
  }
}
```

> [!NOTE]
> The map key (e.g. `"acme-corp"`) is the aggregation key derived from the
> grouped dimension values. Properties not included in `aggregate` are cleared
> from each entry's `properties` object so the response accurately reflects what
> was grouped on.

---

#### B.2 Cost broken down by workload

Returns one entry per `workload_id` across all tenants and models. Useful for
identifying which workloads are the largest cost drivers cluster-wide.

```
GET /inferenceCost/total?window=1d&aggregate=workload_id
```

---

#### B.3 Cost broken down by user

Returns one entry per `user_id`. Users who sent requests without the
`x-llm-d-user-id` header appear with `user_id: "unknown"`.

```
GET /inferenceCost/total?window=1d&aggregate=user_id
```

---

#### B.4 Cost for a single tenant, broken down by workload

Scopes the result to one tenant and further groups by workload. Useful for a
per-tenant billing breakdown. Multiple filter terms are joined with `+`.

```
GET /inferenceCost/total?window=1d&aggregate=workload_id&filter=tenant_id:"acme-corp"
```

Example response:

```json
{
  "data": {
    "inferenceCosts": {
      "batch-summariser": {
        "properties": { "tenantId": "acme-corp", "workloadId": "batch-summariser" },
        "window": { "start": "2025-06-04T00:00:00Z", "end": "2025-06-05T00:00:00Z" },
        "costBasis": "allocation",
        "totalCost": 2.10,
        "inputCost": 1.26,
        "outputCost": 0.84,
        "promptTokens": 590000,
        "generationTokens": 167000,
        "totalTokens": 757000,
        "costPerMillionTokens": 2.77,
        "inputCostPerMillionTokens": 2.14,
        "outputCostPerMillionTokens": 5.02,
        "cacheSavingsFraction": 0.21,
        "allocationMethod": "compute_time"
      },
      "chat-frontend": {
        "properties": { "tenantId": "acme-corp", "workloadId": "chat-frontend" },
        "window": { "start": "2025-06-04T00:00:00Z", "end": "2025-06-05T00:00:00Z" },
        "costBasis": "allocation",
        "totalCost": 2.17,
        "inputCost": 1.30,
        "outputCost": 0.87,
        "promptTokens": 610000,
        "generationTokens": 173000,
        "totalTokens": 783000,
        "costPerMillionTokens": 2.77,
        "inputCostPerMillionTokens": 2.13,
        "outputCostPerMillionTokens": 5.03,
        "cacheSavingsFraction": 0.15,
        "allocationMethod": "compute_time"
      }
    },
    "window": { "start": "2025-06-04T00:00:00Z", "end": "2025-06-05T00:00:00Z" }
  }
}
```

---

#### B.5 Cost broken down by tenant and model

Groups by both `tenant_id` and `model_name`. Each entry represents costs for one
tenant/model pair — useful for showback reports where per-model rates differ.

```
GET /inferenceCost/total?window=1d&aggregate=tenant_id,model_name
```

The aggregation key in the response map is the slash-joined combination of the
grouped values, e.g. `"acme-corp/meta-llama/Llama-3.1-8B-Instruct"`.

---

#### B.6 Cost for a single tenant and workload, broken down by user

Scopes to one tenant + workload and groups by `user_id`. Shows individual user
spend within a specific workload. Multiple filter terms are joined by `+`:

```
GET /inferenceCost/total?window=1d&aggregate=user_id&filter=tenant_id:"acme-corp"+workload_id:"chat-frontend"
```

---

#### B.7 Cost for all workloads under a given model, scoped to a namespace

Isolates one llm-d coordinator deployment (by namespace) and shows per-workload
costs for a specific model. The `+` joins two filter terms with AND semantics:

```
GET /inferenceCost/total?window=7d&aggregate=workload_id&filter=namespace:"llm-d-prod"+model_name:"meta-llama/Llama-3.1-8B-Instruct"
```

---

#### B.8 Daily timeseries of per-tenant costs over a week

Uses the `/timeseries` endpoint with `accumulate=day` to get a day-by-day
breakdown. The response contains one `InferenceCostSet` per day in
`inferenceCostSets`.

```
GET /inferenceCost/timeseries?window=lastweek&aggregate=tenant_id&accumulate=day
```

Example response shape:

```json
{
  "data": {
    "inferenceCostSets": [
      {
        "inferenceCosts": {
          "acme-corp": { "costBasis": "allocation", "totalCost": 3.91, "..." : "..." }
        },
        "window": { "start": "2025-05-25T00:00:00Z", "end": "2025-05-26T00:00:00Z" }
      },
      {
        "inferenceCosts": {
          "acme-corp": { "costBasis": "allocation", "totalCost": 4.27, "..." : "..." }
        },
        "window": { "start": "2025-05-26T00:00:00Z", "end": "2025-05-27T00:00:00Z" }
      }
    ],
    "window": { "start": "2025-05-25T00:00:00Z", "end": "2025-06-01T00:00:00Z" }
  }
}
```

---

#### B.9 Usage-basis cost for a tenant (no idle/shared-infra included)

Pass `costBasis=usage` to get costs based on actual resource consumption only,
excluding idle allocation and shared infrastructure overhead.

```
GET /inferenceCost/total?window=1d&aggregate=tenant_id&costBasis=usage&filter=tenant_id:"acme-corp"
```

---

#### B.10 Model-level view (pre-feature, no attribution dimensions)

When `aggregate` is omitted or set to `model_name`, only model-level entries are
returned. This is the same view available before this feature is enabled.

```
GET /inferenceCost/total?window=1d&aggregate=model_name
```

Model-level entries have no `tenantId`, `workloadId`, or `userId` in their
`properties` object (omitted by `omitempty`). Dimension-level entries always
carry all three.

> [!NOTE]
> Attribution must be enabled (`COORDINATOR_REQUESTATTRIBUTION_ENABLED=true`)
> for dimension-level entries to appear. When attribution is disabled, all
> requests return model-level entries only — existing behaviour is fully
> preserved.

---

### 2 Key design decisions

| Decision | Chosen approach | Rationale |
|---|---|---|
| **Source metrics** | `llm_d_coordinator_request_input_tokens_attributed_sum` + `llm_d_coordinator_request_output_tokens_attributed_sum` | The coordinator emits dedicated attributed histograms distinct from the un-attributed `request_input_tokens` histogram (render-step only). The `_sum` series accumulate total tokens and behave as monotonically increasing counters — the `queryCounterDelta` pattern applies directly. |
| **Cost join formula** | `promptTokens × (inputCostPerM / 1M) + completionTokens × (outputCostPerM / 1M)` | Token sums are the cost drivers. Request count is not used — per-token rates differ between input and output. |
| **Calculator call order** | Calculator runs on model entries first; `buildDimensionCosts` uses the resulting rates | No need to re-run cost split logic per dimension entry. |
| **`user_id` label semantics** | Always present; value is `"unknown"` when the header was absent | The coordinator always emits a label value. OpenCost never needs to handle a missing `user_id` label. This simplifies PromQL (single query, no absent-label fallback path) and ensures `InferenceDimensionKey.UserID` is never an empty string. |
| **`CollectMetrics` signature** | Unchanged `([]*InferenceCost, error)`; `InferenceDimensionResult` stored as `Collector` field; `BuildDimensionCosts` is a separate public method | Calculator must run between collect and dimension-cost build; keeping `CollectMetrics` signature stable avoids breaking `runner.go`, `queryservice.go`, and test code that mocks the interface. |
| **`serving_model` join key** | Label on attributed histograms, populated by the coordinator from the vLLM decode response body; joined on `serving_model:namespace` in OpenCost | The coordinator intercepts the decode response and extracts the `model` field — the same authoritative value vLLM returns and that `vllm:prompt_tokens_total{model_name}` carries. Falls back to `model_name` (requested model) on error paths. |
| **`requested_model` excluded from group-by** | Omitted from `InferenceDimensionKey` and PromQL `sum by (…)` | Cost rate is driven by `serving_model`; including `requested_model` inflates cardinality without adding cost signal. Attribution by what was *served*, not what was *requested*, is the correct billing model. |
| **Coverage warning denominator** | `llm_d_coordinator_request_input_tokens_sum` (coordinator's un-attributed histogram) | The coordinator owns prompt token counts. In the coordinator model, `vllm:prompt_tokens_total` is not the right denominator because it counts tokens at the vLLM level, not the coordinator entry point. The coordinator's un-attributed histogram is the ground truth for the number of requests that passed through the attribution path. |
| **Backward compatibility** | New fields are empty-string by default; feature is opt-in via `COORDINATOR_REQUESTATTRIBUTION_ENABLED=true` | Deployments without attribution configured are completely unaffected. |

---

### Appendix A — Multi-coordinator Deployment Issues

A cluster may contain multiple independent llm-d coordinator deployments, each in its
own namespace with its own coordinator, vLLM pods, and InferencePool. OpenCost is
deployed once per cluster. Multi-deployment concerns are handled naturally: the attributed
histograms carry a `namespace` label (populated from `COORDINATOR_NAMESPACE` on the
coordinator pod), and the `serving_model:namespace` composite key partitions dimension
token data per deployment automatically — no manual per-namespace configuration is
required.

One limitation pre-dates this plan and remains open:

#### Known limitation — Shared infra label is a single global value ⚠️

**The problem:** `INFERENCE_SHARED_INFRA_LABEL` and `INFERENCE_SHARED_INFRA_LABEL_VALUE`
identify shared infrastructure pods whose costs are distributed across model pods. If two
llm-d deployments use different label schemes for their shared infrastructure, there is no
way to express different values per namespace.

**Severity:** Low in practice — llm-d standardises the `llm-d.ai/inference-shared=true`
label across all deployments, so divergence is unlikely.

**Optional future resolution:** A per-namespace configuration mechanism (e.g. an
annotation on the InferencePool or a dedicated ConfigMap) could allow OpenCost to
discover the shared infra label per namespace. However, consuming this would require
restructuring the allocation query logic to filter and aggregate per namespace rather than
in a single cluster-wide pass — a more invasive change deferred to a future iteration.

#### What works correctly without changes

- **Model-level cost collection** — vLLM and allocation queries key by `(model_name, namespace)`, naturally partitioning deployments.
- **`buildDimensionCosts()` join** — keys on `serving_model:namespace`, costs attributed to the correct deployment.
- **Coverage warning** — compares attributed token sums vs coordinator un-attributed totals per `model_name:namespace`, each deployment checked independently.
- **API filtering** — `?filter=namespace:"llm-d-prod"` isolates one deployment's costs from another.
