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
// InferenceDimensionKey identifies a (tenant_id, workload_id, target_model_name, namespace)
// combination, and optionally a user_id dimension. UserID is the value of the x-llm-d-user-id
// header when present; it is empty when the coordinator's prometheus_user_id_label gate is false
// (the default), because in that configuration the metric Vec is registered without the user_id
// label, so attributed counter series have no user_id label at all and the PromQL sum by omits
// it — the decoded key's UserID field is left as the zero value "". When the gate is true and
// the header was absent, the coordinator emits "unknown" and UserID is that value.
type InferenceDimensionKey struct {
    UserID          string
    TenantID        string
    WorkloadID      string
    TargetModelName string
    Namespace       string
}

// InferenceDimensionTokens holds the prompt and completion token counts for one
// dimension key, sourced from llm_d_coordinator_request_input_tokens_attributed_total and
// llm_d_coordinator_request_output_tokens_attributed_total respectively.
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
// llm_d_coordinator_request_input_tokens_attributed_total and
// llm_d_coordinator_request_output_tokens_attributed_total,
// broken down by user_id, tenant_id, workload_id, target_model_name, and namespace.
QueryInferenceDimensionTokens(start, end time.Time) *Future[InferenceDimensionResult]
```

Stubs required in: `core/pkg/source/noop.go`, `record.go`, `mock.go`, and `modules/collector-source/pkg/collector/metricsquerier.go`.
Decoder required in: `core/pkg/source/decoders.go` (`DecodeInferenceDimensionResult`).

### 1.3 Prometheus queries (`modules/prometheus-source/pkg/prom/inference_queries.go`)

#### 1.3.1 `QueryInferenceDimensionTokens` — counter delta strategy

`llm_d_coordinator_request_input_tokens_attributed_total` and
`llm_d_coordinator_request_output_tokens_attributed_total` are **counters**. They
accumulate the total tokens observed across all requests and increase monotonically.
OpenCost queries them using the same `increase`-based delta pattern as
`queryCounterDelta`.

Whether `user_id` is present as a label on these series depends on the coordinator's
`prometheus_user_id_label` configuration gate (default: `false`):

- **Gate off (default):** the metric is registered without the `user_id` label at all.
  The PromQL `sum by` omits `user_id`; the resulting `InferenceDimensionKey.UserID` is
  `""` (empty). The `"unknown"` sentinel is **not** emitted in this configuration.
- **Gate on:** the metric is registered with the `user_id` label. The coordinator emits
  `"unknown"` when the `x-llm-d-user-id` header is absent, so `UserID` is never empty
  for series produced in this configuration.

`QueryInferenceDimensionTokens` issues **two queries per token type** (four total:
input + output, each at start + end of window). When `prometheus_user_id_label` is true
the `sum by` includes `user_id`; when false it is omitted:

- **End-of-window query** (input tokens, gate **on**):
  ```promql
  sum by (user_id, tenant_id, workload_id, target_model_name, namespace) (
    increase(llm_d_coordinator_request_input_tokens_attributed_total[<window>m] @ <end_unix>)
  )
  ```
- **End-of-window query** (input tokens, gate **off**):
  ```promql
  sum by (tenant_id, workload_id, target_model_name, namespace) (
    increase(llm_d_coordinator_request_input_tokens_attributed_total[<window>m] @ <end_unix>)
  )
  ```
- **Start-of-window query** (narrow lookback to anchor the delta, same conditional grouping):
  ```promql
  -- gate on:
  sum by (user_id, tenant_id, workload_id, target_model_name, namespace) (
    increase(llm_d_coordinator_request_input_tokens_attributed_total[2m] @ <start_unix>)
  )
  -- gate off:
  sum by (tenant_id, workload_id, target_model_name, namespace) (
    increase(llm_d_coordinator_request_input_tokens_attributed_total[2m] @ <start_unix>)
  )
  ```
  Delta = end value − start value per key. Negative delta (counter reset) → use end value.

Where `<window>m` = `windowDuration.Minutes()` (minimum 2), matching the
`queryCounterDelta` convention. Both queries are issued at `effectiveEnd`
(clamped to `time.Now()`) to avoid future-timestamp errors.

The same pattern applies to `llm_d_coordinator_request_output_tokens_attributed_total`.
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
// UserID is "" when prometheus_user_id_label is false (the default) because the
// attributed counter Vec is registered without the user_id label; the PromQL sum by
// omits it, and the decoded InferenceDimensionKey.UserID is the zero-value "".
// UserID is "unknown" only when the gate is true and the x-llm-d-user-id header
// was absent from the request (the coordinator's attributionHeader() helper always
// returns "unknown" rather than "" for absent headers).
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
// For each (user_id, tenant_id, workload_id, target_model_name, namespace) in dimensionTokens:
//   look up model cost by target_model_name:namespace
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
// sum(llm_d_coordinator_request_input_tokens_attributed_total[model:ns])
//   / llm_d_coordinator_request_input_tokens_sum[model:ns] < 0.9
// → log.Warnf("InferenceCost: attribution coverage low for model=%s ns=%s (%.0f%% of coordinator tokens attributed)")
```

> [!NOTE]
> When the `render` step is not in the coordinator pipeline (text-only OpenAI-format
> requests), `llm_d_coordinator_request_input_tokens` is zero for those requests. In
> that case the coverage ratio will undercount. The warning threshold is configurable;
> operators in render-less deployments should lower it or disable it to avoid false
> alerts. The attributed counter always captures prompt tokens from the decode response
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

`user_id` is included in the gauge label set regardless of the coordinator's
`prometheus_user_id_label` gate. When the gate is on, it carries the actual user value
or `"unknown"` (for requests with no `x-llm-d-user-id` header). When the gate is off,
it carries `""` — the empty string that results when `InferenceDimensionKey.UserID` is
unpopulated because the attributed counter series had no `user_id` label.

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
| `modules/prometheus-source/pkg/prom/inference_queries.go` | Add | `QueryInferenceDimensionTokens` (counter `increase` delta strategy), `queryDimensionCounterDelta`, decoder, `mergeDimensionDeltas` |
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
| `core/pkg/source/decoders_test.go` | `TestDecodeInferenceDimensionResult` — gate-on case: `user_id = "unknown"` (header absent, label present); gate-off case: `user_id` label absent from series, `UserID = ""` in decoded key |
| `pkg/inferencecost/collector_test.go` | `buildDimensionCosts` unit tests: basic join formula, zero model cost (no panic), empty attribution result; gate-on with `user_id = "unknown"`; gate-off with `user_id = ""` (label absent); coverage warning when attributed token sums diverge from coordinator un-attributed total |
| `pkg/inferencecost/aggregate_test.go` | Aggregation by `user_id`/`tenant_id`/`workload_id`; filter by `tenant_id:"acme-corp"`; gate-on aggregation with `user_id = "unknown"`; gate-off aggregation with `user_id = ""` |
| `pkg/inferencecost/exporter_test.go` | `llm_dimension_hourly_cost` emitted correctly; gate-on: `user_id = "unknown"` emitted with that string value; gate-off: `user_id` label absent from emitted series |
| `modules/prometheus-source/pkg/prom/inference_queries_test.go` | `queryDimensionCounterDelta` unit tests: both token types present; one absent (graceful degradation); gate-on: `user_id = "unknown"` in result; gate-off: `user_id = ""` in result (label not in `sum by`) |

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

Returns one entry per `user_id`. Requires `prometheus_user_id_label: true` on the
coordinator; when the gate is off the attributed counter series carry no `user_id`
label, so all results collapse into a single `""` bucket and per-user breakdown is
not meaningful. When the gate is on, users who sent requests without the
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
| **Source metrics** | `llm_d_coordinator_request_input_tokens_attributed_total` + `llm_d_coordinator_request_output_tokens_attributed_total` | The coordinator emits dedicated attributed counters distinct from the un-attributed `request_input_tokens` histogram (render-step only). Being true counters they accumulate total tokens directly — the `queryCounterDelta` (`increase`) pattern applies. |
| **Cost join formula** | `promptTokens × (inputCostPerM / 1M) + completionTokens × (outputCostPerM / 1M)` | Token sums are the cost drivers. Request count is not used — per-token rates differ between input and output. |
| **Calculator call order** | Calculator runs on model entries first; `buildDimensionCosts` uses the resulting rates | No need to re-run cost split logic per dimension entry. |
| **`user_id` label semantics** | Conditional on `prometheus_user_id_label` gate (default: off). Gate on: label present, value is `"unknown"` when the header was absent. Gate off: metric registered without the label; `InferenceDimensionKey.UserID` is `""` and per-user breakdown via Prometheus is not meaningful. | The coordinator's Vec label set is fixed at registration time; the gate is the only clean way to omit the label in cardinality-sensitive deployments. Per-user attribution remains available via the structured log regardless of this setting. |
| **`CollectMetrics` signature** | Unchanged `([]*InferenceCost, error)`; `InferenceDimensionResult` stored as `Collector` field; `BuildDimensionCosts` is a separate public method | Calculator must run between collect and dimension-cost build; keeping `CollectMetrics` signature stable avoids breaking `runner.go`, `queryservice.go`, and test code that mocks the interface. |
| **`target_model_name` join key** | Label on attributed counters, populated by the coordinator from the vLLM decode response body; joined on `target_model_name:namespace` in OpenCost; matches the EPP label name exactly | The coordinator intercepts the decode response and extracts the `model` field — the same authoritative value vLLM returns. Falls back to `model_name` (requested model) on error paths. Using `target_model_name` (not `serving_model`) keeps coordinator and EPP metric label sets aligned so the two can be joined. |
| **`requested_model` excluded from group-by** | Omitted from `InferenceDimensionKey` and PromQL `sum by (…)` | Cost rate is driven by `target_model_name`; including `requested_model` inflates cardinality without adding cost signal. Attribution by what was *served*, not what was *requested*, is the correct billing model. |
| **Coverage warning denominator** | `llm_d_coordinator_request_input_tokens_sum` (coordinator's un-attributed histogram) | The coordinator owns prompt token counts. In the coordinator model, `vllm:prompt_tokens_total` is not the right denominator because it counts tokens at the vLLM level, not the coordinator entry point. The coordinator's un-attributed histogram is the ground truth for the number of requests that passed through the attribution path. |
| **Backward compatibility** | New fields are empty-string by default; feature is opt-in via `COORDINATOR_REQUESTATTRIBUTION_ENABLED=true` | Deployments without attribution configured are completely unaffected. |

---

### Appendix A — Multi-coordinator Deployment Issues

A cluster may contain multiple independent llm-d coordinator deployments, each in its
own namespace with its own coordinator, vLLM pods, and InferencePool. OpenCost is
deployed once per cluster. Multi-deployment concerns are handled naturally: the attributed
counters carry a `namespace` label (populated from `COORDINATOR_NAMESPACE` on the
coordinator pod), and the `target_model_name:namespace` composite key partitions dimension
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
- **`buildDimensionCosts()` join** — keys on `target_model_name:namespace`, costs attributed to the correct deployment.
- **Coverage warning** — compares attributed token sums vs coordinator un-attributed totals per `model_name:namespace`, each deployment checked independently.
- **API filtering** — `?filter=namespace:"llm-d-prod"` isolates one deployment's costs from another.
