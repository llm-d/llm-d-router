# QoS Contract

**Type:** `qos-contract`
**Interfaces:** `requesthandling.RequestHeaderProcessor`, `requesthandling.ResponseHeaderProcessor`, `requesthandling.ResponseBodyProcessor`
**Stability:** Alpha

Enforces per-team QoS contracts ahead of flow control. Until `mode: enforce-limits` is set it serves each team at its committed class and only records downgrades and rejections; see [Report-Only Mode](#report-only-mode). Each team is committed to a service class, and each class is a flow-control priority band. A team's contract is its size limits (`maxPromptTokens`, `maxOutputTokens`) and its commits: its concurrency (`maxConcurrency`) and its daily weighted-token budget (`dailyTokens`). Traffic within the contract is served at the team's class. Traffic outside it moves one class down, and is rejected when no lower class exists: `400` above a size limit, `429` above a commit.

## What It Does

The plugin runs in the RequestHeader phase, before flow control queues the request. For each request it:

1. Resolves the team from `teamHeader` (default: the fairness ID header). A missing or unlisted team uses `defaultTeam`.
2. Checks the contract. A request is within it when its prompt is at most `maxPromptTokens`, its output cap is at most `maxOutputTokens`, the team has a free `maxConcurrency` slot, and the reservation `input + outputTokenWeight * maxOutput` fits in what is left of `dailyTokens`. Such a request takes a slot, held from this phase, queueing included, to completion, and is charged the reservation against the daily budget.
3. Moves a request outside the contract to the next lower class, without taking a slot or a charge. In the lowest class it is rejected: with `400` when it is above a size limit, else with `429`.
4. Sets the request priority to the served class and sets the fairness ID to the team when no fairness ID was supplied.

At end of stream the reservation of a committed request is settled against the reported usage: unused tokens are refunded to the daily budget, and usage above the reservation is charged to it. A response with no usage block keeps the reservation as the charge. A request that ends without a response body (a flow-control rejection, a scheduling failure, a client disconnect) is refunded in full when its stream ends.

Until settlement, output is charged at its cap, not at a running estimate, because usage is reported only at end of stream. Reserving the cap keeps requests that stay within their caps from together spending more than the budget. The cost is that in-flight requests hold more of the budget than they will use, so near the end of the budget a request can be downgraded while reservations that will be partly refunded hold the rest.

Input tokens are estimated as request bytes / 4, because tokenization runs after flow control. The bytes include the JSON structure, so the estimate runs high for a short prompt with large request fields (tool definitions, many stop sequences) and low for multi-byte UTF-8 text. The daily charge is corrected at settlement; the `maxPromptTokens` check uses the estimate. `maxOutput` is the request's output cap, else the team's `maxOutputTokens`, else `defaultOutputReservation`.

### Why a RequestHeader Plugin

The plugin is not an `Admitter` and does not replace flow-control admission. Both run later in the director: flow control queues and dispatches the request in the band its priority selects, and `Admitter` plugins run after flow control has dispatched it. The contract decision has to come first:

- A downgrade changes the priority, and so the band flow control queues the request in. Only a RequestHeader plugin can still change it.
- A rejection in the lowest class answers at once. Rejected later, the request would first wait in the lowest band, and its `429` would arrive only after the queue wait.
- The concurrency slot counts queued requests, so it is taken before queueing, in the same check that decides the downgrade or rejection.

## Responses

Successful responses carry:

| Header | Value |
|---|---|
| `x-llm-d-qos-class` | Class that served the request |
| `x-llm-d-qos-concurrency-remaining` | Unused concurrency slots |
| `x-llm-d-qos-slo-status` | Team's promise status at admission; see [SLO Validation](#slo-validation) |

The concurrency header is present only for a team with a `maxConcurrency` commit, and the SLO status header only for a request within the contract of a team with a `promise`.

A `429` adds `retry-after` (seconds: `1` for concurrency, until the next UTC day for the daily budget) and `x-llm-d-request-dropped-reason: rejected-over-quota`, and reports the team's committed class in `x-llm-d-qos-class`. A `400` reports `rejected-size-limit` and the team's committed class.

## Report-Only Mode

The plugin starts in `mode: report-only` (the default). It serves every request at its team's committed class: it sets the class priority and, when the request has none, the fairness ID, as `enforce-limits` mode does. Every decision is computed and recorded in `llm_d_epp_qos_contract_decisions_total{mode="report-only"}`, but a request outside the contract is neither moved down a class nor rejected, and the plugin adds no response headers. Commits are counted as they would be when enforced: a request recorded as `within_contract` holds a concurrency slot and a daily charge until it completes, and one recorded as `downgraded` or `rejected_*` holds neither. Use it to compare commits against real traffic, then set `mode: enforce-limits` to apply the decisions.

## SLO Validation

`RequestHeader` has two entry points, run in order for the request's team:

1. **Limits enforcement** decides the class from the team's limits and commits, as described above.
2. **SLO validation** checks whether the team's latency promise is met. It runs only for a request served within the contract, the traffic the promise covers, and changes no decision.

A team's `promise` sets any of three latency objectives, each a percentile and a threshold, held while the team stays within its limits:

| Objective | Validated against | Notes |
|---|---|---|
| `ttft` | `llm_d_epp_request_ttft_seconds{streaming="true"}` | Receipt at the EPP to first token, including flow-control queue wait. Non-streaming TTFT is end-to-end latency, so it is excluded. |
| `tpot` | `llm_d_epp_request_streaming_tpot_seconds` | Mean time per output token of each streamed request. |
| `itl` | `llm_d_epp_request_streaming_itl_seconds` | Time between consecutive streamed chunks. |

The plugin cannot query Prometheus per request, so a background loop validates every promise each `interval` and caches the result. Prometheus is the source because it holds the traffic of every EPP replica, not only this one's. For each objective it queries, over `window`:

- **guaranteed**: the team's traffic at its committed class, `{fairness_id=<team>, priority=<committed class priority>}`. The promise is validated on it.
- **all**: all of the team's traffic, at any class, for comparison. Traffic moved down a class carries the lower priority, so it falls in `all` only.

```promql
histogram_quantile(0.95, sum by (le) (rate(llm_d_epp_request_ttft_seconds_bucket{fairness_id=~"(team-a)(/.*)?",priority="100",streaming="true"}[5m])))
```

An objective is `met` when the latency at its percentile is at most `thresholdMs`, `violated` when above, `insufficient_data` when the window holds fewer than `minRequests` guaranteed requests, and `unknown` when a query failed. A team's status is the most severe of its objectives, in the order `violated`, `unknown`, `insufficient_data`, `met`. A status older than three intervals reads as `unknown`, so a stalled loop is not reported as a current verdict.

At admission, SLO validation records the team's current status in `llm_d_epp_qos_slo_admissions_total` and, in `enforce-limits` mode, returns it in `x-llm-d-qos-slo-status`.

The fairness ID pattern also matches a composed `<team>/<program>` fairness ID. The default team's promise covers every fairness ID the contract does not list.

```yaml
plugins:
- type: qos-contract
  parameters:
    mode: enforce-limits
    classes:
    - {name: critical, priority: 100}
    - {name: standard, priority: 10}
    defaultTeam:
      class: standard
    teams:
    - id: team-a
      class: critical
      maxConcurrency: 20
      maxPromptTokens: 8192
      maxOutputTokens: 4096
      promise:
        ttft: {percentile: p95, thresholdMs: 2000}
        tpot: {percentile: p95, thresholdMs: 33}
        itl:  {percentile: p95, thresholdMs: 50}
    sloValidation:
      prometheusURL: https://thanos-querier.openshift-monitoring.svc:9091
      bearerTokenFile: /var/run/secrets/kubernetes.io/serviceaccount/token
      caFile: /var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt
      labelSelector: namespace="llm-d"
      window: 5m
      interval: 1m
```

| Field | Default | Description |
|---|---|---|
| `teams[].promise.ttft`, `.tpot`, `.itl` | none | `{percentile, thresholdMs}`. `percentile` is `p95`, `p99.9` and so on. At least one is required in a `promise`. `defaultTeam` may have a promise too. |
| `sloValidation.prometheusURL` | required | Prometheus or Thanos querier HTTP API base URL. |
| `sloValidation.bearerTokenFile` | none | Token file, read for every query. |
| `sloValidation.caFile` | none | CA bundle for the Prometheus server certificate. |
| `sloValidation.insecureSkipVerify` | `false` | Skip server certificate verification. |
| `sloValidation.interval` | `1m` | How often promises are validated. |
| `sloValidation.window` | `5m` | PromQL range a promise is evaluated over. |
| `sloValidation.queryTimeout` | `10s` | Bound on one validation cycle. |
| `sloValidation.minRequests` | `20` | Fewest guaranteed requests in the window for a verdict. |
| `sloValidation.labelSelector` | none | Extra label matchers, without braces, added to every query. Use it to select this pool's EPP series. |

`sloValidation` is required when a team has a promise, and a promise requires `teamHeader` to be the fairness ID header, because the series are labeled by fairness ID.

Limitations:

- `histogram_quantile` interpolates within a bucket, and caps a quantile in the `+Inf` bucket at the largest finite bound, so a threshold above that bound can never be violated. Choose thresholds within the histogram's buckets; the TPOT and ITL buckets are set by EPP flags.
- In `report-only` mode every request is served at its team's committed class, so `guaranteed` and `all` cover the same traffic.
- With `composeFairnessIDs`, each program is its own fairness ID, and programs past the fairness ID label cap are folded into the overflow value, which matches no team.
- A client-supplied fairness ID header names the team, as for the limits; authenticate it upstream.

## Configuration

```yaml
plugins:
- type: qos-contract
  parameters:
    mode: enforce-limits
    outputTokenWeight: 2
    defaultOutputReservation: 1000
    classes:              # highest first; priorities must strictly decrease
    - {name: critical, priority: 100}
    - {name: standard, priority: 10}
    - {name: batch,    priority: -10}
    defaultTeam:
      class: standard
    teams:
    - id: team-a
      class: critical
      maxConcurrency: 20
      maxPromptTokens: 8192
      maxOutputTokens: 4096
    - id: evals
      class: batch
      dailyTokens: 50000000
```

| Field | Default | Description |
|---|---|---|
| `mode` | `report-only` | `report-only` serves each team at its committed class and records downgrades and rejections without applying them; `enforce-limits` applies them. |
| `teamHeader` | `x-llm-d-inference-fairness-id` | Request header carrying the team ID. |
| `outputTokenWeight` | `2` | Cost of one output token relative to one input token. Must be `> 0`. |
| `defaultOutputReservation` | `1000` | Output tokens reserved when neither the request nor the team sets a cap. |
| `classes` | required | Service classes, highest first. Each maps to a flow-control priority band. |
| `defaultTeam` | required | Contract for unlisted teams. All unlisted teams share its limits. |
| `teams[].id` | required | Team ID, matched against `teamHeader`. |
| `teams[].class` | required | Class the team's committed traffic is served at. |
| `teams[].maxConcurrency` | `0` | Committed concurrent requests, queued or running. `0` is unlimited. |
| `teams[].maxPromptTokens` | `0` | Estimated prompt tokens above which a request moves one class down. `0` is unlimited. |
| `teams[].maxOutputTokens` | `0` | Output cap above which a request moves one class down, and the default reservation. `0` is unlimited. |
| `teams[].dailyTokens` | `0` | Committed weighted-token budget per UTC day. `0` is unlimited. |

Configure a flow-control priority band for each class priority, or rely on dynamic band provisioning from the default band templates.

## Limitations

- Concurrency counters and daily budgets are held in memory per EPP replica, and no state is shared across replicas. With `N` replicas behind a load balancer, divide each commit by `N`. The configuration fixes the per-replica share, so when `N` changes, through EPP autoscaling or a rolling update, the total commit changes with it until the configuration is updated: scaling from 2 to 3 replicas raises it by half. A replica that starts or restarts mid-day begins with its full daily budget.
- The daily budget covers one fixed UTC day, the period daily token quotas are usually stated for. It does not spread load across the day: a team can spend the whole budget in the first hour, and its traffic is then downgraded, or rejected in the lowest class, until 00:00 UTC. `maxConcurrency` bounds a team's burst, not the budget.
- The team ID comes from a client header. Authenticate it upstream, for example at the gateway, before trusting it for quota.

## Observability

| Metric | Type | Labels | Description |
|---|---|---|---|
| `llm_d_epp_qos_contract_decisions_total` | Counter | `team`, `class`, `served_class`, `decision`, `reason`, `mode` | Decisions: `within_contract`, `downgraded`, `rejected_concurrency`, `rejected_size`, `rejected_daily`, and the limit that caused each downgrade or rejection. |
| `llm_d_epp_qos_charged_tokens_total` | Counter | `team`, `served_class` | Weighted tokens consumed after refunds, including downgraded traffic. Showback. |
| `llm_d_epp_qos_inflight_requests` | Gauge | `team` | Requests counted against the concurrency commit. |
| `llm_d_epp_qos_slo_status` | Gauge | `team`, `class`, `slo`, `status` | Last validation of each objective: `1` for the current `status` (`met`, `violated`, `insufficient_data`, `unknown`), `0` for the others. |
| `llm_d_epp_qos_slo_observed_seconds` | Gauge | `team`, `class`, `slo`, `scope` | Latency at the promised percentile over the window, for `scope="guaranteed"` and `scope="all"`. Absent when there is no data. |
| `llm_d_epp_qos_slo_window_requests` | Gauge | `team`, `class`, `slo` | Guaranteed requests in the window the objective was validated on. |
| `llm_d_epp_qos_slo_validation_errors_total` | Counter | `team`, `slo` | Failed Prometheus queries. |
| `llm_d_epp_qos_slo_admissions_total` | Counter | `team`, `class`, `status` | Requests within the contract, by the team's promise status at admission. |

The SLO gauges are the same on every EPP replica, since each reads the same Prometheus; take `max` rather than `sum` across replicas. The admissions counter is per replica; sum it.

The `team` label shares the fairness ID cardinality bound.

The metrics are served on the EPP `/metrics` endpoint with the other EPP metrics; see [Router pod / EPP](../../../../../../../docs/metrics.md#router-pod--epp) for the port and scrape authentication.

### Decisions

`llm_d_epp_qos_contract_decisions_total` counts one decision per request:

| Label | Value |
|---|---|
| `team` | Team resolved from `teamHeader`, or the `defaultTeam` ID |
| `class` | The team's committed class |
| `served_class` | Class the request is served at: `class` when `within_contract`, the next lower class when `downgraded`, empty for a rejection |
| `decision` | `within_contract`, `downgraded`, `rejected_size` (`400`), `rejected_concurrency` or `rejected_daily` (`429`) |
| `reason` | Limit the request exceeded: `size` (`maxPromptTokens` or `maxOutputTokens`), `daily` (`dailyTokens`) or `concurrency` (`maxConcurrency`). Empty when `within_contract` |
| `mode` | `report-only` when the decision was only recorded, `enforce-limits` when it was applied |

A request over several limits reports one reason, the first of `size`, `daily` and `concurrency`; the same order decides its rejection. For a rejection `reason` repeats what the `decision` name already says, so one query can group downgrades and rejections by cause.

In report-only mode the counter is the report of what enforcement would do; every request is served at its team's committed class.

### PromQL

Requests per second that would be moved down or rejected, per team, decision and limit, in report-only mode:

```promql
sum by (team, decision, reason) (rate(llm_d_epp_qos_contract_decisions_total{mode="report-only",decision!="within_contract"}[5m]))
```

Fraction of each team's traffic outside its contract over a day. Sum over EPP replicas, since each replica decides on its own share of the traffic:

```promql
sum by (team) (increase(llm_d_epp_qos_contract_decisions_total{mode="report-only",decision!="within_contract"}[1d]))
/
sum by (team) (increase(llm_d_epp_qos_contract_decisions_total{mode="report-only"}[1d]))
```

To size the commits before setting `mode: enforce-limits`, compare per-replica demand with them, since each replica holds its own commits:

- Concurrency: `llm_d_epp_qos_inflight_requests` counts only requests within the contract, so it never exceeds `maxConcurrency`. Decisions with `reason="concurrency"` mean demand is above it. A request also over a size limit or the daily budget reports that reason instead, so the count is a lower bound.
- Daily budget: decisions with `reason="daily"` mean the budget ran out. To see by how much, compare `max by (team) (increase(llm_d_epp_qos_charged_tokens_total[1d]))` with `dailyTokens`. The charged tokens include downgraded traffic, so they measure the team's total demand rather than what fit in the budget. The budget resets at 00:00 UTC, so a rolling `[1d]` window only approximates one budget day.
