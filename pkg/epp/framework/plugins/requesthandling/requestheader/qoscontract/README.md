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

The concurrency header is present only for a team with a `maxConcurrency` commit.

A `429` adds `retry-after` (seconds: `1` for concurrency, until the next UTC day for the daily budget) and `x-llm-d-request-dropped-reason: rejected-over-quota`, and reports the team's committed class in `x-llm-d-qos-class`. A `400` reports `rejected-size-limit` and the team's committed class.

## Report-Only Mode

The plugin starts in `mode: report-only` (the default). It serves every request at its team's committed class: it sets the class priority and, when the request has none, the fairness ID, as `enforce-limits` mode does. Every decision is computed and recorded in `llm_d_epp_qos_contract_decisions_total{mode="report-only"}`, but a request outside the contract is neither moved down a class nor rejected, and the plugin adds no response headers. Commits are counted as they would be when enforced: a request recorded as `within_contract` holds a concurrency slot and a daily charge until it completes, and one recorded as `downgraded` or `rejected_*` holds neither. Use it to compare commits against real traffic, then set `mode: enforce-limits` to apply the decisions.

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
