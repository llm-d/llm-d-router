# QoS Admission

**Type:** `qos-admission`
**Interfaces:** `requesthandling.RequestHeaderProcessor`, `requesthandling.ResponseHeaderProcessor`, `requesthandling.ResponseBodyProcessor`
**Stability:** Alpha

Enforces per-team QoS contracts ahead of flow control. It records decisions without acting on them until `mode: enforce` is set; see [Shadow Mode](#shadow-mode). Each team is committed to a service class, and each class is a flow-control priority band. A team's contract is its size limits (`maxPromptTokens`, `maxOutputTokens`) and its commits: its concurrency (`maxConcurrency`) and its daily weighted-token budget (`dailyTokens`). Traffic within the contract is served at the team's class. Traffic outside it moves one class down, and is rejected when no lower class exists: `400` above a size limit, `429` above a commit.

## What It Does

The plugin runs in the RequestHeader phase, before flow control queues the request. For each request it:

1. Resolves the team from `teamHeader` (default: the fairness ID header). A missing or unlisted team uses `defaultTeam`.
2. Checks the contract. A request is within it when its prompt is at most `maxPromptTokens`, its output cap is at most `maxOutputTokens`, the team has a free `maxConcurrency` slot, and the reservation `input + outputTokenWeight * maxOutput` fits in what is left of `dailyTokens`. Such a request takes a slot, held from admission, queueing included, to completion, and is charged the reservation against the daily budget.
3. Moves a request outside the contract to the next lower class, without taking a slot or a charge. In the lowest class it is rejected: with `400` when it is above a size limit, else with `429`.
4. Sets the request priority to the served class and sets the fairness ID to the team when no fairness ID was supplied.

At end of stream the reservation of a committed request is settled against the reported usage: unused tokens are refunded to the daily budget, and usage above the reservation is charged to it. A response with no usage block keeps the reservation as the charge. A request that ends without a response body (a flow-control rejection, a scheduling failure, a client disconnect) is refunded in full when its stream ends.

Input tokens are estimated as request bytes / 4, because tokenization runs after flow control. `maxOutput` is the request's output cap, else the team's `maxOutputTokens`, else `defaultOutputReservation`.

## Responses

Successful responses carry:

| Header | Value |
|---|---|
| `x-llm-d-qos-class` | Class that served the request |
| `x-llm-d-qos-concurrency-remaining` | Unused concurrency slots |

The concurrency header is present only for a team with a `maxConcurrency` commit.

A `429` adds `retry-after` (seconds: `1` for concurrency, until the next UTC day for the daily budget) and `x-llm-d-request-dropped-reason: rejected-over-quota`, and reports the team's committed class in `x-llm-d-qos-class`. A `400` reports `rejected-size-limit` and the team's committed class.

## Shadow Mode

The plugin starts in `mode: shadow` (the default). Every decision is computed and recorded in `llm_d_epp_qos_admission_decisions_total{mode="shadow"}`, but the plugin never rejects, never changes the priority or fairness ID, and adds no response headers. Commits are counted as they would be when enforced: a request recorded as `admitted` holds a concurrency slot and a daily charge until it completes, and one recorded as `downgraded` or `rejected_*` holds neither. Use it to compare commits against real traffic, then set `mode: enforce` to apply the decisions.

## Configuration

```yaml
plugins:
- type: qos-admission
  parameters:
    mode: enforce
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
| `mode` | `shadow` | `shadow` records decisions without acting on them; `enforce` applies them. |
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

- Concurrency counters and daily budgets are held per EPP replica. With `N` replicas behind a load balancer, divide each commit by `N`.
- The team ID comes from a client header. Authenticate it upstream, for example at the gateway, before trusting it for quota.

## Observability

| Metric | Type | Labels | Description |
|---|---|---|---|
| `llm_d_epp_qos_admission_decisions_total` | Counter | `team`, `class`, `served_class`, `decision`, `mode` | Decisions: `admitted`, `downgraded`, `rejected_concurrency`, `rejected_size`, `rejected_daily`. |
| `llm_d_epp_qos_charged_tokens_total` | Counter | `team`, `served_class` | Weighted tokens consumed after refunds, including downgraded traffic. Showback. |
| `llm_d_epp_qos_inflight_requests` | Gauge | `team` | Requests counted against the concurrency commit. |

The `team` label shares the fairness ID cardinality bound.

The metrics are served on the EPP `/metrics` endpoint with the other EPP metrics; see [Router pod / EPP](../../../../../../../docs/metrics.md#router-pod--epp) for the port and scrape authentication.

### Decisions

`llm_d_epp_qos_admission_decisions_total` counts one decision per request:

| Label | Value |
|---|---|
| `team` | Team resolved from `teamHeader`, or the `defaultTeam` ID |
| `class` | The team's committed class |
| `served_class` | Class the request is served at: `class` when `admitted`, the next lower class when `downgraded`, empty for a rejection |
| `decision` | `admitted`, `downgraded`, `rejected_size` (`400`), `rejected_concurrency` or `rejected_daily` (`429`) |
| `mode` | `shadow` when the decision was only recorded, `enforce` when it was applied |

In shadow mode the counter is the report of what enforcement would do; every request is still served at its original priority.

### PromQL

Requests per second that would be moved down or rejected, per team, in shadow mode:

```promql
sum by (team, decision) (rate(llm_d_epp_qos_admission_decisions_total{mode="shadow",decision!="admitted"}[5m]))
```

Fraction of each team's traffic outside its contract over a day. Sum over EPP replicas, since each replica decides on its own share of the traffic:

```promql
sum by (team) (increase(llm_d_epp_qos_admission_decisions_total{mode="shadow",decision!="admitted"}[1d]))
/
sum by (team) (increase(llm_d_epp_qos_admission_decisions_total{mode="shadow"}[1d]))
```

To size the commits before setting `mode: enforce`, compare per-replica demand with them, since each replica holds its own commits:

- Concurrency: `llm_d_epp_qos_inflight_requests` counts only admitted requests, so it never exceeds `maxConcurrency`. A gauge that stays at the commit while the team has `downgraded` or `rejected_concurrency` decisions means demand is above it. A `downgraded` decision does not record which limit was exceeded.
- Daily budget: compare `max by (team) (increase(llm_d_epp_qos_charged_tokens_total[1d]))` with `dailyTokens`. The charged tokens include downgraded traffic, so they measure the team's total demand rather than what fit in the budget. The budget resets at 00:00 UTC, so a rolling `[1d]` window only approximates one budget day.
