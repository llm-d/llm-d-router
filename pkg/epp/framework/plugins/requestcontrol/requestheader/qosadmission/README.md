# QoS Admission

**Type:** `qos-admission`
**Interfaces:** `requestcontrol.RequestHeaderProcessor`, `requestcontrol.ResponseHeaderProcessor`, `requestcontrol.ResponseBodyProcessor`
**Stability:** Alpha

Enforces per-team QoS contracts ahead of flow control. Each team is committed to a service class, and each class is a flow-control priority band. A team's contract is its size limits (`maxPromptTokens`, `maxOutputTokens`) and its commits: its concurrency (`maxConcurrency`) and its daily weighted-token budget (`dailyTokens`). Traffic within the contract is served at the team's class. Traffic outside it moves one class down, and is rejected when no lower class exists: `400` above a size limit, `429` above a commit.

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

With `mode: shadow` every decision is computed and recorded in `llm_d_epp_qos_admission_decisions_total{mode="shadow"}`, but the plugin never rejects, never changes the priority, and adds no response headers. Use it to compare commits against real traffic before enforcing.

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
| `mode` | `enforce` | `enforce` or `shadow`. |
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
