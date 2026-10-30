# QoS Admission

**Type:** `qos-admission`
**Interfaces:** `requestcontrol.RequestHeaderProcessor`, `requestcontrol.ResponseHeaderProcessor`, `requestcontrol.ResponseBodyProcessor`
**Stability:** Alpha

Enforces per-team QoS contracts ahead of flow control. Each team is committed to a service class, and each class is a flow-control priority band. Traffic within the team's commit is served at that class. Traffic above the commit moves one class down, and is rejected with `429` when no lower class exists.

## What It Does

The plugin runs in the RequestHeader phase, before flow control queues the request. For each request it:

1. Resolves the team from `teamHeader` (default: the fairness ID header). A missing or unlisted team uses `defaultTeam`.
2. Rejects with `400` a prompt larger than `maxPromptTokens`, or an output cap larger than `maxOutputTokens`.
3. Rejects with `429` a request that would exceed the team's `dailyTokens`.
4. Rejects with `429` a request that would exceed the team's `maxConcurrency`. A slot is held from admission, queueing included, to completion.
5. Charges the reservation `input + outputTokenWeight * maxOutput` to the team's token bucket. If the bucket cannot cover it, the request moves to the next lower class without a charge. In the lowest class it is rejected with `429`.
6. Sets the request priority to the served class and sets the fairness ID to the team when no fairness ID was supplied.

At end of stream the reservation is settled against the reported usage: unused tokens are refunded, and usage above the reservation becomes bucket debt. A response with no usage block keeps the reservation as the charge. A request that ends without a response body (a flow-control rejection, a scheduling failure, a client disconnect) is refunded in full when its stream ends.

Input tokens are estimated as request bytes / 4, because tokenization runs after flow control. `maxOutput` is the request's output cap, else the team's `maxOutputTokens`, else `defaultOutputReservation`.

## Responses

Successful responses carry:

| Header | Value |
|---|---|
| `x-llm-d-qos-class` | Class that served the request |
| `x-ratelimit-remaining-tokens` | Weighted tokens left in the team's bucket |
| `x-ratelimit-reset-tokens` | Seconds until the bucket is full |
| `x-llm-d-qos-concurrency-remaining` | Unused concurrency slots |

Rate headers are present only for a team with a commit, and the concurrency header only for a team with a cap.

A `429` adds `retry-after` (seconds) and `x-llm-d-request-dropped-reason: rejected-over-quota`, and reports the team's committed class in `x-llm-d-qos-class`. A `400` reports `rejected-size-limit`.

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
      commitTokensPerMinute: 200000
      burstSeconds: 30
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
| `teams[].commitTokensPerMinute` | `0` | Committed weighted-token rate. `0` disables the token bucket. |
| `teams[].burstSeconds` | `30` | Bucket capacity, in seconds of committed rate. |
| `teams[].maxConcurrency` | `0` | Concurrent requests, queued or running. `0` is unlimited. |
| `teams[].maxPromptTokens` | `0` | Estimated prompt tokens above which a request is rejected. `0` is unlimited. |
| `teams[].maxOutputTokens` | `0` | Output cap above which a request is rejected, and the default reservation. `0` is unlimited. |
| `teams[].dailyTokens` | `0` | Weighted-token budget per UTC day. `0` is unlimited. |

Configure a flow-control priority band for each class priority, or rely on dynamic band provisioning from the default band templates.

## Limitations

- Buckets, counters and budgets are held per EPP replica. With `N` replicas behind a load balancer, divide each commit by `N`.
- The team ID comes from a client header. Authenticate it upstream, for example at the gateway, before trusting it for quota.

## Observability

| Metric | Type | Labels | Description |
|---|---|---|---|
| `llm_d_epp_qos_admission_decisions_total` | Counter | `team`, `class`, `served_class`, `decision`, `mode` | Decisions: `admitted`, `downgraded`, `rejected_quota`, `rejected_concurrency`, `rejected_size`, `rejected_daily`. |
| `llm_d_epp_qos_charged_tokens_total` | Counter | `team`, `served_class` | Weighted tokens charged after refunds. Showback. |
| `llm_d_epp_qos_inflight_requests` | Gauge | `team` | Requests counted against the concurrency cap. |
| `llm_d_epp_qos_bucket_tokens` | Gauge | `team` | Bucket level at its last update. |

The `team` label shares the fairness ID cardinality bound.
