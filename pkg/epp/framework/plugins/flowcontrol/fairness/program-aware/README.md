# Program-Aware Fairness Policy

**Type:** `program-aware-fairness`
**Interfaces:** `flowcontrol.FairnessPolicy`, `requestcontrol.PreRequest`, `requestcontrol.ResponseBodyProcessor`

The program-aware fairness policy schedules per-program queues using aggregated per-program metrics rather than per-request attributes. Programs are identified by the fairness ID header (`x-llm-d-inference-fairness-id`) on each request, allowing distinct workflows or tenants sharing an inference pool to compete on equal footing at the workflow level.

## Why choose this policy?

Choose this policy when:

* **Workflow-level fairness matters**: Distinct agentic workflows or tenants share the same pool, and you want fair allocation between them rather than between individual requests.
* **Token cost varies widely between flows**: A program issuing a few large requests should not dominate a program issuing many small ones, or vice versa.
* **Programs may go idle and return**: Idle programs should not retain accumulated service indefinitely; on return they should compete on near-equal footing with persistently active ones.

## What it does

* **Identifies programs** via the fairness ID header carried on each request.
* **Tracks per-program metrics** through the request lifecycle: queue wait time, dispatched count, in-flight count, last completion time, and (LAS) attained service in weighted tokens.
* **Selects a queue to dispatch from** using the configured strategy: `las` (Least Attained Service), where programs with the lowest accumulated service score highest, or `turn-priority`, which favors deeper sessions.
* **Decays attained service** in wall-clock time so a long-idle program is not penalized indefinitely. Decay is applied lazily whenever a program's service is read or accumulated, so it does not depend on the program being visited by the dispatch loop. The rate is set by an explicit half-life.
* **Evicts idle program state** on a periodic sweep so per-program memory and Prometheus label series do not accumulate forever.

## Strategies

### `las`

Orders programs by least attained service, so a program that has consumed fewer weighted tokens is dispatched first.

### `turn-priority`

Orders programs by how deep their session is. A session that has already taken many turns most likely still has its prefix in the KV cache, so serving it again costs less prefill and finishing it frees the slot sooner.

Depth alone would starve shallow sessions, so head wait is scored alongside it under `turnPriorityTimeWeight`:

```
score = turnNumber + turnPriorityTimeWeight * headWaitSeconds
```

Head wait is unbounded, so a shallow program overtakes any deeper rival once it has waited long enough. The two terms are a raw count and raw seconds, so `turnPriorityTimeWeight` converts one second of waiting into a number of turns. A value tuned for one traffic profile does not carry over to a workload with a different depth or wait distribution.

A program that has been idle beyond `turnPriorityInactivitySeconds` counts as turn one and re-earns depth, on the grounds that a session dormant that long has stopped competing for its own prefix. That threshold is separate from `evictionTtlSeconds`, which governs only when per-program bookkeeping is reclaimed.

The default of `120` leaves a margin over the longest activity observed within a session, so a program still taking turns keeps its depth. On an agentic trace replay averaging 48 turns per session, mean think time between turns was about 2.5s and the slowest turn-to-turn cycle reached 96s under the heaviest load tested. A cycle includes service and queueing, so the post-completion gap this threshold measures is smaller. Workloads whose sessions pause for longer need a higher value.

Under contention a prefix is evicted by capacity pressure while its program is still active, and ordering by turn depth is what keeps those prefixes resident. A low prefix-cache hit rate is therefore weak evidence for lowering this value.

## Unit of Fairness

**Attained service**, measured as a weighted sum of input and output tokens consumed by the program. Output tokens are weighted twice as much as input tokens to reflect their relative compute cost.

## Inputs consumed

* **Program identity**: From the request's `FairnessID` field, which the framework populates from the fairness ID header.
* **Queue state**: Reads queue length and the head item's enqueue time from the `FlowQueueAccessor`.
* **Token usage**: From the `Response.Usage` field on stream completion.

## Configuration

```yaml
plugins:
  - type: program-aware-fairness
    parameters:
      strategy: las
      lasWeightService: 0.8
      lasWeightHeadWait: 0.2
      lasHalfLifeSeconds: 60
      evictionTtlSeconds: 3600
      evictionSweepSeconds: 300

flowControl:
  defaultPriorityBand:
    fairnessPolicyRef: program-aware-fairness
```

| Field | Default | Description |
|---|---|---|
| `strategy` | `las` | Scoring strategy: `las` or `turn-priority`. |
| `lasWeightService` | `0.8` | Weight on the inverted attained-service signal. Higher values prioritize underserved programs more aggressively. |
| `lasWeightHeadWait` | `0.2` | Weight on the head-of-queue age. Acts as a tiebreaker on cold start when programs have equal attained service. |
| `lasHalfLifeSeconds` | `60` | Wall-clock half-life of attained service. `0` disables decay, making attained service cumulative for the program's lifetime. |
| `turnPriorityTimeWeight` | `0.05` | (`turn-priority`) Score contribution per second of head wait, against one point per turn of depth. `0` orders on depth alone. |
| `turnPriorityInactivitySeconds` | `120` | (`turn-priority`) A program idle for longer than this counts as turn one. `0` disables the reset. |
| `evictionTtlSeconds` | `3600` | A program with no completion in this window is evicted from the metrics map. |
| `evictionSweepSeconds` | `300` | How often the eviction sweep runs. Must be `> 0`. |

A complete sample is shipped at [`deploy/config/sim-program-aware-config.yaml`](../../../../../../../deploy/config/sim-program-aware-config.yaml).

## Observability

The plugin exports two shared collectors and one strategy-owned collector under the `llm_d_epp` Prometheus subsystem:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `program_aware_jains_fairness_index` | Gauge | none | Jain's Fairness Index over the average wait time per program. `1.0` indicates perfectly equal waits. |
| `program_aware_avg_wait_time_milliseconds` | GaugeVec | `program_id` | Cumulative running mean of flow-control queue wait time per program. |
| `program_aware_attained_service_tokens` | GaugeVec | `program_id` | Time-decayed attained service per program, in weighted tokens. Written by the LAS strategy. |

`program_aware_attained_service_tokens` is written at request completion. Decay is folded in
lazily, so an idle program's gauge holds the value from its last completion until the program
completes another request. Scheduling reads the decayed value directly, so a flat series does not
mean the scheduler is scoring the program on stale service.

## Trade-offs

* **Abandoned requests block eviction**: Requests abandoned after dispatch leave `inFlight` non-zero, and the eviction sweep skips any program with non-zero `inFlight`, so its `ProgramMetrics` entry and Prometheus series persist indefinitely.
* **Memory and label-series growth**: A new program ID adds a `ProgramMetrics` entry plus per-program Prometheus label series. The eviction sweep bounds growth, but a workload with rapidly churning program IDs (e.g. a fresh ID per request) will see TTL-bounded accumulation. Choose a TTL that matches your churn rate.
* **Turn-priority weighting is scale-dependent**: the score sums a turn count and a wait in seconds, so `turnPriorityTimeWeight` converts between the two rather than balancing them proportionally. A value tuned for one traffic profile does not carry over to another with a different depth or wait distribution.
* **Decay accrues during generation**: Decay is continuous wall-clock time, including while a program's requests are in flight; the service consumed by an in-flight request is added back, post-decay, when it completes. With half-lives well above typical generation times this effect is negligible.

## Related Documentation

* [Fairness Overview](../README.md)
* [Flow Control User Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md)
