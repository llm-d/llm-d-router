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
* **Selects a queue to dispatch from** using the configured strategy. Currently `las` (Least Attained Service) is supported; programs with the lowest accumulated service score highest.
* **Decays attained service** for inactive programs so a long-idle program is not penalized indefinitely. Both wall-clock half-life and per-Pick factor decay are supported.
* **Evicts idle program state** on a periodic sweep so per-program memory and Prometheus label series do not accumulate forever.

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
| `strategy` | `las` | Scoring strategy. Only `las` is supported. |
| `lasWeightService` | `0.8` | Weight on the inverted attained-service signal. Higher values prioritize underserved programs more aggressively. |
| `lasWeightHeadWait` | `0.2` | Weight on the head-of-queue age. Acts as a tiebreaker on cold start when programs have equal attained service. |
| `lasDecayFactor` | `0.99997` | Per-Pick decay factor applied to inactive programs when `lasHalfLifeSeconds` is `0`. Must be in `(0, 1]`. Coupled to Pick rate. |
| `lasHalfLifeSeconds` | `0` | Wall-clock half-life of attained service for inactive programs. When `> 0` it overrides `lasDecayFactor`. |
| `evictionTtlSeconds` | `3600` | A program with no completion in this window is evicted from the metrics map. |
| `evictionSweepSeconds` | `300` | How often the eviction sweep runs. Must be `> 0`. |

A complete sample is shipped at [`deploy/config/sim-program-aware-config.yaml`](../../../../../../../deploy/config/sim-program-aware-config.yaml).

## Observability

The plugin exports two shared collectors and one strategy-owned collector under the `llm_d_router_epp` Prometheus subsystem:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `program_aware_jains_fairness_index` | Gauge | none | Jain's Fairness Index over the average wait time per program. `1.0` indicates perfectly equal waits. |
| `program_aware_avg_wait_time_milliseconds` | GaugeVec | `program_id` | Cumulative running mean of flow-control queue wait time per program. |
| `program_aware_attained_service_tokens` | GaugeVec | `program_id` | Time-decayed attained service per program, in weighted tokens. Written by the LAS strategy. |

## Trade-offs

* **Abandoned requests block eviction**: Requests abandoned after dispatch leave `inFlight` non-zero, and the eviction sweep skips any program with non-zero `inFlight`, so its `ProgramMetrics` entry and Prometheus series persist indefinitely.
* **Memory and label-series growth**: A new program ID adds a `ProgramMetrics` entry plus per-program Prometheus label series. The eviction sweep bounds growth, but a workload with rapidly churning program IDs (e.g. a fresh ID per request) will see TTL-bounded accumulation. Choose a TTL that matches your churn rate.
* **Decay tuning depends on workload**: `lasDecayFactor` is per-Pick, so its effective half-life depends on the cluster's pick rate. Use `lasHalfLifeSeconds` for predictable wall-clock decay.

## Related Documentation

* [Fairness Overview](../README.md)
* [Flow Control User Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md)
