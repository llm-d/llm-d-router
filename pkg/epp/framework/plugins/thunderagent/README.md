# ThunderAgent Plugin

Program-aware admission control for agentic workloads, consolidated into one plugin. A program is an agent trajectory identified by the request `FairnessID` (filled by the `agent-identity` plugin from session headers). The plugin tracks each program's KV token footprint in a single shared table and hooks that table into every layer that needs it:

| Slot | Interface | Behavior |
|---|---|---|
| Scheduling profile | `Scorer` | Sticky 1.0 for the program's bound pod; least token load for unbound programs; abstains for anonymous traffic. |
| `flowControl.saturationDetector` | `SaturationDetector` | Maintains the per-pod fit view (working set against real scraped KV capacity) and always reports unsaturated; the gate lives in the fairness policy. |
| `defaultPriorityBand.fairnessPolicyRef` | `FairnessPolicy` | The admission gate: turns of admitted programs always dispatch; new programs dispatch only when their projected footprint fits a pod, smallest first, with a starvation guard. |
| Request lifecycle | `PreRequest` / `ResponseBodyProcessor` | Token accounting (usage totals plus in-flight estimates), pod binding, session-final release. |

One named instance is referenced from all slots, so the accounting, the binding, and the admission decisions cannot drift apart.

## Why not engine-reported KV utilization

`vllm:kv_cache_usage_perc` counts only the blocks held by requests the engine is currently running. A program between turns (waiting on a tool) still owns its context in the prefix cache, but those blocks sit on the free list and are reported as unused. On an agentic workload most programs are idle at any instant, so engine-reported utilization stays near zero while the cache is in fact full. The committed-token footprint tracked here counts idle programs, which is the quantity that determines whether the next program fits.

## Accounting

- A program's footprint is the larger of the `usage.total_tokens` of its most recent completed request and the bytes-per-token estimate of a request still in flight. Values replace rather than accumulate, and in-flight and committed tokens are never summed: each agentic turn resends its whole history, so a turn's prefill covers the previous context and the two numbers describe the same KV. The estimator is refined with momentum from every usage report.
- Per-pod capacity is `block_size * num_gpu_blocks` scraped from the engine's `cache_config_info` by the datalayer metrics extractor, handling heterogeneous pools. When absent, the configured `capacityTokens` is used, unless `requireRealCapacity` is set, in which case the pod is treated as unable to admit.
- A program with no request in flight can have its committed tokens decayed with `actingHalfLifeSeconds`, reflecting engine-side eviction of its blocks.
- Program state is released by the session-final header, or by the idle TTL sweep.

## Admission

The flow controller's saturation gate is a band-level head-of-line block: while a detector reports saturated, nothing dispatches, including the turns of programs already admitted. Those turns are what complete trajectories and free capacity, and their footprint is already counted in the working set, so this plugin never gates there. The `Saturation` hook always returns 0.0 and instead refreshes a per-pod fit view: `capacity` (real scraped, or the configured fallback) and `working set + bufferTokensPerProgram * bound programs`. Per-stage detector calls (prefill, decode) each maintain their own pods; pods no stage reports for 5s age out.

Gating happens in the fairness policy, where an admitted program's turn can be told apart from a new program's first turn:

- A REASONING program (bound, dispatched at least once) always dispatches: smallest decayed footprint first, oldest head as the tiebreaker. This favors the program closest to finishing, because a completed trajectory releases its KV.
- A RESUMING program (shed, returning) is fit-checked like a new one -- its footprint stopped counting when it was shed -- but outranks never-admitted programs: it is mid-trajectory with sunk work, finishing it releases capacity permanently, and a newcomer's smaller arrival size is transient anyway. Class priority applies among fitting candidates only: a resuming program that does not fit never blocks a fitting newcomer.
- A NEW program dispatches only when a pod has room for its projected footprint (`request bytes / bytes-per-token + bufferTokensPerProgram`) within `utilThreshold * capacity`, counting live admission reservations. An admitted program reserves that room until `PreRequest` binds it, so back-to-back dispatch cycles cannot over-admit into the same gap.
- Any head waiting past `headWaitStarvationMs` dispatches regardless of class, size, or fit, oldest first; for a held program this is the forced-admission backstop.

When only non-fitting new programs wait, the policy picks nothing and the requests stay queued; that is the hold.

There is no ACTING class. Upstream ThunderAgent proactively pauses idle programs, so its resume queue can hold a program with no request outstanding; that program is ACTING. Flow control here only ever queues actual requests, so every candidate has a request pending and is REASONING by upstream's definition.

## Session lifecycle

A request carrying the session-final header (`x-session-final: true` by default) marks its program; when that turn's response completes (and no other turn of the program is in flight), the program's accounting and binding are dropped immediately instead of waiting for the idle TTL. A standalone release request (final header on an empty prompt) works as a tiny turn. `x-parent-session-id` is recorded on subagent programs for observability only.

## Configuration

```yaml
- type: thunder-agent
  name: thunder
  parameters:
    capacityTokens: 524288
    utilThreshold: 0.9
    actingHalfLifeSeconds: 60
    bufferTokensPerProgram: 100
    headWaitStarvationMs: 30000
    evictionTtlSeconds: 3600
    evictionSweepSeconds: 300
    sessionFinalHeader: x-session-final
```

| Field | Default | Description |
|---|---|---|
| `capacityTokens` | `4194304` | Per-pod KV capacity in tokens when `cache_config_info` is not scraped. |
| `requireRealCapacity` | `false` | Treat a pod without scraped capacity as having no admission room instead of using `capacityTokens`. |
| `utilThreshold` | `0.9` | Admission fit ceiling: a new program must fit a pod within this fraction of its capacity. |
| `actingHalfLifeSeconds` | `0` | Half-life of an idle program's committed tokens. `0` disables decay. |
| `bufferTokensPerProgram` | `100` | Growth room reserved per program, in the working set and in the admission fit check. Size it near the context a session accumulates over its turns. |
| `shedIdleSeconds` | `0` | Enables shedding: an over-ceiling pod unbinds programs idle at least this long, smallest first, until back under the fit ceiling. `0` disables. |
| `headWaitStarvationMs` | `30000` | Promotes any queue head waiting this long ahead of class and size order. `0` disables the guard, which lets a large program starve behind smaller ones. |
| `evictionTtlSeconds` | `3600` | A program with no in-flight request and no activity in this window is dropped. |
| `evictionSweepSeconds` | `300` | How often the eviction sweep runs. |
| `sessionFinalHeader` | `x-session-final` | Header marking a session's last turn. |
| `parentSessionHeader` | `x-parent-session-id` | Header carrying a subagent's parent session ID. |
| `profileName` | primary profile | Scheduling profile whose picked endpoint a program is attributed to. |

See `deploy/config/thunderagent-config.yaml` for the full pipeline, including the `agent-identity` plugin that fills `FairnessID` and the `utilization-detector` filter that triggers re-placement off overloaded pods.

## Deviations from upstream ThunderAgent

- **Starvation guard.** Smallest-first ordering can starve a large program indefinitely while small ones keep arriving. Upstream accepts that; `headWaitStarvationMs` bounds it.
- **No bin packing.** Upstream re-places a resumed program with best-fit-decreasing across backends. Here resumption re-runs the scheduling scorers and the sticky branch pulls the program back to its bound pod; re-placement happens only when a filter removes that pod.
- **Shed is opt-in and accounting-only.** Upstream pauses idle programs to reclaim capacity (the ACTING state). Here `shedIdleSeconds` enables the equivalent: an over-ceiling pod unbinds its smallest idle programs, whose next turn then re-enters admission as a new program (fit-checked with its known committed footprint, re-placed, held if nothing fits). Shedding changes bindings and accounting only; the engine's LRU decides when the shed program's KV actually leaves. Off by default because it is a bet worth taking only when admission-time sizing cannot hold, such as contexts growing far past `bufferTokensPerProgram`.
- **TTL rejection.** Flow control rejects a queued request with 429 after `flowControl.defaultRequestTTL` (60s default) where upstream force-admits after 1800s. For patient agent harnesses, raise the TTL in the deployment config; the starvation guard dispatches heads long before typical TTLs.
- **Single-replica accounting.** The program table is in-process. Running multiple EPP replicas splits the accounting, the same limitation as upstream's single router instance.

## Observability

Metrics (`thunder_agent_*`, all Alpha): per-pod `pod_utilization` and `pod_capacity_tokens{source="real|fallback"}` gauges, a `programs{state}` gauge, and counters `holds_total{class}`, `releases_total{class}`, `starvation_promotions_total`, `rebinds_total`, `sheds_total`, `session_final_releases_total`, `ttl_evictions_total`.

To verify the plugin engaged during a run: `thunder_agent_holds_total` must be greater than zero and `thunder_agent_pod_utilization` must have reached `utilThreshold`. `releases_total{class="reasoning"}` exceeding `releases_total{class="new"}` under pressure confirms the admission ordering drove dispatch. The generic flow control metrics (`flow_control_queue_size`, `flow_control_request_queue_duration_seconds`) witness holds independently; `flow_control_pool_saturation` reads 0 by design with this plugin as the detector.

The state dump endpoint reports sanitized aggregates (program counts, per-pod tokens, estimator ratio). Program IDs come from a user-controlled header and are never exported in metrics, logs, or dumps.
