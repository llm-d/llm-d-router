# ThunderAgent Plugin

Session level admission control for agentic workloads, a minimal
implementation of ThunderAgent (arXiv 2602.13692). A session is an agent
trajectory, identified by the request FairnessID. Without an explicit
fairness header, the director fills it from the `agent-identity` plugin,
which reads the session headers Claude Code, OpenCode and Codex already
send; other clients send a header listed in its `additionalSessionHeaders`.
Requests with neither are not tracked. No engine changes.

The problem: an agent session resends its whole history every turn, so it is
cheap only while its KV blocks survive on one pod. When a pod holds more
session state than its KV capacity, the engine evicts idle sessions' blocks
and every turn becomes a full prefill. The engine's own KV utilization
cannot detect this: blocks of a session waiting on a tool call sit on the
free list and are reported as unused.

## How it works

One named plugin instance fills every slot, so accounting and decisions
cannot drift apart:

| Slot | Role |
|---|---|
| scheduling profile (`Scorer`) | pin a session to the pod that holds its KV; abstain otherwise |
| `flowControl.saturationDetector` | refresh the per-pod ledger each dispatch cycle and run the pause sweep every `pauseSweepSeconds`; always reports unsaturated, because the controller's own gate would also block admitted sessions' turns |
| `defaultPriorityBand.fairnessPolicyRef` | the admission gate |
| request lifecycle (`PreRequest`, `ResponseBody`) | token accounting, pod binding |

Ledger: each session's KV footprint is the larger of the
`usage.total_tokens` of its last completed turn and the byte estimate of a
turn in flight (each turn resends the history, so the two describe the same
KV). Capacity is `block_size * num_gpu_blocks` scraped from the engine, with
a configured fallback.

Gate: turns of admitted sessions always dispatch. A pod whose working set
exceeds `utilThreshold * capacity` gets its idle sessions paused, smallest
first. A paused session's next turn waits until its own pod has room again
(strict origin affinity: a session never moves, so its warm prefix is never
abandoned; the exceptions are its pod leaving the pool or being filtered out
of the scheduling candidates). A new session is
admitted onto the pod with the most room, and holds until one fits it. Any
head waiting past `headWaitStarvationMs` is force-admitted.

## Configuration

See `deploy/config/thunderagent-config.yaml` for the full pipeline. All
parameters:

| Field | Default | Description |
|---|---|---|
| `capacityTokens` | `4194304` | Per-pod KV capacity in tokens when `cache_config_info` is not scraped. |
| `utilThreshold` | `1.0` | Fit ceiling as a fraction of capacity, for both the sweep and admission. |
| `idleDecayHalfLifeSeconds` | `1` | Idle-session decay half-life in the admission view; 0 disables decay. A value shorter than a typical tool call admits sessions the pod cannot hold; consider 30 to 60. |
| `pauseSweepSeconds` | `5` | Pause sweep interval; 0 sweeps every dispatch cycle. |
| `headWaitStarvationMs` | `1800000` | Forced-admission backstop; 0 disables it. |
| `evictionTtlSeconds` | `3600` | Idle session state retention; must exceed `headWaitStarvationMs`. |

## Observability

Metrics (`llm_d_epp_thunder_agent_*`, all Alpha): `endpoint_working_set_tokens{endpoint,view}`
(compare the undecayed view against the engine's KV utilization: a large
working set over a low utilization is the KV-thrashing signature),
`endpoint_capacity_tokens{endpoint}`, `sessions{state}`, `releases_total{class}`,
`holds_total{class}`, `pauses_total`, `resumes_total`,
`starvation_promotions_total`. To verify the gate engaged during a run,
`pauses_total` and `holds_total` must be greater than zero. Session ids
never appear in metrics or logs.

## Limitations

- The ledger is in-process; running multiple EPP replicas splits the
  accounting.
- Anonymous traffic (no session id) passes the gate untracked; pair the
  plugin with a general load scorer to place it.
- A session is assumed to have at most one request in flight. If one sends
  overlapping requests, the first to finish clears the in-flight estimate
  while the others still run, so the session can be counted as idle and
  paused.
- There is no explicit end-of-session signal: a finished session occupies
  its ledger entry until the idle TTL expires, so the working set
  overestimates by recently ended sessions. An end-of-session marker is a
  candidate follow-up once clients can send one reliably.
