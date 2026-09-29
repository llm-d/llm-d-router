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
| `flowControl.saturationDetector` | refresh the per-pod ledger each dispatch cycle; always reports unsaturated, because the controller's own gate would also block admitted sessions' turns |
| `defaultPriorityBand.fairnessPolicyRef` | the admission gate |
| request lifecycle (`PreRequest`, `ResponseBody`) | token accounting, pod binding |

Ledger: each session's KV footprint is the larger of the
`usage.total_tokens` of its last completed turn and the byte estimate of a
turn in flight (each turn resends the history, so the two describe the same
KV). Capacity is `block_size * num_gpu_blocks` scraped from the engine, with
a configured fallback.

Gate: a pod's room is `utilThreshold * capacity` minus its working set.
Turns of admitted sessions always dispatch. A paused session's next turn
waits until its own pod has room again (strict origin affinity: a session
never moves, so its warm prefix is never abandoned; the exceptions are its
pod leaving the pool or being filtered out of the scheduling candidates). A
new session is admitted onto a pod with room, and holds until one fits it.
Among the sessions that fit, admitted goes before paused before new; a
session that fits no pod does not block the others, so smaller sessions can
take room a larger paused one waits for. Any head waiting past
`headWaitStarvationMs` is force-admitted. A picked session's room is
reserved until its request reaches the pod.

Idle sessions give up their room on demand, when a session is picked,
longest idle first. A paused or new session may take the room of sessions
idle past `idleLeaseSeconds`; a new session prefers a pod it fits without
pausing anyone. An admitted session's turn that pushes its pod over the
ceiling may take the room of any idle session. A session that gives up its
room is paused: it stops counting against the pod and its next turn must fit
again. Sessions with a turn queued are never paused. There is no periodic
sweep: pausing only matters to the next admission, so it happens there.

## Configuration

See `deploy/config/thunderagent-config.yaml` for the full pipeline. All
parameters:

| Field | Default | Description |
|---|---|---|
| `capacityTokens` | `4194304` | Per-pod KV capacity in tokens when `cache_config_info` is not scraped. |
| `utilThreshold` | `1.0` | Fit ceiling as a fraction of capacity (1.0 = 100%). |
| `idleLeaseSeconds` | `30` | How long an idle session keeps its room against paused and new sessions. Set it to about a typical tool-call duration: shorter pauses sessions about to return, longer holds admissions behind long tool calls. |
| `headWaitStarvationMs` | `1800000` | Forced-admission backstop; 0 disables it. |
| `evictionTtlSeconds` | `3600` | Idle session state retention; must exceed `headWaitStarvationMs`. |

## Observability

Metrics (`llm_d_epp_thunder_agent_*`, all Alpha): `endpoint_working_set_tokens{endpoint}`
(compare it against the engine's KV utilization: a large working set over a
low utilization is the KV-thrashing signature),
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
- There is no explicit end-of-session signal: a finished session stays in
  the working set as an idle session until another session's turn reclaims
  its room or the idle TTL drops it, so the working set gauge overestimates
  by recently ended sessions. An end-of-session marker is a candidate
  follow-up once clients can send one reliably.
- Prefill/decode disaggregation is not supported: every pod the flow
  controller reports enters one ledger, so a new session can be reserved
  onto a prefill pod.
