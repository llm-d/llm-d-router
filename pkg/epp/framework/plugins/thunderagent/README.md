# ThunderAgent Plugin

Session level admission control for agentic workloads, based on ThunderAgent
(arXiv 2602.13692). A session is an agent trajectory, identified by the
session id the `agent-identity` plugin reads from the session headers Claude
Code, OpenCode and Codex send; other clients send a header listed in its
`additionalSessionHeaders`. Requests with no session id are not tracked.
`agent-identity` is a required dependency: configuration loading fails
without it. No engine changes.

Flow control queues requests by the fairness ID, and the gate judges each
queue by the session of its first request. Without an explicit fairness
header the director sets the fairness ID to the agent-identity session, so
each session has its own queue. An explicit fairness header that groups
several sessions (for example a tenant) puts them in one queue, where a
paused session at the front holds the others.

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
`usage.total_tokens` of its last completed turn and the sum of the byte
estimates of its turns in flight (each turn resends the history, so the two
describe the same KV). Capacity is `block_size * num_gpu_blocks` scraped from the engine, with
a configured fallback.

When a pod leaves the pool, the plugin drops it from the ledger; its sessions
keep their footprint and move to the pod that serves their next turn.

Gate: a pod's room is `usableCapacityRatio * capacity` minus its working set.
Turns of admitted sessions always dispatch. A paused session's next turn
waits until its own pod has room again (strict origin affinity: a session
never moves, so its warm prefix is never abandoned; the exceptions are its
pod leaving the pool or being filtered out of the scheduling candidates).
This differs from the original ThunderAgent, which restores a paused session
onto any pod with room; here a paused session waits for its own pod for up to
`forceAdmitTimeout`. A new session is admitted onto a pod with room,
and holds until one fits it.
Among the sessions that fit, admitted goes before paused before new; a
session that fits no pod does not block the others, so smaller sessions can
take room a larger paused one waits for. Any queue's first request waiting past
`forceAdmitTimeout` is force-admitted. A picked session's room is
reserved until its request reaches the pod.

Idle sessions give up their room on demand, when a session is picked,
longest idle first. A paused or new session that does not fit, and an
admitted session's turn that pushes its pod over the ceiling, may take the
room of sessions idle past `idleGracePeriod`; a session that just finished a
turn keeps its room, so the pod can stay over the ceiling until one passes
`idleGracePeriod`. A new session prefers a pod it fits without pausing anyone. A
session that gives up its room is paused: it stops counting against the pod and its next turn must fit
again. Sessions with a turn queued are never paused. Pausing happens only
when a session is picked; an internal background sweep runs every `10s` and
drops sessions idle past `2 * max(forceAdmitTimeout, 5m)`.

## Scorer

Referenced as a scorer in a scheduling profile, the plugin gives 1.0 to the
pod a session is bound to in the ledger and 0.0 to the other candidates. It
returns no scores for requests with no session id, for sessions the ledger
does not know, and when the bound pod is not a candidate; the other scorers
then place the request, and `PreRequest` binds the session to the picked pod.
Its category is `Affinity`.

It does the same job as `session-affinity-scorer` with the `session_id`
strategy, with one difference: the binding comes from the ledger, so a
session stays on the pod whose KV the ledger counts for it. Do not configure
both in one profile.

## Configuration

See `test/e2e/artifacts/config/thunderagent-config.yaml` for a sample
configuration. The config loader registers the plugin under its type name
`thunder-agent` when `name` is omitted, and synthesizes the default scheduling
profile, `max-score-picker`, and `single-profile-handler` automatically. All
parameters:

| Field | Default | Description |
|---|---|---|
| `fallbackCapacityTokens` | `4194304` | Per-pod KV capacity in tokens when `cache_config_info` is not scraped. Must be `> 0`. |
| `usableCapacityRatio` | `1.0` | Share of each pod's KV capacity available to sessions, in `(0, 1]`. |
| `idleGracePeriod` | `"30s"` | How long an idle session keeps its room against paused and new sessions (`"0s"` reclaims idle sessions immediately). Set it to about a typical tool-call duration: shorter pauses sessions about to return, longer holds admissions behind long tool calls. |
| `forceAdmitTimeout` | `"5m"` | Maximum queue wait before a request is admitted even when no pod has room (`"0s"` disables forced admission). Flow control's `defaultRequestTTL` defaults to `60s`, so set `defaultRequestTTL` above `forceAdmitTimeout` or to `"0s"` to avoid shedding held requests before forced admission. |

## Observability

Metrics (`llm_d_epp_thunder_agent_*`, all Alpha): `endpoint_working_set_tokens{endpoint}`
(compare it against the engine's KV utilization: a large working set over a
low utilization is the KV-thrashing signature),
`endpoint_capacity_tokens{endpoint}`, `sessions{state}`, `releases_total{class}`,
`delayed_dispatches_total{class}`, `pauses_total`, `resumes_total`,
`starvation_promotions_total`. To verify the gate engaged during a run,
`pauses_total` and `delayed_dispatches_total` must be greater than zero. Session counts
and each pod's working set and capacity are also available from the
`/debug/plugins/state` endpoint. Session ids never appear in metrics, state
dumps or logs.

## Limitations

- The ledger is in-process; running multiple EPP replicas splits the
  accounting.
- Anonymous traffic (no session id) passes the gate untracked; pair the
  plugin with a general load scorer to place it.
- There is no explicit end-of-session signal: a finished session stays in
  the working set as an idle session until another session's turn reclaims
  its room or the retention TTL drops it, so the working set gauge overestimates
  by recently ended sessions. An end-of-session marker is a candidate
  follow-up once clients can send one reliably.
- Prefill/decode disaggregation is not supported: every pod the flow
  controller reports enters one ledger, so a new session can be reserved
  onto a prefill pod.
