# groupsched — GSCH-CORE

One group-fairness scheduler for **one engine and N engines with the same code**: a
single-pod stack is the N=1 case of the per-engine state, never a branch of its own.

This file is the module map, the pass order, the metric and config surface, and the list of
what was deliberately removed. It is written for someone arriving with no context: read §1
(what the policy does), then §2 (the two rules that are not obvious), then §4 (every knob).
The design record lives in the knowledge bank
(`explore/ring-window-v2/tier-design-data/SCHEDULER-REDESIGN.md`, `…-AS-BUILT.md`,
`SCHEDULER-TRIM-DESIGN.md`, `BRANCH-IDENTITY-V2-DESIGN.md`); this file describes the code as
it is, and the bank describes how it got here.

## 1. What the policy does

A **branch** (branch identity's derived id — the flow queue key) is **seated** on exactly one
engine, and its seat is its footprint on that engine's block index: the exact prefill chain of
its queued turn, deduped against what the engine already holds. Each engine's **seat budget**
is its own measured KV capacity × `groupBudgetFraction` — **the seat budget is the whole
admission control**. A branch with a queued head and no seat waits in ONE shared queue;
newcomers go before returning branches, then the oldest arrival; when room frees, the head
goes to the engine with the most free budget left AFTER the seat (tie → lowest engine id) and
stays there while seated. A resident becomes rotation-eligible after **Q** completed busy
periods (or S of silence), and memory pressure evicts in the budget valve's order.

One external answer, then plain functions: the queue key comes from the thread-identity
plugin; everything else the pass decides is a function of the state it already holds.

| seam | where | answers |
|---|---|---|
| identity | *(the thread-identity plugin)* | who is the branch (= the queue key) |

**One pass, one order** (`Pass`, and the only place a decision is taken):

    reap → drainExiting → syncWaiters → deriveColdCharges → enforce (per engine)
         → rotate / seat → dispatch (+ charge the winner's exact chain)

Everything the pass decides on is derived from state in that pass: the request counts from
the request map, the cold charges from the branches, the queue view from the heads this pass
was handed. Nothing accumulates across passes and no timer decides anything.

### Files

| file | contents |
|---|---|
| `scheduler.go` | `Config`, the state machine, the pass, the request map, settle/abandon/chunk, the snapshot, the event ring |
| `admission.go` | the shared waiter queue: the seat loop, the per-engine candidate facts |
| `blockindex.go` | the per-engine block index: two refcounts (used / stay), the exact cost, the seat |
| `filter.go` | the placement Filter (pin dispatch to the seat engine) + the lock-light pin/request table |
| `plugin.go` | the framework plugin: FairnessPolicy + Filter + SaturationDetector, capacity learning, every hook |
| `metrics.go` | the nineteen Prometheus series |
| tests | `scoped_hooks_test.go` (the F13 class, §2b), `rule_o_test.go` (§2c), `invariant_test.go` (randomized, N=1 and N=2, every invariant after every event), `released_window_test.go`, `chargeddrain_test.go`, `cascade_test.go`, `coldcap_test.go`, `blockindex_test.go`, `multipod_test.go`, `scheduler_test.go`, `replay_test.go`, `config_conformance_test.go`, `metrics_events_test.go`, `metrics_wire_test.go`, `observability_test.go`, `integration_test.go` |

## 2. The rules that are not obvious

### 2a. The released window counts as IN FLIGHT

A request that a Pick has released and PreRequest has not yet brought to a pod is a turn ON
ITS WAY, and it counts as in flight everywhere: tenure eligibility, the engine-idle rule,
every victim class, the drain, the cold-charge derivation, the in-flight cap. The window is
the COMMON case — every dispatch passes through it and Pick runs at ~1 kHz — so a branch on
its way to the pod looks IDLE to any rule that ignores it (measured at N=1 and N=2: evicted
and re-seated by the observed seat, used 1200 against a 950-block budget; a marked branch
drained before its request ran, used 1500 against 1100; a quantum-complete branch rotated out
mid-window).

A window is **live for at most S** (`idleSeconds`, the tenure rule's own silence bound),
evaluated in the decision pass — no timer, no new constant. Past S it is presumed dead: it
stops counting, the branch becomes evictable on demand, and the cold charge it held is
released by the derivation. **A valve-held head therefore waits at most S**, not "within one
Pick" — that is the bounded cost of the rule, and it is the honest bound.

One request is one record (`reqRec`), keyed by request id: it opens at the Pick that released
it (carrying the branch, the exact chain and the engine the observation later names) and ends
at its settle, its abandon, its branch's exit or the reap. Every hook the framework SCOPES
(Filter, PreRequest, ResponseBody, RequestAbandoned) resolves its branch from that record —
see §2b.

### 2b. The scoped hooks, and where the branch comes from

**The framework confines a plugin's request attributes.** `datalayer.ScopeRequest` /
`ScopeInvocation` hand a plugin a request whose attribute access is limited to the keys the
plugin declares: an undeclared read RESOLVES AS ABSENT, an undeclared write is DROPPED. That
is applied at `Filter`, `Scorer`, `PreRequest`, the profile/scheduling points and every
`DataProducer`. The flow-control `Pick` path applies NO scope (`processor.go:
fairnessP.Pick(ctx, flowGroup)`).

**So the policy reads a request attribute in exactly one hook.** `Pick` takes the branch id,
the arrival class and the client's own request id from the flow controller's own queue item —
the only place they are readable. Every later hook resolves the branch from the scheduler's
OWN record of the release, by request id; the Filter resolves it through the pin table's
request mirror, which the same release writes, so it still never takes the scheduler's mutex.
The plugin declares `Produces() = nil` and `Consumes()` empty — honestly: it publishes nothing
onto a request, and the one key it reads (branch identity's) cannot be declared, because the
DAG's layer validator rejects a consumer whose producer sits in a later layer (branch identity
is a `PreRequest` plugin, this policy is flow-control layer 0). That is the upstream draft in
§6; until it lands, the request map IS the design.

**Branch identity is REQUIRED, and it is the only accepted source.** `validateBranchProducer`
refuses to start without it, and refuses an agent-identity-only config explicitly: an agent (a
session) id is a MULTI-CHAIN member — its parallel sub-agents are separate chains — while the
index keeps ONE chain per tracked id (R3), so seating by agent id would charge the sub-agents'
blocks to one record and UNDER-ACCOUNT the engine. Without branch identity every request is
anonymous and Pick passes every head through unseated, which looks healthy in every outcome
metric — hence fail closed rather than a degraded queue key.

Why this matters: reading the branch from an attribute under a scope returns `""`, and the
policy then looks perfectly healthy — nothing seated, nothing starved, pass-through in every
outcome metric. It shipped once. `scoped_hooks_test.go` is the test class that catches it: at
N=1 and N=2, for an arm's request, an explicit-fairness-header request, an anonymous request
and a branch-plus-agent-identity config, it drives Pick through the REAL flow controller and
Filter/PreRequest/ResponseBody/Abandon through the REAL scope, and fails on any violation
counted or any branch that fails to resolve.

### 2c. RULE O — the empty engine admits one

A head whose **exact cost exceeds every engine's seat budget** cannot fit an engine that is
empty, because the budget is 0.95 of the cache — and vLLM itself refuses to start unless the
model's max-model-len fits its own KV capacity, so any request the engine accepts physically
fits its WHOLE cache. Without the rule such a head waited for ever, and (strict head-of-line)
so did every branch behind it. The rule, in the order the code applies it:

1. **trigger** — the cost exceeds the budget on EVERY engine; otherwise placement is
   unchanged;
2. an **empty engine** takes it, subject to the cold valve as usual (an empty engine holds no
   charge and a capped charge always fits the cap, so the valve is never the blocker);
3. with no empty engine, the ordinary demand-driven rotation runs on the engine placement
   picks anyway (most free budget, then id), and it **converges on that one engine**: every
   eviction raises its free budget, so it stays the pick until it is empty;
4. the seat charges the **full exact chain** — no capping trick, the index stays exact, and
   the engine reads over budget, which is the truth;
5. **no thrash** — pressure never takes an engine's SOLE staying resident whose own footprint
   exceeds the budget (victim selection returns nil and the pass ends); it leaves by TENURE,
   like everyone else;
6. while it is seated the engine's free budget is negative, so nothing else seats there.

The separate "wait, never evict" guard is dropped for this case (an eviction leads to a seat);
it stays for the valve-held head, where evicting frees budget and never the valve.

Declared, not a defect: **a seated branch whose head has no rendered tokens** is served
through its SEAT (a seat is an admission right), not by the pass-through that a no-token head
of an unseated branch takes. It is the render-failure case, it is counted
(`anomalies_total{kind="no_tokens_head"}`) and its growth is uncharged — the rare path where
the index learns the turn's cost one settle late.

## 3. The metric surface

**Nineteen series**, all `llm_d_epp_group_*`; every per-engine one carries `engine` = the
endpoint id, so `sum(...)` is the whole-policy number with one engine and with N.

| series | what it says |
|---|---|
| `seats{engine}` | seated branches (residencies) |
| `ne_blocks{engine}` | distinct blocks charged to the budget: every seated branch's set, **exiting included** until they drain (the USED side) |
| `budget_blocks{engine}` | the seat budget (fraction × this engine's num_gpu_blocks) |
| `waiters_shared` | the ONE shared admission queue's depth (no engine label: a waiter belongs to no engine) |
| `exiting_branches{engine}` / `exiting_blocks{engine}` / `exiting_oldest_age_seconds{engine}` | the exiting ledger: who is on the way out, how much of `ne_blocks` they hold, and how long the oldest has been leaving (a value that only grows is a drain that is not happening) |
| `cold_inflight_tokens{engine}` / `cold_cap_tokens{engine}` | the cold-entry charge and its DERIVED cap (the budget's own headroom) |
| `cold_valve_held_total{engine}` | Pick PASSES in which the head was held by that engine's valve (a pass count: Pick runs at ~1 kHz) |
| `capacity_missing{engine}` / `capacity_learned_total{via,engine}` | the startup liveness alarm and the acquisition audit |
| `dispatches_total{engine}` | dispatches served per engine; `engine="nocapacity"` counts the pre-capacity pass-through (no engine served those) |
| `exits_total{engine,reason}` | DEPARTURES by reason: `quantum-complete`, `idle`, `engine-idle`, `exiting-drain`, `budget-idle`, `budget-noqueue`, `budget-exiting`, `engine-gone`. A mid-flight mark is a STATE (the exiting ledger), never an exit: the departure is booked once, as `exiting-drain` |
| `entry_wait_seconds{engine,kind}` | seat waits, `kind=` `new` or `reentry`, ONE histogram (top bucket 13,107 s so nothing lands only in `+Inf`) |
| `engine_gone_total` / `rebinds_total` | engine teardowns / seat moves that repaired an affinity violation |
| `affinity_violations_total{from,to}` | requests that landed on an engine other than the one they were charged to — fired from BOTH sites (the PreRequest observation, which rebinds, and the settle). An invariant: must be 0 |
| `anomalies_total{kind}` | the should-be-zero ledger: `no_tokens_head`, `no_request_id`, `stray_settle`. Each names its EVENT; all three read 0 on a healthy stack |

The SEAT/EXIT event ring (dumped in `DumpState`, logged at V(DEBUG)) carries the branch id, the
engine, `fairness_id` (the queue key) and `wire_id` (the client's own `x-correlation-id`, else
`x-request-id`) — the field that joins a seat to a request record, since the derived branch id
joins to nothing on the harness side.

## 4. Every config key (14)

| key | default | what it is |
|---|---|---|
| `groupBudgetFraction` | 0.95 | THE one policy constant, with two readers: the seat budget (fraction × measured KV), and the cold-entry stagger cap, which is its complement. The wave sweeps it |
| `quantum` | 4 | the tenure bound in completed busy periods. Q8 is the same code with 8 |
| `idleSeconds` (S) | 20 | the silence bound: a resident with no queued turn and no activity for S is rotation-eligible. It is also the released window's bound (§2a) |
| `inFlightCap` | 64 | one branch's concurrent requests (a request-count safety valve) |
| `blockSizeTokens` | 64 | a WIRING FACT, not a knob: it must equal the engine's `--block-size`; a measured mismatch fails loudly per engine |
| `inflightReapSeconds` | 9300 | the belt over a termination nobody reported: a request record older than this is dropped. It MUST sit above the client's own timeout (the campaign harness runs 9000 s; every pin sets 9300)|


A key the strict decoder does not know is a CrashLoop, so `config_conformance_test.go` pins
the emitter's blocks against `Config` in both directions, values included.

### Constants in the code

| constant | value | where | why |
|---|---|---|---|
| `eventRingSize` | 2048 | `scheduler.go` | the SEAT/EXIT ring bound (memory). A wrap is COUNTED in `Snapshot.EventRingDropped` (a dump field, not a series) |
| cold-entry cap | DERIVED | `engineGroup.coldCapTokens` | `floor((1 − fraction) × num_gpu_blocks) × blockSizeTokens` per engine; the CHARGE is capped at it, which makes `cold_inflight_tokens ≤ cold_cap_tokens` an invariant and "one cold entry is always allowed" a consequence |
| `maxChainBlocks` | 2^20 | `plugin.go` | effectively unbounded: a cap would silently stop counting a long prompt's tail |
| `defaultInflightReapSeconds` | 9300 | `scheduler.go` | the harness timeout (9000 s) + headroom |
| the released window | S | `scheduler.windowLive` | not a new number: the tenure rule's own silence bound |

Rule: a new number in a decision path needs a table row saying why it cannot be derived.

## 5. What was removed (and where it lives)

Everything below was deleted from this branch and still exists on the archive branch
`dsai/groupsched-tier-aware` — nothing was deleted from git history. Each may return only as
a NEW IMPLEMENTATION of ONE module interface plus ONE config value, with evidence.

* **entry orders**: HRRN (F5) and hrrn-bounded (F5′), and `hrrn-tier` with the CPU-offload tier
  shadow — one implementation per module.
* **tenure variants**: dyn-size quantum, quantum ceilings, MLFQ-2 pools, rotate-when-free, PAT,
  the ledger model.
* **victim variants**: spec order, the F6″ residue order, the residue model, member-stable
  homing.
* **sizing variants**: the byte-derived entry estimate, `sizing: "gpu-plus-tier"` /
  `"tier-inclusive"`, `cpuTierBytes`/`kvBlockBytes`. The seat budget is not a module any more:
  it is `groupBudgetFraction × num_gpu_blocks`, inline where the capacity is learned.
* **a dispatch gate and a safety net**: the seat budget is the admission control and an admitted
  turn runs; the framework's stock utilisation detector is not in a core arm.
* **the goldens and the byte-equality replay machinery**: replaced by
  `TestDeterminism_OneAndTwoEngines` and the randomized invariant suite.
* **`ttlSeconds`**: no timer frees a seat nobody asked for, and a waiting head keeps its place.

## 6. Known live risk / open items

* **UPSTREAM DRAFT (not opened).** The DAG's layer validator
  (`datalayer/data_graph.go`, `pluginToLayerExecutionOrder`) rejects a dependency whose
  producer sits in a LATER layer than its consumer, which is why this policy may not declare
  branch identity's keys: branch identity writes them at `RequestHeader` (a request-control
  hook) while the policy is flow-control layer 0 — even though `RequestHeader` runs BEFORE
  admission, so the value is always there in time. The layer model should admit "a producer
  whose hook runs before my hook", not just "a producer in my layer or lower".
* **RULE O is new** (§2c): it changes placement for one class of head, and its no-thrash
  clause means an engine can sit over budget for a residency. Watch `ne_blocks − budget_blocks`
  per engine and `exits_total{reason="quantum-complete"}` when a deployment's requests can
  exceed 0.95 of the cache.
* **The cold valve bound is S** (§2a): a leaked release window delays admission by at most S,
  engine-local.
* The evidence behind the memory-only detector: vLLM's `kv_cache_usage_perc` is `ref_cnt > 0`
  blocks, so it is a valid preemption-headroom signal — but it LAGS newly admitted large
  requests and can read LOW against a seat × exact-chain accounting (prefix sharing). It is
  off by default, and it is not wired.
* `idleSeconds` is a time constant. Its measured replacement candidate (a branch's own
  think-gap history) was retired with the F-arms; it stays a knob.
