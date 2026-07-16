# Flow Control: In-Flight Request Eviction

Tracking issue: [#1119](https://github.com/llm-d/llm-d-router/issues/1119)

## Summary

Flow control today can shape traffic only before dispatch. Once a request reaches a model server,
its resource footprint (KV-cache blocks, batch slots) is held until the request completes,
regardless of what higher-priority demand arrives afterward. This proposal adds the missing half:
a conservative, demand-driven mechanism that terminates in-flight sheddable requests when
higher-priority requests are queued behind pool saturation.

The mechanism is minimal by design. It reuses the eviction plumbing already merged (tracking,
victim policies, the ext_proc signaling path) and adds one new component: a builtin reclamation
controller that decides when and how much to evict. The controller is built around the two facts
that make this problem hard (the saturation signal is delayed, and request footprints are
heterogeneous and partly unknowable) and follows one rule: never act twice on the same stale
signal.

## Motivation

The flow control layer is the admission choke point for an InferencePool: requests wait in
priority-banded queues, and a dispatch loop releases them, highest priority first, gated by a pool
saturation signal against per-priority ceilings (`UsageLimitPolicy`).

Dispatch is better understood as granting a lease on pool capacity. The lease is held from
dispatch until the request completes and the engine garbage-collects its KV blocks. Because there
is no revocation, the pool's occupancy reflects historical admission order, not current demand
priority. A burst of high-priority traffic arriving after a period of low-priority steady state
finds the pool full of low-priority leases, and the dispatch gate can do nothing but wait for
natural churn: a delay that is unbounded and depends on output lengths that cannot be known in
advance.

There are two controls for this, and they compose rather than compete:

- **Holdback** (preventive): grant less than full capacity to lower priorities, reserving headroom
  for future high-priority arrivals. Implemented today by `UsageLimitPolicy` (e.g. the
  `priority-holdback-policy` plugin). Cost: reserved headroom is stranded capacity whenever the
  high-priority burst does not come.
- **Eviction** (corrective): revoke lower-priority leases when high-priority demand is actually
  blocked. Cost: revoked work is destroyed and may be retried.

A deployment tunes the mix. Aggressive holdback makes eviction rare; eviction bounds the damage
when holdback's reserve was absent or insufficient. What eviction buys: it converts the
priority-inversion window from natural churn time (unbounded, output-length dependent) to
revocation round-trip time (bounded, measurable).

### Goals

- Bound the time a high-priority request waits behind lower-priority in-flight work.
- Reuse the merged eviction plumbing (`pkg/epp/flowcontrol/eviction/`, filter/ordering plugins,
  the ext_proc eviction channel) without interface changes.
- Guarantee stability: the controller must not over-evict when the saturation signal lags the
  actions already taken.
- Fail safe: outside its validity conditions (below), eviction degrades to a no-op or to a
  strictly rate-bounded loss, never to unbounded destruction.

### Non-Goals (v1)

- **SLO/deadline-aware triggering.** Blocked demand is the trigger; urgency modeling comes later.
- **Per-endpoint reclamation.** The saturation gauge is pool-level; v1 reclaims at pool level.
- **Footprint-accurate sizing.** v1 uses a mean-footprint estimate; exact per-request footprints
  arrive with the future capacity-ledger work.
- **Partial eviction** (KV offload / resumable requests) and **retry orchestration**. Evicted
  requests are terminated with a retryable error; the client owns retry.
- **Local-buffer reordering.** Ensuring an evicting request is scheduled ahead of requests already
  buffered on the model server requires engine priority support (see Future Work).

## Proposal

### The model: admission grants leases; eviction revokes them

Every component in this design plays one of five roles:

| Role | Responsibility | Component |
|---|---|---|
| Tracker | Which leases exist (dispatch to confirmed termination) | `RequestEvictor` tracked set, fed by Director PreRequest/ResponseBody hooks |
| Actuator | How a lease is revoked | `Evictor` (`ImmediateResponseEvictor`: close `EvictCh`, ext_proc sends ImmediateResponse, upstream resets, engine aborts) |
| Selector | Which leases are revocable, and in what order | `EvictionFilterPolicy` + `EvictionOrderingPolicy` |
| Controller | When to revoke, and how much | New: builtin reclamation controller (this doc) |
| Signals | Demand and capacity | Blocked priority bands with queued items; `SaturationDetector` |

The two signals have opposite trust profiles, and the design leans on each only where it is
trustworthy. Demand (queued items in blocked bands) is EPP-internal, fresh, and exact; it decides
*whether* to evict. Capacity (the saturation gauge) is external, delayed, and approximate; it
sizes *how much*, and is never trusted to confirm that a past action worked.

Confirmation comes instead from an EPP-local event: when an evicted request's ext_proc stream
terminates, the lease is confirmed dead. Delivery mechanism: the tracker invokes a registered
eviction-terminated listener, exactly once per evicted request, from its stream-termination
cleanup path (which the handler's deferred completion runs even on abnormal exit); the controller
registers its confirmation handler as that listener at construction. The lifecycle of a revoked
lease has two lag stages:

```
active --(revocation issued)--> revoking --(stream closed)--> reclaiming --(engine GC + scrape)--> reclaimed
```

- *revoking*: the eviction signal is sent but the request stream has not yet terminated.
  Observable locally.
- *reclaiming*: the stream is dead but the engine has not yet freed KV blocks, or the freed
  capacity has not yet appeared in scraped metrics. Observable only as eventual gauge movement.

The failure mode to avoid is double-counting capacity that is still in one of these two
stages. The controller therefore maintains **pending-reclaim debits**: one record per issued
revocation, keyed by request ID, holding the capacity credited to it at issue time. The pacing
gate refuses new decisions until every debit has expired, and sizing additionally subtracts
unretired debits from any new deficit, so capacity in these stages is never counted twice (see
Sizing for the relationship between the two defenses).

### Validity conditions

Eviction is coherent only when all four hold. The controller does not verify them at runtime;
deployments enabling eviction accept them. Outside them, eviction must be safe, and it is: the
worst case is that sheddable requests are terminated without the expected capacity benefit, at a
rate bounded by the per-decision revocation cap (`maxRevocationsPerDecision`, defined under
Sizing) per confirmed reclaim round-trip.

1. **Choke-point invariant.** All admission to the pool passes through flow control, so the
   tracker's view of in-flight leases is complete.
2. **Shallow local buffers.** The pool's queueing lives in the EPP, not on model servers. When
   this holds, capacity freed on an endpoint is won by newly dispatched work rather than absorbed
   by that endpoint's local FCFS backlog. With the utilization detector this is flow control's
   own steady-state behavior (it gates on engine queue depth). With the recommended concurrency
   detector, which never observes engine queue depth, the condition holds only if the detector's
   concurrency limit is calibrated at or below engine capacity; an over-provisioned limit lets
   deep local backlogs form, which is exactly the state this condition excludes.
3. **Revocation reaches the engine.** The actuator chain is: ext_proc ImmediateResponse
   terminates the response to the client, Envoy resets the upstream stream, and the engine's
   abort-on-client-disconnect frees the request. Each link is standard Envoy and engine behavior,
   but the chain's end-to-end latency is deployment-dependent and is the one link the simulation
   in Validation does not exercise. An end-to-end measurement of issue-to-abort latency in a live
   environment is part of the implementation's acceptance criteria: eviction remains an
   experimental feature (within the experimental flow control layer that hosts it) until that
   measurement exists.
4. **Cross-flow priority semantics.** `priority < 0` is a contract: the request may be terminated
   mid-stream in favor of higher-priority demand. This is a user-facing QoS contract and belongs
   in release notes, not only plugin docs.

## Design Details

### Trigger

The dispatch loop already detects the condition eviction exists for: it stops on Head-of-Line
blocking when `saturation >= ceilings[i]` for priority band `i` (see `dispatchCycle`,
`pkg/epp/flowcontrol/controller/internal/processor.go`). On that break, the controller first
checks the pacing gate (below); in the common saturated-but-gated case the entire evaluation is
one comparison. If the gate is open, it scans for eligible demand:

```
victimPriority = priority of the next lease the selector would revoke (skip if none)
for each band at or below the break point, in descending priority order:
    if band priority <= victimPriority:  stop        (no later band can dominate the victim)
    if band ceiling == 0:                skip        (unattainable; see below)
    if saturation < band ceiling:        skip        (band is not blocked)
    if band has queued requests:         eligible demand found; decide against this band
```

Stated as a predicate: eligible demand is a blocked band with queued requests whose priority is
strictly greater than the victim head's priority. Notes:

- Each candidate band's ceiling is checked directly against the current saturation; the scan does
  not assume ceilings are monotone across bands (the `UsageLimitPolicy` interface does not
  contract monotonicity, and a custom policy may violate it).
- Bands with a ceiling of zero are excluded. A zero ceiling is a policy statement that the band
  must not dispatch regardless of load; no amount of reclamation can unblock it, so treating its
  queue as demand would destroy sheddable leases in a loop for no benefit. A near-zero ceiling,
  by contrast, is attainable and produces a correspondingly deep reclaim target; that is the
  configured policy taking effect, bounded per epoch by the cap and re-checked against live
  demand at each decision.
- Strict priority dominance (`demand priority > victim priority`) prevents same-band churn: the
  gate must never re-grant at priority *p* what the controller just revoked at priority *p*.
- There is no deadline, SLO, or wait-time heuristic. Blocked demand is necessary and sufficient.
  Urgency modeling is future policy work, not runtime work.

### Sizing: deficit in gauge units, with pending-reclaim debits

All quantities are dimensionless, in the saturation gauge's own units. Definitions:

- `leasesTracked`: the number of leases in the tracker, evictable or not; the mean-footprint
  estimate is over the whole tracked population.
- Debit states: a debit is *outstanding* from issue until its revocation is confirmed or timed
  out, then *cooling* until `confirmationGrace` has elapsed from that retirement, then expired
  and removed. `pendingReclaim` is the sum of outstanding and cooling debits; the
  `flow_control_pending_reclaim` gauge reports this sum.
- `maxRevocationsPerDecision`: the per-decision revocation cap (default 2).

The decision procedure:

```
if leasesTracked == 0 or no lease is evictable:   no decision (no victims)
deficit = saturation - ceiling(the eligible demand band)
credit  = saturation / leasesTracked              (mean lease footprint estimate)
net     = deficit - pendingReclaim
if net < 0:                                       no decision (outstanding debits over-cover)
if net == 0 and pendingReclaim > 0:               no decision (outstanding debits cover exactly)
n       = max(1, ceil(net / credit))
n       = min(n, maxRevocationsPerDecision, evictable leases)
issue n revocations
```

The `max(1, ...)` handles the boundary: the dispatch gate blocks on `saturation >= ceiling`, so at
exact equality the deficit is zero, and rounding to zero revocations would leave the band blocked
with no reclamation. One debit is recorded per lease actually evicted (the actuator may evict
fewer than `n` if the victim heap drains or an eviction attempt fails and the lease is
re-tracked), each at the issue-time `credit` value. A debit retains that value until it is
retired, even though `credit` drifts across decisions; the resulting error is bounded by the
per-decision cap and corrected at the next decision.

Relationship between the debits and the pacing gate: under the confirmation-gated pacing below,
the gate opens only once every debit has expired, so `pendingReclaim` is provably zero at
decision time and `net` always equals the deficit; the two skip branches above never execute in
normal operation. They are retained for two reasons. The debit records are the bookkeeping the
gate's own checks are computed from, and the subtraction makes sizing safe independently of
pacing: it is defense in depth against any out-of-band decision, and it is the invariant a future
relaxation of the gate would rely on (see the sliding-window entry under Alternatives, which
shifts the staleness defense entirely onto these debits).

Design notes:

- `deficit` aims just below the blocked band's ceiling: enough that the dispatch gate resumes for
  the band whose demand justified the destruction, and no deeper. The gate and the controller
  form a hysteresis pair around the same gauge; this coupling rule keeps them from fighting.
- `credit` is a crude mean-footprint estimate. It is, however, self-consistent in the gauge's
  units, and estimation error is bounded by the per-decision cap. The future capacity ledger
  replaces `credit` with exact per-lease footprints; nothing else in this loop changes (see
  Future Work).
- Sizing is never denominated in request counts (e.g. "evict one per queued request"). Queued and
  in-flight requests have unrelated footprints; count-based sizing evicts proportionally to demand
  burst size rather than to the capacity actually needed.

### Pacing: confirmation-gated

The controller may issue a new batch of revocations only when:

1. every previously issued revocation is **confirmed**: the tracker's eviction-terminated
   listener has fired for it, moving its debit into a cooling stage; and
2. `confirmationGrace` has elapsed since the last confirmation. Cooling debits expire after the
   grace, which covers the reclaiming stage (engine abort, KV GC, metrics scrape, staleness
   window) so the gauge has had a chance to reflect reality before the next deficit is computed.

A fixed cooldown fails in both directions. Shorter than the reclaim round-trip, it re-fires
against a gauge that has not yet absorbed prior actions, producing an over-eviction spiral bounded
only by running out of victims. Longer than necessary, it delays legitimate reclamation.

**Wedge protection.** If a revoked stream never terminates (hung client, proxy anomaly), the gate
would arm off forever. After `confirmationTimeout`, the revocation is treated as confirmed: its
debit enters the same cooling stage (so the grace clock restarts from the retirement) and a
dedicated metric is incremented. The evicted lease was already removed from the victim heap at
issue time, so a wedged lease is never re-selected; if its stream later terminates, the late
confirmation is ignored. Cost accounting: the wedged lease still holds real capacity after its
debit expires, so the next decision recomputes a deficit that includes it and kills fresh victims
to cover it. Timeouts therefore cost bounded extra destroyed work in addition to degraded
reclamation throughput; a nonzero timeout rate means the actuator path is unhealthy.

**Detector pairing.** The two saturation detectors err in opposite directions when a lease is
revoked:

- `concurrency-detector`: its in-flight counters decrement the moment the stream terminates, so
  the gauge *leads* physical reclamation. Failure mode: dispatch resumes slightly early and the
  new request briefly waits in the endpoint's local buffer. Cheap.
- `utilization-detector`: scraped KV/queue metrics reflect reclamation only after engine GC and
  the next scrape, so the gauge *lags*. Failure mode if paced too fast: destroyed work. Expensive.

A destructive actuator should be paired with the leading gauge, and the EPP derives the grace
accordingly (see Configuration). The stakes of that derivation are described in Validation: a
grace below the sensor's free-visibility lag kills a multiple of the required leases on shallow
demand, while against the lagging sensor even a correctly matched grace reclaims slowly enough
that natural churn can outperform eviction. The grace is the dominant throughput parameter; the
per-decision cap has little effect because sizing at the ceiling boundary issues only the deficit
regardless of the cap. Pairing eviction with the concurrency detector is therefore the
recommended deployment.

### Victim selection

Unchanged from what is merged:

- `sheddable-eviction-filter`: only `priority < 0` leases are revocable.
- `priority-then-time-eviction-order-policy`: lowest priority first; ties broken newest
  `DispatchTime` first (least invested work, fewest generated tokens discarded).

One known tension, accepted for v1: newest-first minimizes wasted work but frees the least
capacity per revocation, maximizing the number of victims needed. Footprint-aware selection
(fewest victims covering the deficit) requires per-lease footprints and is future work.

The filter and ordering policies are evaluated at lease-track time and maintained in a heap
(`EvictionQueue`). This is adequate for v1 because both shipped policies depend only on immutable
attributes. It is not the long-term policy surface: the heap structurally restricts policies to
static attributes (progress-based protection or endpoint-aware ordering cannot be expressed), and
pairwise comparators cannot express subset selection at all. See Future Work.

### Fate of evicted requests

The request is terminated with HTTP 429 (the ImmediateResponse status for evicted requests),
reason `evicted`, via the existing `EvictionRegistry` reason plumbing. No EPP-side retry in v1;
clients own retry policy. (Terminology caution: `QueueOutcomeEvictedTTL` and related queue
outcomes refer to pre-dispatch queue removal and are unrelated to lease revocation.)

### Placement and concurrency

The controller is a builtin component rather than a plugin: a small struct owned by the flow
controller's processor, evaluated synchronously on the HoL-break path of `dispatchCycle`. It is
cheap (a few comparisons; occasionally `EvictN` with small `n`), runs on the single-writer
processor goroutine, and holds one global pacing state. Confirmations arrive from ext_proc
handler goroutines via the eviction-terminated listener, so the pacing state is the one piece of
controller state that requires synchronization. This mirrors the existing division:
`RequestEvictor` is likewise "wired directly by the EPP, not a user-configurable plugin."

Why the sizing logic is not pluggable in v1: a sizing-policy interface fixed now would freeze
assumptions the dynamics have not yet justified, and the static-attribute constraint baked into
the current filter/ordering interfaces is the in-repo precedent for that failure mode.
`UsageLimitPolicy` was extracted after the legacy admission check proved the shape; the
reclamation policy interface should follow the same path.

### Configuration

The public API surface is a single field under `flowControl`:

| Field | Default | Meaning |
|---|---|---|
| `enableEviction` | `false` | Master switch. Enables the eviction wiring (evictor, registry, Director hooks) and the reclamation controller. |

The controller's remaining parameters are internal, because their correct values are properties of
the deployment rather than preferences:

- **Confirmation grace** is the paired sensor's confirmation-to-visibility lag bound, derived at
  wiring time from the selected saturation detector. For the concurrency detector, the gauge
  decrements in the same event chain as the confirmation, so the grace is 10ms (ten of the
  dispatch loop's 1ms ticks, buying jitter margin). For the utilization detector (and,
  conservatively, any custom detector), it is the metrics refresh interval + the staleness
  threshold + a fixed engine abort/GC budget (250ms). A mismatched grace in either direction has
  measured costs; see Validation.
- **Per-decision cap** (`maxRevocationsPerDecision`, 2) bounds damage from footprint
  misestimation. Sizing is deficit-bound in practice (see Validation), so the cap rarely binds
  and is not worth a knob.
- **Confirmation timeout** (10s) retires unconfirmed revocations as described under Wedge
  protection.

### Observability

- `flow_control_revocations_issued_total` (counter)
- `flow_control_revocations_total{outcome=confirmed|timed_out}` (counter; the outcome label is
  terminal and exclusive: every issued revocation eventually increments exactly one outcome, and
  a timed-out revocation whose stream later terminates does not additionally count as confirmed)
- `flow_control_reclaim_target` (gauge: the raw deficit `saturation - ceiling`, before debits are
  subtracted) and `flow_control_pending_reclaim` (gauge: the sum of outstanding and cooling
  debits, per the definitions under Sizing)
- `flow_control_revocation_confirmation_seconds` (histogram: issue to stream termination)
- Existing: pool saturation gauge, per-outcome request counters (reason `evicted`).

The useful operational signature is the pair (reclaim target, confirmation latency). A
persistently positive target with low confirmation latency means demand exceeds sheddable supply;
high confirmation latency means the actuator path (client abort handling) is the bottleneck.

### Validation

The controller design is validated in a closed-loop simulation: a synthetic pool of capacity
units, occupied by sheddable leases, with a configurable lag between a lease's confirmed
termination and the freed capacity appearing in the saturation gauge. The reclamation loop under
test is the real one; only the pool and sensor are synthetic, which means the actuator chain of
validity condition 3 is out of scope here and is covered by the end-to-end
acceptance criteria stated there. The simulation harness is part of the implementation work
that follows this proposal and makes these results reproducible when it lands. Evaluated across
sensor-lag profiles, grace values, per-decision caps, and burst shapes (single urgent request,
sticky batch, sustained arrivals), with eviction-disabled baselines:

- Relief times match the closed-form epoch model (confirmation round-trip + grace per decision)
  to within timer jitter, including the exact-boundary case (zero deficit) that the minimum-one
  rule handles.
- The grace is the dominant throughput parameter: against a leading (concurrency-style) sensor,
  tightening the grace from lagging-sensor scale to dispatch-tick scale improves burst goodput by
  orders of magnitude, without over-eviction.
- A grace below the sensor's free-visibility lag over-evicts on shallow demand, killing a
  multiple of the required leases; a grace at or above the lag over-evicted in no scenario.
- Against a lagging (utilization-style) sensor, even a correctly matched grace reclaims slowly
  enough that natural lease churn can outperform eviction, which is the basis for the
  concurrency-detector pairing recommendation.
- The per-decision cap has negligible effect: sizing at the ceiling boundary is deficit-bound and
  issues only one or two revocations per decision regardless of the cap.

## Alternatives Considered

**Count-based demand sizing with a deadline heuristic and fixed cooldown** (draft PR
[#1360](https://github.com/llm-d/llm-d-router/pull/1360)). Rejected on stability grounds rather
than mechanism (its wiring is sound and is reused here): (a) sizing by
`min(queuedCount, evictable)` assumes footprint parity between queued and in-flight requests and
scales destruction with burst size; (b) a fixed cooldown shorter than the reclaim round-trip
re-fires against a stale gauge, and with (a) the spiral is bounded only by exhausting the
sheddable population; (c) the `remaining <= elapsed * overloadRatio` deadline trigger contains no
drain-rate model, so near the common operating point it acts only after roughly half the deadline
budget is burned, which the reclaim round-trip then consumes; (d) demand was evaluated only for
the first blocked band, starving lower bands whenever a higher band is registered but empty.

**Fixed-k paced eviction** (evict at most k per confirmed epoch, no deficit estimate). Simpler and
equally stable, but reclaims at a fixed rate regardless of overload depth; under a deep burst it
visibly under-delivers. The deficit-debit controller costs little more and degenerates to fixed-k
when the estimate saturates the cap.

**Sliding-window pacing** (allow new decisions while fewer than W revocations are outstanding,
with staleness defense shifted onto the pending-reclaim debits). Prototyped and evaluated in the
closed-loop simulation; no measurable benefit at any window size on either sensor profile. The
result is structural: sizing is deficit-bound, and at the ceiling boundary the deficit is one or
two mean footprints, so the epoch rate, not the gate, limits reclaim throughput. A window becomes
worth revisiting only alongside a demand-aware sizing policy that can want more than the
instantaneous deficit.

**A pluggable reclamation-policy interface now.** Rejected for v1; see Placement.

**Pure holdback (no eviction).** Already available via `priority-holdback-policy`; strands
capacity in proportion to the reserve, and provides no recourse once low-priority leases occupy
the pool. Holdback and eviction compose; neither substitutes for the other.

## Future Work

- **Vector-valued reclamation (capacity ledger).** A planned redesign
  ([`flow-control-capacity-ledger.md`](flow-control-capacity-ledger.md)) models each request as a
  resource footprint (prefill compute; KV tokens; batch slots) held against per-endpoint ledgers
  that roll up to the pool, with a hold-then-lease lifecycle. In that world this controller
  survives intact with three type upgrades: the deficit becomes per-dimension, computed from a
  hold-fit failure (the blocked request's footprint minus available capacity; no ceilings, no
  thresholds); `credit` is replaced by exact per-lease footprints; the pending-reclaim debits
  become the ledger's released-but-unacknowledged accounting. The v1 ceiling-relative deficit is
  a scalar stand-in for hold-fit deficit; the ceiling semantics must not ossify into the contract.
- **Set-aware victim selection.** Replace heap-order selection with
  `VictimSelector(candidates, deficit) -> victims`: evaluated on demand against live lease views,
  able to express progress-based protection, endpoint-aware targeting, and minimum-waste coverage
  of a multi-dimensional deficit. The current filter/ordering policies become one trivial
  implementation of it.
- **Engine priority passthrough.** Forwarding request priority to the engine (vLLM supports
  priority scheduling/preemption) lets the engine preempt locally while EPP eviction handles the
  cross-endpoint case, and closes validity condition 2's residual gap (deep local buffers).
  Requires co-design with the engine community.
- **Partial eviction** (`PartialEvictionPolicy` in
  [#1119](https://github.com/llm-d/llm-d-router/issues/1119)): KV offload and resumption, as an
  alternative `Evictor` implementation; the mechanism axis is already pluggable.
- **Urgency-aware triggering**: deadline/SLO inputs to the demand predicate, as policy, once the
  base dynamics are proven in production.
