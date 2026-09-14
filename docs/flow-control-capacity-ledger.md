# Flow Control North Star: The Capacity Ledger

Status: **exploratory north star**. This document records direction, not commitment. Each major
section carries a confidence label:

- **Proposed**: settled shape; implementation may begin against it.
- **Directional**: the argument is made and the seam is fixed, but the mechanism behind the seam
  is expected to evolve.
- **Open**: known problem, no chosen answer.

Related: [`flow-control-eviction.md`](flow-control-eviction.md). The v1 eviction design is the
scalar projection of this design; its controller survives this redesign with type upgrades only.

## Summary

Flow control today reasons about pool capacity through a single delayed scalar (the saturation
gauge). This document proposes replacing that with a **capacity ledger**: every request is modeled
as a multi-dimensional resource **footprint**, held as a **lease** against per-endpoint ledgers
that roll up to a pool ledger. Admission, holdback, and eviction all become bookkeeping operations
against the ledger rather than threshold comparisons against a gauge.

The engine's own scheduler already runs this accounting model inside each replica. This design
raises it to pool scope, the one place the engine cannot.

The design also unifies the QoS story: tiers are admission against different confidence levels of
the same ledger. Guaranteed traffic reserves against the pessimistic bound; sheddable traffic is
statistically multiplexed against the expected value; and revocation (eviction) is the enforcement
mechanism that makes the overcommit safe.

## Motivation: what a scalar gauge cannot do

*(Proposed.)*

Recurring capacity-management defects in the flow control layer trace to the same root:
heterogeneous, multi-dimensional, lifecycle-varying resource claims are collapsed into one delayed
dimensionless number.

- **Eviction sizing** ([#1119](https://github.com/llm-d/llm-d-router/issues/1119)): "how many
  requests must be evicted" is unanswerable in gauge units. Requests have unrelated footprints,
  and the gauge reflects an eviction only after abort, GC, and a scrape. The v1 eviction design
  works around this with a mean-footprint estimate and pending-reclaim debits, a deliberately
  crude scalar shadow of this design.
- **Holdback stranding**: ceilings reserve a fraction of a gauge, not capacity for a class of
  footprints; the reserve cannot be sized to expected burst demand.
- **Token-mode under-count**: in-flight token accounting that releases at first-token conflates
  the prefill-compute claim (which does end at TTFT) with the KV-residency claim (which does not),
  admitting far past capacity during decode-heavy load. The bug is a lifecycle distinction the
  scalar cannot express.
- **Admission/scheduling interference**: gauges built from means mask per-endpoint skew; gating on
  a pool mean shapes the load scorers see and can hold the system at pathological equilibria.

The remedy is to account in the units the hardware enforces.

## The resource model

*(Proposed: the residency axes and the lifecycle split. Directional: the prefill axis. Open: the
shared-resource extension.)*

### Why these axes

Continuous-batching inference has three distinct per-request physical bottlenecks, each with a
distinct saturation mode and a distinct lifecycle. These are the ledger's vector dimensions;
shared, non-additive resources (see LoRA below) are handled outside the vector.

| Axis | Physical bottleneck | Saturates as | Lifecycle |
|---|---|---|---|
| `Prefill` (tokens) | SM compute: prompt evaluation is high-intensity GEMM over the input | TTFT spikes; decode preemption | **Transient**: claimed at admission, released the moment the first decode step is scheduled (TTFT) |
| `Residency.KVTokens` | VRAM capacity: KV history storage (PagedAttention blocks) | OOM / swap thrashing | **Persistent**: claimed at admission, released at end of stream |
| `Residency.Slots` | HBM bandwidth: decode is memory-bound; weights stream per active sequence per token | TPOT degradation; scheduler queue saturation (`max_num_seqs`) | **Persistent**: as above |

The transient/persistent split is the central modeling decision. Prefill compute and KV residency
have different release events and different failure modes; any accounting that merges them is
wrong in one direction or the other. The token-mode under-count above is this error observed in
production configuration.

The two residency axes have hardware-enforced stock limits (the block pool; `max_num_seqs`), which
is what earns them the Proposed label. The prefill axis is different in kind: prefill compute is a
rate, not a stock, and treating chunked-prefill tokens as inventory claimed against a pool-level
limit leaves that limit's semantics and units undefined. The axis is therefore Directional: the
lifecycle split it encodes is settled, but its limit model is not.

### Types

Two representations, one translation seam:

```go
// Prediction carries the per-request quantities the translation is computed from.
type Prediction struct {
    PromptTokens int64 // ISL, known at admission
    OutputTokens int64 // predicted OSL (see Stochastic layer)
    CachedTokens int64 // prefix-cache hit, known at scheduling; zero at admission
    Branching    int64 // decode width (best_of, beam, n)
    BlockSize    int64 // engine KV block size
    ChunkSize    int64 // engine chunked-prefill cap
}

// Footprint is the hardware-agnostic resource claim of one request. Pool-level
// admission reasons exclusively in these units.
type Footprint struct {
    Prefill   int64      // transient: unshared prompt tokens, capped by chunked-prefill size
    Residency Residency  // persistent: held until end of stream
}

type Residency struct {
    KVTokens int64 // (prompt - cached prefix) + predicted output * branching
    Slots    int64 // concurrent sequences (branching factor; maps to engine max_num_seqs)
}

// EngineFootprint is the engine-specific physical claim on one replica: block-granular,
// fragmentation- and copy-on-write-aware. Endpoint-level commits use these units.
type EngineFootprint struct {
    Prefill   int64
    Residency EngineResidency // KVBlocks, Slots
}

type Translator interface {
    ToFootprint(p Prediction) Footprint
    ToEngineFootprint(p Prediction) EngineFootprint
}
```

Design rules:

- **The Translator is a calibrated estimator.** Block-exact math (copy-on-write duplication,
  fragmentation, prefix-boundary rounding) couples the router to engine-version internals with no
  contract. Estimates carry error acknowledged by the reconciliation layer; the long-term fix is
  an engine API reporting actual per-request block usage (see Engine co-design).
- **Underflow is an error.** Footprints support coordinate-wise `Add`/`Sub`; underflow is
  surfaced as ledger corruption rather than silently clamped. Correctness comes from zero-sum
  discipline (a lease releases exactly what it committed).
- **Shared resources stay out of the vector** *(Open)*. LoRA adapter slots are set-union-scoped
  (the first request pays, co-tenants ride free), which breaks per-request additivity. They
  require a reference-counted side ledger, not a fourth coordinate.

## The ledger architecture

*(Proposed.)*

### Hold, then lease

Admission is a two-phase reservation protocol:

```
        TryAcquireHold(footprint)          Commit(endpoint, engineFootprint)
request -------------------------> HOLD ----------------------------------> LEASE
                                    |  TTL expiry / cancel                    |
                                    v                                        |  ReleasePrefill (TTFT)
                                 dropped                                     v
                                                               LEASE (residency only)
                                                                              |  Release (EOS)     natural
                                                                              |  Revoke (eviction)  forced
                                                                              v
                                                             reclaiming --> reclaimed
```

- A **hold** is a tentative, TTL-bounded, endpoint-unbound reservation taken before scheduling, in
  `Footprint` units. Holds exist only for the scheduling window (from the admission decision to
  commit or cancellation), not for queued requests, so the holds table stays small and a deep
  queue cannot zero out available capacity. Holds close the admit-then-schedule race at pool
  granularity: capacity checked at admission cannot be double-promised while scheduling runs. The
  race is not closed at endpoint granularity (two holds can each pass the fit check against the
  same lone endpoint); that residual is caught at commit time. TTL expiry cancels the admission
  and the request is rejected to the client, the same outcome as failing the admission check: the
  TTL reclaims capacity from scheduling stalls rather than acting as a queueing mechanism.
- A **lease** is the committed claim, bound to an endpoint, in `EngineFootprint` units. The
  *escalation guard* enforces `commit <= hold` per dimension, compared in logical units: the
  committed footprint's blocks are converted at the endpoint's block size, rounding the committed
  side up, so the guard is typed consistently at the logical/physical boundary and rounding can
  never excuse an escalation. Scheduling may not discover a larger footprint than admission
  approved.
- **Two release events**, per the lifecycle split: `ReleasePrefill` at TTFT frees the transient
  axis; `Release` at end of stream frees residency. **Revocation** (eviction) is a forced release:
  the same ledger operation with a different initiator, entering the same reclaiming/reclaimed
  accounting (released by the EPP, not yet acknowledged freed by the engine).
- Pool admission requires both an **aggregate check** (pool-wide available capacity covers the
  footprint) and a **fit check** (at least one healthy endpoint can hold it). Aggregate room with
  no single endpoint able to fit the request is not admissible capacity.

### Endpoint and pool ledgers

- `EndpointLedger`: the deterministic map of committed leases on one replica, plus the
  reclaiming-state accounting for released-but-unacknowledged capacity. Hot-path reads are
  lock-free snapshots.
- `PoolLedger`: registration/draining of endpoints, the holds table, and the roll-up:
  `Available = sum(limits) - sum(committed) - sum(holds) - sum(reclaiming)`.
- Every consumer that today reads the saturation gauge becomes a view over the ledger: the
  dispatch gate asks "does the head request's hold fit"; holdback reserves footprint-denominated
  headroom per tier; the eviction controller computes a per-dimension deficit from a hold-fit
  failure. The `SaturationDetector` abstraction survives as a derived, backwards-compatible view
  (saturation approximately equals the max over dimensions of used/limit), not as the source of
  truth.

### What this does to the eviction controller

Nothing structural; that invariance is the design goal (see `flow-control-eviction.md`). Three
type upgrades:

| v1 (scalar) | Ledger world |
|---|---|
| `deficit = saturation - ceiling` | per-dimension deficit: `blocked hold's Footprint - Available` |
| `credit = saturation / leases` (mean estimate) | exact per-lease `EngineFootprint` from the ledger |
| pending-reclaim debits (controller-local) | the ledger's reclaiming-state accounting |

Victim selection graduates from heap-order to subset selection: choose the minimum-waste set of
revocable leases whose footprints cover the deficit vector (`VictimSelector(candidates, deficit)`).

## Reconciliation: two sources of truth

*(Directional. The seam and the argument are fixed; the estimator behind the seam is not.)*

The ledger is a predicted view (footprints are estimates over predicted output lengths); scraped
engine telemetry is a delayed view of reality. Something must close the loop. The design
principle: correct the ledger at the events that reveal actual values, and model only the one
quantity no event reveals (time to release).

Filtering approaches (Kalman-style or observer-based bias estimation) are rejected. They earn
their complexity when a system has continuous hidden dynamics observable only indirectly. Here,
capacity changes in discrete jumps at knowable events (commit, TTFT, EOS, abort), and the dominant
reconciliation errors are systematic (output-length over-prediction, prefix-cache discounts,
translation drift) and correctable at those events. A filter also faces an identification problem:
it cannot distinguish "predictions run 20% high" from "three requests completed and the scrape has
not landed," and the guard machinery that distinction demands grows without bound. An estimator
that needs that much protection signals a model mismatch.

Event truth-up instead:

- **At scheduling**: actual cached-prefix tokens are known; replace the zero-cache-hit pessimistic
  prefill/KV estimate.
- **At EOS**: actual output length is known; the lease releases its committed footprint exactly
  (zero-sum), and the prediction error feeds the predictor, not the ledger.
- **At abort/eviction**: the revocation event marks the lease reclaiming; engine acknowledgment
  (completion/abort counters, block counts) retires it. *(Open: engines count aborts and natural
  completions in different metrics; the acknowledgment channel must include both or reclaiming
  entries stall.)*
- **Per scrape**: telemetry validates the roll-up and catches drift (translation error, missed
  events); persistent per-endpoint discrepancy is surfaced as calibration error on the Translator
  rather than silently absorbed.

## Stochastic layer: hazard-based release modeling

*(Directional: the framing and its uses. Open: estimator choice.)*

After truth-up, one quantity remains uncertain: when each active lease will release. Output length
is a random variable; everything the ledger wants to know about the future is a function of its
distribution.

### The framing

Let `L` be a request's output length (tokens). Define, per flow (or model/workload class):

- Survival: `S(n) = P(L > n)`, the probability a request generating its n-th token continues.
- Hazard: `h(n) = P(L = n | L >= n)`, the completion intensity at age n.
- Mean residual life: `m(n) = E[L - n | L > n]`, the expected remaining tokens given age n.

A lease at decode age `n` then has an expected remaining residency time of roughly
`m(n) / decode_rate`, and the pool has an expected capacity supply schedule. For horizon `t`:

```
ExpectedRelease(t) = sum over active leases i of P(L_i <= n_i + r_i*t | L_i > n_i) * footprint_i
```

(`n_i` = current age, `r_i` = decode rate; `footprint_i` is the lease's committed footprint, an
approximation of the release amount that truth-up corrects at EOS.) This is the quantity a scalar
gauge cannot provide: not how full the pool is, but how fast it will empty.

### Why age-conditioning is justified

If output lengths were geometric (memoryless), age would carry no information: `m(n)` constant,
every lease equally close to completion, and dispatch-time ordering heuristics vacuous.
Empirically, LLM output-length distributions are strongly non-memoryless (mode near typical
response lengths, heavy right tails from long generations), so `h(n)` and `m(n)` vary with age and
age-conditioned decisions dominate age-blind ones. This is the first-principles justification for
hazard modeling over both filtering (which models the wrong noise) and static heuristics (which
discard the age information).

### Where it plugs in

Each consumer reads the same two views of the ledger: the **guaranteed bound** (deterministic,
worst-case) and the **expected view** (hazard-discounted).

- **Tiered admission.** Guaranteed-tier holds are checked against the pessimistic bound: capacity
  is reserved, with OOM-shield semantics. Sheddable-tier holds may be checked against the expected
  view: capacity is statistically multiplexed (overcommitted), and revocation is the enforcement
  mechanism that makes the overcommit safe. When the gamble loses (a tail event: releases arrive
  slower than predicted), sheddable leases are revoked to restore the guaranteed tier's bound.
  This is the architecture of airline overbooking and effective-bandwidth admission in telecom,
  applied to KV residency. In this frame, eviction is a prerequisite for efficient admission
  rather than a repair mechanism bolted onto it.
- **Eviction: wait-vs-evict.** Revoke only when expected natural supply misses demand:
  `ExpectedRelease(t) < deficit` for the blocked demand's tolerance horizon `t`. This is the
  principled answer to "how many to evict"; often the answer is zero because capacity frees itself
  within the tolerance. It is the term the v1 eviction design approximates with
  confirmation-gated pacing.
- **Holdback sizing.** Reserve headroom sized to expected burst demand over the reclaim horizon
  (arrival model * footprint distribution), replacing hand-tuned ceiling fractions.
- **Prediction.** The same distributions serve as priors for per-request OSL prediction
  (`m(0)` is the unconditional mean), tightened per-request by any upstream predictor; EOS
  truth-up supplies the training signal for free.

### Open questions

- Estimator form: empirical per-flow survival curves (histogram or Kaplan-Meier style; cheap,
  assumption-free) vs. parametric fits (compact; extrapolate tails better). Likely start
  empirical.
- Conditioning variables: flow identity, prompt length, model. How much stratification before data
  sparsity dominates?
- Non-stationarity: workload mix shifts; windowing/decay policy for the curves.
- Decode-rate variability under load: the rate itself depends on batch occupancy, a second-order
  coupling ignored until first-order value is proven.

## Migration path

*(Proposed.)*

Each stage subsumes the previous stage's bookkeeping; no stage requires rework of the eviction
controller or the dispatch gate's structure.

1. **Scalar eviction**: gauge-unit deficit and pending-reclaim debits
   (`flow-control-eviction.md`).
2. **Dual ledger**: `KVTokens` + `Slots` held dispatch-to-EOS with TTFT prefill release. This is
   the two-axis degenerate Footprint; it fixes the token-mode under-count and the dispatch race,
   and needs token estimation but no block-level translation.
3. **Footprint ledger**: full types, hold-then-lease protocol, event truth-up;
   `SaturationDetector` becomes a derived view.
4. **Stochastic layer**: hazard curves, expected-release schedules, tiered admission against dual
   confidence levels; `VictimSelector` with deficit-covering subset selection.
5. **Engine co-design**: the engine reports actual per-request block usage (retiring Translator
   estimation) and accepts priority (local preemption; EPP handles only the cross-endpoint case).

## Relationship to existing components

- `concurrency-detector` + `inflight-load-producer` are a proto-ledger (stage 2 grows out of
  them); `utilization-detector` becomes a reconciliation input rather than the primary gauge.
- `UsageLimitPolicy` survives as the tier-policy seam; its ceilings become footprint-denominated
  reserves in stage 4.
- The eviction plumbing (`RequestEvictor`, `Evictor`, `EvictionRegistry`, ext_proc channel) is
  unchanged throughout; only selection and sizing upgrade.

## Prior art and what is claimed

Every component has direct precedent: multi-dimensional resource vectors (Borg/Kubernetes,
DRF), two-phase TTL'd reservations (slot booking, allocators), overcommit with revocation (airline
overbooking, statistical multiplexing and effective bandwidth in telecom), survival analysis
(reliability theory). The claimed contribution is the synthesis and its placement: engine-grade,
lifecycle-split, revocation-capable capacity accounting at the fleet choke point.
