/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package groupsched

// THE STATE MACHINE'S NOUNS: the thread, the waiter, the request record, the engine group, and the
// Scheduler that owns them — plus the engine life cycle and the whole-policy Snapshot.
//
// THE LOCK CONVENTION, stated once, for every file in this package:
//   - EXPORTED methods take s.mu themselves and are safe to call from anywhere;
//   - UNEXPORTED methods ASSUME the caller already holds it — no name says so, because there is no
//     other kind of unexported method here;
//   - the BlockIndex has no lock of its own: it is called only under s.mu.

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/go-logr/logr"
)

// ---- the queue-head snapshot (waiting state comes from the flow-control
// queues at Pick time, never from a hook) ----

// Head is one waiting queue head as seen at Pick: the thread id (the fairness unit AND the
// queue key), the request id, the arrival, and the request's exact prefill chains.
type Head struct {
	// Thread is the group policy's unit: the derived thread id, or "" for anonymous traffic
	// (never tracked, never seated).
	Thread  string
	ReqID   string
	Arrival time.Time
	// Chains is the request's EXACT prefill chains, computed once from the tokens the token
	// producer rendered at arrival (one chain per prompt).
	Chains []Chain
	// FairnessID is the request's flow-control identity — its QUEUE KEY.
	FairnessID string
	// WireID is the client's OWN id for this request, from its headers (x-correlation-id, else
	// x-request-id): the one field that joins a SEAT/EXIT line to the client's own request records,
	// whatever the queue key turned out to be.
	WireID string
	// Returner reports whether this request is NOT its thread's first turn — the fairness class,
	// from thread identity's arrival record.
	Returner bool
}

// ---- per-thread scheduling state ----

// thread is one seated residency: a record that exists exactly as long as the thread holds a
// seat on ONE engine: the seat IS the binding.
type thread struct {
	id string
	// seatTime orders exits: earliest-seated leaves first.
	seatTime time.Time
	// seatTurns counts settled turns in this residency (the residency histogram's unit).
	seatTurns int
	// inflight and pending are DERIVED each decision point from the request map (derive-
	// don't-accumulate): a leaked increment cannot persist past a reconciliation.
	inflight int
	pending  int
	// busyPeriods counts completed busy periods: the thread's in-flight count reaching zero and
	// staying there, which — one thread, one chain — is the whole of its work draining.
	busyPeriods int
	// lastCompletion is the last time the thread's in-flight count returned to zero.
	lastCompletion time.Time
	// lastActivity is the last arrival, dispatch or completion.
	lastActivity time.Time
	// coldEntry is this thread's outstanding cold-entry charge in TOKENS against the stagger cap;
	// released EXACTLY once, at the first of: prefill complete, any termination, unseat.
	coldEntry int64
	coldDone  bool
	// seatObs is the observability of the turn that SEATED this thread, measured at the seat before
	// its own reservation: the first dispatch after the seat reports it instead of a post-insert
	// view (which would count the reservation as residency). Cleared by that dispatch.
	seatObs *dispatchObservables
	// waitingSince is the thread's oldest queued head: the wait its victims and its SEAT event
	// are measured from, kept OLDEST across fan-out re-arrivals.
	waitingSince time.Time
	// headChains / headReturner are the LAST queued head observed for this thread, from the queue
	// view of a decision pass.
	headChains   []Chain
	headReturner bool
	// exitingSince is when this thread was marked exiting (the mid-flight class): the age of a
	// retiree is the time between "nothing more will be admitted for it" and "its blocks are off
	// the budget", i.e. how long the engine pays for a seat it has already written off.
	exitingSince time.Time
	// fairnessID is the last fairness id (queue key) this thread was seen with — the queue key
	// its SEAT/EXIT events name (see Head.FairnessID).
	fairnessID string
	// wireID is the last client request id this thread was seen with: the field a SEAT/EXIT line
	// joins a request record on (see Head.WireID).
	wireID string
}

// busy reports whether the thread has work in progress: a request at a pod, or a LIVE released
// window (a turn that a Pick released and PreRequest has not yet brought to a pod).
func (m *thread) busy() bool { return m.inflight > 0 || m.pending > 0 }

// noQueuedHead reports that this thread has nothing waiting in the queue. It is NOT the same
// question as "no turn waiting": a thread whose head a Pick has already released has no queued
// head and a turn on the way.
func (m *thread) noQueuedHead() bool { return m.waitingSince.IsZero() }

// reqRec is ONE tracked request's record, and the scheduler's only memory of a request that is
// not in a queue.
type reqRec struct {
	// thread is the thread that released the request (never empty: anonymous traffic is not
	// tracked).
	thread string
	// chains is the prefill chain the release was charged with.
	chains []Chain
	// engine is the engine the request landed on, once an observation named one.
	engine string
	// released is the Pick that released the request.
	released time.Time
	// pod is when the request reached a pod (PreRequest); zero while it is pending.
	pod time.Time
	// the OBSERVABLES of this turn, captured at dispatch for the span PreRequest may set. Nothing
	// in a decision reads them: the queue wait is dispatch − arrival, and the predicted cache is
	// the resident prefix read BEFORE the charge (after it, it would always be everything).
	queueWait       time.Duration
	predictedCached int64
	promptTokens    int64
}

// waiter is one head in the SHARED waiter queue.
type waiter struct {
	id    string
	since time.Time // the OLDEST arrival: a fan-out re-arrival does not reset it
	// returner is the fairness class: this thread's queued turn is NOT its first one.
	returner bool
	// chains is the head's exact prefill chain set.
	chains []Chain
	// fairnessID rides with the head (see Head.FairnessID) so a SEAT event can name the queue key
	// even before the thread has a record, and wireID rides with it for the join to the client's
	// own request record (see Head.WireID).
	fairnessID string
	wireID     string
}

// engineGroup is everything that is a fact about ONE engine's HBM: its block index (dedup is per
// engine — a block on A is worthless to B), its budget, its measured capacity, its seats, its
// cold-entry charge and its counters.
type engineGroup struct {
	// id is the endpoint id the EPP discovered.
	id string

	index *BlockIndex // attached at creation (block size from config)
	// budgetBlocks is what this engine may seat, in blocks: groupBudgetFraction x THIS engine's
	// measured capacity. 0 until THIS engine's num_gpu_blocks arrives.
	budgetBlocks int
	numBlocks    int
	// blockSizeMismatch is the loud per-engine wiring failure (a measured CacheBlockSize that
	// differs from the config's BlockSizeTokens).
	blockSizeMismatch error

	threads map[string]*thread
	// coldInFlight is this engine's outstanding cold-entry charge against its own stagger cap, so
	// N engines allow N concurrent cold prefills and no more.
	coldInFlight int64

	counters struct {
		served int64 // dispatches served on this engine
		exits  map[string]int64
	}
}

// coldCapTokens is THIS engine's cold-ENTRY stagger cap, in tokens. It is NOT a knob: it is the
// seat budget's own headroom — floor((1 - groupBudgetFraction) x num_gpu_blocks) blocks x
// blockSizeTokens — so the valve can only ever spend the margin the budget already reserved.
func (g *engineGroup) coldCapTokens(fraction float64, blockSize int) int64 {
	if g.numBlocks <= 0 || blockSize <= 0 {
		return 0
	}
	headroomBlocks := int64((1 - fraction) * float64(g.numBlocks))
	return headroomBlocks * int64(blockSize)
}

// newEngineGroup builds one engine's group: its block index, and the counters it will report.
func newEngineGroup(id string, blockSizeTokens int) (*engineGroup, error) {
	idx, err := NewBlockIndex(blockSizeTokens)
	if err != nil {
		return nil, err
	}
	g := &engineGroup{id: id, index: idx,
		threads: map[string]*thread{}}
	g.counters.exits = map[string]int64{}
	return g, nil
}

// engineLabel is the metrics/event label value for an engine id.
func engineLabel(id string) string { return id }

// Scheduler is the group scheduler state machine over an engine set.
type Scheduler struct {
	mu  sync.Mutex
	cfg Config
	// metrics is the Prometheus surface. Every observation is a DIRECT call on it, and every
	// method is nil-safe, so a bare Scheduler (the unit tests) records nothing without any
	// wiring.
	metrics *groupMetrics
	// logger carries the two lines a run's artifacts have to be able to point at: an affinity
	// violation and an engine disappearing under its threads. The zero Logger is safe.
	logger logr.Logger
	// onEvent is the ONLY observer: the forensic seat/exit/rebind stream. The plugin logs it at
	// V(DEBUG) and the tests collect it; nothing else watches the pass.
	onEvent func(Event)
	// groups holds one engineGroup per DISCOVERED engine, keyed by endpoint id.
	groups map[string]*engineGroup
	// groupOrder is the sorted key list (deterministic iteration: every cross-engine decision is
	// taken in id order, never map order).
	groupOrder []string

	blockSize int

	// bindings is the thread→engine association.
	bindings map[string]string

	// waiters is THE SHARED WAITER QUEUE: one entry per thread with a queued head and no seat.
	waiters map[string]*waiter

	// reqs is EVERY tracked request, keyed by request id: the record written when a Pick releases
	// a turn and dropped at its settle, its abandon, its reap, or its thread's exit.
	reqs map[string]*reqRec

	// now is the virtual clock of the CURRENT decision pass (set at the top of every public
	// decision entry point).
	now time.Time

	counters struct {
		// anomalies counts the EVENTS THE POLICY CANNOT ACT ON, by kind (anomalyKinds).
		anomalies map[string]int64
		// affinityViolations counts requests that LANDED on an engine other than the one they were
		// seated on (CORRECTNESS: the observed endpoint wins, so each violation is repaired by
		// moving the seat; a nonzero steady rate means the placement path is not honouring the pin).
		affinityViolations int64
		engineGone         int64
		exits              map[string]int64
		served             int64
		fcfsServed         int64
	}
}

// NewScheduler builds the state machine.
func NewScheduler(cfg Config) (*Scheduler, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	s := &Scheduler{cfg: cfg, logger: logr.Discard(), blockSize: cfg.BlockSizeTokens,
		groups:   map[string]*engineGroup{},
		bindings: map[string]string{},
		reqs:     map[string]*reqRec{},
		waiters:  map[string]*waiter{}}
	s.counters.exits = map[string]int64{}
	s.counters.anomalies = map[string]int64{}
	return s, nil
}

// groupFor returns (creating) the group for an engine id.
func (s *Scheduler) groupFor(id string) (*engineGroup, error) {
	if g, ok := s.groups[id]; ok {
		return g, nil
	}
	g, err := newEngineGroup(id, s.blockSize)
	if err != nil {
		return nil, err
	}
	s.groups[id] = g
	s.rebuildGroupOrder()
	// this engine's zero-valued COUNTER series exist from its first scrape, whatever its capacity
	// turns out to be (a budgetless or block-size-mismatched group included)
	s.metrics.preInstantiateEngine(id)
	return g, nil
}

// rebuildGroupOrder re-derives the sorted engine id list every cross-engine decision iterates in.
func (s *Scheduler) rebuildGroupOrder() {
	ids := make([]string, 0, len(s.groups))
	for id := range s.groups {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	s.groupOrder = ids
}

// groupOf is the thread's SEAT engine (nil when the thread is unseated).
func (s *Scheduler) groupOf(thread string) *engineGroup {
	id, ok := s.bindings[thread]
	if !ok {
		return nil
	}
	return s.groups[id]
}

// activeGroups is the deterministic engine list every decision pass iterates: every
// discovered engine, in id order.
func (s *Scheduler) activeGroups() []*engineGroup {
	out := make([]*engineGroup, 0, len(s.groupOrder))
	for _, id := range s.groupOrder {
		out = append(out, s.groups[id])
	}
	return out
}

// SetNumBlocksFor attaches ONE engine's measured capacity.
func (s *Scheduler) SetNumBlocksFor(engine string, blockSizeTokens, numBlocks int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	g, err := s.groupFor(engine)
	if err != nil {
		return err
	}
	if blockSizeTokens > 0 && blockSizeTokens != s.blockSize {
		g.blockSizeMismatch = fmt.Errorf(
			"groupsched[%s]: measured CacheBlockSize %d != configured BlockSizeTokens %d (the token processor's value); refusing the capacity",
			engineLabel(g.id), blockSizeTokens, s.blockSize)
		return g.blockSizeMismatch
	}
	if g.numBlocks > 0 || numBlocks <= 0 {
		return nil // first capacity wins, PER ENGINE
	}
	g.numBlocks = numBlocks
	// The seat budget: the deployment's own statement about how much of this engine's KV one
	// group may hold.
	g.budgetBlocks = int(s.cfg.GroupBudgetFraction * float64(numBlocks))
	return nil
}

// HasCapacity reports whether ANY engine has a measured num_gpu_blocks (the pre-capacity FCFS
// pass-through's whole-stack condition).
func (s *Scheduler) HasCapacity() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, g := range s.groups {
		if g.numBlocks > 0 {
			return true
		}
	}
	return false
}

// HasCapacityFor reports whether THAT engine's capacity has arrived. Until it has, the engine
// takes no seats: an engine without metrics would serve nothing while looking healthy.
func (s *Scheduler) HasCapacityFor(engine string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	g, ok := s.groups[engine]
	return ok && g.numBlocks > 0
}

// GroupLabel resolves an endpoint key to the label the metrics and events carry for it.
func (s *Scheduler) GroupLabel(engine string) string {
	return engineLabel(engine)
}

// BlockSizeMismatches is every mismatched engine, in id order.
func (s *Scheduler) BlockSizeMismatches() map[string]error {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]error{}
	for _, id := range s.groupOrder {
		if err := s.groups[id].blockSizeMismatch; err != nil {
			out[engineLabel(id)] = err
		}
	}
	return out
}

// FCFSServed counts a pre-capacity dispatch: it belongs to no engine (a pre-capacity stack has
// none to attribute it to), so it is only counted.
func (s *Scheduler) FCFSServed() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counters.fcfsServed++
}

// ---- the seat/exit event log (spans cannot see seats, so every run needs it) ----

// Event is one seat/exit/rebind decision, per id, with reason and timestamp.
type Event struct {
	T      time.Time `json:"t"`
	Kind   string    `json:"kind"`   // SEAT | EXIT | REBIND | AFFINITY-VIOLATION
	ID     string    `json:"id"`     // the thread id the decision was taken for
	Engine string    `json:"engine"` // the engine label the decision happened on
	Detail string    `json:"detail"` // reason=… turns=… / kind=… wait=… / from=… to=… / fairness_id=… wire_id=…
}

// recordEvent books one seat/exit event: the engine attribution plus the observer. The Event is
// the forensic stream (the adapter logs it at V(DEBUG)); nothing keeps a copy.
func (s *Scheduler) recordEvent(kind, id, detail string, now time.Time) {
	ev := Event{T: now, Kind: kind, ID: id, Detail: detail}
	if g := s.groupOf(id); g != nil {
		ev.Engine = engineLabel(g.id)
	}
	if s.onEvent != nil {
		s.onEvent(ev)
	}
}

// anomalyNoTokens opens the anomaly kinds: each names ONE event the policy cannot act on, and each is
// counted under its own name in anomalies_total. This one is the render failure — a Pick pass saw a
// thread head with no rendered tokens, so the head could not be seated or charged.
const (
	// anomalyNoTokens: a Pick pass saw a thread head with no rendered tokens, so it cannot be
	// seated or charged (the render failed; thread identity counts that too).
	anomalyNoTokens = "no_tokens_head"
	// anomalyNoRequestID: a Pick released a turn for a request with no id.
	anomalyNoRequestID  = "no_request_id"
	anomalyReaped       = "reaped_inflight"
	anomalyReapedWindow = "reaped_window"
)

// anomalyKinds is every kind, in a fixed order (the metric pre-instantiates one zero series
// per kind: a zero is a fact, an absent series is a question).
var anomalyKinds = []string{anomalyNoTokens, anomalyNoRequestID, anomalyReaped, anomalyReapedWindow}

// noteAnomaly books one anomaly and tells the observer.
func (s *Scheduler) noteAnomaly(kind string) {
	if s.counters.anomalies == nil {
		s.counters.anomalies = map[string]int64{}
	}
	s.counters.anomalies[kind]++
	s.metrics.ObserveAnomaly(kind)
}

// countExit books ONE departure on the engine AND in the global tally (Snapshot.Exits
// stays the whole-policy view).
func (s *Scheduler) countExit(g *engineGroup, reason string) {
	if g != nil {
		g.counters.exits[reason]++
	}
	s.counters.exits[reason]++
}

// Snapshot is the whole-policy view of the state machine: DumpState serialises it and the metrics
// refresh reads it, so it carries exactly what either of them reports.
type Snapshot struct {
	Seats int
	// NEBlocks is the USED side: every seated thread's distinct blocks, exiting threads INCLUDED
	// (the engine pins them until their requests drain).
	NEBlocks     int
	BudgetBlocks int
	// WaitersShared is the SHARED waiter queue's size — the whole policy's admission queue.
	WaitersShared int
	Exits         map[string]int64
	Served        int64
	FCFSServed    int64
	// Engines is the PER-ENGINE section.
	Engines map[string]EngineSnapshot `json:"engines,omitempty"`
	// AffinityViolations counts requests that LANDED off their thread's seat engine (the
	// correctness counter: each one is repaired by the observed endpoint winning).
	AffinityViolations int64 `json:"affinityViolations"`
	EngineGone         int64 `json:"enginesGone"`
	// Anomalies is the should-be-zero ledger, by kind (anomalyKinds): every entry is an event the
	// policy could not act on.
	Anomalies map[string]int64 `json:"anomalies,omitempty"`
}

// EngineSnapshot is ONE engine's group state: seats, the budget, the exit mix and the exiting
// ledger.
type EngineSnapshot struct {
	ID    string `json:"id"`
	Seats int    `json:"seats"`
	// NEBlocks is the USED side (exiting threads included): see Snapshot.NEBlocks.
	NEBlocks     int   `json:"neBlocks"`
	BudgetBlocks int   `json:"budgetBlocks"`
	NumBlocks    int   `json:"numBlocks"`
	ColdInFlight int64 `json:"coldInFlightTokens"`
	// ColdCapTokens is the DERIVED cold-entry stagger cap (the budget's own headroom, in tokens).
	ColdCapTokens int64 `json:"coldCapTokens"`
	// The EXITING ledger (the enforcement view's own inputs, per engine): threads on their way
	// out, the blocks of the used side they still hold, and how long the oldest one has been
	// leaving.
	ExitingThreads    int              `json:"exitingThreads"`
	ExitingBlocks     int              `json:"exitingBlocks"`
	ExitingOldestAgeS float64          `json:"exitingOldestAgeSeconds"`
	Served            int64            `json:"served"`
	Exits             map[string]int64 `json:"exits,omitempty"`
	BlockSizeErr      string           `json:"blockSizeMismatch,omitempty"`
}

// Snapshot is the whole-policy view of the state machine: DumpState serialises it and the scrape
// collector reads it. It takes the caller's clock because one field in it — the oldest exiting
// thread's age — is a DURATION, and the scheduler's own clock only advances when the pass runs: an
// idle stack would otherwise report a frozen age, and that gauge is the drain warning.
func (s *Scheduler) Snapshot(now time.Time) Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.snapshot(now)
}

// snapshot is Snapshot under the caller's lock.
func (s *Scheduler) snapshot(now time.Time) Snapshot {
	snap := Snapshot{Exits: map[string]int64{},
		Served: s.counters.served, FCFSServed: s.counters.fcfsServed,
		AffinityViolations: s.counters.affinityViolations,
		EngineGone:         s.counters.engineGone,
		Engines:            map[string]EngineSnapshot{}}
	for k, v := range s.counters.exits {
		snap.Exits[k] = v
	}
	if len(s.counters.anomalies) > 0 {
		snap.Anomalies = map[string]int64{}
		for k, v := range s.counters.anomalies {
			snap.Anomalies[k] = v
		}
	}
	var seats, ne int
	for _, id := range s.groupOrder {
		g := s.groups[id]
		label := engineLabel(id)
		exitingMembers, exitingOldest := 0, time.Duration(0)
		for id, m := range g.threads {
			if !g.index.Exiting(id) {
				continue
			}
			exitingMembers++
			if age := now.Sub(m.exitingSince); age > exitingOldest {
				exitingOldest = age
			}
		}
		es := EngineSnapshot{ID: label, Seats: len(g.threads),
			ExitingThreads: exitingMembers, ExitingBlocks: g.index.ExitingBlocks(),
			ExitingOldestAgeS: exitingOldest.Seconds(),
			NEBlocks:          g.index.UniqueBlocks(),
			BudgetBlocks:      g.budgetBlocks, NumBlocks: g.numBlocks,
			ColdInFlight: g.coldInFlight, Served: g.counters.served,
			ColdCapTokens: g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize),
		}
		if g.blockSizeMismatch != nil {
			es.BlockSizeErr = g.blockSizeMismatch.Error()
		}
		if len(g.counters.exits) > 0 {
			es.Exits = map[string]int64{}
			for k, v := range g.counters.exits {
				es.Exits[k] = v
			}
		}
		snap.Engines[label] = es
		seats += es.Seats
		ne += es.NEBlocks
		snap.BudgetBlocks += g.budgetBlocks
	}
	snap.Seats, snap.NEBlocks = seats, ne
	snap.WaitersShared = len(s.waiters)
	return snap
}
