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

// THE CORE MECHANISM SUITE.

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// ---- shared helpers ----

// keys builds a deterministic distinct-key set of n blocks.
func keys(n int, seed Hash) []Hash {
	out := make([]Hash, 0, n)
	prev := seed
	for i := 0; i < n; i++ {
		prev = prev*0x9E3779B97F4A7C15 ^ Hash(i+1) ^ 0xC2B2AE3D27D4EB4F
		out = append(out, prev)
	}
	return out
}

// seedFor derives a DISTINCT block key set per (thread, size) — two threads must not share
// blocks by accident, or the index's dedup would hide the second one's footprint.
func seedFor(id string) Hash {
	h := Hash(1469598103934665603)
	for i := 0; i < len(id); i++ {
		h = (h ^ Hash(id[i])) * 1099511628211
	}
	return h
}

// chainOf is the EXACT prefill chain a head of n blocks of a thread's own content has: the
// blocks are the thread's own keys, so two threads share nothing unless a test gives them the
// same seed on purpose.
func chainOf(id string, blocks int) []Chain {
	return []Chain{{Keys: keys(blocks, seedFor(id)), Tokens: blocks * 64}}
}

// chainPrefixedBy is a chain whose first prefixBlocks keys are another thread's: the shared-
// prefix shape (a fork, or a turn that grows an existing prompt).
func chainPrefixedBy(prefix string, prefixBlocks int, own string, ownBlocks int) []Chain {
	k := append(keys(prefixBlocks, seedFor(prefix)), keys(ownBlocks, seedFor(own))...)
	return []Chain{{Keys: k, Tokens: (prefixBlocks + ownBlocks) * 64}}
}

// head is one queue head for thread id (ago = how long it has waited), carrying the thread's
// exact chain of n blocks.
func head(id string, ago time.Duration, blocks int) Head {
	return Head{Thread: id, ReqID: "r-" + id, Arrival: time.Now().Add(-ago),
		Chains: chainOf(id, blocks)}
}

// headWithChain is head for a test that builds the chain itself (shared prefixes).
func headWithChain(id string, ago time.Duration, chains []Chain) Head {
	return Head{Thread: id, ReqID: "r-" + id, Arrival: time.Now().Add(-ago), Chains: chains}
}

// dispatch drives the released-request path: Pick chose the head, the framework picked the
// engine, PreRequest records the in-flight request.
func dispatch(t *testing.T, s *Scheduler, id string) {
	t.Helper()
	s.Pass(nil, s.now)
	// the pass clock, not time.Now(): the request reaches the pod inside the fixture's own
	// timeline, which is what makes the released window's liveness (and every derived count)
	// answerable in the same units the passes use.
	s.releaseThread(id, "r-"+id, nil, s.now)
	s.Dispatched("r-"+id, s.now)
}

// settleChunk is the first response chunk (prefill complete) + the natural settle.
func settle(s *Scheduler, id string, prompt, cached int) {
	s.Chunk(id)
	s.Settle(s.reqOf(id), "", time.Now())
}

// ---- the request map, for a test that declares a thread's request state directly ----
// A request is tracked by ONE record (scheduler.go's reqRec), so a test that wants "this
// thread has a request at a pod" writes that record the way PreRequest does and lets the
// thread's derived counts follow it — never the fields directly, which is exactly the
// duplication the record removed.

// inFlight replaces m's at-a-pod records with reqID (the caller holds s.mu).
func inFlight(s *Scheduler, m *thread, now time.Time, reqID string) {
	noInFlight(s, m)
	addInFlight(s, m, now, reqID)
}

// addInFlight adds one at-a-pod record to m (the caller holds s.mu).
func addInFlight(s *Scheduler, m *thread, now time.Time, reqID string) {
	s.reqs[reqID] = &reqRec{thread: m.id, released: now, pod: now}
	m.inflight++
}

// noInFlight drops m's at-a-pod records (the caller holds s.mu).
func noInFlight(s *Scheduler, m *thread) {
	for rid, rec := range s.reqs {
		if rec.thread == m.id && !rec.pod.IsZero() {
			delete(s.reqs, rid)
		}
	}
	m.inflight = 0
}

// pending replaces m's released-but-not-at-a-pod record with reqID ("" clears it).
func pending(s *Scheduler, m *thread, now time.Time, reqID string) {
	for rid, rec := range s.reqs {
		if rec.thread == m.id && rec.pod.IsZero() {
			delete(s.reqs, rid)
		}
	}
	m.pending = 0
	if reqID == "" {
		return
	}
	s.reqs[reqID] = &reqRec{thread: m.id, released: now}
	m.pending = 1
}

// releasedWindow is the decision pass that lands between a release and PreRequest in
// production (Pick runs at ~1 kHz; a fixture that dispatches immediately has no window at all,
// which is how every rule that read the window as idle stayed hidden).
func releasedWindow(s *Scheduler, now time.Time) {
	s.Pass(nil, now)
}

// engineSet is one twin fixture: n engines each with measured capacity.
type engineSet struct {
	s     *Scheduler
	ids   []string
	now   time.Time
	sim   int
	clock time.Time
}

// forEngines runs fn at N=1 and at N=2 with identical per-engine capacity.
func forEngines(t *testing.T, cfg Config, fn func(t *testing.T, fx *engineSet)) {
	t.Helper()
	forEnginesCap(t, cfg, 1000, fn)
}

// forEnginesCap is forEngines with a chosen per-engine measured capacity in blocks.
func forEnginesCap(t *testing.T, cfg Config, capacity int, fn func(t *testing.T, fx *engineSet)) {
	t.Helper()
	for _, n := range []int{1, 2} {
		n := n
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			s, err := NewScheduler(cfg)
			if err != nil {
				t.Fatal(err)
			}
			fx := &engineSet{s: s, clock: time.Unix(1793000000, 0).UTC()}
			for i := 0; i < n; i++ {
				id := fmt.Sprintf("e%d", i)
				fx.ids = append(fx.ids, id)
				if err := s.SetNumBlocksFor(id, 64, capacity); err != nil {
					t.Fatal(err)
				}
			}
			fx.now = fx.clock
			fn(t, fx)
		})
	}
}

// thread finds a seated thread and the engine it holds, on any engine.
func (fx *engineSet) thread(t *testing.T, id string) (*thread, *engineGroup) {
	t.Helper()
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	for _, g := range fx.allGroups() {
		if m, ok := g.threads[id]; ok {
			return m, g
		}
	}
	t.Fatalf("thread %s is not seated on any engine", id)
	return nil, nil
}

// g is the fixture's FIRST engine group (the single-engine assertions' target; at N=2 a test
// that means "any engine" must say so explicitly).
func (fx *engineSet) g() *engineGroup { return fx.s.groups[fx.ids[0]] }

// allGroups is every engine group of the fixture, in id order.
func (fx *engineSet) allGroups() []*engineGroup { return fx.s.activeGroups() }

// tick advances the fixture's virtual clock by d and runs the scheduler's ticker.
func (fx *engineSet) tick(d time.Duration) {
	fx.clock = fx.clock.Add(d)
	fx.s.Pass(nil, fx.clock)
}

// pick runs one Pass over the given heads at the fixture's clock, returning only the index.
func (fx *engineSet) pick(hs ...Head) int {
	i := fx.s.Pass(hs, fx.clock)
	return i
}

// pickEngine runs one Pick and returns the engine it served from.
func (fx *engineSet) pickEngine(hs ...Head) (string, int) {
	i := fx.s.Pass(hs, fx.clock)
	if i < 0 {
		return "", i
	}
	return fx.s.engineOfDispatch(hs[i].ReqID), i
}

// seatOn seats one thread on a NAMED engine through the public path (the seat-time choice must
// agree with the ask, or the test's premise is broken).
func (fx *engineSet) seatOn(t *testing.T, engine, id string, blocks int, ago time.Duration) {
	t.Helper()
	fx.seatOne(t, id, blocks, ago)
	if got := fx.s.bindings[id]; got != engine {
		t.Fatalf("thread %s seated on %s, wanted %s", id, got, engine)
	}
}

// tighten makes room scarce on the FIRST engine and absent everywhere else: the test seats its
// own threads first, then every OTHER engine is filled to its budget with one 950-block
// filler (the most-free-budget rule puts each one there).
func (fx *engineSet) tighten(t *testing.T) {
	t.Helper()
	for i, e := range fx.ids {
		if i == 0 {
			continue
		}
		id := "closed-" + e
		fx.seatOne(t, id, 950, 5*time.Second)
	}
	fx.s.mu.Lock()
	for _, g := range fx.allGroups() {
		for _, m := range g.threads {
			m.lastActivity = fx.clock
		}
	}
	fx.s.mu.Unlock()
}

// newSched is the legacy single-engine fixture// newSched is the legacy single-engine fixture
// (the "" engine id, capacity 1000).
func newSched(t *testing.T) *Scheduler {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocks(64, 1000); err != nil { // budget 950 blocks
		t.Fatal(err)
	}
	return s
}

// seatOne seats one thread on the fixture through the public path.
func (fx *engineSet) seatOne(t *testing.T, id string, blocks int, ago time.Duration) {
	t.Helper()
	h := head(id, ago, blocks)
	if i := fx.pick(h); i < 0 {
		t.Fatalf("thread %s must be seated", id)
	}
	dispatch(t, fx.s, id)
}

// ---- the tests ----

// A thread with a queued head dispatches immediately once seated; an unknown thread is never
// seated by dispatching to it (non-threads never run unseated).
func TestMemberServedAndUnknownNeverSeated(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		if i := fx.pick(head("m", time.Second, 100)); i != 0 {
			t.Fatalf("a seated thread's head must dispatch, got %d", i)
		}
		if !fx.s.Seated("m") {
			t.Fatal("the thread must be seated")
		}
		settle(fx.s, "m", 100, 100)
		if fx.s.SeatRecords() != 1 {
			t.Fatalf("only the seated thread has a record, got %d", fx.s.SeatRecords())
		}
	})
}

// Newcomers go first among the WAITING heads (D1 as f0p ran it), and the seat goes to the
// newcomer even though the returning thread has waited longer.
func TestNewcomerBeforeReturnerThenOldest(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// fill EVERY engine's budget so exactly ONE of the two queued heads can be seated (the twin
		// rule: at N=2 the second engine must not simply take it) the filler lands on e0 (id tie),
		// THEN the other engines are closed off
		fx.seatOne(t, "filler", 800, 5*time.Second)
		settle(fx.s, "filler", 800, 800)
		fx.tighten(t)
		// the returning thread's turn is a continuation; the other's is its first
		ret := head("ret", 30*time.Second, 100)
		ret.Returner = true
		hs := []Head{ret, head("new", time.Second, 100)}
		fx.pick(hs...)
		if !fx.s.Seated("new") {
			t.Fatal("the newcomer must be seated first (D1 newcomer-first)")
		}
		if fx.s.Seated("ret") {
			t.Fatal("only one seat fits: the returning thread must wait behind the newcomer")
		}
	})
}

// Among two WAITING heads of the same class, the OLDEST arrival is the head, and it is the one
// that gets the seat.
func TestOldestWaiterIsTheHead(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// the filler lands on e0 (id tie), THEN the other engines are closed off
		fx.seatOne(t, "filler", 800, 5*time.Second)
		settle(fx.s, "filler", 800, 800)
		fx.tighten(t)
		hs := []Head{head("young", time.Second, 100), head("old", 30*time.Second, 100)}
		fx.pick(hs...)
		if !fx.s.Seated("old") {
			t.Fatal("the oldest waiting head must be seated first")
		}
		if fx.s.Seated("young") {
			t.Fatal("the younger head must wait")
		}
	})
}

// The quantum Q is the tenure rule: a thread that completed Q busy periods is rotation-
// eligible, one mid-quantum is not.
func TestQuantumIsTheTenureRule(t *testing.T) {
	for _, q := range []int{4, 8} {
		q := q
		t.Run(fmt.Sprintf("Q=%d", q), func(t *testing.T) {
			cfg := defaultConfig()
			cfg.Quantum = q
			forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
				fx.seatOne(t, "m", 400, time.Second)
				settle(fx.s, "m", 400, 400) // one busy period completes (main chain drains)
				fx.s.mu.Lock()
				m := fx.g().threads["m"]
				bp := m.busyPeriods
				_, eligible := fx.s.eligible(m, fx.clock)
				fx.s.mu.Unlock()
				if bp != 1 {
					t.Fatalf("one drained main-chain request is one busy period, got %d", bp)
				}
				if eligible {
					t.Fatalf("Q=%d: a thread with 1 busy period must NOT be rotation-eligible", q)
				}
				for i := 1; i < q; i++ {
					fx.s.mu.Lock()
					m.busyPeriods++
					fx.s.mu.Unlock()
				}
				fx.s.mu.Lock()
				reason, ok := fx.s.eligible(m, fx.clock)
				fx.s.mu.Unlock()
				if !ok || reason != "quantum-complete" {
					t.Fatalf("Q=%d: %d busy periods must be eligible (quantum-complete), got %q %v", q, q, reason, ok)
				}
			})
		})
	}
}

// A thread in flight is NEVER eligible: rotation must not evict running work.
func TestInFlightNeverEligible(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		m.busyPeriods = 100 // way past any Q
		m.inflight = 1
		_, ok := fx.s.eligible(m, fx.clock)
		fx.s.mu.Unlock()
		if ok {
			t.Fatal("a thread with a request in flight must never be rotation-eligible")
		}
	})
}

// The budget valve (MEMORY PRESSURE): when an engine's footprint grows past its budget, the
// victim is the DONE thread (no queued turn) — never the thread that is waiting to run (f0p's
// order).
func TestVictimOrderPrefersTheDoneMember(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// order matters: the first thread lands on e0 (id tie), THEN the other engines are closed
		// off, so the second thread has to share e0
		fx.seatOne(t, "done", 450, 5*time.Second)
		settle(fx.s, "done", 450, 450) // finished its turn: no queued head
		fx.tighten(t)
		fx.seatOne(t, "busy", 450, 4*time.Second)
		// `busy` wants to run again (a queued head) and its next prompt is 200 blocks longer: 450 +
		// 650 > 950 budget => the valve must fire
		fx.s.mu.Lock()
		fx.g().threads["busy"].waitingSince = fx.clock.Add(-time.Second)
		fx.g().threads["busy"].lastActivity = fx.clock
		fx.s.mu.Unlock()
		if i := fx.pick(head("busy", time.Second, 650)); i != 0 {
			t.Fatalf("the thread's grown head must dispatch, got %d", i)
		}
		dispatch(t, fx.s, "busy") // the growth lands
		fx.s.Pass(nil, fx.clock)  // an empty-queue Pick: enforcement runs there now
		fx.s.mu.Lock()
		_, doneSeated := fx.g().threads["done"]
		_, busySeated := fx.g().threads["busy"]
		fx.s.mu.Unlock()
		if doneSeated {
			t.Fatal("the DONE thread (no queued turn) must be the budget victim")
		}
		if !busySeated {
			t.Fatal("the thread with a queued turn must survive (f0p's valve order)")
		}
		if snap := fx.s.Snapshot(fx.clock); snap.Exits["budget-noqueue"] == 0 && snap.Exits["budget-idle"] == 0 {
			t.Fatalf("the exit must be a budget-class exit, got %v", snap.Exits)
		}
	})
}

// No eviction while in flight: a thread with a running request is marked exiting at most,
// never removed, until the drain completes.
func TestNoEvictionWhileInFlight(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 400, 2*time.Second)
		fx.s.Chunk("m") // m's prefill landed (its turn keeps decoding); the cold slot is free
		fx.tighten(t)   // every other engine is closed, so m2 must share m's budget
		fx.seatOne(t, "m2", 400, time.Second)
		// m's first turn is still running when its second (grown) head is served
		if i := fx.pick(head("m", time.Millisecond, 700)); i != 0 {
			t.Fatalf("the thread's second head must be served (in-flight cap), got %d", i)
		}
		// THE RELEASED WINDOW: between the release and PreRequest nothing has reached a pod yet, and
		// in production a ~1 kHz Pick lands HERE.
		fx.s.Pass(nil, fx.clock)
		if !fx.s.Seated("m") {
			t.Fatal("a released window is in flight: the thread must never be evicted inside it")
		}
		dispatch(t, fx.s, "m")
		fx.s.Pass(nil, fx.clock) // the next pass enforces the growth
		if !fx.s.Seated("m") {
			t.Fatal("a thread in flight must never be evicted (marked exiting at most)")
		}
		settle(fx.s, "m", 400, 0)
		settle(fx.s, "m", 700, 0)
		fx.s.Pass(nil, fx.clock)
		if fx.s.Seated("m") {
			t.Fatal("after the drain the exiting thread must leave")
		}
	})
}

// m8: a thread evicted while it has a QUEUED head re-enters the shared queue with THAT HEAD's
// cost and class — not with the chain the engine last ran, and not with a forced "returner".
func TestExitRequeuesWithTheQueuedHead(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second) // it ran a 100-block turn
		queued := head("m", time.Second, 300)
		queued.Returner = false // a NEWCOMER's turn, whatever the thread's history
		fx.s.mu.Lock()
		g := fx.g()
		m := g.threads["m"]
		fx.s.syncQueue([]Head{queued}, fx.clock)
		ran := g.index.Chains("m")
		fx.s.mu.Unlock()
		if len(ran) == 0 {
			t.Fatal("premise: the engine accounts the chain it is running")
		}
		if sameChains(ran, queued.Chains) {
			t.Fatal("premise: the queued head must differ from the turn that ran")
		}
		fx.s.mu.Lock()
		fx.s.exit(g, m, "test", fx.clock)
		w := fx.s.waiters["m"]
		fx.s.mu.Unlock()
		if w == nil {
			t.Fatal("a thread with a queued head must re-enter the shared queue")
		}
		if !sameChains(w.chains, queued.Chains) {
			t.Fatal("the re-queue carried the WRONG chain: it must be the queued head's own cost, never the turn that ran")
		}
		if w.returner {
			t.Fatal("the re-queued head must keep its own class: a newcomer's turn was forced to returner=true")
		}
	})
}

// overOf is the engine's overage on enforcement's own view (the stay side).
func overOf(fx *engineSet) int64 {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return fx.s.overBudget(fx.g())
}

// AN EXITING MEMBER'S BLOCKS STAY ON THE USED SIDE UNTIL IT DRAINS.
func TestExitingMemberStaysOnTheUsedSideUntilDrain(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		for i := range fx.ids {
			fx.seatOne(t, fmt.Sprintf("m%d", i), 400, time.Duration(5+i)*time.Second)
		}
		min, g := fx.thread(t, "m0")
		fx.s.mu.Lock()
		// in flight and marked exiting (the mid-flight pressure class)
		inFlight(fx.s, min, fx.clock, "r-m0")
		freeBefore := fx.s.freeBudget(g)
		neBefore := g.index.UniqueBlocks()
		g.index.MarkExiting("m0")
		neAfter := g.index.UniqueBlocks()
		freeAfter := fx.s.freeBudget(g)
		fx.s.mu.Unlock()
		if neAfter != neBefore {
			t.Fatalf("marking a thread exiting released %d blocks from the used side (NE %d -> %d)",
				neBefore-neAfter, neBefore, neAfter)
		}
		if freeAfter != freeBefore {
			t.Fatalf("the mark freed budget (free %d -> %d): the engine still holds those blocks",
				freeBefore, freeAfter)
		}
		// the drain is what releases them
		settle(fx.s, "m0", 400, 400)
		fx.s.mu.Lock()
		defer fx.s.mu.Unlock()
		if _, seated := g.threads["m0"]; seated {
			t.Fatal("a drained exiting thread must leave")
		}
		if got := g.index.UniqueBlocks(); got >= neBefore {
			t.Fatalf("the drain did not release the thread's blocks: NE %d want < %d", got, neBefore)
		}
	})
}

// An exiting thread that cannot be freed must END the enforcement pass: marking is idempotent
// and its blocks stay charged, so re-picking it would spin the loop forever under the
// scheduler's mutex.
func TestExitingVictimDoesNotSpinEnforcement(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 900, 2*time.Second)
		m, g := fx.thread(t, "m")
		fx.s.mu.Lock()
		inFlight(fx.s, m, fx.clock, "r-m")
		g.index.MarkExiting("m")
		// way over budget with the only thread already exiting
		g.budgetBlocks = 100
		fx.s.mu.Unlock()
		// The enforcement pass runs BEFORE Tick's queue re-sync, so a clock this far ahead leaves
		// the queue view untrustworthy — the one path that names an already-exiting thread as the
		// victim.
		fx.clock = fx.clock.Add(60 * time.Second)
		done := make(chan struct{})
		go func() {
			fx.s.Pass(nil, fx.clock)
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatal("Tick spun: an exiting in-flight victim was re-picked forever")
		}
		if !fx.s.Seated("m") {
			t.Fatal("the exiting thread must still be seated (it drains, it is not evicted)")
		}
	})
}

// The cold-entry valve: one cold entry is ALWAYS allowed (R11); a second waits until the
// first's prefill completes — and the cap is PER ENGINE, so at N=2 the second oversized entry
// may seat immediately on the sibling engine.
func TestColdEntryValveIsPerEngine(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		big := 1000 // twice the derived cap (500 blocks), far inside the budget
		if got, want := fx.g().coldCapTokens(0.95, 64), int64(500*64); got != want {
			t.Fatalf("derived cold cap: %d want %d", got, want)
		}
		if i := fx.pick(head("big1", time.Second, big)); i != 0 {
			t.Fatal("a big newcomer must be seated when nothing cold is in flight (R11)")
		}
		dispatch(t, fx.s, "big1")
		h2 := []Head{head("big2", time.Second, big)}
		i := fx.pick(h2...)
		if len(fx.ids) == 1 {
			if i != -1 {
				t.Fatalf("the second big entry must wait while the first's prefill is cold (got %d)", i)
			}
			fx.s.Chunk("big1")
			if i := fx.pick(h2...); i != 0 {
				t.Fatal("the second big entry must seat after the first's prefill completes")
			}
			return
		}
		// N=2: the cap is per engine, so the sibling has its own cold slot
		if i != 0 {
			t.Fatalf("the second cold entry must seat on the SIBLING engine (per-engine cap), got %d", i)
		}
		fx.s.mu.Lock()
		e1, e2 := fx.s.bindings["big1"], fx.s.bindings["big2"]
		fx.s.mu.Unlock()
		if e1 == e2 {
			t.Fatalf("two oversized cold entries must not share an engine's cold slot: both on %s", e1)
		}
	})
}

// Anonymous traffic (R9) is served when no thread head is dispatchable, is never tracked, and
// never outranks a thread's head.
func TestAnonymousPassThrough(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		anon := Head{Thread: "", ReqID: "anon-1", Arrival: fx.clock.Add(-time.Second)}
		if i := fx.pick(anon); i != 0 {
			t.Fatalf("an anonymous head must be served when no thread head is dispatchable, got %d", i)
		}
		if fx.s.SeatRecords() != 0 {
			t.Fatal("anonymous traffic must not be tracked")
		}
		fx.seatOne(t, "m", 10, time.Second)
		anonLater := Head{Thread: "", ReqID: "anon-2", Arrival: fx.clock}
		if i := fx.pick(anonLater, head("m", time.Second, 10)); i != 1 {
			t.Fatalf("the thread's head must outrank the anonymous one: got %d", i)
		}
	})
}

// Hold the line: a blocked head does not let a later, smaller head jump it.
func TestHoldTheLine(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// both residents must live on ONE engine, so the tests' own threads are seated first (the
		// first lands on e0 by the id tie) and THEN the other engines are closed off; a resident
		// that still lands elsewhere makes the shape invalid
		fx.seatOne(t, "m1", 450, 3*time.Second)
		settle(fx.s, "m1", 450, 450)
		fx.tighten(t)
		fx.seatOne(t, "m2", 450, 2*time.Second)
		m1, g1 := fx.thread(t, "m1")
		m2, g2 := fx.thread(t, "m2")
		if g1 != g2 {
			t.Skip("residents landed on different engines; the hold-the-line shape needs them together")
		}
		fx.s.mu.Lock()
		m1.waitingSince = fx.clock.Add(-3 * time.Second)
		m1.lastActivity = fx.clock
		m2.waitingSince = fx.clock.Add(-2 * time.Second)
		m2.lastActivity = fx.clock
		fx.s.mu.Unlock()
		hs := []Head{
			head("m1", 3*time.Second, 400),
			head("bighead", 2*time.Second, 900),
			head("small", 100*time.Millisecond, 100),
			head("m2", 2*time.Second, 400),
		}
		for round := 0; round < 4; round++ {
			i := fx.pick(hs...)
			if i == 2 {
				t.Fatal("hold the line violated: small jumped the blocked big head")
			}
			if i >= 0 && hs[i].Thread != "" {
				dispatch(t, fx.s, hs[i].Thread)
				settle(fx.s, hs[i].Thread, 400, 400)
			}
		}
	})
}

// No starvation: under steady load every thread with a queued turn is eventually seated.
func TestNoStarvation(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		w := newWorkload(t, fx)

		// THE SHAPE — every engine FULL, and the same shape at any engine count: one small anchor
		// per engine (a headless thread that holds 40 of the 950 blocks) and three residents per
		// engine on top of it.
		for _, engine := range fx.ids {
			w.seatStatic(t, "anchor-"+engine, engine, 40)
		}
		var residents []string
		for i := 0; i < 3*len(fx.ids); i++ {
			residents = append(residents, fmt.Sprintf("m%d", i))
		}
		for _, id := range residents {
			w.arrive(id, fx.clock.Add(-time.Second))
		}
		// each resident completes one turn, so every later arrival is a continuation
		settled := func() bool {
			for _, id := range residents {
				if !w.everSeated(id) || w.served[id] == 0 {
					return false
				}
			}
			return true
		}
		for round := 0; round < 400 && !settled(); round++ {
			hs, ids := w.heads(fx.clock)
			if i := fx.pick(hs...); i >= 0 {
				w.dispatchTurn(ids[i])
			}
			w.step(roundStep)
		}
		if !settled() {
			for _, id := range residents {
				t.Logf("resident %s seated=%v turns=%d", id, w.everSeated(id), w.served[id])
			}
			t.Fatal("the fixture could not seat and turn over its residents")
		}
		// NON-VACUITY, per engine: the budgets really are full, so every waiter's seat costs a
		// rotation (the check is the fixture's, not the scheduler's)
		fx.s.mu.Lock()
		for _, g := range fx.s.activeGroups() {
			if free := fx.s.freeBudget(g); free >= 300 {
				fx.s.mu.Unlock()
				t.Fatalf("the fixture is not contended on %s: %d blocks free", g.id, free)
			}
		}
		fx.s.mu.Unlock()

		// eight threads nobody has served: each has ONE queued turn (its first) from now until it
		// dispatches
		var waiters []string
		for i := 0; i < 8; i++ {
			waiters = append(waiters, fmt.Sprintf("w%d", i))
		}
		for i, id := range waiters {
			w.arrive(id, fx.clock.Add(-time.Duration(len(waiters)-i)*100*time.Millisecond))
		}
		hs, ids := w.heads(fx.clock)
		if i := fx.pick(hs...); i >= 0 {
			w.dispatchTurn(ids[i])
		}

		for cycles := 0; cycles < 80; cycles++ {
			for round := 0; round < 12; round++ {
				hs, ids := w.heads(fx.clock)
				if i := fx.pick(hs...); i >= 0 {
					if !fx.s.Seated(hs[i].Thread) {
						t.Fatalf("%s was dispatched without a seat", ids[i])
					}
					w.dispatchTurn(ids[i])
				}
				w.step(roundStep)
			}
			done := 0
			for _, id := range waiters {
				if w.everSeated(id) {
					done++
				}
			}
			if done == len(waiters) {
				break
			}
		}
		for _, id := range waiters {
			if !w.everSeated(id) {
				t.Errorf("starvation: %s never held a seat", id)
			}
		}
		// CONTENTION, restated from the scheduler's own books: the budgets were full when the
		// waiters arrived and nothing frees blocks except an exit, so every waiter's seat must have
		// cost at least one eviction
		exits := int64(0)
		for _, n := range fx.s.Snapshot(fx.clock).Exits {
			exits += n
		}
		if exits < int64(len(waiters)) {
			t.Errorf("%d exits for %d waiter seats: the fixture was not contended", exits, len(waiters))
		}
		t.Logf("%d/%d waiters seated on %v; exits=%v", len(waiters), len(waiters), fx.ids, fx.s.Snapshot(fx.clock).Exits)
	})
}

// C8, MADE TO BITE: a thread with a QUEUED TURN is not idle — even silent past S, and even
// while that turn cannot dispatch (the in-flight cap holds it).
func TestWaitingMemberIsNotIdleEvicted(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 400, time.Second)
		settle(fx.s, "m", 400, 400)
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		m.lastActivity = fx.clock.Add(-time.Hour) // silent past S
		for i := 0; i < fx.s.cfg.InFlightCap; i++ {
			// the cap is full, so the head below stays QUEUED while the pass reads it
			addInFlight(fx.s, m, fx.clock, fmt.Sprintf("r-cap-%d", i))
		}
		fx.s.mu.Unlock()
		h := head("m", time.Minute, 400)
		if i := fx.pick(h); i >= 0 {
			t.Fatalf("the in-flight cap must hold the head, got %d", i)
		}
		fx.s.mu.Lock()
		waiting := !fx.g().threads["m"].waitingSince.IsZero()
		reason, victim := fx.s.rotationVictim(fx.g(), fx.clock)
		fx.s.mu.Unlock()
		if !waiting {
			t.Fatal("premise: the pass must have seen the thread's queued head")
		}
		if victim != nil {
			t.Fatalf("a thread with a queued turn was rotation-eligible as %q (C8: a queued turn is not idle)", reason)
		}
		if !fx.s.Seated("m") {
			t.Fatal("a thread with a queued turn must not be evicted (C8)")
		}
		// and a Tick alone decides nothing (R5): no decision point, no eviction — the gate being
		// closed can never manufacture one
		fx.tick(5 * time.Minute)
		if !fx.s.Seated("m") {
			t.Fatal("a Tick must not evict: decisions live in Pick (R5)")
		}
	})
}

// NO RECORD BEFORE PreRequest (R7): a request that is picked and never reaches a pod leaves no
// in-flight record at all — the thread carries the released window instead, and the terminal
// callback the handler forces for every scheduled request closes it.
func TestNoRecordBeforePreRequest(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		fx.s.mu.Lock()
		inFlight := fx.g().threads["m"].inflight
		fx.s.mu.Unlock()
		if inFlight != 1 {
			t.Fatalf("after PreRequest the thread has %d requests in flight, want 1", inFlight)
		}
		// a second turn is picked but never dispatched
		h2 := head("m", time.Second, 100)
		h2.ReqID = "r-m-2"
		if i := fx.pick(h2); i != 0 {
			t.Fatalf("the seated thread's head must be picked, got %d", i)
		}
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		afterPick, released := m.inflight, ""
		for reqID, rec := range fx.s.reqs {
			if rec.thread == "m" && rec.pod.IsZero() {
				released = reqID
			}
		}
		fx.s.mu.Unlock()
		if afterPick != 1 {
			t.Fatalf("a picked-but-not-dispatched request made an in-flight record: %d want 1", afterPick)
		}
		if released != "r-m-2" {
			t.Fatalf("the released window names %q, want r-m-2 (it is the fact that keeps the thread busy)", released)
		}
		if !fx.s.Seated("m") {
			t.Fatal("the thread must stay seated while its request is on its way")
		}
		// the terminal callback closes the window, and the thread is free again
		fx.s.Settle("r-m-2", "", fx.clock)
		fx.s.mu.Lock()
		releasedAfter := ""
		for reqID, rec := range fx.s.reqs {
			if rec.thread == "m" && rec.pod.IsZero() {
				releasedAfter = reqID
			}
		}
		fx.s.mu.Unlock()
		if releasedAfter != "" {
			t.Fatalf("the terminal callback left the released window open (%q)", releasedAfter)
		}
	})
}

// A leaked cold charge is re-derived away at the next decision point (derive-
// don't-accumulate).
func TestLeakedColdChargeVanishesOnDerive(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		fx.s.mu.Lock()
		fx.g().coldInFlight += 999999
		fx.s.derive()
		got := fx.g().coldInFlight
		cap := fx.g().coldCapTokens(fx.s.cfg.GroupBudgetFraction, fx.s.blockSize)
		fx.s.mu.Unlock()
		if got > cap {
			t.Fatalf("the derived cold charge must replace the leak, got %d (cap %d)", got, cap)
		}
	})
}

// The SEAT/EXIT event log carries the fairness id and the engine (the join key the E4a
// instrumentation exists for).
func TestSeatExitEventsCarryFairnessIDAndEngine(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		h := head("m", time.Second, 100)
		h.FairnessID = "thread-7"
		events := newEventCollector(fx.s)
		if i := fx.pick(h); i != 0 {
			t.Fatal("the thread must seat")
		}
		dispatch(t, fx.s, "m")
		evs := events.events()
		var seat, exit *Event
		for i := range evs {
			switch evs[i].Kind {
			case "SEAT":
				seat = &evs[i]
			}
		}
		if seat == nil {
			t.Fatal("a SEAT event must be recorded")
		}
		if seat.Engine == "" {
			t.Fatalf("the SEAT event must carry the engine, got %q", seat.Engine)
		}
		if want := "fairness_id=thread-7"; !contains(seat.Detail, want) {
			t.Fatalf("the SEAT event must carry the wire identity (%s), got %q", want, seat.Detail)
		}
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		fx.s.exit(fx.g(), m, "test", fx.clock)
		fx.s.mu.Unlock()
		evs = events.events()
		for i := range evs {
			if evs[i].Kind == "EXIT" {
				exit = &evs[i]
			}
		}
		if exit == nil || !contains(exit.Detail, "fairness_id=thread-7") {
			t.Fatalf("the EXIT event must carry the fairness id, got %+v", exit)
		}
	})
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// BUG 2: the MID-QUANTUM victim order never takes a seat back from a thread that has not
// served a turn in this residency while a thread that HAS served one is available.
func TestClass3VictimPrefersAServedMember(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// two residents, both with a queued head (so class 2 finds nothing), neither eligible by the
		// tenure rule
		fx.seatOne(t, "served", 200, 9*time.Second)
		settle(fx.s, "served", 200, 200) // served one turn
		fx.seatOne(t, "fresh", 200, 8*time.Second)
		for _, id := range []string{"served", "fresh"} {
			m, _ := fx.thread(t, id)
			fx.s.mu.Lock()
			m.waitingSince = fx.clock.Add(-time.Second)
			m.lastActivity = fx.clock
			m.inflight = 0
			m.busyPeriods = 0
			fx.s.mu.Unlock()
		}
		var reason string
		var victim *thread
		fx.s.mu.Lock()
		reason, victim = fx.s.budgetVictim(fx.g(), fx.clock)
		fx.s.mu.Unlock()
		if victim == nil {
			t.Fatal("a victim must exist")
		}
		if victim.id == "fresh" {
			t.Fatalf("BUG 2: a thread that has served ZERO turns must not be taken before one that has (reason=%s)", reason)
		}
		if victim.id != "served" {
			t.Fatalf("expected the served thread, got %s", victim.id)
		}
	})
}

// BUG 3 (the C8 race): a head that arrives just before a Tick protects its thread, and a STALE
// queue view must never justify an idle eviction.
func TestStaleQueueViewNeverIdleEvicts(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 200, time.Second)
		settle(fx.s, "m", 200, 200)
		// (a) a head arrives, Pick sees it, then a Tick lands just after -> protected
		fx.pick(head("m", time.Second, 200))
		fx.tick(25 * time.Second) // silence past S, but the head is queued
		if !fx.s.Seated("m") {
			t.Fatal("BUG 3: a head present at the last sync must protect its thread from an idle eviction")
		}
		// (b) no Pick has run for longer than S (the flow controller's gate is closed): the queue
		// view is stale, so silence is NOT evidence and nothing may be evicted
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		m.waitingSince = time.Time{} // the flag went stale exactly as in the race
		m.lastActivity = fx.clock.Add(-time.Hour)
		fx.s.mu.Unlock()
		fx.tick(5 * time.Minute) // Tick re-syncs from the LAST snapshot (which has the head)
		if !fx.s.Seated("m") {
			t.Fatal("BUG 3: a stale queue view must never justify an idle eviction")
		}
	})
}

// ONE DEPARTURE, ONE BOOKING.
func TestExitingIsCountedAndRotationNeverPicksInflight(t *testing.T) {
	cfg := defaultConfig()
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		var hooked []string
		fx.s.onEvent = func(ev Event) {
			if ev.Kind == "EXIT" {
				reason, _, _ := strings.Cut(ev.Detail, " ")
				hooked = append(hooked, strings.TrimPrefix(reason, "reason="))
			}
		}
		// a thread filling most of the budget, in flight — and a SIBLING on the SAME engine, also in
		// flight, so the growth cannot be RULE O's sole oversize resident (that one is deliberately
		// not a pressure victim at all: see TestRuleOWaitsForTenureNotPressure)
		fx.seatOne(t, "run", 800, 5*time.Second)
		fx.s.Chunk("run") // its prefill landed: the cold valve is free for the sibling
		fx.closeEnginesExcept(t, fx.ids[0], 950)
		fx.seatOne(t, "sib", 100, 4*time.Second)
		// over budget by growth: the pressure module takes the mid-flight class
		fx.s.mu.Lock()
		m := fx.g().threads["run"]
		m.waitingSince = time.Time{}
		m.lastActivity = fx.clock.Add(-time.Hour) // silent, so class 2 has a victim...
		fx.s.mu.Unlock()
		// ...but the thread is in flight, so class 2 skips it: force class 4 by taking the engine
		// over budget with the thread's own next prompt (400 blocks longer), which sync charges in
		// the same pass enforcement runs in
		fx.dispatchBig(t, "run", 1200)
		fx.s.mu.Lock()
		fx.s.enforceBudget(fx.g(), fx.clock)
		fx.s.mu.Unlock()
		// (b) the rotation never returns an in-flight thread
		fx.s.mu.Lock()
		_, rv := fx.s.rotationVictim(fx.g(), fx.clock)
		fx.s.mu.Unlock()
		if rv != nil && rv.inflight > 0 {
			t.Fatal("BUG 4(b): a rotation victim must never have requests in flight")
		}
		// (a) the MARK is state, not an exit: it must not reach the exit hook at all, and the
		// snapshot must show it
		for _, r := range hooked {
			if r == "exiting" || r == "budget-exiting" {
				t.Fatalf("the mark of a mid-flight thread was booked as an EXIT (%v): one departure, one booking", hooked)
			}
		}
		var marked int
		fx.s.mu.Lock()
		for id := range fx.g().threads {
			if fx.g().index.Exiting(id) {
				marked++
			}
		}
		fx.s.mu.Unlock()
		if marked == 0 && len(hooked) == 0 {
			t.Fatal("the mid-flight thread was neither marked exiting nor exited: nothing happened at all")
		}
		if len(hooked) == 1 && hooked[0] == "exiting-drain" {
			t.Fatalf("a thread cannot be drained while it is in flight (%v)", hooked)
		}
	})
}

// dispatchBig serves a thread's next turn whose PROMPT is `blocks` long (the head carries the
// whole chain, so the growth is whatever the new prompt adds), through the public API.
func (fx *engineSet) dispatchBig(t *testing.T, id string, blocks int) {
	t.Helper()
	h := head(id, time.Second, blocks)
	h.ReqID = "r-" + id + "-grow"
	// the growth is charged at SYNC, inside this pass, so enforcement may already have condemned
	// the thread by the time the dispatch is chosen — the pass order is sync → enforce →
	// rotate/seat → dispatch, and a prompt that no longer fits must not run.
	if i := fx.pick(h); i >= 0 {
		fx.s.releaseThread(id, h.ReqID, nil, fx.clock)
		fx.s.Dispatched(h.ReqID, fx.clock)
	}
}

// When no thread head is dispatchable, the pass-through fallback serves the OLDEST anonymous
// (or uncostable) head.
func TestPassThroughServesTheOldest(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		young := Head{Thread: "", ReqID: "anon-young", Arrival: fx.clock}
		old := Head{Thread: "", ReqID: "anon-old", Arrival: fx.clock.Add(-30 * time.Second)}
		blind := Head{Thread: "blind", ReqID: "r-blind", Arrival: fx.clock.Add(-20 * time.Second)}
		if i := fx.pick(young, old, blind); i != 1 {
			t.Fatalf("the oldest pass-through head is the second one, got %d", i)
		}
	})
}

// SetNumBlocks attaches the measured capacity for the fixture's ONE engine (the "" engine id):
// the production path names engines, a single-engine test does not have to.
func (s *Scheduler) SetNumBlocks(blockSizeTokens, numBlocks int) error {
	return s.SetNumBlocksFor("", blockSizeTokens, numBlocks)
}

// THE ENGINE-IDLE RULE NEVER FIRES WHILE A THREAD IS DISPATCHABLE (the must-keep lesson M19/
// F7 had no regression test).
func TestEngineIdleNeverFiresWhileAThreadIsDispatchable(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		var exits []string
		fx.s.onEvent = func(ev Event) {
			if ev.Kind == "EXIT" {
				reason, _, _ := strings.Cut(ev.Detail, " ")
				exits = append(exits, strings.TrimPrefix(reason, "reason="))
			}
		}

		fx.seatOne(t, "res", 800, 5*time.Second)
		settle(fx.s, "res", 800, 800) // nothing in flight, nothing silent: only its HEAD says it is alive
		// N=2: the blocked waiter must not escape to a sibling engine
		fx.closeEnginesExcept(t, fx.ids[0], 950)

		dispatchable := head("res", time.Second, 800)     // its own next turn, queued NOW
		blocked := head("new", 500*time.Millisecond, 300) // 300 blocks against 150 free
		engine, i := fx.pickEngine(dispatchable, blocked)
		if i < 0 {
			t.Fatal("the resident's queued turn must dispatch")
		}
		if got := fx.s.bindings["res"]; got != engine {
			t.Fatalf("the served engine is %s, want the resident's %s", got, engine)
		}
		if len(exits) != 0 {
			t.Fatalf("a dispatchable thread was evicted (%v): the engine-idle rule fired while a resident could use its seat", exits)
		}
		if !fx.s.Seated("res") {
			t.Fatal("the dispatchable resident must keep its seat")
		}
		if fx.s.Seated("new") {
			t.Fatal("the blocked waiter cannot fit: it must not be seated by an eviction it should not have caused")
		}
	})
}
