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

// THE MULTI-ENGINE SUITE — the SP intent tests (T1-T4) plus the per-engine invariants the
// multi-pod wave is scored on.

import (
	"fmt"
	"testing"
	"time"
)

// T1: PLACEMENT IS MOST-FREE-BUDGET.
func TestT1SeatGoesToTheMostFreeBudget(t *testing.T) {
	cfg := defaultConfig()
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		if len(fx.ids) < 2 {
			// N=1: the same code with one candidate — the head lands on it.
			h := head("m", time.Second, 100)
			if e, i := fx.pickEngine(h); i != 0 || e != fx.ids[0] {
				t.Fatalf("N=1: the only engine must take the head, got %q/%d", e, i)
			}
			return
		}
		// engine-a is 300 blocks fuller than engine-b, so b has more free budget
		fx.seatOn(t, fx.ids[0], "big", 300, 3*time.Second)
		// the resident's prefill lands, so its engine's cold-entry slot is free again (the derived
		// cap is 5 % of the engine, and an outstanding prefill owns it)
		fx.s.Chunk("big")
		h := head("m", time.Second, 100)
		e, i := fx.pickEngine(h)
		if i != 0 {
			t.Fatalf("the head must be seated and dispatched, got %d", i)
		}
		if e != fx.ids[1] {
			t.Fatalf("T1: the seat must go to the engine with the MOST FREE BUDGET (%s), got %s", fx.ids[1], e)
		}
	})
}

// T2: THE BINDING LASTS ONLY WHILE SEATED.
func TestT2ExitUnbindsAndReentryReseatsAfresh(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 300, 3*time.Second)
		first := fx.s.bindings["m"]
		if first == "" {
			t.Fatal("a seated thread must be bound")
		}
		m, g := fx.thread(t, "m")
		fx.s.mu.Lock()
		fx.s.exit(g, m, "test", fx.clock)
		_, bound := fx.s.bindings["m"]
		fx.s.mu.Unlock()
		if bound {
			t.Fatal("T2: an exit must DROP the binding (seat = pin)")
		}
		if len(fx.ids) < 2 {
			return
		}
		// fill the first engine so the re-entry can only fit on the other one
		other := fx.ids[0]
		if first == fx.ids[0] {
			other = fx.ids[1]
		}
		fx.seatOn(t, first, "fill", 900, 2*time.Second)
		h := head("m", 900*time.Second, 300)
		e, i := fx.pickEngine(h)
		if i < 0 {
			t.Fatal("the re-entering thread must be seated somewhere")
		}
		if e == first {
			t.Fatalf("T2: the re-entry must be re-placed where there is room (%s), got the old engine %s", other, e)
		}
	})
}

// T3: THE e4 REGRESSION.
func TestT3e4RegressionNoLockInOnQueueDepth(t *testing.T) {
	cfg := defaultConfig()
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		// four seats, each with a queued follow-up head (the residents want to run)
		for i := 0; i < 4; i++ {
			id := fmt.Sprintf("s%d", i)
			fx.seatOne(t, id, 100, time.Duration(10+i)*time.Second)
			fx.s.Chunk(id) // each resident's prefill has landed: its cold slot is free
			m, _ := fx.thread(t, id)
			fx.s.mu.Lock()
			m.waitingSince = fx.clock.Add(-time.Duration(5+i) * time.Second)
			m.lastActivity = fx.clock
			fx.s.mu.Unlock()
		}
		fx.s.Pass(nil, fx.clock)
		before := fx.s.Snapshot(fx.clock)
		for i := 0; i < 6; i++ {
			id := fmt.Sprintf("q%d", i)
			h := head(id, time.Duration(60-i)*time.Second, 50)
			if _, j := fx.pickEngine(h); j < 0 {
				t.Fatalf("head %d must be seated: 3 %% KV and a deep waiting queue are not saturation", i)
			}
			// the head is dispatched by the same pass: its prefill lands, which is what frees the
			// engine's single cold-entry slot for the next newcomer
			fx.s.Chunk(id)
		}
		after := fx.s.Snapshot(fx.clock)
		if after.Seats <= before.Seats {
			t.Fatalf("T3: the policy must KEEP seating (%d -> %d seats)", before.Seats, after.Seats)
		}
	})
}

// PER-ENGINE SIZING: each engine's budget is its OWN capacity × the fraction, shown by two
// engines with different capacities.
func TestPerEngineBudgetFromOwnCapacity(t *testing.T) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("a", 64, 1000); err != nil { // budget 950
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("b", 64, 2000); err != nil { // budget 1900
		t.Fatal(err)
	}
	snap := s.Snapshot(s.now)
	if snap.Engines["a"].BudgetBlocks != 950 || snap.Engines["b"].BudgetBlocks != 1900 {
		t.Fatalf("per-engine budgets wrong: a=%d b=%d", snap.Engines["a"].BudgetBlocks, snap.Engines["b"].BudgetBlocks)
	}
	if snap.BudgetBlocks != 2850 {
		t.Fatalf("the whole-policy budget must be the sum, got %d", snap.BudgetBlocks)
	}
}

// PER-ENGINE DEDUP, and (R9) the shared prefix that pulls a thread to the engine which already
// holds it: the same block keys on two engines are two footprints (a block on A is worthless
// to B), and a thread whose prompt is already resident somewhere costs nothing there — so it
// lands THERE, not on the emptier engine.
func TestPerEngineDedupAndIndependence(t *testing.T) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range []string{"a", "b"} {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Unix(1793000000, 0).UTC()
	shared := keys(200, 42)
	for _, id := range []string{"m1", "m2"} {
		// the SAME chain: two threads whose prompts share every block
		h := headWithChain(id, time.Second, []Chain{{Keys: shared, Tokens: 200 * 64}})
		if i := s.Pass([]Head{h}, now); i != 0 {
			t.Fatalf("%s must seat", id)
		}
		releasedWindow(s, now)
		s.releaseThread(id, "r-"+id, nil, now)
		s.Dispatched("r-"+id, now)
		s.Chunk(id) // its prefill lands: the cold charge is released, the valve reopens
	}
	// R9: m2 costs 0 on the engine that already holds its blocks and 200 on the other, so both
	// threads are on ONE engine and its 200 blocks are counted ONCE (dedup within an engine) —
	// the empty engine holds nothing.
	snap := s.Snapshot(s.now)
	if len(snap.Engines) != 2 {
		t.Fatalf("both engines must be tracked, got %d", len(snap.Engines))
	}
	if snap.Engines["a"].NEBlocks != 200 || snap.Engines["b"].NEBlocks != 0 {
		t.Fatalf("a shared prefix must pull the second thread to the first engine: a=%d b=%d, want 200/0",
			snap.Engines["a"].NEBlocks, snap.Engines["b"].NEBlocks)
	}
	if snap.Engines["a"].Seats != 2 {
		t.Fatalf("both threads must be seated on a, got %d", snap.Engines["a"].Seats)
	}
	if snap.NEBlocks != 200 {
		t.Fatalf("the whole-policy NE counts the shared chain once (200), got %d", snap.NEBlocks)
	}
	// DEDUP IS PER ENGINE: fill engine a, and the same chain must seat on b and count AGAIN there
	// — the same block keys on two engines are two footprints. the filler continues the shared
	// chain (200 + 750 of its own), so it costs 750 on the engine that holds the prefix and 950
	// on the empty one: it fills a
	filler := headWithChain("filler", time.Second, []Chain{{
		Keys: append(append([]Hash{}, shared...), keys(750, 55)...), Tokens: 950 * 64}})
	if i := s.Pass([]Head{filler}, now); i != 0 {
		t.Fatal("the filler must seat")
	}
	releasedWindow(s, now)
	s.releaseThread("filler", "r-filler", nil, now)
	s.Dispatched("r-filler", now)
	s.Chunk("filler")
	s.mu.Lock()
	fillerEngine := s.bindings["filler"]
	s.mu.Unlock()
	if fillerEngine != "a" {
		t.Fatalf("the filler must take a (it continues a's chain), it took %q", fillerEngine)
	}
	m3 := headWithChain("m3", time.Second, []Chain{{Keys: shared, Tokens: 200 * 64}})
	if i := s.Pass([]Head{m3}, now); i != 0 {
		t.Fatal("the third copy must seat on b (a is full)")
	}
	releasedWindow(s, now)
	s.releaseThread("m3", "r-m3", nil, now)
	s.Dispatched("r-m3", now)
	snap = s.Snapshot(s.now)
	if snap.Engines["b"].NEBlocks != 200 {
		t.Fatalf("the same chain on the second engine counts again: b=%d want 200", snap.Engines["b"].NEBlocks)
	}
	if snap.NEBlocks != 1150 { // a: 200 + 750 filler, b: 200
		t.Fatalf("the whole-policy NE must sum the per-engine footprints, got %d want 1150", snap.NEBlocks)
	}
}

// R9: PLACEMENT IS DECIDED ON THE FREE BUDGET AFTER THE SEAT, so a fork follows its parent's
// prefix even when the other engine is emptier BEFORE the seat — no affinity rule, nothing to
// tune, because the cost of a shared prefix is already zero.
func TestPlacementIsDedupAware(t *testing.T) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range []string{"a", "b"} {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Unix(1793000000, 0).UTC()
	seat := func(id string, chains []Chain) string {
		t.Helper()
		h := headWithChain(id, time.Second, chains)
		if i := s.Pass([]Head{h}, now); i != 0 {
			t.Fatalf("%s must seat", id)
		}
		releasedWindow(s, now)
		s.releaseThread(id, "r-"+id, nil, now)
		s.Dispatched("r-"+id, now)
		s.Chunk(id) // its prefill lands: the cold charge goes, the valve reopens
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.bindings[id]
	}
	parent := keys(300, 77)
	if got := seat("parent", []Chain{{Keys: parent, Tokens: 300 * 64}}); got != "a" {
		t.Fatalf("the parent must seat on a (the id tie), it took %q", got)
	}
	// b gets 350 blocks of its own: a 650 free, b 600
	if got := seat("fillb", chainOf("fillb", 350)); got != "b" {
		t.Fatalf("fillb must seat on b, it took %q", got)
	}
	// a continues its own chain (300 + 100): 100 of cost there, 400 elsewhere
	if got := seat("filla", []Chain{{
		Keys: append(append([]Hash{}, parent...), keys(100, 88)...), Tokens: 400 * 64}}); got != "a" {
		t.Fatalf("filla must seat on a, it took %q", got)
	}
	// now: a 550 free, b 600 free.
	s.mu.Lock()
	freeA := s.freeBudget(s.groups["a"])
	freeB := s.freeBudget(s.groups["b"])
	s.mu.Unlock()
	if freeB <= freeA {
		t.Fatalf("premise: b must be the emptier engine before the seat (a=%d b=%d)", freeA, freeB)
	}
	forkKeys := append(append([]Hash{}, parent...), keys(100, 99)...)
	if got := seat("fork", []Chain{{Keys: forkKeys, Tokens: 400 * 64}}); got != "a" {
		t.Fatalf("the fork must land where its prefix is resident (a), it took %q — placement is not dedup-aware", got)
	}
}

// ENGINE GONE: the engine's threads are exited ("engine-gone") and their heads go back to the
// SHARED queue, where a survivor seats them.
func TestEngineGoneReleasesMembersToTheSharedQueue(t *testing.T) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range []string{"a", "b"} {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Unix(1793000000, 0).UTC()
	for _, id := range []string{"m1", "m2", "m3"} {
		if i := s.Pass([]Head{head(id, time.Second, 300)}, now); i != 0 {
			t.Fatalf("%s must seat", id)
		}
		releasedWindow(s, now)
		s.releaseThread(id, "r-"+id, nil, now)
		s.Dispatched("r-"+id, now)
		s.Chunk(id)
		s.Settle("r-"+id, "", now)
	}
	gone := s.bindings["m1"]
	if n := s.RemoveEngine(gone, now); n == 0 {
		t.Fatal("the teardown must exit the engine's threads")
	}
	if snap := s.Snapshot(s.now); snap.EngineGone != 1 {
		t.Fatalf("the engine-gone counter must be 1, got %d", snap.EngineGone)
	}
	if _, ok := s.groups[gone]; ok {
		t.Fatal("the group must be gone")
	}
	// a thread of the dead engine re-enters and must be seated on the survivor
	survivor := "a"
	if gone == "a" {
		survivor = "b"
	}
	m1 := head("m1", 30*time.Second, 300)
	i := s.Pass([]Head{m1}, now)
	e := s.engineOfDispatch(m1.ReqID)
	if i < 0 {
		t.Fatal("the released thread must be seatable on the survivor")
	}
	if e != survivor {
		t.Fatalf("the re-entry must land on the survivor %s, got %s", survivor, e)
	}
}

// THE FILTER PINS DISPATCH TO THE SEAT ENGINE: one candidate, never empty, and a pass-through
// for unseated or anonymous threads.
func TestFilterPinsTheSeatEngine(t *testing.T) {
	cfg := defaultConfig()
	plugin := newTestPlugin(t, cfg, 2, 1000)
	s := plugin.sched
	now := time.Unix(1793000000, 0).UTC()
	if i := s.Pass([]Head{head("m", time.Second, 200)}, now); i != 0 {
		t.Fatal("the thread must seat")
	}
	seat := s.bindings["m"]
	pods := podsOf(schedulerEndpoint("a"), schedulerEndpoint("b"))
	got := plugin.Filter(t.Context(), requestFor("m"), pods)
	if len(got) != 1 || endpointKey(got[0].GetMetadata()) != seat {
		t.Fatalf("the Filter must narrow to the seat engine %s, got %d candidate(s)", seat, len(got))
	}
	// anonymous: pass everything through
	if got := plugin.Filter(t.Context(), requestFor(""), pods); len(got) != 2 {
		t.Fatalf("anonymous traffic must pass through, got %d", len(got))
	}
	// unseated thread: pass through (no placement opinion yet)
	if got := plugin.Filter(t.Context(), requestFor("stranger"), pods); len(got) != 2 {
		t.Fatalf("an unseated thread must pass through, got %d", len(got))
	}
	// the seat engine is not among the candidates: NEVER empty — pass all through and let the
	// observed endpoint win
	onlyOther := podsOf(schedulerEndpoint("zzz"))
	if got := plugin.Filter(t.Context(), requestFor("m"), onlyOther); len(got) != 1 {
		t.Fatalf("a missing seat engine must not empty the candidate list, got %d", len(got))
	}
	// one candidate: nothing to narrow
	if got := plugin.Filter(t.Context(), requestFor("m"), podsOf(schedulerEndpoint("a"))); len(got) != 1 {
		t.Fatalf("a single candidate must pass through, got %d", len(got))
	}
	// an exit drops the pin
	m, g := func() (*thread, *engineGroup) {
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, gg := range s.groups {
			if mm, ok := gg.threads["m"]; ok {
				return mm, gg
			}
		}
		return nil, nil
	}()
	s.mu.Lock()
	s.exit(g, m, "test", now)
	s.mu.Unlock()
	if _, bound := plugin.sched.SeatEngineOfRequest(""); bound {
		t.Fatal("an untracked request must resolve to no seat")
	}
	if got, ok := plugin.sched.bindings["m"]; ok && got != "" {
		t.Fatalf("an exit must drop the seat binding, still bound to %q", got)
	}
}

// GOLDEN AT N=2.
func TestMultiEngineGoldenProperties(t *testing.T) {
	run := func(engines int, capacity int) (string, Snapshot) {
		cfg := defaultConfig()
		s, err := NewScheduler(cfg)
		if err != nil {
			t.Fatal(err)
		}
		// the golden scenario names engine-a; a second engine is added for N=2
		for i := 0; i < engines; i++ {
			id := fmt.Sprintf("engine-%c", 'a'+i)
			if err := s.SetNumBlocksFor(id, 64, capacity); err != nil {
				t.Fatal(err)
			}
		}
		trace := runScenarioOn(t, s)
		return trace, s.Snapshot(s.now)
	}
	oneA, snapOne := run(1, 1000)
	oneB, _ := run(1, 1000)
	if oneA != oneB {
		t.Fatal("the golden scenario must be deterministic at N=1")
	}
	twoA, snapTwo := run(2, 1000)
	twoB, _ := run(2, 1000)
	if twoA != twoB {
		t.Fatal("the golden scenario must be deterministic at N=2 (no map-order decision left)")
	}
	// conservation: two engines of the same size serve MORE (two budgets), never fewer, and the
	// totals are sane multiples rather than double-counted figures
	if snapTwo.Served < snapOne.Served {
		t.Fatalf("N=2 must serve at least as many dispatches as N=1 (%d < %d)", snapTwo.Served, snapOne.Served)
	}
	if snapTwo.BudgetBlocks != 2*snapOne.BudgetBlocks {
		t.Fatalf("the whole-policy budget must be the per-engine sum: %d vs %d", snapTwo.BudgetBlocks, snapOne.BudgetBlocks)
	}
	var seats int
	for _, es := range snapTwo.Engines {
		seats += es.Seats
		// THE BUDGET, I1 + R4: admission counts the blocks an exiting thread still pins (the engine
		// holds them until its drain), so the USED side may exceed the budget by exactly those and
		// by nothing else.
		if es.NEBlocks > es.BudgetBlocks+es.ExitingBlocks {
			t.Fatalf("engine %s is over its budget at the end of the run: used %d > budget %d + exiting %d",
				es.ID, es.NEBlocks, es.BudgetBlocks, es.ExitingBlocks)
		}
		// the derived cold-entry cap is per engine and the charge never exceeds it
		if es.ColdCapTokens <= 0 {
			t.Fatalf("engine %s has no derived cold-entry cap", es.ID)
		}
		if es.ColdInFlight > es.ColdCapTokens {
			t.Fatalf("engine %s: cold charge %d exceeds its derived cap %d", es.ID, es.ColdInFlight, es.ColdCapTokens)
		}
	}
	if seats != snapTwo.Seats {
		t.Fatalf("the whole-policy seat count must be the per-engine sum (%d vs %d)", seats, snapTwo.Seats)
	}
}

// THE COLD VALVE NEVER EVICTS (the defect the core fixes): a head that fits an engine's BUDGET
// but is held by the cold-entry valve WAITS for the prefill slot.
func TestColdValveHeldHeadNeverEvicts(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		// ONE resident per ENGINE seats (its cold charge alone fills the cap, and the placement
		// spreads them one per engine because a closed valve is not a candidate), and their prefills
		// stay in flight (no Chunk), so EVERY engine's valve has no room for another entry
		for i := range fx.ids {
			id := fmt.Sprintf("res%d", i)
			fx.seatOne(t, id, 600, time.Duration(5+i)*time.Second)
		}
		before := fx.s.Snapshot(fx.clock)
		// the residents' prefills are RUNNING (in flight, no chunk), which is what keeps their cold
		// charge live — and a running thread is never a rotation victim, so nothing can be evicted
		// for the head that waits.
		for attempt := 0; attempt < 4; attempt++ {
			h := head(fmt.Sprintf("cold%d", attempt), time.Second, 900)
			_, i := fx.pickEngine(h)
			if i >= 0 {
				t.Fatalf("attempt %d: a valve-held head must not be dispatched, got %d", attempt, i)
			}
		}
		after := fx.s.Snapshot(fx.clock)
		if after.Seats != before.Seats {
			t.Fatalf("THE VALVE MUST NEVER EVICT: seats moved %d -> %d for a head that only the valve held",
				before.Seats, after.Seats)
		}
		if len(after.Exits) != len(before.Exits) {
			t.Fatalf("no exit may be booked for a valve-held head: %v -> %v", before.Exits, after.Exits)
		}
		// R11 liveness: the valve opens when the residents' prefills complete
		for i := range fx.ids {
			fx.s.Chunk(fmt.Sprintf("res%d", i))
		}
		if _, i := fx.pickEngine(head("coldX", time.Second, 900)); i < 0 {
			t.Fatal("R11: once the cold prefills land, the waiting head must be seatable")
		}
	})
}

// AT N=2 THE HEAD MOVES TO THE SIBLING rather than waiting: engine A's valve refuses it,
// engine B both fits by budget AND allows it.
func TestColdValveHeldHeadSeatsOnTheSibling(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		if len(fx.ids) < 2 {
			t.Skip("needs two engines: the sibling is the point")
		}
		// one engine takes the first cold entry (fits AND allowed: nothing cold in flight there), so
		// ITS valve is now closed while the sibling's is open
		fx.seatOne(t, "coldA", 300, 5*time.Second)
		engA := fx.s.bindings["coldA"]
		// the next head fits the budget on both; A's valve refuses it and B's allows
		h := head("next", time.Second, 50)
		_, i := fx.pickEngine(h)
		if i < 0 {
			t.Fatal("the head must seat (on the SIBLING), not wait")
		}
		if got := fx.s.bindings["next"]; got == engA {
			t.Fatalf("the head was refused by %s's cold valve and must go to the sibling, not %s", engA, got)
		}
	})
}
