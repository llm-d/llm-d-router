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

// RULE O — THE EMPTY ENGINE ADMITS ONE (the oversize-request liveness fix).

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// pickAndLand releases a turn for the head and brings it to a pod (the live order: Pick, a
// decision pass inside the window, then PreRequest).
func pickAndLand(t *testing.T, fx *engineSet, h Head) string {
	t.Helper()
	engine, i := fx.pickEngine(h)
	if i < 0 {
		t.Fatalf("the head must be served, got %d", i)
	}
	fx.s.releaseThread("", h.ReqID, nil, fx.clock)
	fx.s.DispatchedOn(h.ReqID, engine, fx.clock)
	return engine
}

// seatIncumbent places a tenure-eligible resident on a NAMED engine directly: the
// convergence family needs both engines occupied, and the public seat path would choose the
// engine for us.
func seatIncumbent(fx *engineSet, engine, id string, blocks, busyPeriods int) {
	g := fx.s.groups[engine]
	g.index.TrySeat(id, chainOf(id, blocks), g.budgetBlocks)
	g.threads[id] = &thread{id: id, seatTime: fx.clock,
		lastActivity: fx.clock, busyPeriods: busyPeriods, coldDone: true}
	fx.s.bindings[id] = engine
}

// settleTurn completes one turn of a thread the fixture dispatched by hand (one busy period).
func settleTurn(fx *engineSet, thread string, reqID string, blocks int) {
	fx.s.Settle(reqID, "", fx.clock)
}

// TestRuleOEmptyEngineAdmitsOneOversizeHead is families 1, 4 and 6 at once: the oversize head
// is seated on an EMPTY engine, the seat charges its full exact chain (so the engine reads
// over budget), and nothing else seats on that engine while it is there — at N=2 the small
// head goes to the other engine, at N=1 it waits (family 5 is its own test).
func TestRuleOEmptyEngineAdmitsOneOversizeHead(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		budget := int64(fx.g().budgetBlocks)
		big := head("big", time.Second, int(budget)+250) // over the budget on every engine
		small := head("small", 500*time.Millisecond, 100)

		// ONE pass sees both heads: big is the queue head, the small one is behind it
		engine, i := fx.pickEngine(big, small)
		if i < 0 {
			t.Fatal("RULE O: an oversize head must be admitted on an EMPTY engine")
		}
		fx.s.releaseThread("", big.ReqID, nil, fx.clock)
		fx.s.DispatchedOn(big.ReqID, engine, fx.clock)

		if !fx.s.Seated("big") {
			t.Fatal("the oversize thread must hold a seat")
		}
		g := fx.s.groups[engine]
		if got, want := int64(g.index.UniqueBlocks()), budget+250; got != want {
			t.Fatalf("the seat must charge the FULL exact chain: used %d, want %d", got, want)
		}
		if used := int64(g.index.UniqueBlocks()); used <= budget {
			t.Fatalf("an oversize seat reads %d blocks against a %d-block budget: the engine must read OVER budget", used, budget)
		}

		// the small head: at N=2 it seats on the OTHER (still empty) engine; at N=1 it waits
		if len(fx.ids) == 1 {
			if fx.s.Seated("small") {
				t.Fatal("nothing may seat beside an oversize resident: its engine's free budget is negative")
			}
			if i := fx.pick(small); i >= 0 {
				t.Fatal("a small head must not be dispatchable while it cannot be seated")
			}
		} else {
			fx.pick(small) // the head is still queued: one more pass admits it elsewhere
			if !fx.s.Seated("small") {
				t.Fatal("at N=2 the small head must seat on the other engine")
			}
			if got := fx.s.bindings["small"]; got == engine {
				t.Fatalf("the small head seated beside the oversize resident (engine %s)", got)
			}
		}
	})
}

// TestRuleOEvictionsConvergeOnOneEngine is family 3: with incumbents everywhere and no empty
// engine, the evictions all land on ONE engine — the one placement picks (most free budget,
// then id) — whose free budget rises with each exit until it is empty and takes the head.
func TestRuleOEvictionsConvergeOnOneEngine(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// the exit stream IS the observation (the event carries the engine and the reason)
		type exit struct{ engine, reason string }
		var exits []exit
		fx.s.onEvent = func(ev Event) {
			if ev.Kind != "EXIT" {
				return
			}
			reason, _, _ := strings.Cut(ev.Detail, " ")
			exits = append(exits, exit{ev.Engine, strings.TrimPrefix(reason, "reason=")})
		}

		// two tenure-eligible incumbents on EVERY engine, so any engine could be emptied
		fx.s.mu.Lock()
		for _, engine := range fx.ids {
			for j := 0; j < 2; j++ {
				seatIncumbent(fx, engine, fmt.Sprintf("inc-%s-%d", engine, j), 400, fx.s.cfg.Quantum)
			}
		}
		fx.s.mu.Unlock()

		budget := int64(fx.g().budgetBlocks)
		big := head("big", time.Minute, int(budget)+250)
		engine, i := fx.pickEngine(big)
		if i < 0 {
			t.Fatal("RULE O: the head must be seated once an engine is freed for it")
		}
		fx.s.releaseThread("", big.ReqID, nil, fx.clock)
		fx.s.DispatchedOn(big.ReqID, engine, fx.clock)

		if !fx.s.Seated("big") {
			t.Fatal("the oversize head must hold the seat it freed")
		}
		if got := fx.s.bindings["big"]; got != engine {
			t.Fatalf("the head seated on %s, but placement picked %s", got, engine)
		}
		// CONVERGENCE: every eviction happened on the engine that took the head, and it is now empty
		// of incumbents.
		for _, e := range exits {
			if e.engine != engine {
				t.Fatalf("an eviction for ONE oversize head landed on %s, not on the engine that took it (%s): the order did not converge", e.engine, engine)
			}
		}
		if want := 2; len(exits) != want {
			t.Fatalf("the engine held 2 incumbents: want %d exits on it, got %d (%v)", want, len(exits), exits)
		}
		for _, other := range fx.ids {
			if other == engine {
				continue
			}
			g := fx.s.groups[other]
			if len(g.threads) != 2 {
				t.Fatalf("engine %s lost residents (%d left) for a head that seated elsewhere", other, len(g.threads))
			}
		}
	})
}

// TestRuleOWaitsForTenureNotPressure is family 5: once the oversize thread holds its engine,
// no amount of pressure passes touches it — not the enforcement sweep, not a waiting small
// head — and it is not even MARKED.
func TestRuleOWaitsForTenureNotPressure(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		budget := int64(fx.g().budgetBlocks)
		big := head("big", time.Minute, int(budget)+250)
		engine := pickAndLand(t, fx, big)
		if !fx.s.Seated("big") {
			t.Fatal("fixture: the oversize thread must be seated")
		}
		// the other engines are FULL, so a waiting head cannot escape to one of them (at N=1 there
		// is none): the pressure this test applies is pressure on the oversize engine
		fx.closeEnginesExcept(t, engine, 950)

		var exits int
		fx.s.onEvent = func(ev Event) {
			if ev.Kind == "EXIT" {
				exits++
			}
		}

		// many passes with growth and a waiting small head in the queue
		for round := 0; round < 40; round++ {
			fx.clock = fx.clock.Add(100 * time.Millisecond)
			small := head(fmt.Sprintf("small-%d", round), time.Second, 100)
			fx.pick(small)
			// every few rounds the resident's own next turn grows into a new chain
			if round%7 == 3 {
				grown := headWithChain("big", time.Second, chainOf("big", int(budget)+400))
				grown.ReqID = fmt.Sprintf("r-big-%d", round)
				if i := fx.pick(grown); i >= 0 {
					fx.s.releaseThread("", grown.ReqID, nil, fx.clock)
					fx.s.DispatchedOn(grown.ReqID, engine, fx.clock)
					settleTurn(fx, "big", grown.ReqID, int(budget)+400)
				}
			}
		}

		if !fx.s.Seated("big") {
			t.Fatal("RULE O: pressure must never take the engine's sole oversize resident")
		}
		if exits != 0 {
			t.Fatalf("the sole oversize resident was exited %d times: that is the thrash the rule forbids", exits)
		}
		g := fx.s.groups[engine]
		if g.index.Exiting("big") {
			t.Fatal("the sole oversize resident was MARKED exiting: the mark is the thrash, one pass later")
		}
		// and no small head slipped onto the engine beside it
		for _, id := range []string{"small-0", "small-39"} {
			if fx.s.Seated(id) {
				t.Fatalf("%s seated on an engine whose free budget is negative", id)
			}
		}
	})
}

// TestRuleOTenureFreesTheEngineForAWaiter is family 5's other half at BOTH engine counts: the
// oversize thread leaves by TENURE once its Q busy periods are done, and the waiting head then
// takes the engine — the stall ends through the normal rotation, not through pressure.
func TestRuleOTenureFreesTheEngineForAWaiter(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		budget := int64(fx.g().budgetBlocks)
		big := head("big", time.Minute, int(budget)+250)
		engine := pickAndLand(t, fx, big)
		if !fx.s.Seated("big") {
			t.Fatal("fixture: the oversize thread must be seated")
		}
		// one engine only, at both N: the waiter must take THIS engine once it is free
		fx.closeEnginesExcept(t, engine, 950)

		small := head("small", time.Second, 100)
		if i := fx.pick(small); i >= 0 {
			t.Fatal("the small head cannot be seated beside the oversize resident")
		}

		// the entry turn completes, then Q further busy periods make the resident rotation-eligible
		settleTurn(fx, "big", big.ReqID, int(budget)+250)
		for turn := 1; turn <= fx.s.cfg.Quantum; turn++ {
			h := headWithChain("big", time.Second, chainOf("big", int(budget)+250))
			h.ReqID = fmt.Sprintf("r-big-%d", turn)
			if i := fx.pick(h); i < 0 {
				t.Fatalf("turn %d: the seated thread's own turn must dispatch", turn)
			}
			fx.s.releaseThread("", h.ReqID, nil, fx.clock)
			fx.s.DispatchedOn(h.ReqID, engine, fx.clock)
			settleTurn(fx, "big", h.ReqID, int(budget)+250)
		}

		// the waiter is still there: the tenure rule now releases the resident, and it seats
		if i := fx.pick(small); i < 0 {
			t.Fatalf("the waiting head must take the engine the tenure rule released (engine %s)", engine)
		}
		if fx.s.Seated("big") {
			t.Fatal("the resident completed its quantum: it must have been rotated out")
		}
		if !fx.s.Seated("small") {
			t.Fatal("the waiting head must hold the seat the rotation freed")
		}
		if got := fx.s.bindings["small"]; got != engine {
			t.Fatalf("the waiter seated on %s, want the freed engine %s", got, engine)
		}
	})
}

// TestRuleOIsDeterministic: the same oversize scenario decides the same way twice (the
// placement tie-break and the eviction order are total orders, so nothing here reads map
// order).
func TestRuleOIsDeterministic(t *testing.T) {
	run := func() string {
		s, err := NewScheduler(defaultConfig())
		if err != nil {
			t.Fatal(err)
		}
		fx := &engineSet{s: s, clock: time.Unix(1793000000, 0).UTC()}
		for i := 0; i < 2; i++ {
			id := fmt.Sprintf("e%d", i)
			fx.ids = append(fx.ids, id)
			if err := s.SetNumBlocksFor(id, 64, 1000); err != nil {
				t.Fatal(err)
			}
		}
		var trace string
		s.onEvent = func(ev Event) {
			if ev.Kind == "EXIT" {
				reason, _, _ := strings.Cut(ev.Detail, " ")
				trace += ev.Engine + ":" + strings.TrimPrefix(reason, "reason=") + ";"
			}
		}
		fx.s.mu.Lock()
		for _, engine := range fx.ids {
			for j := 0; j < 2; j++ {
				seatIncumbent(fx, engine, fmt.Sprintf("inc-%s-%d", engine, j), 400, s.cfg.Quantum)
			}
		}
		fx.s.mu.Unlock()
		engine, i := fx.pickEngine(head("big", time.Minute, 1200))
		trace += fmt.Sprintf("seat=%s:%d", engine, i)
		return trace
	}
	first := run()
	for i := 0; i < 3; i++ {
		if again := run(); again != first {
			t.Fatalf("run %d decided differently: %q vs %q", i, again, first)
		}
	}
}

// TestSoleOversizeStaysSeatedWithoutWaiters: the (b) fix — the sole-oversize guard runs BEFORE
// the rotation-victim class.
func TestSoleOversizeStaysSeatedWithoutWaiters(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		budget := int64(fx.g().budgetBlocks)
		// Seat the oversize thread by Rule O and run it past Q busy periods.
		h := head("big", time.Second, int(budget)+250)
		engine, i := fx.pickEngine(h)
		if i < 0 {
			t.Fatal("Rule O did not seat the oversize head")
		}
		fx.s.releaseThread("", h.ReqID, nil, fx.clock)
		fx.s.DispatchedOn(h.ReqID, engine, fx.clock)
		for j := 0; j < 5; j++ { // Q + 1 busy periods
			settleTurn(fx, "big", h.ReqID, int(budget)+250)
			h = head("big", time.Second, int(budget)+250)
			if _, k := fx.pickEngine(h); k < 0 {
				t.Fatalf("turn %d: the oversize thread lost its seat mid-residency", j)
			}
			fx.s.releaseThread("", h.ReqID, nil, fx.clock)
			fx.s.DispatchedOn(h.ReqID, engine, fx.clock)
		}
		// Settle the final turn: the thread must be idle (not in flight) for the tenure rule to see
		// it as eligible.
		settleTurn(fx, "big", h.ReqID, int(budget)+250)
		// Many Pick passes with NO waiters: the thread must stay seated.
		small := head("small", 500*time.Millisecond, 100)
		for p := 0; p < 20; p++ {
			if i := fx.pick(); i != -1 {
				t.Fatalf("pass %d: something was dispatched with no waiters (index %d)", p, i)
			}
			if !fx.s.Seated("big") {
				t.Fatalf("pass %d: the sole oversize thread was evicted with no waiters", p)
			}
		}
		// A waiter arrives: tenure rotation makes room for it.
		fx.pick(small)
		for p := 0; p < 3 && !fx.s.Seated("small"); p++ {
			fx.pick(small)
		}
		if !fx.s.Seated("small") {
			t.Fatal("the waiter never seated — the oversize thread did not rotate out by tenure")
		}
		// At N=1 the waiter must take the oversize thread's engine (it rotated out by tenure).
		if smallEngine := fx.s.bindings["small"]; smallEngine == engine && fx.s.Seated("big") {
			t.Fatal("the waiter took the oversize engine but the oversize thread is still seated")
		}
	})
}
