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

// THE RELEASED WINDOW — every way it closes, and the one eviction it may justify.

import (
	"context"
	"testing"
	"time"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// the two release-path extension points this plugin takes part in.
var (
	_ fwkrc.DataProducer     = (*GroupScheduler)(nil)
	_ fwkrc.RequestAbandoned = (*GroupScheduler)(nil)
)

// closeOtherEngines fills every engine but the first, so "the engine cannot take it" is true
// at N=1 and N=2 alike (the 10,000-block capacity fixture needs its own filler size).
func (fx *engineSet) closeEnginesExcept(t *testing.T, keep string, blocks int) {
	t.Helper()
	for _, e := range fx.ids {
		if e == keep {
			continue
		}
		id := "closed-" + e
		fx.seatOne(t, id, blocks, 3*time.Second)
		if got := fx.s.bindings[id]; got != e {
			t.Fatalf("%s must hold %s, it holds %q", id, e, got)
		}
	}
}

// engineOf is where a thread is seated.
func (fx *engineSet) engineOf(t *testing.T, id string) string {
	t.Helper()
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return fx.s.bindings[id]
}

// openWindow seats thread "m" and then releases a SECOND turn that never reaches a pod: one
// request in flight, one window open.
func openWindow(t *testing.T, fx *engineSet, reqID string) {
	t.Helper()
	fx.seatOne(t, "m", 100, time.Second)
	h := head("m", time.Second, 100)
	h.ReqID = reqID
	if i := fx.pick(h); i != 0 {
		t.Fatalf("the seated thread's head must be picked, got %d", i)
	}
	if got := windowOf(t, fx, "m"); got != reqID {
		t.Fatalf("the window must name %q, it names %q", reqID, got)
	}
}

// windowOf is the request the thread has released and not yet brought to a pod ("" when none):
// the record's own view, never a copy of it.
func windowOf(t *testing.T, fx *engineSet, id string) string {
	t.Helper()
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	for reqID, rec := range fx.s.reqs {
		if rec.thread == id && rec.pod.IsZero() {
			return reqID
		}
	}
	return ""
}

// Every EVENT that proves a request is no longer only "on its way" closes the window.
func TestReleasedWindowClosesOnItsEvents(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		t.Run("PreRequest", func(t *testing.T) {
			openWindow(t, fx, "r-a")
			fx.s.releaseThread("m", "r-a", nil, fx.clock)
			fx.s.Dispatched("r-a", fx.clock)
			if got := windowOf(t, fx, "m"); got != "" {
				t.Fatalf("PreRequest must close the window, it is %q", got)
			}
		})
	})
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		t.Run("terminal-callback", func(t *testing.T) {
			openWindow(t, fx, "r-b")
			// the handler forces this call for every request it scheduled a pod for
			fx.s.Settle("r-b", "", fx.clock)
			if got := windowOf(t, fx, "m"); got != "" {
				t.Fatalf("the terminal callback must close the window, it is %q", got)
			}
		})
	})
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		t.Run("abandon-hook", func(t *testing.T) {
			openWindow(t, fx, "r-c")
			// the director's hook for a request that left the path before PreRequest
			fx.s.Abandon("r-c", fx.clock)
			if got := windowOf(t, fx, "m"); got != "" {
				t.Fatalf("RequestAbandoned must close the window, it is %q", got)
			}
		})
	})
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		t.Run("engine-gone", func(t *testing.T) {
			openWindow(t, fx, "r-d")
			fx.s.RemoveEngine(fx.ids[0], fx.clock)
			if fx.s.Seated("m") {
				t.Fatal("the teardown must unseat the thread")
			}
		})
	})
}

// A window that no event ever closes (the paths that reach no hook: no endpoint candidates at
// the director's entry, an item finalized while it was being dispatched) must not be able to
// freeze anything: it never bars the tenure rule, the engine-idle rule or the in-flight cap.
func TestStaleWindowNeverBarsLiveness(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		openWindow(t, fx, "r-stale")
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		noInFlight(fx.s, m) // its other request is over
		// the release is an HOUR old: past S, so the window is presumed dead (a live one counts as
		// in flight and is protected — see thread.busy)
		pending(fx.s, m, fx.clock.Add(-time.Hour), "r-stale")
		m.lastActivity = fx.clock.Add(-time.Hour) // silent
		fx.s.mu.Unlock()
		// one pass with no heads: the released turn is no longer queued (the next Pick in production
		// clears it the same way)
		fx.pick()
		reason, victim := fx.s.rotationVictim(fx.g(), fx.clock)
		if victim == nil {
			t.Fatal("a dead window must not stop the tenure rule: a silent thread is releasable")
		}
		if reason != "idle" {
			t.Fatalf("the dead window's thread is releasable as %q, want idle", reason)
		}
		if victim.id != "m" {
			t.Fatalf("the victim is %q, want m", victim.id)
		}
	})
}

// A LIVE window (a release a few milliseconds old) is IN FLIGHT: a valve-held head waits for
// it (S has not elapsed), and no rule may evict the thread out from under its request.
func TestLiveWindowIsNotEvicted(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "holder", 600, time.Second)                // its cold charge fills the engine's cap
		fx.closeEnginesExcept(t, fx.engineOf(t, "holder"), 9000) // no sibling may take the head
		fx.s.mu.Lock()
		m := fx.g().threads["holder"]
		noInFlight(fx.s, m)
		pending(fx.s, m, fx.clock, "r-live") // a request on its way, released just now
		m.lastActivity = fx.clock
		fx.s.mu.Unlock()
		fx.pick() // the released turn is no longer queued
		before := fx.s.Snapshot(fx.clock)
		if _, i := fx.pickEngine(head("cold", time.Second, 900)); i >= 0 {
			t.Fatalf("a valve-held head must not be served over a live window, got %d", i)
		}
		after := fx.s.Snapshot(fx.clock)
		if after.Seats != before.Seats || len(after.Exits) != len(before.Exits) {
			t.Fatalf("nothing may be evicted for a live window: seats %d->%d exits %v->%v",
				before.Seats, after.Seats, before.Exits, after.Exits)
		}
		if got := windowOf(t, fx, "holder"); got != "r-live" {
			t.Fatalf("the live window must still be open, it is %q", got)
		}
	})
}

// A DEAD window (> S) frees its valve BY ITSELF: the derivation releases a charge nothing
// explains any more, so the head that the valve was holding is served — and NOTHING is evicted
// for it.
func TestDeadWindowFreesTheValveWithoutEviction(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "holder", 600, time.Second)                // charges min(cost, cap) = the whole cap
		fx.closeEnginesExcept(t, fx.engineOf(t, "holder"), 9000) // no sibling may take it instead
		fx.s.mu.Lock()
		m := fx.g().threads["holder"]
		noInFlight(fx.s, m)
		pending(fx.s, m, fx.clock.Add(-21*time.Second), "r-dead") // dead: 21s > S
		m.lastActivity = fx.clock.Add(-21 * time.Second)
		fx.s.mu.Unlock()
		fx.pick() // the released turn is no longer queued; the charge has nothing to explain it
		if got := fx.g().coldInFlight; got != 0 {
			t.Fatalf("a dead window's charge must be released by the derivation, coldInFlight=%d", got)
		}
		if _, i := fx.pickEngine(head("cold", time.Second, 900)); i < 0 {
			t.Fatal("the head the valve was holding must be served once the dead window's charge is released")
		}
		if !fx.s.Seated("holder") {
			t.Fatal("nothing may be evicted for a head that fits the budget: the holder keeps its seat")
		}
		if exits := fx.s.Snapshot(fx.clock).Exits; len(exits) != 0 {
			t.Fatalf("a dead window needs no eviction at all, exits=%v", exits)
		}
	})
}

// THE BUG1 REGRESSION: a valve-held head never evicts a thread that is NOT the holder — not
// even a releasable one.
func TestValveHeldHeadNeverEvictsANonHolder(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		// an innocent, releasable thread whose charge is already gone (seated FIRST: the holder's
		// charge will fill the valve)
		fx.seatOne(t, "innocent", 100, 2*time.Second)
		fx.s.Chunk("innocent") // its prefill landed: no cold charge left to explain
		// the holder keeps its charge (in flight, so nothing can release it)
		fx.seatOne(t, "holder", 600, time.Second)
		fx.closeEnginesExcept(t, fx.engineOf(t, "holder"), 9000)
		fx.s.mu.Lock()
		m := fx.g().threads["innocent"]
		noInFlight(fx.s, m)
		m.lastActivity = fx.clock.Add(-time.Hour) // it looks idle — but it holds no charge
		fx.s.mu.Unlock()
		before := fx.s.Snapshot(fx.clock)
		if _, i := fx.pickEngine(head("cold", time.Second, 900)); i >= 0 {
			t.Fatalf("a valve-held head must not be served by evicting a NON-holder, got %d", i)
		}
		after := fx.s.Snapshot(fx.clock)
		if after.Seats != before.Seats {
			t.Fatalf("BUG1: seats moved %d -> %d for a valve-held head", before.Seats, after.Seats)
		}
		if !fx.s.Seated("innocent") || !fx.s.Seated("holder") {
			t.Fatal("BUG1: a thread was evicted for a valve-held head")
		}
	})
}

// Two releases of the same thread before either reaches a pod (a retry merged into one queue)
// are TWO records — the thread is held busy by either of them — and a callback ends only the
// record it NAMES, in either order, leaving nothing open at the end.
func TestTwoReleasesClearOnlyTheirOwnWindow(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		openWindow(t, fx, "r-first")
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		// the retry released a second request before the first reached a pod
		fx.s.reqs["r-second"] = &reqRec{thread: "m", released: fx.clock}
		m.pending++
		fx.s.mu.Unlock()
		// a callback for the FIRST request ends the first record only
		fx.s.Abandon("r-first", fx.clock)
		if got := windowOf(t, fx, "m"); got != "r-second" {
			t.Fatalf("a callback for the first request closed the second's window: it names %q", got)
		}
		// and the matching one ends the second
		fx.s.Abandon("r-second", fx.clock)
		if got := windowOf(t, fx, "m"); got != "" {
			t.Fatalf("the matching callback must close the window, it is %q", got)
		}
		// a late callback for an ended request invents nothing
		fx.s.Abandon("r-first", fx.clock)
		if got := windowOf(t, fx, "m"); got != "" {
			t.Fatalf("a late callback reopened the window (%q)", got)
		}
	})
	// the same, in the other order
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		openWindow(t, fx, "r-first")
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		fx.s.reqs["r-second"] = &reqRec{thread: "m", released: fx.clock}
		m.pending++
		fx.s.mu.Unlock()
		fx.s.Abandon("r-second", fx.clock)
		if got := windowOf(t, fx, "m"); got != "r-first" {
			t.Fatalf("the older window must still be open, it names %q", got)
		}
		fx.s.Abandon("r-first", fx.clock)
		if got := windowOf(t, fx, "m"); got != "" {
			t.Fatalf("the matching callback must close the window, it is %q", got)
		}
	})
	_ = context.Background()
}

// usedWithinBudget fails when the engine is over its budget with nothing exiting to excuse it.
func usedWithinBudget(t *testing.T, g *engineGroup) {
	t.Helper()
	used, budget, exiting := g.index.UniqueBlocks(), g.budgetBlocks, int(g.index.ExitingBlocks())
	if used > budget+exiting {
		t.Fatalf("the engine is over its budget with nothing on the way out to excuse it: used %d, budget %d, exiting %d", used, budget, exiting)
	}
}

// (Two tests were removed here at S5b-1: they built a record WITHOUT a seat to exercise
// seatObserved and its `case g == nil`. The invariant makes that state impossible — a
// release requires a seat, and every unseat sweeps the thread's records — so the tests could not
// be ported; the production thread they covered is deleted at S5b-2.)
func TestEngineIdleRuleNeverEvictsALiveWindow(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 1000, func(t *testing.T, fx *engineSet) {
		engine := fx.ids[0]
		fx.seatOne(t, "m", 900, 2*time.Second)
		fx.s.Chunk("m")
		fx.closeEnginesExcept(t, engine, 950) // no sibling may take the new head instead
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		noInFlight(fx.s, m) // nothing at a pod: the window is m's ONLY claim
		pending(fx.s, m, fx.clock, "r-live")
		fx.s.mu.Unlock()
		before := fx.s.Snapshot(fx.clock).Seats
		// a head that does not fit the budget: the old rule read "nothing in flight, nothing
		// dispatchable" and evicted m for it
		if _, i := fx.pickEngine(head("cold", time.Second, 500)); i >= 0 {
			t.Fatalf("a head that does not fit must wait, got %d", i)
		}
		if !fx.s.Seated("m") {
			t.Fatal("the engine-idle rule evicted a thread whose request is on its way")
		}
		if after := fx.s.Snapshot(fx.clock).Seats; after != before {
			t.Fatalf("no seat may move for a live window: %d -> %d", before, after)
		}
		if exits := fx.s.Snapshot(fx.clock).Exits; len(exits) != 0 {
			t.Fatalf("nothing may leave for a live window, exits=%v", exits)
		}
		usedWithinBudget(t, fx.g())
	})
}

// (ii) A MARKED thread is not DRAINED inside a live window: the drain used to release it
// before its request ran, and the observed seat then put 1500 blocks back against 1100.
func TestMarkedThreadIsNotDrainedInsideALiveWindow(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 1000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 900, 2*time.Second)
		fx.s.Chunk("m")
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		noInFlight(fx.s, m)
		pending(fx.s, m, fx.clock, "r-live") // a turn on its way
		fx.g().index.MarkExiting("m")        // the pressure pass already marked it
		fx.s.mu.Unlock()
		fx.pick() // the pass runs the drain
		if !fx.s.Seated("m") {
			t.Fatal("a marked thread must not be drained while its window is live: its request is on its way")
		}
		if exits := fx.s.Snapshot(fx.clock).Exits; exits["exiting-drain"] != 0 {
			t.Fatalf("the drain released a thread mid-window, exits=%v", exits)
		}
		usedWithinBudget(t, fx.g())
		// past S the window is presumed dead and the drain lets it go — exactly once
		fx.tick(time.Duration(fx.s.cfg.IdleSeconds+1) * time.Second)
		if fx.s.Seated("m") {
			t.Fatal("a dead window must not hold a marked thread's seat")
		}
		if exits := fx.s.Snapshot(fx.clock).Exits; exits["exiting-drain"] != 1 {
			t.Fatalf("the departure must be booked exactly once, exits=%v", exits)
		}
	})
}

// (iii) NO ROTATION EXIT inside a live window, even for a thread the tenure rule would release
// on its own (Q complete).
func TestNoRotationExitInsideALiveWindow(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 1000, func(t *testing.T, fx *engineSet) {
		engine := fx.ids[0]
		fx.seatOne(t, "m", 800, 2*time.Second)
		fx.s.Chunk("m")
		fx.closeEnginesExcept(t, engine, 950)
		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		m.busyPeriods = fx.s.cfg.Quantum // the rotation quantum is complete
		noInFlight(fx.s, m)
		pending(fx.s, m, fx.clock, "r-live") // and a turn is on its way
		m.lastActivity = fx.clock.Add(-time.Hour)
		fx.s.mu.Unlock()
		fx.s.mu.Lock()
		reason, victim := fx.s.rotationVictim(fx.g(), fx.clock)
		fx.s.mu.Unlock()
		if victim != nil {
			t.Fatalf("the tenure rule released %q (%s) inside a live window", victim.id, reason)
		}
		// and a blocked head does not get the seat by eviction either
		if _, i := fx.pickEngine(head("cold", time.Second, 500)); i >= 0 {
			t.Fatalf("a head that does not fit must wait, got %d", i)
		}
		if !fx.s.Seated("m") {
			t.Fatal("a quantum-complete thread with a live window must keep its seat until the window ends")
		}
		if exits := fx.s.Snapshot(fx.clock).Exits; len(exits) != 0 {
			t.Fatalf("no rotation exit may be taken inside a live window, exits=%v", exits)
		}
		usedWithinBudget(t, fx.g())
	})
}

// A LATE PreRequest closes only what was COUNTED.
func TestLatePreRequestDoesNotEatAnotherLiveWindow(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, 5*time.Second)
		h1 := head("m", time.Second, 100)
		h1.ReqID = "r-1"
		if i := fx.pick(h1); i < 0 {
			t.Fatal("m's first released turn must dispatch on its seat")
		}
		if got := fx.s.pendingForTest("m"); got != 1 {
			t.Fatalf("one live window, pending=%d", got)
		}
		// r-1's window outlives S: the pass presumes it dead and stops counting it
		fx.clock = fx.clock.Add(time.Duration(fx.s.cfg.IdleSeconds+1) * time.Second)
		fx.pick()
		if got := fx.s.pendingForTest("m"); got != 0 {
			t.Fatalf("the dead window must stop counting, pending=%d", got)
		}
		// a SECOND turn is released now, so its window is live
		h2 := head("m", time.Second, 100)
		h2.ReqID = "r-2"
		if i := fx.pick(h2); i < 0 {
			t.Fatal("m's second released turn must dispatch")
		}
		if got := fx.s.pendingForTest("m"); got != 1 {
			t.Fatalf("one live window again, pending=%d", got)
		}
		// the LATE PreRequest for r-1 arrives: it is in flight now, but r-2 is still on its way
		fx.s.releaseThread("m", "r-1", chainOf("m", 100), fx.clock)
		fx.s.DispatchedOn("r-1", "", fx.clock)
		if got := fx.s.pendingForTest("m"); got != 1 {
			t.Fatalf("the late arrival consumed another live window: pending=%d, want 1", got)
		}
	})
}
