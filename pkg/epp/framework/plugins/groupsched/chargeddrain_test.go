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

// THE CHARGED DRAIN'S RELEASE INVARIANT, ONE TEST PER EXIT PATH.

import (
	"fmt"
	"testing"
	"time"
)

// ---- helpers ----

// neOn is the engine's budget-side (NE) block count, under the scheduler's lock.
func neOn(fx *engineSet, g *engineGroup) int {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return g.index.UniqueBlocks()
}

// uniqueOn is the all-seated block count (the other side of the same coin).
func uniqueOn(fx *engineSet, g *engineGroup) int {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return g.index.UniqueBlocks()
}

// chargedPath runs one path test at N=1 and N=2 with the fixture every path needs: "keep" (a
// survivor whose charge must never be touched) and "gone" (the thread that becomes exiting),
// BOTH on the fixture's first engine. fx.tighten closes every other engine with a filler at
// its whole budget, so both threads land on e0 at N=1 AND at N=2 — the per-engine assertions
// then mean the same thing at both counts, and the sibling's filler is the cross-engine check.
func chargedPath(t *testing.T, cfg Config, path string, fn func(t *testing.T, fx *engineSet, g *engineGroup)) {
	t.Helper()
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "keep", 400, 5*time.Second) // e0 (the tie goes to e0)
		fx.s.Chunk("keep")
		fx.tighten(t) // every other engine full: "gone" has one home
		fx.seatOne(t, "gone", 300, 5*time.Second)
		g := fx.g()
		for _, id := range []string{"keep", "gone"} {
			if _, ok := g.threads[id]; !ok {
				t.Fatalf("fixture: %q must be seated on %s", id, g.id)
			}
		}
		if got, want := neOn(fx, g), 700; got != want {
			t.Fatalf("fixture: NE = %d want %d", got, want)
		}
		fn(t, fx, g)
	})
}

// markExitingInflight makes "gone" an exiting thread with one in-flight request, exactly as
// the pressure valve's mid-flight class does, and asserts the mark moved no counter (the whole
// point of (iii): it is a mark, not a release).
func markExitingInflight(t *testing.T, fx *engineSet, g *engineGroup, reqID string) {
	t.Helper()
	fx.s.mu.Lock()
	m := g.threads["gone"]
	before, beforeU := g.index.UniqueBlocks(), g.index.UniqueBlocks()
	inFlight(fx.s, m, fx.clock, reqID)
	g.index.MarkExiting("gone")
	n, u := g.index.UniqueBlocks(), g.index.UniqueBlocks()
	fx.s.mu.Unlock()
	if n != before {
		t.Fatalf("MarkExiting moved the budget counter: NE %d -> %d (the mark must move nothing)", before, n)
	}
	if u != beforeU {
		t.Fatalf("MarkExiting moved the used counter: %d -> %d", beforeU, u)
	}
	if stay := g.index.StayBlocks(); stay == beforeU {
		t.Fatalf("MarkExiting must move the STAY side (still %d): the thread's blocks are leaving", stay)
	}
	if !fx.s.Seated("gone") {
		t.Fatal("a marked thread is still seated: it drains, it is not evicted")
	}
}

// assertReleased is the assertion every path test ends with: "gone" is unseated, its 300
// blocks are off the engine's used side, "keep"'s 400 are untouched, the sibling is untouched,
// and no counter diverged.
func assertReleased(t *testing.T, fx *engineSet, g *engineGroup, path string) {
	t.Helper()
	if fx.s.Seated("gone") {
		t.Fatalf("%s: the exiting thread is still seated — its blocks are still charged (LEAK)", path)
	}
	if got, want := neOn(fx, g), 400; got != want {
		why := "over-released (somebody else's charge was taken)"
		if got > want {
			why = "never released (LEAK)"
		}
		t.Fatalf("%s: NE = %d want %d — %s", path, got, want, why)
	}
	if got, want := uniqueOn(fx, g), 400; got != want {
		t.Fatalf("%s: all-seated blocks = %d want %d", path, got, want)
	}
	if !fx.s.Seated("keep") {
		t.Fatalf("%s: the release took the OTHER thread's seat (asymmetry)", path)
	}
	for _, id := range fx.ids[1:] {
		if got, want := neOn(fx, fx.s.groups[id]), 950; got != want {
			t.Fatalf("%s: sibling %s NE = %d want %d (a release crossed engines)", path, id, got, want)
		}
	}
	fx.s.mu.Lock()
	g.index.checkInvariant() // panics on a double release / divergence
	fx.s.mu.Unlock()
}

// ---- path 1: the ordinary drain (the last in-flight request settles) ----

func TestChargedDrainSettleReleases(t *testing.T) {
	chargedPath(t, defaultConfig(), "settle-drain", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone")
		// the request terminates: the finalizer removes the record, sees the thread exiting and
		// drained, and unseats it.
		fx.s.Settle("r-gone", g.id, fx.clock)
		assertReleased(t, fx, g, "settle-drain")
	})
}

// A settle that does NOT drain the thread (a second request still running) must not release:
// the blocks are still pinned.
func TestChargedDrainWaitsForTheLastRequest(t *testing.T) {
	chargedPath(t, defaultConfig(), "settle-drain-last", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r1")
		fx.s.mu.Lock()
		m := g.threads["gone"]
		addInFlight(fx.s, m, fx.clock, "r2")
		fx.s.mu.Unlock()

		fx.s.Settle("r1", g.id, fx.clock)
		if !fx.s.Seated("gone") {
			t.Fatal("one of two in-flight requests settled: the thread must stay charged")
		}
		if got := neOn(fx, g); got != 700 {
			t.Fatalf("NE = %d want 700 while one request still runs", got)
		}
		fx.s.Settle("r2", g.id, fx.clock)
		assertReleased(t, fx, g, "settle-drain (last)")
	})
}

// A LATE/DUPLICATE settle of an id the record no longer holds closes nothing, so it must not be
// read as a drain (and must not exit anybody).
func TestChargedDrainSettleWithNoRecordDoesNotRelease(t *testing.T) {
	chargedPath(t, defaultConfig(), "settle-no-record", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r1")
		fx.s.mu.Lock()
		m := g.threads["gone"]
		addInFlight(fx.s, m, fx.clock, "r2")
		fx.s.mu.Unlock()

		fx.s.Settle("not-a-known-id", g.id, fx.clock)
		if !fx.s.Seated("gone") {
			t.Fatal("an unknown req id removed nothing, so nothing drained: the seat must stand")
		}
		if got := neOn(fx, g); got != 700 {
			t.Fatalf("NE = %d want 700 after a settle with no record", got)
		}
	})
}

// ---- path 2: engine-gone (the engine disappears under the thread) ----

func TestChargedDrainEngineGoneReleases(t *testing.T) {
	chargedPath(t, defaultConfig(), "engine-gone", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone")
		if n := fx.s.RemoveEngine(g.id, fx.clock); n == 0 {
			t.Fatal("the teardown must exit the threads it tears down")
		}
		if got, want := neOn(fx, g), 0; got != want {
			t.Fatalf("engine-gone: NE = %d want %d — a dead engine's charge leaked", got, want)
		}
	})
}

// ---- path 3: the reap — a termination nobody observed ----
// This is the class the reviewer pointed at: the settle is NOT the only way an in-flight count
// reaches zero. reapRequests drops ids whose termination was never observed — a client
// cancel whose forced EndOfStream never reached the plugin, or a request that errored before
// its first chunk.
func TestChargedDrainReapOfCancelledRequestReleases(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60 // the campaign shape's bound, scaled down
	cfg.IdleSeconds = 1e5
	chargedPath(t, cfg, "reap-cancelled", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-cancelled")
		fx.tick(61 * time.Second) // the confirmed-id reap bound
		assertReleased(t, fx, g, "reap (cancel, no settle)")
	})
}

// A request that died between the RELEASE and PreRequest (an error before the first chunk).
func TestChargedDrainReleasedWindowReleases(t *testing.T) {
	cfg := defaultConfig()
	cfg.IdleSeconds = 5
	chargedPath(t, cfg, "released-window", func(t *testing.T, fx *engineSet, g *engineGroup) {
		fx.s.mu.Lock()
		m := g.threads["gone"]
		charged := m.coldEntry
		noInFlight(fx.s, m)
		pending(fx.s, m, fx.clock, "r-gone") // released at the last Pick, never reached a pod
		g.index.MarkExiting("gone")
		fx.s.mu.Unlock()
		if charged == 0 {
			t.Fatal("fixture: the entry's cold charge must still be outstanding (no chunk, no settle)")
		}
		// a LIVE window is in flight: the thread must not be released out from under it
		if i := fx.pick(); i != -1 {
			t.Fatalf("no heads: nothing to dispatch, got %d", i)
		}
		if !fx.s.Seated("gone") {
			t.Fatal("a LIVE window must keep the thread: its request may be at the pod any moment")
		}
		// past S the window is presumed dead, and the same derivation releases it
		fx.tick(6 * time.Second)
		assertReleased(t, fx, g, "dead released window (error before PreRequest)")
		if got := g.coldInFlight; got != 0 {
			t.Fatalf("the cold-entry charge leaked on the released-window path: %d want 0", got)
		}
	})
}

// A LEAKED cold charge is recovered by the derivation at the next decision point, and nothing
// is evicted for it: the engine's valve is closed for everyone while the charge stands, so a
// charge nobody can explain must go — but a charge a head or a live request explains must
// stay.
func TestLeakedColdChargeIsReleasedWithoutEvicting(t *testing.T) {
	cfg := defaultConfig()
	cfg.IdleSeconds = 1e5
	chargedPath(t, cfg, "leaked-charge", func(t *testing.T, fx *engineSet, g *engineGroup) {
		fx.s.mu.Lock()
		// "keep" is innocent: it holds blocks and nothing else is wrong with it
		keep := g.threads["keep"]
		keepCold := keep.coldEntry
		// "gone" leaked: it was charged at its seat, then its request vanished without a chunk, a
		// settle or a PreRequest.
		m := g.threads["gone"]
		noInFlight(fx.s, m)
		m.waitingSince = time.Time{}
		leaked := m.coldEntry
		fx.s.mu.Unlock()
		if leaked == 0 {
			t.Fatal("fixture: the leaked charge must be outstanding")
		}
		if i := fx.pick(); i != -1 {
			t.Fatalf("no heads: nothing to dispatch, got %d", i)
		}
		fx.s.mu.Lock()
		after := g.threads["gone"].coldEntry
		keepAfter := g.threads["keep"].coldEntry
		seated := len(g.threads)
		fx.s.mu.Unlock()
		if after != 0 {
			t.Fatalf("the leaked charge survived a Pick (still %d)", after)
		}
		if keepAfter != keepCold {
			t.Fatalf("the innocent thread's charge moved: %d -> %d", keepCold, keepAfter)
		}
		if seated != 2 {
			t.Fatalf("the derivation evicted a thread for a charge: %d seats, want 2", seated)
		}
	})
}

// ---- path 4: TTL, and the idle exit while already exiting ----
// A thread that is exiting with nothing in flight is NOT a rotation victim: the victim rules
// skip it (it is already leaving, and marking it again frees nothing).
func TestChargedDrainHeadlessIdleReleases(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60
	chargedPath(t, cfg, "ttl", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone")
		fx.s.mu.Lock()
		m := g.threads["gone"]
		noInFlight(fx.s, m)
		m.lastActivity = fx.clock
		m.waitingSince = time.Time{} // headless: nothing queued for it
		fx.s.mu.Unlock()
		fx.tick(31 * time.Second) // past the TTL
		assertReleased(t, fx, g, "ttl")
	})
}

// The same thread WITH a queued head still on the books (the common case at c128: it has its
// next turn waiting).
func TestChargedDrainExitingWithQueuedHeadReleases(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60
	chargedPath(t, cfg, "exiting-with-head", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone")
		fx.s.mu.Lock()
		m := g.threads["gone"]
		noInFlight(fx.s, m)
		m.lastActivity = fx.clock
		m.waitingSince = fx.clock.Add(-10 * time.Second) // its next turn is queued
		fx.s.mu.Unlock()
		// a Pick with no heads: the reconciliation runs, nothing can be re-admitted
		if i := fx.pick(); i != -1 {
			t.Fatalf("no heads: nothing to dispatch, got %d", i)
		}
		assertReleased(t, fx, g, "exiting-with-head")
	})
}

// ... and while it DOES have a request in flight it must never be dispatched for its queued
// head: the seat is not being handed out twice on the way out.
func TestChargedDrainExitingIsNeverDispatched(t *testing.T) {
	chargedPath(t, defaultConfig(), "exiting-no-dispatch", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone")
		if _, i := fx.pickEngine(head("gone", 10*time.Second, 300)); i >= 0 {
			t.Fatal("an exiting thread must not be dispatched for its queued head")
		}
		if got := neOn(fx, g); got != 700 {
			t.Fatalf("the thread is still draining: NE = %d want 700", got)
		}
	})
}

// ---- path 5: the release is exactly once, and a fresh residency starts clean ----

// The release point is UNSET and idempotent: unseating an exiting thread releases both
// counters (the used side and, for an exiting thread, nothing more on the stay side), a second
// unseat is a harmless no-op rather than a second release (which would underflow and panic),
// and the next residency starts non-exiting and charges once.
func TestChargedDrainUnseatIsExactlyOnce(t *testing.T) {
	chargedPath(t, defaultConfig(), "unseat-once", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone")

		fx.s.mu.Lock()
		g.index.Unseat("gone")
		g.index.checkInvariant()
		n := g.index.UniqueBlocks()
		fx.s.mu.Unlock()
		if got, want := neOn(fx, g), 400; got != want {
			t.Fatalf("unseating a seated exiting thread: NE = %d want %d", got, want)
		}
		if n != 400 {
			t.Fatalf("used after the unseat: %d want 400 (the survivor's)", n)
		}
		// the second release must be a no-op, not an underflow
		fx.s.mu.Lock()
		g.index.Unseat("gone")
		g.index.checkInvariant()
		ok, _ := g.index.TrySeat("gone", chainOf("gone", 400), 950)
		n = g.index.UniqueBlocks()
		exiting := g.index.Exiting("gone")
		fx.s.mu.Unlock()
		if !ok {
			t.Fatal("the thread must be re-seatable after its release")
		}
		if got, want := n, 800; got != want {
			// the fresh residency charges its own chain: 400 (survivor) + 400 (its own)
			t.Fatalf("a fresh residency: NE = %d want %d (charged once, on its own blocks)", got, want)
		}
		if exiting {
			t.Fatal("a fresh residency must not inherit the exiting flag")
		}
	})
}

// ---- path 6: the other direction — a mark on a thread that holds no charge ----

// MarkExiting refuses an unseated session (a seated thread is the only thing that can be
// exiting), so no path can release a charge that was never taken.
func TestChargedDrainMarkOnUnseatedRecordIsANoOp(t *testing.T) {
	chargedPath(t, defaultConfig(), "mark-unseated", func(t *testing.T, fx *engineSet, g *engineGroup) {
		fx.s.mu.Lock()
		g.index.MarkExiting("never-seated")
		g.index.MarkExiting("keep-then-gone")
		g.index.checkInvariant()
		n := g.index.UniqueBlocks()
		fx.s.mu.Unlock()
		if got, want := n, 700; got != want {
			t.Fatalf("marking an unseated id moved the budget counter: %d want %d", got, want)
		}
		if g.index.Exiting("keep-then-gone") {
			t.Fatal("an unseated session is not exiting")
		}
	})
}

// TestReapPathsIncrementTheirOwnAnomaly: each reap path (released-window vs in-flight)
// increments its own anomalies_total kind and only its own.
func TestReapPathsIncrementTheirOwnAnomaly(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// A released-window record: dispatched at Pick but never arrived at a pod.
		h1 := head("wthread", time.Second, 100)
		if _, i := fx.pickEngine(h1); i < 0 {
			t.Fatal("the head must be picked")
		}
		// Do NOT call DispatchedOn — the record stays released (no pod).
		fx.tick(time.Duration(defaultConfig().InflightReapSeconds) * time.Second)
		fx.tick(time.Second) // one more pass to run the reap
		if got := fx.s.counters.anomalies["reaped_window"]; got != 1 {
			t.Fatalf("reaped_window = %d, want 1 (the released-window reap must count)", got)
		}
		if got := fx.s.counters.anomalies["reaped_inflight"]; got != 0 {
			t.Fatalf("reaped_inflight = %d, want 0 (the window reap must not fire the in-flight kind)", got)
		}

		// An in-flight record: dispatched AND arrived at a pod, then aged past the reap.
		h2 := head("ithread", time.Second, 100)
		engine, i := fx.pickEngine(h2)
		if i < 0 {
			t.Fatal("the head must be picked")
		}
		fx.s.releaseThread("", h2.ReqID, nil, fx.clock)
		fx.s.DispatchedOn(h2.ReqID, engine, fx.clock)
		fx.tick(time.Duration(defaultConfig().InflightReapSeconds) * time.Second)
		fx.tick(time.Second)
		if got := fx.s.counters.anomalies["reaped_inflight"]; got != 1 {
			t.Fatalf("reaped_inflight = %d, want 1 (the in-flight reap must count)", got)
		}
		if got := fx.s.counters.anomalies["reaped_window"]; got != 1 {
			t.Fatalf("reaped_window = %d, want 1 (only the first thread's window reap; the in-flight reap must not fire the window kind)", got)
		}
	})
}

// TestReapDoesNotStealALiveWindow is the re-checker's probe for I1 on the REAP path. A record can
// stop existing without a settle (a window nobody observed dying), and when the reap deletes it the
// record's count and its place in the thread's window slice must go with it. The bug this pins: the
// reap left `counted` true behind and the record in m.windows, so the NEXT settle's expireWindows
// found the dead record and took one off a LIVE window — the thread then read not-busy and exited
// (or rotated) under work that was still in flight.
func TestReapDoesNotStealALiveWindow(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60 // scaled down; the bound only has to exceed S for the probe
	cfg.IdleSeconds = 20
	chargedPath(t, cfg, "reap-live-window", func(t *testing.T, fx *engineSet, g *engineGroup) {
		fx.s.mu.Lock()
		m := g.threads["gone"]
		// TWO windows on one thread: an OLD one the reap will take, and a LIVE one it must not.
		fx.s.release(m, "r-old", nil, fx.clock.Add(-120*time.Second), dispatchObservables{})
		fx.s.release(m, "r-live", nil, fx.clock.Add(-5*time.Second), dispatchObservables{})
		before := m.pending
		fx.s.mu.Unlock()
		if before != 2 {
			t.Fatalf("fixture: two live windows must count twice, got pending=%d", before)
		}

		fx.tick(1 * time.Second) // r-old is already 121 s old (past the 60 s bound); r-live is 2 s old
		fx.s.mu.Lock()
		_, oldExists := fx.s.reqs["r-old"]
		_, liveExists := fx.s.reqs["r-live"]
		after := m.pending
		windows := len(m.windows)
		fx.s.mu.Unlock()
		if oldExists {
			t.Fatal("the reap must delete the record that outlived the bound")
		}
		if !liveExists {
			t.Fatal("the reap must not delete a record inside the bound")
		}
		// r-live is 2 s old: inside S, so it counts, and the reap must have left exactly that one.
		counted := 0
		fx.s.mu.Lock()
		for _, rec := range fx.s.reqs {
			if rec.thread == m.id && rec.counted {
				counted++
			}
		}
		fx.s.mu.Unlock()
		if after != counted {
			t.Fatalf("pending=%d but only %d record(s) count: the reap took a count that belonged to another window", after, counted)
		}
		if after != 1 {
			t.Fatalf("pending=%d after the reap: the live window's count was taken", after)
		}
		if windows != 1 { // r-live is the thread's only open window
			t.Fatalf("the window slice holds %d pointer(s) for 1 open window: a dead pointer survived", windows)
		}
	})

	// THE EXIT PROBE: with the reap bound configured, a thread holding a LIVE window must not be
	// released by an exit decision — the reap of another record must not have moved its count.
	chargedPath(t, cfg, "reap-exit-over-live-window", func(t *testing.T, fx *engineSet, g *engineGroup) {
		fx.s.mu.Lock()
		m := g.threads["gone"]
		fx.s.release(m, "r-old", nil, fx.clock.Add(-120*time.Second), dispatchObservables{})
		fx.s.release(m, "r-live", nil, fx.clock.Add(-1*time.Second), dispatchObservables{})
		g.index.MarkExiting("gone")
		fx.s.mu.Unlock()
		fx.tick(1 * time.Second) // r-old reaped (121 s); r-live is 2 s old and still inside S
		if fx.pick() != -1 {
			t.Fatal("no heads: nothing to dispatch")
		}
		if !fx.s.Seated("gone") {
			t.Fatal("a thread whose reap left the counts consistent must still be seated here")
		}
	})
}

// TestWindowsSliceIsBounded pins the other half of the reap fix: the windows slice is the thread's
// liveness list, so a record that stops existing must leave it. Before the fix 1000 abandons left
// 1000 dead pointers behind (len 1000, 0 records) — a memory leak on the per-request path that also
// re-ran a dead pointer on every settle.
func TestWindowsSliceIsBounded(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "gone", 300, 5*time.Second)
		fx.s.mu.Lock()
		m := fx.g().threads["gone"]
		// start from an empty record set: the seat's own dispatch left one released window behind,
		// and this probe counts only the abandons it makes.
		for rid, rec := range fx.s.reqs {
			if rec.thread == m.id {
				delete(fx.s.reqs, rid)
			}
		}
		m.pending, m.inflight, m.windows = 0, 0, nil
		for i := 0; i < 1000; i++ {
			id := fmt.Sprintf("r%d", i)
			fx.s.reqs[id] = &reqRec{thread: m.id, released: fx.clock, counted: true}
			m.pending++
			m.windows = append(m.windows, fx.s.reqs[id])
			// the abandon: the record drops with nobody having settled it
			fx.s.dropRequest(id, fx.clock)
		}
		left := 0
		for _, rec := range fx.s.reqs {
			if rec.thread == m.id {
				left++
			}
		}
		pending, windows := m.pending, len(m.windows)
		fx.s.mu.Unlock()
		if left != 0 {
			t.Fatalf("%d record(s) left after 1000 abandons", left)
		}
		if pending != 0 {
			t.Fatalf("pending=%d after 1000 abandons: the count outlived its records", pending)
		}
		if windows != 0 {
			t.Fatalf("the windows slice holds %d dead pointer(s): it must be pruned on every drop", windows)
		}
	})
}

// TestReapOfAnInFlightRecordLeavesNoWindowPointer pins the last path that closed a window without
// leaving the slice: a record that REACHED A POD is not a window, so the thread's window slice must
// have dropped it at PreRequest — and the reap of that record at the bound must therefore change
// neither the slice nor any count. Before atPod unlinked, a reaped in-flight record left a dangling
// pointer in m.windows until the thread exited (rare at 9300 s, same class as the reap bug).
func TestReapOfAnInFlightRecordLeavesNoWindowPointer(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60
	chargedPath(t, cfg, "reap-inflight-window", func(t *testing.T, fx *engineSet, g *engineGroup) {
		fx.s.mu.Lock()
		m := g.threads["gone"]
		// one OPEN window and one record that reached a pod, both on the same thread
		fx.s.release(m, "r-open", nil, fx.clock, dispatchObservables{})
		fx.s.release(m, "r-at-pod", nil, fx.clock.Add(-120*time.Second), dispatchObservables{})
		rec := fx.s.reqs["r-at-pod"]
		fx.s.atPod(rec, m, g.id, fx.clock.Add(-120*time.Second)) // reached the pod long ago
		windowsBefore, pendingBefore := len(m.windows), m.pending
		fx.s.mu.Unlock()
		if windowsBefore != 1 {
			t.Fatalf("fixture: only the open window may sit in the slice, got %d", windowsBefore)
		}
		if pendingBefore != 1 {
			t.Fatalf("fixture: one open window counts once, got pending=%d", pendingBefore)
		}

		fx.tick(1 * time.Second) // the in-flight record is 121 s old and is reaped; the open window is 1 s old
		fx.s.mu.Lock()
		_, stillThere := fx.s.reqs["r-at-pod"]
		windowsAfter, pendingAfter := len(m.windows), m.pending
		fx.s.mu.Unlock()
		if stillThere {
			t.Fatal("the in-flight record must be reaped at the bound")
		}
		if windowsAfter != windowsBefore {
			t.Fatalf("the reap of an at-a-pod record moved the window slice: %d -> %d", windowsBefore, windowsAfter)
		}
		if pendingAfter != pendingBefore {
			t.Fatalf("the reap of an at-a-pod record moved the count: %d -> %d", pendingBefore, pendingAfter)
		}
	})
}
