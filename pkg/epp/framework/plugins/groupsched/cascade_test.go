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

// THE ENFORCEMENT CASCADE, and the TWO VIEWS that end it.

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// ---- ENFORCEMENT: one exiting thread covering the overage ends the pass ----

// The engine is over budget by (thread's blocks), and that thread is ALREADY exiting:
// enforcement must mark NOBODY else.
func TestEnforcementDoesNotMarkAnotherWhileOneIsLeaving(t *testing.T) {
	chargedPath(t, defaultConfig(), "no-cascade", func(t *testing.T, fx *engineSet, g *engineGroup) {
		// three threads in flight on one engine; the engine's budget is exactly full
		fx.s.Chunk("gone") // its prefill completed: the cold valve is free for the third seat
		fx.seatOne(t, "m2", 250, 5*time.Second)
		markExitingInflight(t, fx, g, "r-gone") // the 300-block thread leaves
		fx.s.mu.Lock()
		// give the sibling a request too (mid-flight)
		inFlight(fx.s, g.threads["m2"], fx.clock, "r-m2")
		// THE OVERAGE IS EXACTLY THE EXITING MEMBER'S OWN FOOTPRINT: 950 total seated (400 + 300 +
		// 250) against a budget of 100 more... make it precise:
		g.budgetBlocks = 700 // NE is 950: over by 250, and "gone" (300) already covers it
		fx.s.mu.Unlock()

		fx.s.mu.Lock()
		fx.s.enforceBudget(g, fx.clock)
		m2Exiting := g.index.Exiting("m2")
		fx.s.mu.Unlock()
		if m2Exiting {
			t.Fatal("an exiting thread already covers the overage: enforcement marked ANOTHER in-flight thread (the cascade)")
		}
		if !g.index.Exiting("gone") {
			t.Fatal("the thread on its way out must still be the one leaving")
		}
	})
}

// ... and when the overage is LARGER than what is already leaving, the pass takes ONE more
// victim — not the whole engine: after that mark the pass is done again.
func TestEnforcementTakesOneMoreWhenTheOverageExceedsTheRetirees(t *testing.T) {
	chargedPath(t, defaultConfig(), "one-more", func(t *testing.T, fx *engineSet, g *engineGroup) {
		fx.s.Chunk("gone")
		fx.seatOne(t, "m2", 250, 5*time.Second)
		markExitingInflight(t, fx, g, "r-gone")
		fx.s.mu.Lock()
		inFlight(fx.s, g.threads["m2"], fx.clock, "r-m2")
		// NE 950, budget 300: over by 650; "gone" excuses 300, so 350 is still over — more than m2's
		// 250, so exactly ONE further mark can be justified... and after that (250 excused, 100
		// left) the pass must stop: nobody else may be marked while the two retirees alone carry the
		// footprint.
		g.budgetBlocks = 300
		fx.s.mu.Unlock()

		fx.s.mu.Lock()
		fx.s.enforceBudget(g, fx.clock)
		marked := 0
		for id := range g.threads {
			if g.index.Exiting(id) {
				marked++
			}
		}
		fx.s.mu.Unlock()
		if marked != 2 {
			t.Fatalf("%d threads are exiting, want exactly 2 (one retiree + the one mark the overage justifies)", marked)
		}
	})
}

// ---- ADMISSION: the same blocks are still OCCUPIED for a new seat ----

// The heart of the two views: a head that would fit only if the exiting thread's blocks were
// free must WAIT — and then be admitted the moment that thread actually drains.
func TestAdmissionRefusesWhatOnlyTheRetireesBlocksWouldMakeRoomFor(t *testing.T) {
	chargedPath(t, defaultConfig(), "admission-refuses", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone") // NE = 700 of a 950 budget
		fx.s.mu.Lock()
		// shrink the budget so that the newcomer only fits if "gone"'s 300 were free
		g.budgetBlocks = 650
		fx.s.mu.Unlock()

		h := head("newcomer", time.Second, 100)
		if got := neOn(fx, g); got != 700 {
			t.Fatalf("premise: NE = %d", got)
		}
		fx.pick(h)
		if fx.s.Seated("newcomer") {
			t.Fatal("a newcomer was seated on room that only exists if the exiting thread's blocks are free")
		}
		// the drain: now the room is real, and the same head is admitted
		fx.s.Settle("r-gone", g.id, fx.clock)
		fx.s.mu.Lock()
		g.budgetBlocks = 950
		fx.s.mu.Unlock()
		fx.pick(h)
		if !fx.s.Seated("newcomer") {
			t.Fatal("after the drain the room is real: the head must be admitted")
		}
	})
}

// ---- the livelock guard that (iii) added must still work ----

// An engine whose ONLY thread is exiting and in flight, with a queue view too old to trust:
// the pass must return instead of re-picking the same victim forever (the guard the accounting
// fix added — the cascade fix must not remove the reason for it).
func TestExitingVictimDoesNotSpinEnforcementAfterTheCascadeFix(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 900, 2*time.Second)
		m, g := fx.thread(t, "m")
		fx.s.mu.Lock()
		inFlight(fx.s, m, fx.clock, "r-m")
		g.index.MarkExiting("m")
		g.budgetBlocks = 100
		fx.s.mu.Unlock()
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

// exitReasonOf pulls the reason out of a trace's EXIT line.
func exitReasonOf(line string) (string, bool) {
	i := strings.Index(line, "EXIT reason=")
	if i < 0 {
		return "", false
	}
	r := line[i+len("EXIT reason="):]
	if j := strings.IndexAny(r, " \t"); j >= 0 {
		r = r[:j]
	}
	return r, true
}

// ---- the replay differential: the EXIT COMPOSITION, not just the dispatch count ----

// TestExitCompositionIsNotACascade REPORTS the replay's exit composition and mid-flight marks
// per dispatch.
func TestExitCompositionIsNotACascade(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{"default arm (f0p config)", defaultConfig()},
		{"saturated arm (CORE modules, 0.60 seat budget)", func() Config {
			c := defaultConfig()
			c.GroupBudgetFraction = 0.6 // the same memory pressure, with the modules that ship
			return c
		}()},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			trace := runScenarioCfg(t, 2600, tc.cfg)
			exits := map[string]int{}
			total, dispatches := 0, 0
			for _, line := range strings.Split(trace, "\n") {
				if reason, ok := exitReasonOf(line); ok {
					exits[reason]++
					total++
				}
				if strings.Contains(line, " PICK ") && !strings.Contains(line, "PICK -1") {
					dispatches++
				}
			}
			if total == 0 || dispatches == 0 {
				t.Fatalf("the scenario produced %d exits and %d dispatches", total, dispatches)
			}
			marks := exits["exiting-drain"]
			perDispatch := float64(marks) / float64(dispatches)
			t.Logf("%d exits (%.1f %% exiting-drain), %d dispatches, marks/dispatch %.1f %%",
				total, 100*float64(marks)/float64(total), dispatches, 100*perDispatch)
			// A COARSE CEILING only, an order of magnitude below the cascade's own signature (81 %
			// marks per dispatch live): it catches "enforcement is marking threads in bulk", not a
			// policy difference between trees.
			if perDispatch > 0.25 {
				t.Fatalf("%d mid-flight marks for %d dispatches (%.1f %%): enforcement is marking threads in bulk "+
					"(the cascade measured 81 %% live)", marks, dispatches, 100*perDispatch)
			}
		})
	}
}

// ---- R4: THE STAY SIDE, with a SHARED block ----

// TestSharedBlockWithAStayingThreadIsNotExcused is the defect R4 fixes.
func TestSharedBlockWithAStayingThreadIsNotExcused(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		shared := keys(3, seedFor("fork-shared")) // 3 blocks the parent and the fork both hold
		privateKeep := keys(2, seedFor("fork-private"))
		privateGone := keys(2, seedFor("fork-private-gone"))

		fx.s.mu.Lock()
		g := fx.s.groups[fx.ids[0]]
		// a staying thread holding shared+privateKeep, and an exiting one holding
		// shared+privateGone: |used| = 5 + 5 - 3 (shared) = 7, |stay| = 5.
		keep := append(append([]Hash{}, shared...), privateKeep...)
		gone := append(append([]Hash{}, shared...), privateGone...)
		g.index.TrySeat("keep", []Chain{{Keys: keep, Tokens: len(keep) * 64}}, 1<<20)
		g.threads["keep"] = &thread{id: "keep", waitingSince: fx.clock}
		g.index.TrySeat("gone", []Chain{{Keys: gone, Tokens: len(gone) * 64}}, 1<<20)
		g.threads["gone"] = &thread{id: "gone"}
		inFlight(fx.s, g.threads["gone"], fx.clock, "r-gone")
		g.index.MarkExiting("gone")
		used, stay := g.index.UniqueBlocks(), g.index.StayBlocks()
		fx.s.mu.Unlock()

		if used != 7 || stay != 5 {
			t.Fatalf("premise: used = %d, stay = %d (want 7 and 5: 3 shared + 2 + 2)", used, stay)
		}
		// the budget leaves a 1-block overage on the STAY side, which the old arithmetic would have
		// excused entirely (|exit| = 5 ≥ overage 1)
		fx.s.mu.Lock()
		g.budgetBlocks = 4
		over := fx.s.overBudget(g)
		fx.s.mu.Unlock()
		if over != 1 {
			t.Fatalf("enforcement's overage = %d, want 1 (stay 5 − budget 4): a shared block an exiting thread holds is NOT a reason to stop evicting", over)
		}

		// THE OLD FORMULA, for the record: it excused |exit| wholesale — used (7) − exit (5) = 2 ≤
		// budget (4) → "nothing to do", and the engine stayed overcommitted by the shared block.
		exiting := 5
		if old := int64(used - exiting - 4); old > 0 {
			t.Fatalf("premise: the excuse-based formula must read <= 0 here, got %d", old)
		}
		// and enforcement keeps the overage visible rather than declaring itself done: with a fresh
		// queue view, a pass over this state does NOT leave |stay| over budget
		fx.s.mu.Lock()
		staysOver := fx.s.overBudget(g) > 0
		fx.s.mu.Unlock()
		if !staysOver {
			t.Fatal("the pass must still see an overage: a shared block an exiting thread holds is not leaving")
		}
	})
}

// ---- THE CASCADE, DEFINED WITHOUT A THRESHOLD ----

// TestEveryEnforcementMarkReducesTheOverage is the property a ratio can only approximate: THE
// CASCADE IS MARKS THAT FREE NOTHING.
func TestEveryEnforcementMarkReducesTheOverage(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 1000, func(t *testing.T, fx *engineSet) {
		// fill the engine to the brim with threads whose blocks are all private, then squeeze the
		// budget so enforcement has real work to do
		for i := 0; i < 6; i++ {
			fx.seatOne(t, fmt.Sprintf("b%d", i), 120, time.Duration(5+i)*time.Second)
			fx.s.Chunk(fmt.Sprintf("b%d", i))
		}
		g := fx.g()
		fx.s.mu.Lock()
		g.budgetBlocks = 300 // over budget by 420 blocks with six 120-block threads
		before := fx.s.overBudget(g)
		fx.s.mu.Unlock()
		if before <= 0 {
			t.Fatalf("premise: the engine must start over budget, got %d", before)
		}

		overage := before
		for step := 0; step < 100; step++ {
			fx.s.mu.Lock()
			over := fx.s.overBudget(g)
			if over <= 0 {
				fx.s.mu.Unlock()
				return // done, and every step in between made progress
			}
			// one enforcement pass
			fx.s.enforceBudget(g, fx.clock)
			after := fx.s.overBudget(g)
			fx.s.mu.Unlock()
			if after >= over {
				t.Fatalf("a pass that changed the ledger left the overage at %d (was %d): a mark or eviction MUST strictly reduce it", after, over)
			}
			overage = after
		}
		t.Fatalf("enforcement did not converge in 100 passes (overage %d): a no-progress loop is the cascade", overage)
	})
}
