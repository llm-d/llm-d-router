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

// THE DERIVED COLD-ENTRY CAP (constants table, README §3).

import (
	"fmt"
	"testing"
	"time"
)

func TestColdCapIsTheBudgetsHeadroomPerEngine(t *testing.T) {
	type comb struct {
		numBlocks int
		fraction  float64
		blockSize int
	}
	combs := []comb{
		{40183, 0.95, 64}, // the campaign engine (2,571,712-token pool)
		{40183, 0.90, 64}, // the fallback arm the root cause's option D proposed
		{40183, 1.0, 64},  // no headroom at all: the cap is 0
		{1000, 0.95, 64},  // the unit-test fixture
		{2072, 0.95, 32},  // a different block size
		{7, 0.95, 64},     // tiny: floor() must not round the headroom up
		{38173, 0.9501, 16},
	}
	for _, c := range combs {
		g := &engineGroup{numBlocks: c.numBlocks}
		g.budgetBlocks = int(c.fraction * float64(c.numBlocks))
		capTokens := g.coldCapTokens(c.fraction, c.blockSize)
		headroomTokens := (1 - c.fraction) * float64(c.numBlocks) * float64(c.blockSize)
		if float64(capTokens) > headroomTokens {
			t.Fatalf("%+v: cap %d exceeds the budget's headroom %.1f tokens — the valve could outspend the margin",
				c, capTokens, headroomTokens)
		}
		if want := int64((1 - c.fraction) * float64(c.numBlocks)); capTokens != want*int64(c.blockSize) {
			t.Fatalf("%+v: cap %d want %d blocks x %d", c, capTokens, want, c.blockSize)
		}
		if capTokens < 0 {
			t.Fatalf("%+v: negative cap %d", c, capTokens)
		}
	}
	// the derived value at the campaign shape is the constant it replaces, to within one block:
	// 2009 blocks x 64 = 128,576 tokens (the knob said 131,072).
	g := &engineGroup{numBlocks: 40183, budgetBlocks: 38173} // int(0.95 x 40183)
	if got := g.coldCapTokens(0.95, 64); got != 128576 {
		t.Fatalf("campaign-shape cap: %d want 128576 (2009 blocks x 64)", got)
	}
}

// The charge is CAPPED at the cap, so the invariant holds even for an entry bigger than the
// whole headroom — which is exactly when the old rule ("one cold entry is always allowed", as
// an EXCEPTION to the cap) let the charge run 1.5x over.
func TestColdChargeIsCappedAtTheCap(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 4000, func(t *testing.T, fx *engineSet) {
		g := fx.g()
		capTokens := g.coldCapTokens(fx.s.cfg.GroupBudgetFraction, fx.s.blockSize)
		if capTokens != 200*64 { // 5 % of 4,000 blocks = 200 blocks
			t.Fatalf("cap %d want %d", capTokens, 200*64)
		}
		big := 1000 // 5x the headroom, well inside the 3,800-block budget
		if i := fx.pick(head("huge", time.Second, big)); i != 0 {
			t.Fatal("an empty valve must admit one entry of any size (R11)")
		}
		dispatch(t, fx.s, "huge")
		if got := g.coldInFlight; got != capTokens {
			t.Fatalf("charge %d want the cap %d (capped, not the entry's 1,000 blocks)", got, capTokens)
		}
		// the exact-count refinement at PreRequest must not push it over either
		snap := fx.s.Snapshot(fx.clock)
		for _, es := range snap.Engines {
			if es.ColdInFlight > es.ColdCapTokens {
				t.Fatalf("engine %s: cold charge %d exceeds its cap %d", es.ID, es.ColdInFlight, es.ColdCapTokens)
			}
		}
		// ... and the second entry waits until the first prefill lands.
		i := fx.pick(head("next", time.Second, 10))
		if len(fx.ids) == 1 && i >= 0 {
			t.Fatalf("a second cold entry must wait while the valve is full, got %d", i)
		}
		for _, es := range fx.s.Snapshot(fx.clock).Engines {
			if es.ColdInFlight > es.ColdCapTokens {
				t.Fatalf("engine %s: cold %d > cap %d", es.ID, es.ColdInFlight, es.ColdCapTokens)
			}
		}
		fx.s.Chunk("huge")
		if len(fx.ids) == 1 {
			if i := fx.pick(head("after", time.Second, 10)); i != 0 {
				t.Fatal("the valve must open when the prefill completes")
			}
		}
	})
}

// A cold charge is never released by anything but its own release points, and the per-engine
// sum is what the invariant is about: two engines at N=2 each carry their own cap, never a
// shared one.
func TestColdCapIsPerEngine(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 4000, func(t *testing.T, fx *engineSet) {
		for i := range fx.ids {
			fx.seatOne(t, fmt.Sprintf("c%d", i), 500, time.Duration(3+i)*time.Second)
		}
		snap := fx.s.Snapshot(fx.clock)
		if len(snap.Engines) != len(fx.ids) {
			t.Fatalf("engines in the snapshot: %d want %d", len(snap.Engines), len(fx.ids))
		}
		for _, es := range snap.Engines {
			if es.ColdCapTokens <= 0 {
				t.Fatalf("engine %s has no derived cap", es.ID)
			}
			if es.ColdInFlight > es.ColdCapTokens {
				t.Fatalf("engine %s: cold %d > cap %d", es.ID, es.ColdInFlight, es.ColdCapTokens)
			}
		}
	})
}
