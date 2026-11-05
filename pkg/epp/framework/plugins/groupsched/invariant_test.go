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

// THE RANDOMIZED INVARIANT TEST.

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"testing"
	"time"
)

// invEvent is one exogenous event of the stream.
type invEvent struct {
	kind   int
	thread string
	blocks int
	engine string
}

const (
	evArrive = iota
	evPick
	evSettle
	evAbort
	evEngineGone
	evTick
)

type invStream struct {
	events  []invEvent
	threads []string
}

// newInvStream builds a seeded event stream: arrivals grow a thread's prompt (a turn), and
// every collision of lifetimes is possible. capacity is the per-engine measured block count,
// so the stream can also produce an OVERSIZE head (a prompt whose exact cost exceeds the seat
// budget on every engine) — RULE O's trigger, which has to be exercised randomly, not only by
// the hand-written cases.
func newInvStream(seed int64, n, capacity int) *invStream {
	rng := rand.New(rand.NewSource(seed))
	s := &invStream{}
	for i := 0; i < 12; i++ {
		s.threads = append(s.threads, fmt.Sprintf("b%02d", i))
	}
	sizes := map[string]int{}
	for i := 0; i < n; i++ {
		b := s.threads[rng.Intn(len(s.threads))]
		switch rng.Intn(20) {
		case 0, 1, 2, 3, 4: // an arrival: a new turn, a bigger prompt
			if rng.Intn(6) == 0 {
				// an OVERSIZE prompt: past the whole seat budget (0.95 of capacity), so it cannot fit an
				// occupied engine and only RULE O can admit it anywhere
				sizes[b] = capacity + 50 + rng.Intn(capacity/2+1)
			} else {
				sizes[b] += 20 + rng.Intn(120)
			}
			s.events = append(s.events, invEvent{kind: evArrive, thread: b, blocks: sizes[b]})
		case 5, 6, 7, 8, 9, 10, 11, 12: // a decision pass
			s.events = append(s.events, invEvent{kind: evPick})
		case 13, 14, 15: // a natural settle
			s.events = append(s.events, invEvent{kind: evSettle, thread: b})
		case 16, 17: // a termination nobody asked for (a cancel / an error before the first chunk)
			s.events = append(s.events, invEvent{kind: evAbort, thread: b})
		case 18: // an engine disappears (a node rotation)
			s.events = append(s.events, invEvent{kind: evEngineGone})
		case 19: // the ~1s ticker
			s.events = append(s.events, invEvent{kind: evTick})
		}
	}
	return s
}

// invRun drives one stream against a fresh scheduler and returns the decision trace.
func invRun(t *testing.T, seed int64, engines, capacity int, events int) string {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	var engineIDs []string
	for i := 0; i < engines; i++ {
		id := fmt.Sprintf("e%d", i)
		engineIDs = append(engineIDs, id)
		if err := s.SetNumBlocksFor(id, 64, capacity); err != nil {
			t.Fatal(err)
		}
	}
	clock := time.Unix(1793000000, 0).UTC()
	stream := newInvStream(seed, events, capacity)

	// the fixture's own model of what is queued and what is running
	queued := map[string]Head{}   // thread -> its pending head
	flying := map[string]string{} // thread -> the dispatched request id
	growing := map[string]int{}   // thread -> the prompt size of its last arrival
	var trace strings.Builder

	settleOne := func(thread, reqID string, natural bool) {
		if _, ok := flying[thread]; !ok {
			return // nothing was running: a stray termination is tolerated, never counted
		}
		s.ChunkAt(thread, clock)
		s.Settle(reqID, "", clock)
		delete(flying, thread)
		trace.WriteString("settle " + thread + "\n")
	}

	for i, ev := range stream.events {
		switch ev.kind {
		case evArrive:
			growing[ev.thread] = ev.blocks
			// the head carries the thread's WHOLE prompt: a turn's chain, not a delta
			queued[ev.thread] = Head{
				Thread:  ev.thread,
				ReqID:   fmt.Sprintf("%s-r%d", ev.thread, ev.blocks),
				Arrival: clock,
				Chains:  chainOf(ev.thread, ev.blocks),
			}
		case evPick:
			hs := make([]Head, 0, len(queued))
			for _, b := range sortedKeysStr(queued) {
				hs = append(hs, queued[b])
			}
			idx := s.Pass(hs, clock)
			if idx < 0 {
				trace.WriteString("pick -1\n")
				break
			}
			h := hs[idx]
			engine := s.engineOfDispatch(h.ReqID)
			trace.WriteString("pick " + h.Thread + "\n")
			if h.Thread == "" {
				break
			}
			if !s.Seated(h.Thread) {
				t.Fatalf("event %d: %s was dispatched without a seat", i, h.Thread)
			}
			// the request left the queue and is NOT yet in flight: PreRequest is the director's own
			// call, on another goroutine, and the ~1 kHz Pick runs in between (the released window).
			delete(queued, h.Thread)
			s.Pass(nil, clock)
			trace.WriteString("dispatch " + h.Thread + " " + engine + "\n")
			s.releaseThread(h.Thread, h.ReqID, h.Chains, clock)
			s.DispatchedOn(h.ReqID, engine, clock)
			flying[h.Thread] = h.ReqID
		case evSettle:
			if rid, ok := flying[ev.thread]; ok {
				settleOne(ev.thread, rid, true)
			}
		case evAbort:
			if rid, ok := flying[ev.thread]; ok {
				settleOne(ev.thread, rid, false)
			}
		case evEngineGone:
			if len(engineIDs) == 0 {
				break
			}
			// deterministic choice: the last engine id
			gone := engineIDs[len(engineIDs)-1]
			n := s.RemoveEngine(gone, clock)
			trace.WriteString(fmt.Sprintf("engine-gone %s %d\n", gone, n))
			engineIDs = engineIDs[:len(engineIDs)-1]
			for b := range flying {
				if !s.Seated(b) {
					delete(flying, b)
				}
			}
		case evTick:
			clock = clock.Add(time.Second)
			s.Pass(nil, clock)
			// a termination nobody observed: the reap drops the id, and the drop IS a drain
			trace.WriteString("tick\n")
		}
		clock = clock.Add(time.Millisecond)
		checkInvariants(t, s, i, capacity)
	}
	// the whole-policy view has to be self-consistent at the end too: the top-level used side is
	// the sum of the per-engine ones, and the seats are the bindings
	snap := s.Snapshot(s.now)
	sumNE, sumSeats := 0, 0
	for id, es := range snap.Engines {
		sumNE += es.NEBlocks
		sumSeats += es.Seats
		if got, want := es.Seats, boundTo(s, id); got != want {
			t.Fatalf("%s: %d seats but %d bindings", id, got, want)
		}
	}
	if snap.NEBlocks != sumNE || snap.Seats != sumSeats {
		t.Fatalf("the snapshot's totals disagree with its own per-engine section: used %d/%d seats %d/%d",
			snap.NEBlocks, sumNE, snap.Seats, sumSeats)
	}
	return trace.String()
}

// checkInvariants asserts every stated invariant against the live state.
func checkInvariants(t *testing.T, s *Scheduler, event int, capacity int) {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("event %d: the index panicked: %v", event, r)
		}
	}()
	seats := map[string]string{}
	for _, g := range s.activeGroups() {
		g.index.checkInvariant() // I1/I7: both refcounts == the recomputed union
		if g.numBlocks <= 0 && len(g.threads) > 0 {
			t.Fatalf("event %d: %d threads seated on %s, whose capacity is unknown (I6)", event, len(g.threads), g.id)
		}
		if cap := g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize); cap > 0 && g.coldInFlight > cap {
			t.Fatalf("event %d: %s cold charge %d exceeds its cap %d (B2)", event, g.id, g.coldInFlight, cap)
		}
		// R4's postcondition: inside the budget, or nothing left to evict
		if over := s.overBudget(g); over > 0 {
			_, victim := s.rotationVictim(g, s.now)
			if victim != nil && !s.soleOversize(g, victim) {
				t.Fatalf("event %d: %s is %d blocks over its budget with an eligible victim %s (R4)", event, g.id, over, victim.id)
			}
			// no victim — or the only candidate is RULE O's sole oversize resident, which pressure must
			// leave alone: the pass is DONE either way
		}
		for id := range g.threads {
			if prev, dup := seats[id]; dup {
				t.Fatalf("event %d: %s is seated on %s AND %s (B10)", event, id, prev, g.id)
			}
			seats[id] = g.id
			if got := s.bindings[id]; got != g.id {
				t.Fatalf("event %d: %s is seated on %s but its binding says %q (B10)", event, id, g.id, got)
			}
		}
	}
	// the shared queue holds exactly the unseated threads, and no seated one
	for id := range s.waiters {
		if _, seated := seats[id]; seated {
			t.Fatalf("event %d: %s is both seated and waiting", event, id)
		}
	}
	// and every binding names an engine that is actually holding the thread
	for id, engine := range s.bindings {
		if got, ok := seats[id]; ok && got != engine {
			t.Fatalf("event %d: binding %s->%s but the seat is on %s", event, id, engine, got)
		}
	}
}

// sortedKeysStr is the deterministic iteration order for the fixture's head set.
func sortedKeysStr(m map[string]Head) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// TestInvariantsHoldUnderRandomEvents is the gate: two engine counts, three seeds, every
// invariant after every event.
func TestInvariantsHoldUnderRandomEvents(t *testing.T) {
	for _, engines := range []int{1, 2} {
		for _, seed := range []int64{7, 11, 23} {
			engines, seed := engines, seed
			t.Run(fmt.Sprintf("N=%d/seed=%d", engines, seed), func(t *testing.T) {
				if tr := invRun(t, seed, engines, 1000, 4000); tr == "" {
					t.Fatal("the stream produced no decisions")
				}
			})
		}
	}
}

// TestDecisionStreamIsDeterministic: the same seed produces the same DECISIONS, twice.
func TestDecisionStreamIsDeterministic(t *testing.T) {
	for _, engines := range []int{1, 2} {
		engines := engines
		t.Run(fmt.Sprintf("N=%d", engines), func(t *testing.T) {
			first := invRun(t, 31, engines, 1000, 3000)
			for i := 2; i <= 3; i++ {
				if again := invRun(t, 31, engines, 1000, 3000); again != first {
					a, b := strings.Split(again, "\n"), strings.Split(first, "\n")
					for k := 0; k < len(a) || k < len(b); k++ {
						ga, wa := "", ""
						if k < len(a) {
							ga = a[k]
						}
						if k < len(b) {
							wa = b[k]
						}
						if ga != wa {
							t.Fatalf("run %d diverges at line %d: %q vs %q", i, k+1, ga, wa)
						}
					}
					t.Fatal("the decision stream is not deterministic")
				}
			}
		})
	}
}

// TestStrictHeadOfLineUnderI2: when the head fits nowhere and has no victim, nothing behind it
// is seated — the head keeps the queue.
func TestStrictHeadOfLineUnderI2(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		w := newWorkload(t, fx)
		// fill every engine so that a big head cannot fit and nothing is evictable yet
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
		for round := 0; round < 400; round++ {
			hs, ids := w.heads(fx.clock)
			if i := fx.pick(hs...); i >= 0 {
				w.dispatchTurn(ids[i])
			}
			w.step(roundStep)
			done := true
			for _, id := range residents {
				if !w.everSeated(id) || w.served[id] == 0 {
					done = false
				}
			}
			if done {
				break
			}
		}
		// EVERY RESIDENT GOES IN FLIGHT and nothing settles again: a thread with a request at a pod
		// is never evictable, and an engine whose residents are all running is not engine-idle — the
		// one state in which a head that fits nowhere truly has no victim.
		for _, id := range residents {
			w.arrive(id, fx.clock.Add(-time.Second))
		}
		for round := 0; round < 200; round++ {
			hs, ids := w.heads(fx.clock)
			i := fx.pick(hs...)
			if i < 0 {
				break
			}
			w.dispatchTurn(ids[i])
		}
		for _, id := range residents {
			if _, flying := w.flying[id]; !flying {
				t.Fatalf("fixture: %s must be in flight, so nothing is evictable", id)
			}
		}
		// now a BIG head arrives (three times any seat budget, so it fits no engine even empty) and
		// a small one behind it: no victim exists anywhere, so the head keeps the queue and the
		// small one must not be seated
		w.setPromptBytes("big", maxBudget(fx)*3*256)
		w.arrive("big", fx.clock.Add(-time.Minute))
		w.arrive("small", fx.clock.Add(-time.Second))
		w.heads(fx.clock) // materialise both
		hs, ids := w.heads(fx.clock)
		if i := fx.pick(hs...); i >= 0 {
			t.Fatalf("the pass served %s while the head fits nowhere and has no victim", ids[i])
		}
		if w.everSeated("small") {
			t.Fatal("a head behind a blocked head was seated: the queue is not strict head-of-line (I2)")
		}
		if w.everSeated("big") {
			t.Fatal("a head bigger than every budget was seated")
		}
	})
}

// maxBudget is the fixture's per-engine seat budget in blocks.
func maxBudget(fx *engineSet) int {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return fx.s.groups[fx.ids[0]].budgetBlocks
}
