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

// ONE PASS, in the three shapes a reviewer asks about first: which head gets served, which head
// ADMISSION prefers when only one seat fits, and what happens when the request lands on an engine
// other than the one it was charged to.

import (
	"testing"
	"time"
)

// epoch is the fixture's own clock: the policy measures everything in the units its passes are
// handed, so a test that mixes time.Now() in is measuring a different timeline.
var epoch = time.Unix(1793000000, 0).UTC()

// newSched is one engine with a known capacity: 0.95 x 1000 = a 950-block seat budget.
func newSched(t *testing.T) *Scheduler {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("", 64, 1000); err != nil {
		t.Fatal(err)
	}
	return s
}

// head is a thread's queued turn: its own blocks, arrived ago.
func head(id string, ago time.Duration, blocks int) Head {
	return Head{Thread: id, ReqID: "r-" + id, Arrival: epoch.Add(-ago), Chains: chain(id, blocks)}
}

// seat seats one thread through the real pass and lands its request on a pod.
func seat(t *testing.T, s *Scheduler, id string, blocks int) {
	t.Helper()
	if i := s.Pass([]Head{head(id, time.Second, blocks)}, epoch); i != 0 {
		t.Fatalf("thread %s must be seated and served, got index %d", id, i)
	}
	s.AtPod("r-"+id, "", epoch)
}

// settle ends the thread's turn: prefill completed (the cold charge is returned) and the request
// finished, so the busy period counts.
func settle(t *testing.T, s *Scheduler, id string) {
	t.Helper()
	s.FirstChunk("r-"+id, epoch)
	s.Settle("r-"+id, "", epoch)
}

// seated is "this thread holds a seat", read straight off the binding the policy keeps.
func seated(s *Scheduler, id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.groupOf(id) != nil
}

// TestMemberServedAndUnknownNeverSeated: a seated thread's queued head is what a pass serves, and
// serving it changes no seat.
func TestMemberServedAndUnknownNeverSeated(t *testing.T) {
	s := newSched(t)
	seat(t, s, "m", 100)
	if i := s.Pass([]Head{head("m", time.Second, 100)}, epoch); i != 0 {
		t.Fatalf("a seated thread's head must dispatch, got %d", i)
	}
	if !seated(s, "m") {
		t.Fatal("the thread must still hold its seat")
	}
	settle(t, s, "m")
	s.mu.Lock()
	records := len(s.groupOf("m").index.seats)
	s.mu.Unlock()
	if records != 1 {
		t.Fatalf("only the seated thread has a seat record, got %d", records)
	}
}

// TestNewcomerBeforeReturnerThenOldest: among the heads WAITING for a seat, a thread's first turn
// goes before a continuation, and among equals the oldest arrival is the head. One 950-block
// budget, 800 already seated, so exactly one of the two 100-block heads fits.
func TestNewcomerBeforeReturnerThenOldest(t *testing.T) {
	s := newSched(t)
	seat(t, s, "filler", 800)
	settle(t, s, "filler")

	ret := head("ret", 30*time.Second, 100)
	ret.Returner = true // thread identity says this is the thread's n-th turn
	s.Pass([]Head{ret, head("new", time.Second, 100)}, epoch)
	if !seated(s, "new") {
		t.Fatal("the newcomer must take the one seat that fits (newcomer-first admission)")
	}
	if seated(s, "ret") {
		t.Fatal("the returning thread must wait behind the newcomer")
	}
}

// TestOldestWaiterIsTheHead: between two heads of the same class, the OLDEST arrival is the head
// and it is the one that gets the seat.
func TestOldestWaiterIsTheHead(t *testing.T) {
	s := newSched(t)
	seat(t, s, "filler", 800)
	settle(t, s, "filler")

	s.Pass([]Head{head("young", time.Second, 100), head("old", 30*time.Second, 100)}, epoch)
	if !seated(s, "old") {
		t.Fatal("the oldest waiting head must be seated first")
	}
	if seated(s, "young") {
		t.Fatal("the younger head must wait")
	}
}

// TestRebindFollowsTheObservedEngine: the framework picks the endpoint, so an observation beats
// the seat-time choice — the thread MOVES, with its blocks, instead of drifting from the engine
// that is really serving it.
func TestRebindFollowsTheObservedEngine(t *testing.T) {
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range []string{"e0", "e1"} {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	if i := s.Pass([]Head{head("m", time.Second, 100)}, epoch); i != 0 {
		t.Fatalf("the thread must be seated, got %d", i)
	}
	s.mu.Lock()
	charged := s.groupOf("m").id
	s.mu.Unlock()
	if charged == "" || charged == "e1" {
		t.Fatalf("fixture: the seat-time choice must be e0, got %q", charged)
	}

	s.AtPod("r-m", "e1", epoch) // the request lands on the OTHER engine

	s.mu.Lock()
	defer s.mu.Unlock()
	if got := s.groupOf("m").id; got != "e1" {
		t.Fatalf("the observed engine must win: bound to %q, want e1", got)
	}
	if got := s.groups[charged].index.ChargedBlocks("m"); got != 0 {
		t.Fatalf("the blocks must leave the engine that is not serving the thread, %d left", got)
	}
	if got := s.groups["e1"].index.ChargedBlocks("m"); got != 100 {
		t.Fatalf("the serving engine must carry the thread's 100 blocks, got %d", got)
	}
	if s.counters.affinityViolations != 1 {
		t.Fatalf("one observation, one violation: %d", s.counters.affinityViolations)
	}
}
