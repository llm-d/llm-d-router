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

// WHO LEAVES, and when: the tenure rule (a thread that completed its quantum) and the budget
// valve (memory pressure picks a victim), plus the one thread that is never a victim.

import (
	"fmt"
	"testing"
	"time"
)

// busyPeriods is the thread's completed-tenure counter, read under the lock.
func busyPeriods(s *Scheduler, id string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.groupOf(id).threads[id].busyPeriods
}

// TestQuantumIsTheTenureRule: Q is the tenure, counted in COMPLETED busy periods — a thread that
// has finished Q turns is rotation-eligible, one mid-quantum is not.
func TestQuantumIsTheTenureRule(t *testing.T) {
	for _, q := range []int{4, 8} {
		t.Run(fmt.Sprintf("Q=%d", q), func(t *testing.T) {
			cfg := defaultConfig()
			cfg.Quantum = q
			s, err := NewScheduler(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := s.SetNumBlocksFor("", 64, 1000); err != nil {
				t.Fatal(err)
			}
			seat(t, s, "m", 400)
			settle(t, s, "m")
			if bp := busyPeriods(s, "m"); bp != 1 {
				t.Fatalf("one drained turn is one busy period, got %d", bp)
			}
			s.mu.Lock()
			_, eligible := s.eligible(s.groupOf("m").threads["m"], epoch)
			s.mu.Unlock()
			if eligible {
				t.Fatalf("Q=%d: a thread with one busy period must not be rotation-eligible", q)
			}
			for i := 1; i < q; i++ { // the rest of the tenure, without waiting for turns
				s.mu.Lock()
				s.groupOf("m").threads["m"].busyPeriods++
				s.mu.Unlock()
			}
			s.mu.Lock()
			reason, eligible := s.eligible(s.groupOf("m").threads["m"], epoch)
			s.mu.Unlock()
			if !eligible || reason != "quantum-complete" {
				t.Fatalf("Q=%d: %d busy periods must be eligible (quantum-complete), got %q %v", q, q, reason, eligible)
			}
		})
	}
}

// TestInFlightNeverEligible: rotation never takes a seat back from running work.
func TestInFlightNeverEligible(t *testing.T) {
	s := newSched(t)
	seat(t, s, "m", 100) // its request is in flight
	s.mu.Lock()
	m := s.groupOf("m").threads["m"]
	m.busyPeriods = 100 // way past any Q
	_, eligible := s.eligible(m, epoch)
	s.mu.Unlock()
	if m.inflight == 0 {
		t.Fatal("fixture: the thread's request must be in flight")
	}
	if eligible {
		t.Fatal("a thread with a request in flight must never be rotation-eligible")
	}
}

// TestVictimOrderPrefersTheDoneMember: under memory pressure the victim is the thread with NO
// queued turn — never the one that is waiting to run.
func TestVictimOrderPrefersTheDoneMember(t *testing.T) {
	s := newSched(t)
	seat(t, s, "done", 450) // 450 of the 950-block budget
	settle(t, s, "done")    // its turn is over: nothing queued
	seat(t, s, "busy", 450) // 900 blocks: inside the budget
	settle(t, s, "busy")

	// `busy` wants to run again, and its next prompt is 200 blocks longer: 450 + 650 > 950, so the
	// budget valve is the only way this head ever runs.
	s.mu.Lock()
	m := s.groupOf("busy").threads["busy"]
	m.waitingSince = epoch.Add(-time.Second)
	m.lastActivity = epoch
	s.mu.Unlock()
	if i := s.Pass([]Head{head("busy", time.Second, 650)}, epoch); i != 0 {
		t.Fatalf("the thread's grown head must dispatch, got %d", i)
	}
	s.AtPod("r-busy", "", epoch) // the growth lands on the engine
	s.Pass(nil, epoch)           // an empty-queue pass: the valve runs here

	if seated(s, "done") {
		t.Fatal("the DONE thread (no queued turn) must be the budget victim")
	}
	if !seated(s, "busy") {
		t.Fatal("the thread with a queued turn must survive the valve")
	}
	if snap := s.Snapshot(epoch); snap.Exits["budget-noqueue"] == 0 && snap.Exits["budget-idle"] == 0 {
		t.Fatalf("the exit must be a budget-class exit, got %v", snap.Exits)
	}
}
