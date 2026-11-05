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

// A SETTLE WITH NO RECORD IS A NON-EVENT, and this file is the two legitimate ways to reach one.
//
// The settle resolves the request's record (threadOfRequest) to find the thread. Two services the
// policy PROVIDES write no record at all, because neither of them is a decision it made:
//   - a thread head the render produced no tokens for is served by the pass-through, last and
//     uncharged (pick.go, "serve the OLDEST anonymous head");
//   - before the measured capacity arrives, Pick serves FCFS from outside the pass entirely
//     (hooks_request.go, "FCFSPick"), so no pass and no release ever saw the request.
//
// Both run, both settle, and neither names a thread: there is nothing to book. The kinds that stay
// are the ones that mean something went wrong — a reaped window, a reaped in-flight turn.

import (
	"testing"
	"time"
)

// TestSettleAfterANoTokenHeadBooksNothing: a head with no rendered tokens passes through unseated
// and uncharged; when its stream ends, the settle finds no record and must leave the ledger alone.
func TestSettleAfterANoTokenHeadBooksNothing(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// no Chains: the render produced no tokens, so the head cannot be seated or charged
		h := Head{Thread: "n", ReqID: "r-n", Arrival: fx.clock}
		if i := fx.pick(h); i != 0 {
			t.Fatalf("a tokenless head must be served by the pass-through, got index %d", i)
		}
		fx.s.mu.Lock()
		_, recorded := fx.s.reqs["r-n"]
		fx.s.mu.Unlock()
		if recorded {
			t.Fatal("the pass-through must not write a record: it is not a decision the policy made")
		}

		fx.s.Settle("r-n", "", fx.clock)

		// the pass-through's own ledger entry (a head with no tokens is a lost render) is legitimate:
		// what must NOT appear is anything the settle added
		anomalies := fx.s.Snapshot(fx.clock).Anomalies
		if len(anomalies) != 1 || anomalies[anomalyNoTokens] != 1 {
			t.Fatalf("a settle over a request the policy never took on was booked as an anomaly: %v", anomalies)
		}
	})
}

// TestSettleAfterAPreCapacityDispatchBooksNothing: before capacity is measured every head is served
// FCFS from outside the pass, so the first settle of a fresh process is a settle with no record.
func TestSettleAfterAPreCapacityDispatchBooksNothing(t *testing.T) {
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(1793000000, 0).UTC()
	if s.HasCapacity() {
		t.Fatal("the fixture must be pre-capacity: that is the window under test")
	}
	heads := []Head{
		head("late", 2*time.Second, 100),
		head("early", 5*time.Second, 100),
	}
	i := FCFSPick(heads) // what Pick does before the measured capacity arrives
	if i != 1 {
		t.Fatalf("the pre-capacity order is FCFS, wanted the older head, got index %d", i)
	}
	s.mu.Lock()
	_, recorded := s.reqs[heads[i].ReqID]
	s.mu.Unlock()
	if recorded {
		t.Fatal("a dispatch from outside the pass must not write a record")
	}

	s.Settle(heads[i].ReqID, "", now)

	anomalies := s.Snapshot(now).Anomalies
	if len(anomalies) != 0 {
		t.Fatalf("the first settle of a pre-capacity process was booked as an anomaly: %v", anomalies)
	}
}

// TestASettleWithNoRecordLeavesTheThreadsAlone pins the rest of the contract: the non-event is
// silent but not a no-op by accident — a request that IS tracked still settles normally in the same
// process, so the early return cannot be swallowing real work.
func TestASettleWithNoRecordLeavesTheThreadsAlone(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		fx.s.Settle("nothing-tracks-this", "", fx.clock)
		m, _ := fx.thread(t, "m")
		fx.s.mu.Lock()
		inflight := m.inflight
		periods := m.busyPeriods
		fx.s.mu.Unlock()
		if inflight != 1 || periods != 0 {
			t.Fatalf("a stray settle touched a live thread: inflight=%d periods=%d", inflight, periods)
		}
		// and the thread's own settle still closes its period
		fx.s.Settle("r-m", "", fx.clock)
		m, _ = fx.thread(t, "m")
		fx.s.mu.Lock()
		periods = m.busyPeriods
		fx.s.mu.Unlock()
		if periods != 1 {
			t.Fatalf("the tracked settle must still count its busy period, got %d", periods)
		}
	})
}
