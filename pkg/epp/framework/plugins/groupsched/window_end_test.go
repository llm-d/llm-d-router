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

// THE END OF A THREAD'S WORK — the window rule applies there too.
//
// A released turn is work: a Pick handed the request to the framework and PreRequest has not
// brought it to a pod yet. thread.busy() is the definition of "this thread still has work", and
// every rule that asks the question at the END of the work has to use it — not the at-a-pod count,
// which is zero for exactly that window.
//
// Three places asked it with the at-a-pod count:
//   - the busy period (a settle counted a period while a second turn was on its way);
//   - the exiting drain at the settle (a marked thread exited out from under its own window);
//   - the reap (a reap stamped lastCompletion while a window was open).
//
// The reference is drainExiting, which has always used busy().

import (
	"testing"
	"time"
)

// releasedSecondTurn releases a second turn for thread "m" and leaves it on its way: the record
// exists, no pod has seen it. It returns the request id.
func releasedSecondTurn(t *testing.T, fx *engineSet, reqID string) string {
	t.Helper()
	h := head("m", time.Second, 100)
	h.ReqID = reqID
	h.Chains = chainOf("m-second", 100)
	if i := fx.pick(h); i < 0 {
		t.Fatalf("the second turn must be released")
	}
	if got := windowOf(t, fx, "m"); got != reqID {
		t.Fatalf("the window must name %q, it names %q", reqID, got)
	}
	return reqID
}

// TestSettleDoesNotCountABusyPeriodOverALiveWindow: the at-pod request settles, but the thread
// still has a turn on its way — the work has not drained, so no busy period has completed.
func TestSettleDoesNotCountABusyPeriodOverALiveWindow(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second) // r-m is at the pod
		second := releasedSecondTurn(t, fx, "r-second")

		fx.s.Settle("r-m", "", fx.clock)

		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		periods, completed := m.busyPeriods, m.lastCompletion
		fx.s.mu.Unlock()
		if periods != 0 {
			t.Fatalf("a settle over a live window counted a busy period (%d): the thread still has %s on its way", periods, second)
		}
		if !completed.IsZero() {
			t.Fatalf("a settle over a live window stamped lastCompletion=%v: the work has not drained", completed)
		}
	})
}

// TestSettleDoesNotExitAThreadOverItsOwnWindow: a marked thread's at-pod request settles; the
// window it holds must keep the seat. Draining it here hands the next PreRequest a thread that is
// no longer seated — the request runs, uncharged and unbound.
func TestSettleDoesNotExitAThreadOverItsOwnWindow(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		second := releasedSecondTurn(t, fx, "r-second")
		fx.s.mu.Lock()
		fx.g().index.MarkExiting("m")
		fx.s.mu.Unlock()

		fx.s.Settle("r-m", "", fx.clock)

		if !fx.s.Seated("m") {
			t.Fatalf("the settle drained an exiting thread that still had %s on its way", second)
		}
		// and the window's request lands where its thread is still seated
		fx.s.Dispatched(second, fx.clock)
		if got := fx.engineOf(t, "m"); got == "" {
			t.Fatalf("the window's own PreRequest landed on an unseated thread")
		}
	})
}

// TestReapDoesNotStampCompletionOverALiveWindow: the reap drops an at-pod record that outlived
// the reap bound (an unobserved termination) — but this thread still has a turn on its way, so its
// work has not drained and lastCompletion must not move.
func TestReapDoesNotStampCompletionOverALiveWindow(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		releasedSecondTurn(t, fx, "r-second")

		// the at-pod request is far older than the reap bound: an unobserved termination
		fx.s.mu.Lock()
		for _, rec := range fx.s.reqs {
			if rec.thread == "m" && !rec.pod.IsZero() {
				rec.pod = fx.clock.Add(-time.Duration(defaultConfig().InflightReapSeconds+60) * time.Second)
			}
		}
		fx.s.mu.Unlock()

		fx.pick() // one pass re-derives and reaps

		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		completed := m.lastCompletion
		fx.s.mu.Unlock()
		if !completed.IsZero() {
			t.Fatalf("a reap stamped lastCompletion=%v while a window was still open: the work has not drained", completed)
		}
		if got := windowOf(t, fx, "m"); got != "r-second" {
			t.Fatalf("the live window must survive the reap, it is %q", got)
		}
	})
}

// TestAbandonedWindowLosesItsPeriod pins the price of the fix, so the panel sees it: if the window
// is ABANDONED instead of reaching a pod, no settle ever counts the period the at-pod request
// carried — the thread's busyPeriods comes out one lower than it would have with the at-a-pod
// count. Rotation does not lose the thread: the next settle-to-idle or the idle rule still
// releases it. Abandon has never booked a period, and booking one there would be a new booking
// site (a policy change, not a fix).
func TestAbandonedWindowLosesItsPeriod(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		second := releasedSecondTurn(t, fx, "r-second")

		fx.s.Settle("r-m", "", fx.clock) // no period: the window is live
		fx.s.Abandon(second, fx.clock)   // the window never reaches a pod

		fx.s.mu.Lock()
		m := fx.g().threads["m"]
		periods := m.busyPeriods
		fx.s.mu.Unlock()
		if periods != 0 {
			t.Fatalf("the abandoned window must not book a period either: busyPeriods=%d", periods)
		}

		// the thread is still releasable: silence (S) reaches it without any settle
		fx.s.mu.Lock()
		m.lastActivity = fx.clock.Add(-time.Duration(fx.s.cfg.IdleSeconds+60) * time.Second)
		fx.s.mu.Unlock()
		reason, victim := fx.s.rotationVictim(fx.g(), fx.clock)
		if victim == nil || victim.id != "m" || reason != "idle" {
			t.Fatalf("an idle thread must still rotate out: reason=%q victim=%v", reason, victim)
		}
	})
}
