/*
Copyright 2026 The Kubernetes Authors.

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

package internal

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	testclock "k8s.io/utils/clock/testing"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// fakeInFlightEvictor is a configurable InFlightEvictor for controller tests.
type fakeInFlightEvictor struct {
	mu             sync.Mutex
	inFlight       int
	evictable      int
	victimPriority int
	hasVictim      bool
	evictErr       error

	evictCalls []int // n per EvictN call
	boundCalls []int // priorityBound per EvictN call
	nextID     int
	listener   func(requestID string)
}

func (f *fakeInFlightEvictor) EvictN(_ context.Context, n int, priorityBound int) ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.evictCalls = append(f.evictCalls, n)
	f.boundCalls = append(f.boundCalls, priorityBound)
	if f.evictErr != nil {
		return nil, f.evictErr
	}
	if n > f.evictable {
		n = f.evictable
	}
	ids := make([]string, 0, n)
	for range n {
		ids = append(ids, fmt.Sprintf("req-%d", f.nextID))
		f.nextID++
	}
	f.evictable -= n
	f.inFlight -= n
	return ids, nil
}

func (f *fakeInFlightEvictor) Stats() (int, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.inFlight, f.evictable
}

func (f *fakeInFlightEvictor) PeekVictimPriority() (int, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.victimPriority, f.hasVictim
}

func (f *fakeInFlightEvictor) SetEvictionTerminatedListener(listener func(requestID string)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.listener = listener
}

func (f *fakeInFlightEvictor) totalEvictCalls() []int {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.evictCalls) == 0 {
		return nil
	}
	out := make([]int, len(f.evictCalls))
	copy(out, f.evictCalls)
	return out
}

func newTestReclamationController(
	evictor *fakeInFlightEvictor,
	clk *testclock.FakeClock,
	cfg ReclamationConfig,
) *ReclamationController {
	return NewReclamationController(cfg, evictor, clk, logr.Discard(), "test-pool")
}

var testReclamationConfig = ReclamationConfig{
	MaxRevocationsPerDecision: 2,
	ConfirmationGrace:         100 * time.Millisecond,
	ConfirmationTimeout:       5 * time.Second,
}

func TestReclamationController_Sizing(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name       string
		inFlight   int
		evictable  int
		saturation float64
		ceiling    float64
		expectedN  []int // expected EvictN call args; empty means no call
	}{
		{
			// credit = 1.2/6 = 0.2; deficit = 0.2; n = 1.
			name:     "SmallDeficit_EvictsOne",
			inFlight: 6, evictable: 6, saturation: 1.2, ceiling: 1.0,
			expectedN: []int{1},
		},
		{
			// credit = 2.0/4 = 0.5; deficit = 1.0; n = 2 = cap.
			name:     "DeepOverload_CappedByMaxPerDecision",
			inFlight: 4, evictable: 4, saturation: 2.0, ceiling: 1.0,
			expectedN: []int{2},
		},
		{
			// n would be 2 but only 1 lease is evictable.
			name:     "CappedByEvictable",
			inFlight: 4, evictable: 1, saturation: 2.0, ceiling: 1.0,
			expectedN: []int{1},
		},
		{
			// Saturation below the ceiling: no deficit, no eviction.
			name:     "NoDeficit_NoEviction",
			inFlight: 6, evictable: 6, saturation: 0.9, ceiling: 1.0,
			expectedN: nil,
		},
		{
			// The dispatch gate blocks at saturation == ceiling, but the deficit is exactly zero.
			// The controller must still issue one revocation or the band deadlocks until churn.
			name:     "ExactBoundary_EvictsOne",
			inFlight: 10, evictable: 10, saturation: 1.0, ceiling: 1.0,
			expectedN: []int{1},
		},
		{
			// No tracked leases: credit is undefined and there are no victims.
			name:     "ZeroLeases_NoEviction",
			inFlight: 0, evictable: 0, saturation: 1.5, ceiling: 1.0,
			expectedN: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			clk := testclock.NewFakeClock(time.Now())
			evictor := &fakeInFlightEvictor{inFlight: tc.inFlight, evictable: tc.evictable}
			rc := newTestReclamationController(evictor, clk, testReclamationConfig)

			rc.Reclaim(context.Background(), tc.saturation, tc.ceiling, 0)

			assert.Equal(t, tc.expectedN, evictor.totalEvictCalls(), "EvictN calls should match expected sizing")
		})
	}
}

func TestReclamationController_GateLifecycle(t *testing.T) {
	t.Parallel()
	clk := testclock.NewFakeClock(time.Now())
	evictor := &fakeInFlightEvictor{inFlight: 4, evictable: 4}
	rc := newTestReclamationController(evictor, clk, testReclamationConfig)

	require.True(t, rc.GateOpen(clk.Now()), "gate should be open initially")

	// Issue a decision: deficit 1.0, credit 0.5, n = 2.
	rc.Reclaim(context.Background(), 2.0, 1.0, 0)
	require.Equal(t, []int{2}, evictor.totalEvictCalls(), "decision should issue revocations")
	assert.False(t, rc.GateOpen(clk.Now()), "gate should be closed while revocations are outstanding")

	// One confirmation is not enough.
	rc.Confirm("req-0")
	assert.False(t, rc.GateOpen(clk.Now()), "gate should stay closed until all revocations confirm")

	// All confirmed, but grace has not elapsed.
	rc.Confirm("req-1")
	assert.False(t, rc.GateOpen(clk.Now()), "gate should stay closed during the grace window")

	// Grace elapsed: gate reopens.
	clk.Step(testReclamationConfig.ConfirmationGrace + time.Millisecond)
	assert.True(t, rc.GateOpen(clk.Now()), "gate should reopen after all confirmations plus grace")
}

func TestReclamationController_ConfirmUnknownID_Ignored(t *testing.T) {
	t.Parallel()
	clk := testclock.NewFakeClock(time.Now())
	evictor := &fakeInFlightEvictor{inFlight: 4, evictable: 4}
	rc := newTestReclamationController(evictor, clk, testReclamationConfig)

	require.NotPanics(t, func() { rc.Confirm("req-unknown") })
	assert.True(t, rc.GateOpen(clk.Now()), "unknown confirmations must not perturb the gate")
}

func TestReclamationController_ConfirmationTimeout_ReopensGate(t *testing.T) {
	t.Parallel()
	clk := testclock.NewFakeClock(time.Now())
	evictor := &fakeInFlightEvictor{inFlight: 4, evictable: 4}
	rc := newTestReclamationController(evictor, clk, testReclamationConfig)

	rc.Reclaim(context.Background(), 2.0, 1.0, 0)
	require.False(t, rc.GateOpen(clk.Now()), "gate closed while outstanding")

	// Confirmation never arrives; the timeout retires the revocations.
	clk.Step(testReclamationConfig.ConfirmationTimeout + time.Millisecond)
	require.False(t, rc.GateOpen(clk.Now()),
		"first check after timeout retires entries but grace restarts from retirement")
	clk.Step(testReclamationConfig.ConfirmationGrace + time.Millisecond)
	assert.True(t, rc.GateOpen(clk.Now()), "gate should reopen after timeout retirement plus grace")

	// A late confirmation for a timed-out revocation is a no-op: no second outcome, and the gate
	// stays open.
	require.NotPanics(t, func() { rc.Confirm("req-0") })
	assert.True(t, rc.GateOpen(clk.Now()), "late confirmation must not re-close the gate")
}

func TestReclamationController_PendingDebit_SuppressesRepeatDecision(t *testing.T) {
	t.Parallel()
	clk := testclock.NewFakeClock(time.Now())
	evictor := &fakeInFlightEvictor{inFlight: 4, evictable: 4}
	rc := newTestReclamationController(evictor, clk, testReclamationConfig)

	rc.Reclaim(context.Background(), 2.0, 1.0, 0)
	require.Equal(t, []int{2}, evictor.totalEvictCalls())

	// Even if a caller bypassed the gate, the pending debit covers the deficit and no further
	// revocations are issued for the same signal.
	rc.Reclaim(context.Background(), 2.0, 1.0, 0)
	assert.Equal(t, []int{2}, evictor.totalEvictCalls(),
		"a second decision against the same stale signal must not issue more revocations")
}

// --- Integration: dispatchCycle -> maybeReclaim ---

// withReclamation attaches a reclamation controller backed by the fake evictor to the harness
// processor.
func withReclamation(h *testHarness, evictor *fakeInFlightEvictor, cfg ReclamationConfig) {
	h.processor.reclamation = NewReclamationController(cfg, evictor, h.clock, logr.Discard(), "test-pool")
}

func TestDispatchCycle_HoLBlock_TriggersReclamation(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t, testCleanupTick)
	evictor := &fakeInFlightEvictor{inFlight: 6, evictable: 3, victimPriority: -1, hasVictim: true}
	withReclamation(h, evictor, testReclamationConfig)

	q := h.addQueue(testFlow) // Priority 10.
	require.NoError(t, q.Add(h.newTestItem("req-blocked", testFlow, testTTL)))

	h.saturationDetector.SaturationFunc = func(context.Context, []fwkdl.Endpoint) float64 { return 1.2 }

	dispatched := h.processor.dispatchCycle(h.ctx)

	assert.False(t, dispatched, "saturated cycle must not dispatch")
	// credit = 1.2/6 = 0.2; deficit = 0.2; n = 1.
	assert.Equal(t, []int{1}, evictor.totalEvictCalls(), "HoL blocking with eligible demand should revoke")
}

func TestDispatchCycle_NoQueuedDemand_NoReclamation(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t, testCleanupTick)
	evictor := &fakeInFlightEvictor{inFlight: 6, evictable: 3, victimPriority: -1, hasVictim: true}
	withReclamation(h, evictor, testReclamationConfig)

	h.addQueue(testFlow) // Registered band, but its queue is empty.

	h.saturationDetector.SaturationFunc = func(context.Context, []fwkdl.Endpoint) float64 { return 1.2 }

	h.processor.dispatchCycle(h.ctx)
	assert.Empty(t, evictor.totalEvictCalls(), "no queued demand means nothing to reclaim for")
}

func TestDispatchCycle_VictimNotStrictlyLower_NoReclamation(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t, testCleanupTick)
	// Victim priority equals the blocked band's priority: same-band churn is forbidden.
	evictor := &fakeInFlightEvictor{inFlight: 6, evictable: 3, victimPriority: testFlow.Priority, hasVictim: true}
	withReclamation(h, evictor, testReclamationConfig)

	q := h.addQueue(testFlow)
	require.NoError(t, q.Add(h.newTestItem("req-blocked", testFlow, testTTL)))

	h.saturationDetector.SaturationFunc = func(context.Context, []fwkdl.Endpoint) float64 { return 1.2 }

	h.processor.dispatchCycle(h.ctx)
	assert.Empty(t, evictor.totalEvictCalls(), "demand must dominate the victim priority strictly")
}

func TestDispatchCycle_NoVictims_NoReclamation(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t, testCleanupTick)
	evictor := &fakeInFlightEvictor{inFlight: 0, evictable: 0, hasVictim: false}
	withReclamation(h, evictor, testReclamationConfig)

	q := h.addQueue(testFlow)
	require.NoError(t, q.Add(h.newTestItem("req-blocked", testFlow, testTTL)))

	h.saturationDetector.SaturationFunc = func(context.Context, []fwkdl.Endpoint) float64 { return 1.2 }

	h.processor.dispatchCycle(h.ctx)
	assert.Empty(t, evictor.totalEvictCalls(), "no evictable leases means no decision")
}

func TestDispatchCycle_GateClosed_SkipsDemandScan(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t, testCleanupTick)
	evictor := &fakeInFlightEvictor{inFlight: 6, evictable: 6, victimPriority: -1, hasVictim: true}
	withReclamation(h, evictor, testReclamationConfig)

	q := h.addQueue(testFlow)
	require.NoError(t, q.Add(h.newTestItem("req-blocked", testFlow, testTTL)))

	h.saturationDetector.SaturationFunc = func(context.Context, []fwkdl.Endpoint) float64 { return 1.2 }

	// First cycle issues; the gate then closes until confirmation.
	h.processor.dispatchCycle(h.ctx)
	require.Equal(t, []int{1}, evictor.totalEvictCalls())

	// Repeated cycles against the same stale gauge must not issue more revocations.
	for range 10 {
		h.processor.dispatchCycle(h.ctx)
	}
	assert.Equal(t, []int{1}, evictor.totalEvictCalls(), "gate must suppress repeat decisions until confirmation")
}

func TestMaybeReclaim_ZeroCeilingBand_NotEligibleDemand(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t, testCleanupTick)
	evictor := &fakeInFlightEvictor{inFlight: 6, evictable: 6, victimPriority: -1, hasVictim: true}
	withReclamation(h, evictor, testReclamationConfig)

	q := h.addQueue(testFlow)
	require.NoError(t, q.Add(h.newTestItem("req-blocked", testFlow, testTTL)))

	// A zero-ceiling band is permanently blocked by policy; no amount of reclamation can unblock
	// it, so its queue must not count as eviction demand.
	h.processor.maybeReclaim(h.ctx, 1.0, []int{testFlow.Priority}, []float64{0}, 0)
	assert.Empty(t, evictor.totalEvictCalls(), "a zero-ceiling band must not trigger reclamation")

	// Positive control: the same demand against an attainable ceiling reclaims.
	h.processor.maybeReclaim(h.ctx, 1.0, []int{testFlow.Priority}, []float64{0.9}, 0)
	assert.Equal(t, []int{1}, evictor.totalEvictCalls(), "an attainable blocked ceiling must reclaim")
}

func TestReclaim_PassesDemandPriorityAsVictimBound(t *testing.T) {
	t.Parallel()
	clk := testclock.NewFakeClock(time.Now())
	evictor := &fakeInFlightEvictor{inFlight: 4, evictable: 4}
	rc := newTestReclamationController(evictor, clk, testReclamationConfig)

	rc.Reclaim(context.Background(), 2.0, 1.0, -1)

	evictor.mu.Lock()
	defer evictor.mu.Unlock()
	require.Equal(t, []int{-1}, evictor.boundCalls,
		"the demand band's priority must reach the actuator as the victim bound")
}
