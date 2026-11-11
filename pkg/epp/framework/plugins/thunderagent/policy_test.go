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

package thunderagent

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
)

// When a paused session's pod leaves the pool, its pause clears without
// incrementing resumes, and its new reservation on a replacement pod accounts
// for its room and pins Score before PreRequest runs.
func TestPausedSessionClearedWithoutResumeWhenPodLeaves(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	podB := l.pod("pod-b", 0, 0)
	l.seed("s1", "pod-a", 500)
	l.pause("s1")
	l.prime(podA)
	require.True(t, l.isPaused("s1"))

	l.removePod("pod-a")
	l.prime(podB)

	// s1 had TurnsTaken=1 on pod-a; now unbound, Pick treats it as new onto pod-b.
	q1 := l.makeQueue("s1", l.now(), 800)
	require.Equal(t, q1, pick(t, l.agent, q1))
	require.False(t, l.isPaused("s1"))
	require.Equal(t, float64(0), testutil.ToFloat64(l.agent.metrics.resumes))
	require.Equal(t, float64(500), l.endpointTokens("pod-b"))

	// A second newcomer requiring 600 tokens holds while s1's unbound reservation
	// holds 500 of pod-b's 1000 capacity.
	q2 := l.makeQueue("s2", l.now(), 2400)
	require.Nil(t, pick(t, l.agent, q2))

	// Score pins s1 to pod-b via the unbound reservation even with Endpoint="".
	schedB, schedC := schedEndpoint("pod-b"), schedEndpoint("pod-c")
	scores := l.agent.Score(context.Background(), l.request("s1", 800), []fwksched.Endpoint{schedB, schedC})
	require.Equal(t, 1.0, scores[schedB])
	require.Equal(t, 0.0, scores[schedC])

	// Completing the turn on pod-b advances TurnsTaken and clears the reservation.
	l.runTurn("s1", "pod-b", 800, 520)
	require.Equal(t, float64(520), l.endpointTokens("pod-b"))
	l.agent.mu.Lock()
	require.Empty(t, l.agent.local)
	l.agent.mu.Unlock()
}

// sweep prunes expired reservations, superseded pause state, and stale
// endpoint entries without counting false resumes.
func TestSweepPrunesStaleLocalAndEndpoints(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s-idle", "pod-a", 100)
	l.seed("s-paused", "pod-a", 200)
	l.pause("s-paused")
	l.prime(podA)

	// Create a reservation for s-new on pod-a.
	qNew := l.makeQueue("s-new", l.now(), 400)
	require.Equal(t, qNew, pick(t, l.agent, qNew))

	// Halfway through reservationTTL, both the pause overlay and reservation remain.
	l.step(reservationTTL / 2)
	l.agent.sweep()
	l.agent.mu.Lock()
	require.Len(t, l.agent.local, 2)
	l.agent.mu.Unlock()

	// Past reservationTTL (while Saturation keeps pod-a fresh), s-new's
	// unconfirmed reservation drops and s-paused remains.
	l.step(reservationTTL)
	l.prime(podA)
	l.agent.sweep()
	l.agent.mu.Lock()
	require.Len(t, l.agent.local, 1)
	_, hasPaused := l.agent.local["s-paused"]
	require.True(t, hasPaused)
	l.agent.mu.Unlock()

	// When s-paused advances TurnsTaken directly (outside Pick), sweep clears
	// its superseded pause overlay without incrementing resumes.
	l.runTurn("s-paused", "pod-a", 400, 250)
	l.agent.sweep()
	require.Equal(t, float64(0), testutil.ToFloat64(l.agent.metrics.resumes))
	l.agent.mu.Lock()
	require.Empty(t, l.agent.local)
	l.agent.mu.Unlock()

	// Past endpointStaleAfter without a Saturation refresh, sweep drops pod-a.
	l.step(endpointStaleAfter + time.Second)
	l.agent.sweep()
	l.agent.mu.Lock()
	require.Empty(t, l.agent.endpoints)
	l.agent.mu.Unlock()
}

// snapshot aggregates running, idle, and paused counts plus working sets
// including active reservations.
func TestSnapshotCountsAndWorkingSet(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	podB := l.pod("pod-b", 16, 100) // 1600 capacity

	l.startTurn("s-run", "pod-a", 800) // 200 in-flight
	l.seed("s-idle", "pod-a", 300)
	l.seed("s-paused", "pod-a", 400)
	l.pause("s-paused")
	l.seed("s-b", "pod-b", 200)
	l.prime(podA, podB)

	// Reserve a new 100-token session onto pod-b (most free room: 1400 vs 500).
	qNew := l.makeQueue("s-new", l.now(), 400)
	require.Equal(t, qNew, pick(t, l.agent, qNew))

	snap := l.agent.snapshot()
	require.Equal(t, 1, snap.running)
	require.Equal(t, 2, snap.idle)
	require.Equal(t, 1, snap.paused)
	require.Equal(t, endpointSnapshot{workingSet: 500, capacity: 1000}, snap.endpoints["default/pod-a"])
	require.Equal(t, endpointSnapshot{workingSet: 300, capacity: 1600}, snap.endpoints["default/pod-b"])
}

// reclaimableLocked skips sessions that have never finished a turn, are
// in-flight, paused, reserved, queued, or still inside idleGracePeriod, while
// accepting an aborted turn whose LastResponseAt is zero.
func TestReclaimableGuards(t *testing.T) {
	l := newTestLedger(testConfig())
	now := l.now()
	idlePast := now.Add(-l.agent.idleGracePeriod - time.Second)
	withinGrace := now.Add(-l.agent.idleGracePeriod + time.Second)

	base := sessionstate.SessionState{
		SessionID:      "s1",
		Endpoint:       "default/pod-a",
		TurnsTaken:     1,
		ContextTokens:  100,
		LastResponseAt: idlePast,
	}
	require.True(t, l.agent.reclaimableLocked(base, nil, now, nil))
	require.False(t, l.agent.reclaimableLocked(base, nil, now, map[string]bool{"s1": true}))

	zeroTurns := base
	zeroTurns.TurnsTaken = 0
	require.False(t, l.agent.reclaimableLocked(zeroTurns, nil, now, nil))

	inFlight := base
	inFlight.InFlightRequests = 1
	require.False(t, l.agent.reclaimableLocked(inFlight, nil, now, nil))

	recent := base
	recent.LastResponseAt = withinGrace
	require.False(t, l.agent.reclaimableLocked(recent, nil, now, nil))

	pausedOverlay := &localSession{paused: true, pausedAtTurns: 1}
	require.False(t, l.agent.reclaimableLocked(base, pausedOverlay, now, nil))

	reservedOverlay := &localSession{reservedUntil: now.Add(time.Second), reservedAtTurns: 1}
	require.False(t, l.agent.reclaimableLocked(base, reservedOverlay, now, nil))

	abortedFirstTurn := sessionstate.SessionState{
		SessionID:  "s-aborted",
		Endpoint:   "default/pod-a",
		TurnsTaken: 1,
	}
	require.True(t, l.agent.reclaimableLocked(abortedFirstTurn, nil, now, nil))
}

// Saturation with a bare Endpoint (no EndpointSessions attribute attached yet)
// still enters the pool view and enforces capacity in Pick, matching snapshot().
func TestBareEndpointEnforcesCapacity(t *testing.T) {
	l := newTestLedger(testConfig())
	barePod := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Namespace: "default", Name: "pod-bare"},
	}, &fwkdl.Metrics{CacheBlockSize: 16, CacheNumBlocks: 50}) // 800 token capacity
	l.prime(barePod)

	fits := l.makeQueue("s-fit", l.now(), 2000) // 500 tokens <= 800 capacity
	require.Equal(t, fits, pick(t, l.agent, fits))
	require.Equal(t, float64(500), l.endpointTokens("pod-bare"))

	secondTooBig := l.makeQueue("s-other", l.now(), 1600) // 400 tokens > 300 remaining room
	require.Nil(t, pick(t, l.agent, secondTooBig))

	secondFits := l.makeQueue("s-small", l.now(), 800) // 200 tokens <= 300 remaining room
	require.Equal(t, secondFits, pick(t, l.agent, secondFits))
	require.Equal(t, float64(700), l.endpointTokens("pod-bare"))
}

// Score (which passes nil pods) preserves a live reservation, while sweep
// clears reservations immediately once the last pod leaves the pool.
func TestSweepClearsReservationWhenLastPodLeaves(t *testing.T) {
	l := newTestLedger(testConfig())
	podOnly := l.pod("pod-only", 0, 0)
	l.prime(podOnly)

	qNew := l.makeQueue("s-new", l.now(), 400)
	require.Equal(t, qNew, pick(t, l.agent, qNew))

	schedOnly, schedB := schedEndpoint("pod-only"), schedEndpoint("pod-b")
	scores := l.agent.Score(context.Background(), l.request("s-new", 400), []fwksched.Endpoint{schedOnly, schedB})
	require.Equal(t, 1.0, scores[schedOnly])
	require.Equal(t, 0.0, scores[schedB])

	l.removePod("pod-only")
	l.prime()
	l.agent.sweep()

	l.agent.mu.Lock()
	require.Empty(t, l.agent.local)
	l.agent.mu.Unlock()

	require.Nil(t, l.agent.Score(context.Background(), l.request("s-new", 400), []fwksched.Endpoint{schedOnly, schedB}))
}

// Removing an endpoint from the pool clears paused overlays in snapshot/local
// (even if a request-header state snapshot still carries the departed pod) and
// does not count a resume when the session re-enters on another pod.
func TestRemoveEndpointClearsPaused(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	podB := l.pod("pod-b", 0, 0)
	l.seed("s1", "pod-a", 300)
	l.seed("s2", "pod-a", 300)
	l.pause("s1")
	l.pause("s2")
	l.prime(podA, podB)
	require.Equal(t, 2, l.agent.snapshot().paused)

	// Capture a stale candidate snapshot for s2 before pod-a is removed.
	staleCand := map[string]sessionstate.SessionState{
		"s2": *l.sessions["s2"],
	}

	l.removePod("pod-a")
	l.prime(podB)

	l.agent.mu.Lock()
	_ = l.agent.readPoolLocked(l.now(), staleCand)
	require.Empty(t, l.agent.local)
	l.agent.mu.Unlock()

	snap := l.agent.snapshot()
	require.Equal(t, 0, snap.paused)
	require.False(t, l.isPaused("s1"))
	require.False(t, l.isPaused("s2"))

	q1 := l.makeQueue("s1", l.now(), 800)
	require.Equal(t, q1, pick(t, l.agent, q1))
	require.Equal(t, float64(0), testutil.ToFloat64(l.agent.metrics.resumes))
}
