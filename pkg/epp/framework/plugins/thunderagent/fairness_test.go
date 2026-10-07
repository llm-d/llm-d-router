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

	"github.com/stretchr/testify/require"
)

// A new session is admitted only when a pod has room for its estimate.
func TestNewSessionHeldWhenNoRoom(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 900) // 900 of 1000 used
	primeFitView(a, dlEndpoint("pod-a"))

	big := makeQueue("s2", time.Now(), 2000) // estimate 500 > room 100
	require.Nil(t, pick(t, a, big))

	small := makeQueue("s3", time.Now(), 320) // estimate 80 <= room 100
	require.Equal(t, small, pick(t, a, small))
}

// An admitted session's next turn dispatches even when it pushes its pod
// over the ceiling and no idle session can give up room.
func TestAdmittedBypassesFit(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 950)
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeQueue("s1", time.Now(), 8000) // estimate 2000: 1050 over the room
	require.Equal(t, q, pick(t, a, q))
	require.Equal(t, float64(2000), endpointTokens(a, "default/pod-a"))
}

// A picked turn of an admitted session holds its room until PreRequest:
// dispatched, it is no longer queued and not yet in flight, but must not
// look idle.
func TestAdmittedPickReservesRoom(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 300) // room 700
	idleFor(a, "s1", time.Minute)
	primeFitView(a, dlEndpoint("pod-a"))

	q1 := makeQueue("s1", time.Now(), 1600) // estimate 400: s1 grows to 400
	require.Equal(t, q1, pick(t, a, q1))
	require.Equal(t, float64(400), endpointTokens(a, "default/pod-a"))

	// s1's turn has left the queue; a new session that would fit only by
	// reclaiming s1 holds.
	q2 := makeQueue("s2", time.Now(), 3200) // estimate 800 > room 600
	require.Nil(t, pick(t, a, q2))
	require.False(t, isPaused(a, "s1"))
}

// Admitting a new session that does not fit pauses idle sessions past the
// lease, longest idle first, and only as many as the admission needs.
func TestAdmissionReclaimsLeaseExpiredIdleLongestFirst(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 300)
	seed(t, a, "s2", "pod-a", 300)
	seed(t, a, "s3", "pod-a", 300) // room 100
	idleFor(a, "s1", 2*time.Minute)
	idleFor(a, "s2", time.Minute)
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeQueue("s4", time.Now(), 1400) // estimate 350: needs s1's room
	require.Equal(t, q, pick(t, a, q))
	require.True(t, isPaused(a, "s1"))
	require.False(t, isPaused(a, "s2"))
	require.False(t, isPaused(a, "s3"))
	// s2 and s3 plus the 350 reserved for s4.
	require.Equal(t, float64(950), endpointTokens(a, "default/pod-a"))
}

// An idle session inside its lease keeps its room: the admission holds.
func TestAdmissionKeepsIdleWithinLease(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 900) // idle for less than the lease
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeQueue("s2", time.Now(), 1600) // estimate 400 > room 100
	require.Nil(t, pick(t, a, q))
	require.False(t, isPaused(a, "s1"))
}

// A new session goes to a pod it fits without pausing anyone before a pod
// where it would fit only by reclaiming.
func TestNewSessionPrefersPodWithoutReclaim(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 800) // room 200, 800 reclaimable
	seed(t, a, "s2", "pod-b", 600) // room 400
	idleFor(a, "s1", time.Minute)
	primeFitView(a, dlEndpoint("pod-a"), dlEndpoint("pod-b"))

	q := makeQueue("s3", time.Now(), 1400) // estimate 350
	require.Equal(t, q, pick(t, a, q))
	require.False(t, isPaused(a, "s1"))
	s, _ := sessionOf(a, "s3")
	require.Equal(t, "default/pod-b", s.endpoint.id)
}

// An admitted session's turn that pushes its pod over the ceiling pauses
// sessions idle past the lease, longest idle first. A session that just
// finished a turn keeps its room.
func TestAdmittedGrowthReclaimsPastLease(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 300)
	seed(t, a, "s2", "pod-a", 250)
	seed(t, a, "s3", "pod-a", 450) // pod full
	idleFor(a, "s1", time.Second)
	idleFor(a, "s2", time.Minute)
	primeFitView(a, dlEndpoint("pod-a"))

	// Estimate 650 grows s3 by 200: pausing s2 (250) covers the growth.
	q := makeQueue("s3", time.Now(), 2600)
	require.Equal(t, q, pick(t, a, q))
	require.True(t, isPaused(a, "s2"))
	require.False(t, isPaused(a, "s1"))
}

// When every idle session is inside the lease, an admitted session's turn
// still dispatches and pauses nobody; the pod stays over its ceiling.
func TestAdmittedGrowthKeepsSessionsInsideLease(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 550)
	seed(t, a, "s2", "pod-a", 450) // pod full
	idleFor(a, "s1", time.Second)
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeQueue("s2", time.Now(), 2600) // grows s2 by 200
	require.Equal(t, q, pick(t, a, q))
	require.False(t, isPaused(a, "s1"))
}

// A session with a turn queued is about to be active and is never
// reclaimed, even when it is long idle.
func TestQueuedSessionNotReclaimed(t *testing.T) {
	cfg := testConfig()
	cfg.HeadWaitStarvationMs = 1000
	a := newTestAgent(cfg)
	seed(t, a, "s1", "pod-a", 600) // room 400
	idleFor(a, "s1", time.Minute)
	primeFitView(a, dlEndpoint("pod-a"))

	q1 := makeQueue("s1", time.Now(), 400)
	q2 := makeQueue("s2", time.Now().Add(-2*time.Second), 2000) // starving, estimate 500
	require.Equal(t, q2, pick(t, a, q1, q2))
	require.False(t, isPaused(a, "s1"))
}

// A session larger than the ceiling fits once every other session on its
// pod can give up room, instead of never.
func TestOversizedSessionFitsWhenPodReclaimable(t *testing.T) {
	cfg := testConfig()
	cfg.UtilThreshold = 0.5 // ceiling 500
	a := newTestAgent(cfg)
	seed(t, a, "s1", "pod-a", 600)
	seed(t, a, "s2", "pod-a", 100)
	pause(a, "s1")
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeQueue("s1", time.Now(), 2400) // estimate 600 > ceiling
	require.Nil(t, pick(t, a, q), "s2 is inside its lease")

	idleFor(a, "s2", time.Minute)
	require.Equal(t, q, pick(t, a, q))
	require.True(t, isPaused(a, "s2"))
}

// A paused session resumes only on its own pod, even when another pod has
// room (strict origin affinity); it goes ahead of new sessions once its pod
// has room again.
func TestPausedResumesOnlyOnItsOwnPod(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 600)
	seed(t, a, "s2", "pod-a", 500)
	pause(a, "s2")
	primeFitView(a, dlEndpoint("pod-a"), dlEndpoint("pod-b"))

	// pod-b is empty, but s2 waits for pod-a: room on a is 400 < 500.
	q2 := makeQueue("s2", time.Now(), 2000)
	require.Nil(t, pick(t, a, q2))

	// s1 idles past the TTL and is evicted; pod-a has room again; s2 beats
	// a fitting newcomer.
	a.mgr.mu.Lock()
	a.mgr.sessions["s1"].lastActivity = time.Now().Add(-2 * a.mgr.ttl)
	a.mgr.mu.Unlock()
	a.mgr.sweep(time.Now())
	primeFitView(a, dlEndpoint("pod-a"), dlEndpoint("pod-b"))

	qNew := makeQueue("s9", time.Now().Add(-time.Minute), 320)
	require.Equal(t, q2, pick(t, a, q2, qNew))

	// PreRequest confirms the resume.
	req := newRequest("s2", 2000)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(schedEndpoint("pod-a", 0, 0))))
	require.False(t, isPaused(a, "s2"))
}

// An admission reservation blocks a second admit into the same room until
// it is confirmed or expires.
func TestReservationPreventsDoubleAdmit(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 900) // room 100
	primeFitView(a, dlEndpoint("pod-a"))

	q2 := makeQueue("s2", time.Now(), 320) // estimate 80
	q3 := makeQueue("s3", time.Now(), 320)
	require.Equal(t, q2, pick(t, a, q2, q3))
	// 80 of the 100 are reserved for s2; s3 no longer fits.
	require.Nil(t, pick(t, a, q3))

	// A sweep before the reservation expires keeps it: the idle TTL
	// does not apply to a session that has never dispatched.
	a.mgr.sweep(time.Now())
	primeFitView(a, dlEndpoint("pod-a"))
	_, ok := sessionOf(a, "s2")
	require.True(t, ok)
	require.Nil(t, pick(t, a, q3))

	// The reservation expires unconfirmed; the room frees and s3 fits.
	a.mgr.mu.Lock()
	a.mgr.sessions["s2"].reservedUntil = time.Now().Add(-time.Second)
	a.mgr.mu.Unlock()
	require.Equal(t, q3, pick(t, a, q3))
}

// The starvation backstop force-admits a head past the deadline, fit or not.
func TestStarvationForcesAdmission(t *testing.T) {
	cfg := testConfig()
	cfg.HeadWaitStarvationMs = 1000
	a := newTestAgent(cfg)
	seed(t, a, "s1", "pod-a", 1000)
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeQueue("s2", time.Now().Add(-2*time.Second), 4000)
	require.Equal(t, q, pick(t, a, q))
}

// Without a fit view (the plugin not wired as the saturation detector) the
// gate fails open and only class ordering remains.
func TestFailsOpenWithoutFitView(t *testing.T) {
	a := newTestAgent(testConfig())
	q := makeQueue("s1", time.Now(), 4000000)
	require.Equal(t, q, pick(t, a, q))
}

// A head with no session id is anonymous: it passes the gate untracked, even
// when no pod has room.
func TestAnonymousHeadPasses(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 1000) // pod-a full
	primeFitView(a, dlEndpoint("pod-a"))

	anon := makeFlowQueue("default-flow", "", time.Now(), 400)
	require.Equal(t, anon, pick(t, a, anon))
}

// The gate reads the session of the queue head, not the queue's fairness ID:
// an admitted session queued under a tenant fairness ID dispatches even when
// its pod is full, where an unknown session would hold.
func TestGateUsesHeadSession(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 1000) // pod-a full
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeFlowQueue("tenant-a", "s1", time.Now(), 400)
	require.Equal(t, q, pick(t, a, q))

	unknown := makeFlowQueue("tenant-a", "s9", time.Now(), 400)
	require.Nil(t, pick(t, a, unknown))
}

// An admitted session with a turn already in flight is sized as that turn
// plus the new one, since PreRequest adds the new estimate to what is in
// flight; its room is reserved at that size.
func TestAdmittedSizeAddsTurnsInFlight(t *testing.T) {
	a := newTestAgent(testConfig())
	seed(t, a, "s1", "pod-a", 300)
	_ = startTurn(t, a, "s1", schedEndpoint("pod-a", 0, 0), 1600) // 400 in flight
	primeFitView(a, dlEndpoint("pod-a"))

	q := makeQueue("s1", time.Now(), 1200) // estimate 300: 400 + 300 = 700
	require.Equal(t, q, pick(t, a, q))
	require.Equal(t, float64(700), endpointTokens(a, "default/pod-a"))
}
