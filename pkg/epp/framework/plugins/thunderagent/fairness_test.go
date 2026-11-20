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
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// A new session is admitted only when a pod has room for its estimate.
func TestNewSessionHeldWhenNoRoom(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 900) // 900 of 1000 used
	l.prime(podA)

	big := l.makeQueue("s2", l.now(), 2000) // estimate 500 > room 100
	require.Nil(t, pick(t, l.agent, big))

	small := l.makeQueue("s3", l.now(), 320) // estimate 80 <= room 100
	require.Equal(t, small, pick(t, l.agent, small))
}

// An admitted session's next turn dispatches even when it pushes its pod
// over the ceiling and no idle session can give up room.
func TestAdmittedBypassesFit(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 950)
	l.prime(podA)

	q := l.makeQueue("s1", l.now(), 8000) // estimate 2000: 1050 over the room
	require.Equal(t, q, pick(t, l.agent, q))
	require.Equal(t, float64(2000), l.endpointTokens("pod-a"))
}

// A picked turn of an admitted session holds its room until PreRequest:
// dispatched, it is no longer queued and not yet in flight, but must not
// look idle.
func TestAdmittedPickReservesRoom(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 300) // room 700
	l.idleFor("s1", time.Minute)
	l.prime(podA)

	q1 := l.makeQueue("s1", l.now(), 1600) // estimate 400: s1 grows to 400
	require.Equal(t, q1, pick(t, l.agent, q1))
	require.Equal(t, float64(400), l.endpointTokens("pod-a"))

	// s1's turn has left the queue; a new session that would fit only by
	// reclaiming s1 holds.
	q2 := l.makeQueue("s2", l.now(), 3200) // estimate 800 > room 600
	require.Nil(t, pick(t, l.agent, q2))
	require.False(t, l.isPaused("s1"))
}

// Admitting a new session that does not fit pauses idle sessions past
// idleGracePeriod, longest idle first, and only as many as the admission needs.
func TestAdmissionReclaimsIdlePastGracePeriodLongestFirst(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 300)
	l.seed("s2", "pod-a", 300)
	l.seed("s3", "pod-a", 300) // room 100
	l.idleFor("s1", 2*time.Minute)
	l.idleFor("s2", time.Minute)
	l.prime(podA)

	q := l.makeQueue("s4", l.now(), 1400) // estimate 350: needs s1's room
	require.Equal(t, q, pick(t, l.agent, q))
	require.True(t, l.isPaused("s1"))
	require.False(t, l.isPaused("s2"))
	require.False(t, l.isPaused("s3"))
	// s2 and s3 plus the 350 reserved for s4.
	require.Equal(t, float64(950), l.endpointTokens("pod-a"))
}

// An idle session within idleGracePeriod keeps its room: the admission holds.
func TestAdmissionKeepsIdleWithinGracePeriod(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 900) // idle for less than idleGracePeriod
	l.prime(podA)

	q := l.makeQueue("s2", l.now(), 1600) // estimate 400 > room 100
	require.Nil(t, pick(t, l.agent, q))
	require.False(t, l.isPaused("s1"))
}

// A new session goes to a pod it fits without pausing anyone before a pod
// where it would fit only by reclaiming.
func TestNewSessionPrefersPodWithoutReclaim(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	podB := l.pod("pod-b", 0, 0)
	l.seed("s1", "pod-a", 800) // room 200, 800 reclaimable
	l.seed("s2", "pod-b", 600) // room 400
	l.idleFor("s1", time.Minute)
	l.prime(podA, podB)

	q := l.makeQueue("s3", l.now(), 1400) // estimate 350
	require.Equal(t, q, pick(t, l.agent, q))
	require.False(t, l.isPaused("s1"))

	l.agent.mu.Lock()
	local := l.agent.local["s3"]
	reservedPod := ""
	if local != nil {
		reservedPod = local.reservedEndpoint
	}
	l.agent.mu.Unlock()
	require.Equal(t, "default/pod-b", reservedPod)
}

// An admitted session's turn that pushes its pod over the ceiling pauses
// sessions idle past idleGracePeriod, longest idle first. A session that just
// finished a turn keeps its room.
func TestAdmittedGrowthReclaimsPastGracePeriod(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 300)
	l.seed("s2", "pod-a", 250)
	l.seed("s3", "pod-a", 450) // pod full
	l.idleFor("s1", time.Second)
	l.idleFor("s2", time.Minute)
	l.prime(podA)

	// Estimate 650 grows s3 by 200: pausing s2 (250) covers the growth.
	q := l.makeQueue("s3", l.now(), 2600)
	require.Equal(t, q, pick(t, l.agent, q))
	require.True(t, l.isPaused("s2"))
	require.False(t, l.isPaused("s1"))
}

// When every idle session is within idleGracePeriod, an admitted session's turn
// still dispatches and pauses nobody; the pod stays over its ceiling.
func TestAdmittedGrowthKeepsSessionsWithinGracePeriod(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 550)
	l.seed("s2", "pod-a", 450) // pod full
	l.idleFor("s1", time.Second)
	l.prime(podA)

	q := l.makeQueue("s2", l.now(), 2600) // grows s2 by 200
	require.Equal(t, q, pick(t, l.agent, q))
	require.False(t, l.isPaused("s1"))
}

// A session with a turn queued is about to be active and is never
// reclaimed, even when it is long idle.
func TestQueuedSessionNotReclaimed(t *testing.T) {
	cfg := testConfig()
	cfg.ForceAdmitTimeout = metav1.Duration{Duration: time.Second}
	l := newTestLedger(cfg)
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 600) // room 400
	l.idleFor("s1", time.Minute)
	l.prime(podA)

	q1 := l.makeQueue("s1", l.now(), 400)
	q2 := l.makeQueue("s2", l.now().Add(-2*time.Second), 2000) // starving, estimate 500
	require.Equal(t, q2, pick(t, l.agent, q1, q2))
	require.False(t, l.isPaused("s1"))
}

// A session larger than the ceiling fits once every other session on its
// pod can give up room, instead of never.
func TestOversizedSessionFitsWhenPodReclaimable(t *testing.T) {
	cfg := testConfig()
	cfg.UsableCapacityRatio = 0.5 // ceiling 500
	l := newTestLedger(cfg)
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 600)
	l.seed("s2", "pod-a", 100)
	l.pause("s1")
	l.prime(podA)

	q := l.makeQueue("s1", l.now(), 2400) // estimate 600 > ceiling
	require.Nil(t, pick(t, l.agent, q), "s2 is within idleGracePeriod")

	l.idleFor("s2", time.Minute)
	require.Equal(t, q, pick(t, l.agent, q))
	require.True(t, l.isPaused("s2"))
}

// A paused session resumes only on its own pod, even when another pod has
// room (strict origin affinity); it goes ahead of new sessions once its pod
// has room again.
func TestPausedResumesOnlyOnItsOwnPod(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	podB := l.pod("pod-b", 0, 0)
	l.seed("s1", "pod-a", 600)
	l.seed("s2", "pod-a", 500)
	l.pause("s2")
	l.prime(podA, podB)

	// pod-b is empty, but s2 waits for pod-a: room on a is 400 < 500.
	q2 := l.makeQueue("s2", l.now(), 2000)
	require.Nil(t, pick(t, l.agent, q2))

	// s1 is evicted; pod-a has room again; s2 beats a fitting newcomer.
	l.evict("s1")
	l.prime(podA, podB)

	qNew := l.makeQueue("s9", l.now().Add(-10*time.Second), 320) // older, below the starvation deadline
	require.Equal(t, q2, pick(t, l.agent, q2, qNew))
	require.False(t, l.isPaused("s2"))
	require.Equal(t, float64(1), testutil.ToFloat64(l.agent.metrics.resumes))

	// Completing the turn keeps resumes at 1 across subsequent sweeps.
	l.runTurn("s2", "pod-a", 2000, 550)
	l.agent.sweep()
	require.Equal(t, float64(1), testutil.ToFloat64(l.agent.metrics.resumes))
}

// An admission reservation blocks a second admit into the same room until
// it is confirmed or expires.
func TestReservationPreventsDoubleAdmit(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 900) // room 100
	l.prime(podA)

	q2 := l.makeQueue("s2", l.now(), 320) // estimate 80
	q3 := l.makeQueue("s3", l.now(), 320)
	require.Equal(t, q2, pick(t, l.agent, q2, q3))
	// 80 of the 100 are reserved for s2; s3 no longer fits.
	require.Nil(t, pick(t, l.agent, q3))

	// A sweep before the reservation expires keeps it.
	l.agent.sweep()
	l.prime(podA)
	require.Nil(t, pick(t, l.agent, q3))

	// The reservation expires unconfirmed; the room frees and s3 fits.
	l.step(reservationTTL + time.Second)
	l.prime(podA)
	require.Equal(t, q3, pick(t, l.agent, q3))
}

// The starvation backstop force-admits a head past the deadline, fit or not.
func TestStarvationForcesAdmission(t *testing.T) {
	cfg := testConfig()
	cfg.ForceAdmitTimeout = metav1.Duration{Duration: time.Second}
	l := newTestLedger(cfg)
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 1000)
	l.prime(podA)

	q := l.makeQueue("s2", l.now().Add(-2*time.Second), 4000)
	require.Equal(t, q, pick(t, l.agent, q))
	require.Equal(t, float64(1), testutil.ToFloat64(l.agent.metrics.starvationPromotions))
}

// A zero forceAdmitTimeout disables forced admission even after long waits.
func TestForceAdmitDisabledWhenZero(t *testing.T) {
	cfg := testConfig()
	cfg.ForceAdmitTimeout = metav1.Duration{Duration: 0}
	l := newTestLedger(cfg)
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 1000)
	l.prime(podA)

	q := l.makeQueue("s2", l.now().Add(-10*time.Minute), 4000)
	require.Nil(t, pick(t, l.agent, q))
}

// A sub-second forceAdmitTimeout compares against native time.Duration wait.
func TestForceAdmitSubSecondDuration(t *testing.T) {
	cfg := testConfig()
	cfg.ForceAdmitTimeout = metav1.Duration{Duration: 500 * time.Millisecond}
	l := newTestLedger(cfg)
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 1000)
	l.prime(podA)

	tooFresh := l.makeQueue("s2", l.now().Add(-100*time.Millisecond), 4000)
	require.Nil(t, pick(t, l.agent, tooFresh))

	starving := l.makeQueue("s2", l.now().Add(-600*time.Millisecond), 4000)
	require.Equal(t, starving, pick(t, l.agent, starving))
}

// A zero idleGracePeriod reclaims idle sessions immediately without waiting.
func TestIdleGracePeriodZeroReclaimsImmediately(t *testing.T) {
	cfg := testConfig()
	cfg.IdleGracePeriod = metav1.Duration{Duration: 0}
	l := newTestLedger(cfg)
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 900)
	l.prime(podA)

	q := l.makeQueue("s2", l.now(), 1600) // estimate 400 > room 100
	require.Equal(t, q, pick(t, l.agent, q))
	require.True(t, l.isPaused("s1"))
}

// A force-admitted paused session still reclaims sessions on its pod past
// idleGracePeriod.
func TestStarvingPausedSessionReclaimsIdle(t *testing.T) {
	cfg := testConfig()
	cfg.ForceAdmitTimeout = metav1.Duration{Duration: time.Second}
	l := newTestLedger(cfg)
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 600)
	l.seed("s3", "pod-a", 400) // within idleGracePeriod
	l.seed("s2", "pod-a", 700)
	l.pause("s2")
	l.idleFor("s1", time.Minute)
	l.prime(podA)

	// s2 needs 700; room 0 plus s1's reclaimable 600 is not enough, so it force-admits.
	q := l.makeQueue("s2", l.now().Add(-2*time.Second), 2000)
	require.Equal(t, q, pick(t, l.agent, q))
	require.True(t, l.isPaused("s1"))
	require.False(t, l.isPaused("s3"))
	// s3 plus the 700 reserved for s2.
	require.Equal(t, float64(1100), l.endpointTokens("pod-a"))
	require.Equal(t, float64(1), testutil.ToFloat64(l.agent.metrics.resumes))
}

// Without a fit view (the plugin not wired as the saturation detector) the
// gate fails open and only class ordering remains.
func TestFailsOpenWithoutFitView(t *testing.T) {
	l := newTestLedger(testConfig())
	q := l.makeQueue("s1", l.now(), 4000000)
	require.Equal(t, q, pick(t, l.agent, q))
}

// A head with no session id is anonymous: it passes the gate untracked, even
// when no pod has room.
func TestAnonymousHeadPasses(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 1000) // pod-a full
	l.prime(podA)

	anon := l.makeFlowQueue("default-flow", "", l.now(), 400)
	require.Equal(t, anon, pick(t, l.agent, anon))
}

// The gate reads the session of the queue head, not the queue's fairness ID:
// an admitted session queued under a tenant fairness ID dispatches even when
// its pod is full, where an unknown session would hold.
func TestGateUsesHeadSession(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 1000) // pod-a full
	l.prime(podA)

	q := l.makeFlowQueue("tenant-a", "s1", l.now(), 400)
	require.Equal(t, q, pick(t, l.agent, q))

	unknown := l.makeFlowQueue("tenant-a", "s9", l.now(), 400)
	require.Nil(t, pick(t, l.agent, unknown))
}

// An admitted session with a turn already in flight is sized as that turn
// plus the new one, since PreRequest adds the new estimate to what is in
// flight; its room is reserved at that size.
func TestAdmittedSizeAddsTurnsInFlight(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.seed("s1", "pod-a", 300)
	l.startTurn("s1", "pod-a", 1600) // 400 in flight
	l.prime(podA)

	q := l.makeQueue("s1", l.now(), 1200) // estimate 300: 400 + 300 = 700
	require.Equal(t, q, pick(t, l.agent, q))
	require.Equal(t, float64(700), l.endpointTokens("pod-a"))
}
