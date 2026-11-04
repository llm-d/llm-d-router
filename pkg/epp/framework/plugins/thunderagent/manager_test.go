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

	"github.com/stretchr/testify/require"
)

// t0 is a fixed base time so tests do not depend on the wall clock.
var t0 = time.Unix(1_000_000, 0)

// Binding to a new endpoint moves the same session, with its state, out of
// the old endpoint's working set and into the new one.
func TestBindMovesSession(t *testing.T) {
	m := newSessionManager(testConfig())
	a := m.ensureEndpointLocked("default/ep-a", 1000, t0)
	b := m.ensureEndpointLocked("default/ep-b", 1000, t0)

	s := m.bindLocked("s1", a)
	s.committedTokens = 300
	require.Same(t, s, m.bindLocked("s1", a))

	require.Same(t, s, m.bindLocked("s1", b))
	require.Same(t, b, s.endpoint)
	require.Empty(t, a.sessions)
	require.Same(t, s, b.sessions["s1"])
	require.Equal(t, float64(0), a.occupancy(t0))
	require.Equal(t, float64(300), b.occupancy(t0))
}

// Removing a session drops it from the global index and from its endpoint;
// removing an unknown session is a no-op.
func TestRemoveSession(t *testing.T) {
	m := newSessionManager(testConfig())
	ep := m.ensureEndpointLocked("default/ep-a", 1000, t0)
	m.bindLocked("s1", ep)

	m.removeLocked("s1")
	m.removeLocked("unknown")
	require.Empty(t, m.sessions)
	require.Empty(t, ep.sessions)
}

// A known endpoint keeps its entry and sessions; only capacity and last-seen
// time are refreshed.
func TestEnsureEndpointRefreshes(t *testing.T) {
	m := newSessionManager(testConfig())
	ep := m.ensureEndpointLocked("default/ep-a", 1000, t0)
	m.bindLocked("s1", ep)

	later := t0.Add(time.Minute)
	require.Same(t, ep, m.ensureEndpointLocked("default/ep-a", 2000, later))
	require.Equal(t, float64(2000), ep.capacity)
	require.Equal(t, later, ep.updatedAt)
	require.Len(t, ep.sessions, 1)
}

func TestEstimateTokens(t *testing.T) {
	require.Equal(t, int64(0), estimateTokens(0))
	require.Equal(t, int64(0), estimateTokens(-1))
	require.Equal(t, int64(100), estimateTokens(400))
}

// The snapshot counts sessions by state and reports each endpoint's working
// set and capacity.
func TestSnapshot(t *testing.T) {
	m := newSessionManager(testConfig())
	a := m.ensureEndpointLocked("default/ep-a", 1000, t0)
	b := m.ensureEndpointLocked("default/ep-b", 2000, t0)

	running := m.bindLocked("running", a)
	running.inflightTokens = 100
	running.lastActivity = t0
	idle := m.bindLocked("idle", a)
	idle.committedTokens = 300
	idle.lastActivity = t0
	other := m.bindLocked("other", b)
	other.committedTokens = 50
	other.lastActivity = t0

	snap := m.snapshot(t0)
	require.Equal(t, 1, snap.running)
	require.Equal(t, 2, snap.idle)
	require.Equal(t, endpointGauge{workingSet: 400, capacity: 1000}, snap.endpoints["default/ep-a"])
	require.Equal(t, endpointGauge{workingSet: 50, capacity: 2000}, snap.endpoints["default/ep-b"])
}

// An endpoint with no sessions is dropped once it has not been seen for
// endpointStaleAfter; an endpoint that still holds a session is kept.
func TestSweepDropsStaleEmptyEndpoints(t *testing.T) {
	m := newSessionManager(testConfig())
	m.ensureEndpointLocked("default/empty", 1000, t0)
	held := m.ensureEndpointLocked("default/held", 1000, t0)
	s := m.bindLocked("s1", held)
	s.lastActivity = t0.Add(endpointStaleAfter + time.Second)

	m.sweep(t0.Add(endpointStaleAfter))
	require.Contains(t, m.endpoints, "default/empty")

	m.sweep(t0.Add(endpointStaleAfter + time.Second))
	require.NotContains(t, m.endpoints, "default/empty")
	require.Contains(t, m.endpoints, "default/held")
}

// A session is new until it has an endpoint and a dispatched turn: a session
// Pick reserved onto a pod but PreRequest has not bound yet is still new, and
// so is a paused session whose pod left the pool.
func TestSessionClass(t *testing.T) {
	m := newSessionManager(testConfig())
	ep := m.ensureEndpointLocked("default/ep-a", 1000, t0)

	s := m.bindLocked("s1", ep)
	require.Equal(t, classNew, s.class(), "bound, never dispatched")
	s.turnCount = 1
	require.Equal(t, classAdmitted, s.class())
	s.paused = true
	require.Equal(t, classPaused, s.class())
	s.endpoint = nil
	require.Equal(t, classNew, s.class(), "pod left the pool")
	require.Equal(t, "new", classNew.String())
	require.Equal(t, "admitted", classAdmitted.String())
	require.Equal(t, "paused", classPaused.String())
}

// The accounting rule: a live reservation counts its reserved size, a paused
// session counts nothing, any other session its full size.
func TestFootprint(t *testing.T) {
	s := &session{committedTokens: 300, inflightTokens: 500, turnCount: 1}
	require.Equal(t, float64(500), s.footprint(t0), "in-flight estimate above the committed total")
	s.inflightTokens = 0
	require.Equal(t, float64(300), s.footprint(t0))

	s.paused = true
	require.Equal(t, float64(0), s.footprint(t0))
	s.reservedTokens, s.reservedUntil = 700, t0.Add(reservationTTL)
	require.Equal(t, float64(700), s.footprint(t0), "a live reservation counts even while paused")
	require.Equal(t, float64(0), s.footprint(t0.Add(reservationTTL)), "an expired one does not")
}

// Only an admitted session with no turn in flight and no live reservation,
// idle for at least the idle lease, can give up its room.
func TestReclaimable(t *testing.T) {
	m := newSessionManager(testConfig())
	idle := func() *session {
		return &session{committedTokens: 300, turnCount: 1, lastResponseAt: t0}
	}
	later := t0.Add(time.Minute)
	m.idleLease = time.Minute
	require.True(t, m.reclaimableLocked(idle(), later))
	m.idleLease = 2 * time.Minute
	require.False(t, m.reclaimableLocked(idle(), later), "inside the lease")

	m.idleLease = 0
	s := idle()
	s.inflightTokens = 100
	require.False(t, m.reclaimableLocked(s, later), "turn in flight")
	s = idle()
	s.paused = true
	require.False(t, m.reclaimableLocked(s, later), "already paused")
	s = idle()
	s.reservedUntil = later.Add(time.Second)
	require.False(t, m.reclaimableLocked(s, later), "live reservation")
	s = idle()
	s.turnCount = 0
	require.False(t, m.reclaimableLocked(s, later), "never dispatched")
}

// A session with a turn queued is never offered for reclaim, and the
// reclaimable total counts exactly the sessions reclaim may pause.
func TestIdleSessionsSkipQueued(t *testing.T) {
	m := newSessionManager(testConfig())
	ep := m.ensureEndpointLocked("default/ep-a", 1000, t0)
	for id, tokens := range map[string]int64{"s1": 300, "s2": 200} {
		s := m.bindLocked(id, ep)
		s.committedTokens, s.turnCount, s.lastResponseAt = tokens, 1, t0
	}
	later := t0.Add(time.Minute)

	require.Len(t, m.idleSessionsLocked(ep, later, nil), 2)
	queued := map[string]bool{"s1": true}
	idle := m.idleSessionsLocked(ep, later, queued)
	require.Len(t, idle, 1)
	require.Same(t, m.sessions["s2"], idle[0])
	require.Equal(t, float64(200), m.reclaimableTokensLocked(ep, later, queued))
}

// Reclaim pauses the longest-idle sessions first and stops as soon as the
// room covers the need; with enough room it pauses nobody.
func TestReclaimLongestIdleFirst(t *testing.T) {
	m := newSessionManager(testConfig())
	ep := m.ensureEndpointLocked("default/ep-a", 1000, t0)
	for id, idleFor := range map[string]time.Duration{"old": 3 * time.Minute, "mid": 2 * time.Minute, "new": time.Minute} {
		s := m.bindLocked(id, ep)
		s.committedTokens, s.turnCount = 300, 1
		s.lastResponseAt = t0.Add(-idleFor)
	}

	require.Equal(t, 0, m.reclaimLocked(ep, t0, nil, 400, 350))
	require.Equal(t, 2, m.reclaimLocked(ep, t0, nil, 100, 650))
	require.True(t, m.sessions["old"].paused)
	require.True(t, m.sessions["mid"].paused)
	require.False(t, m.sessions["new"].paused)
}

// A live reservation keeps a never-dispatched session through the sweep;
// once it expires the session is dropped, but a dispatched session whose
// reservation expired stays until the idle TTL.
func TestSweepReservationExpiry(t *testing.T) {
	m := newSessionManager(testConfig())
	ep := m.ensureEndpointLocked("default/ep-a", 1000, t0)
	fresh := m.bindLocked("fresh", ep)
	fresh.reservedTokens, fresh.reservedUntil = 100, t0.Add(reservationTTL)
	resumed := m.bindLocked("resumed", ep)
	resumed.turnCount, resumed.paused, resumed.lastActivity = 1, true, t0
	resumed.reservedTokens, resumed.reservedUntil = 100, t0.Add(reservationTTL)

	m.sweep(t0)
	require.Contains(t, m.sessions, "fresh")

	m.sweep(t0.Add(reservationTTL))
	require.NotContains(t, m.sessions, "fresh")
	require.Contains(t, m.sessions, "resumed")
}

// The snapshot counts paused sessions apart from idle ones, and a paused
// session's tokens are not in its endpoint's working set.
func TestSnapshotCountsPaused(t *testing.T) {
	m := newSessionManager(testConfig())
	ep := m.ensureEndpointLocked("default/ep-a", 1000, t0)
	idle := m.bindLocked("idle", ep)
	idle.committedTokens, idle.turnCount, idle.lastActivity = 300, 1, t0
	paused := m.bindLocked("paused", ep)
	paused.committedTokens, paused.turnCount, paused.lastActivity, paused.paused = 200, 1, t0, true

	snap := m.snapshot(t0)
	require.Equal(t, 1, snap.idle)
	require.Equal(t, 1, snap.paused)
	require.Equal(t, float64(300), snap.endpoints["default/ep-a"].workingSet)
}
