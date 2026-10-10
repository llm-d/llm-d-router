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
	require.Equal(t, float64(0), a.workingSetTokens())
	require.Equal(t, float64(300), b.workingSetTokens())
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

	snap := m.snapshot()
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
