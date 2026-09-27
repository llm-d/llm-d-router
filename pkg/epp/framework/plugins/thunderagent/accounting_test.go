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

// A turn charges its byte estimate while in flight and settles to the
// usage-reported total when it completes.
func TestTurnAccounting(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)

	req := startTurn(t, a, "s1", ep, 400) // 400 bytes / 4 bytes-per-token = 100 tokens
	s, ok := sessionOf(a, "s1")
	require.True(t, ok)
	require.Equal(t, int64(100), s.inflightTokens)
	require.Equal(t, float64(100), endpointTokens(a, "default/pod-a"))

	a.ResponseBody(context.Background(), req, endOfStream(250, 100), nil)
	s, _ = sessionOf(a, "s1")
	require.Equal(t, int64(0), s.inflightTokens)
	require.Equal(t, int64(250), s.committedTokens)
	require.Equal(t, float64(250), endpointTokens(a, "default/pod-a"))
}

// During a later turn the footprint is the max of the committed total and
// the new in-flight estimate, not their sum: the new prefill covers the same
// history.
func TestFootprintIsMaxNotSum(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)

	req := startTurn(t, a, "s1", ep, 400)
	a.ResponseBody(context.Background(), req, endOfStream(300, 100), nil)

	_ = startTurn(t, a, "s1", ep, 800) // in-flight estimate 200 < committed 300
	require.Equal(t, float64(300), endpointTokens(a, "default/pod-a"))

	_ = startTurn(t, a, "s1", ep, 2000) // a newer turn replaces the estimate: in-flight 500 > 300
	require.Equal(t, float64(500), endpointTokens(a, "default/pod-a"))
}

// Idle sessions past the TTL are dropped by maintenance, and a pod that then
// holds nothing is dropped too. Fresh sessions and pods stay.
func TestTTLEviction(t *testing.T) {
	cfg := testConfig()
	a := newTestAgent(cfg)
	ep := schedEndpoint("pod-a", 0, 0)
	runTurn(t, a, "s1", ep, 400, 300)

	a.mgr.mu.Lock()
	a.mgr.sessions["s1"].lastActivity = time.Now().Add(-2 * a.mgr.ttl)
	a.mgr.endpoints["default/pod-a"].updatedAt = time.Now().Add(-2 * endpointStaleAfter)
	a.mgr.mu.Unlock()
	forceMaintenance(a)

	runTurn(t, a, "s2", schedEndpoint("pod-b", 0, 0), 4, 1) // any hook triggers maintenance
	_, ok := sessionOf(a, "s1")
	require.False(t, ok)
	require.Equal(t, float64(-1), endpointTokens(a, "default/pod-a"))
	_, ok = sessionOf(a, "s2")
	require.True(t, ok)
	require.Equal(t, float64(1), endpointTokens(a, "default/pod-b"))
}

// A session with a turn in flight is never TTL-evicted, so the completion
// bookkeeping always finds its state.
func TestTTLKeepsInflightSessions(t *testing.T) {
	a := newTestAgent(testConfig())
	ep := schedEndpoint("pod-a", 0, 0)
	_ = startTurn(t, a, "s1", ep, 400)

	a.mgr.mu.Lock()
	a.mgr.sessions["s1"].lastActivity = time.Now().Add(-2 * a.mgr.ttl)
	a.mgr.mu.Unlock()
	forceMaintenance(a)
	runTurn(t, a, "s2", ep, 4, 1)

	_, ok := sessionOf(a, "s1")
	require.True(t, ok)
}

// Real scraped capacity wins over the configured fallback.
func TestCapacity(t *testing.T) {
	a := newTestAgent(testConfig())
	require.Equal(t, float64(16000), a.endpointCapacity(schedEndpoint("pod-a", 16, 1000).GetMetrics()))
	require.Equal(t, float64(1000), a.endpointCapacity(schedEndpoint("pod-a", 0, 0).GetMetrics()))
}

// Anonymous requests (no session identity) are not tracked.
func TestAnonymousIgnored(t *testing.T) {
	a := newTestAgent(testConfig())
	anon := newRequest("", 400)
	require.NoError(t, a.PreRequest(context.Background(), anon, schedulingResultFor(schedEndpoint("pod-a", 0, 0))))
	a.mgr.mu.Lock()
	n := len(a.mgr.sessions)
	a.mgr.mu.Unlock()
	require.Equal(t, 0, n)
}
