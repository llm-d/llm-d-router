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

package sessionstate

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func registrySessionCount(registry *SessionStateRegistry) int {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	return len(registry.sessions)
}

func setRegistrySessionTimes(t *testing.T, registry *SessionStateRegistry, identity string, firstSeenAt, lastSeenAt time.Time) {
	t.Helper()
	registry.mu.Lock()
	defer registry.mu.Unlock()
	record, ok := registry.sessions[identity]
	require.True(t, ok)
	record.firstSeenAt = firstSeenAt
	record.state.LastSeenAt = lastSeenAt
}

func setRegistryLastResponseAt(t *testing.T, registry *SessionStateRegistry, identity string, lastResponseAt time.Time) {
	t.Helper()
	registry.mu.Lock()
	defer registry.mu.Unlock()
	record, ok := registry.sessions[identity]
	require.True(t, ok)
	record.state.LastResponseAt = lastResponseAt
}

func TestRegistryGetState(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	beforeFirst := time.Now()
	state := registry.GetState("session-a")
	afterFirst := time.Now()
	assert.Equal(t, "session-a", state.SessionID)
	assert.Zero(t, state.TurnsTaken)
	assert.Zero(t, state.Duration)
	assert.False(t, state.LastSeenAt.Before(beforeFirst))
	assert.False(t, state.LastSeenAt.After(afterFirst))
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.CompletedRequests)
	assert.Zero(t, state.TotalInputTokens)
	assert.Zero(t, state.TotalOutputTokens)
	assert.Empty(t, state.Endpoint)
	assert.Zero(t, state.ContextTokens)
	assert.Zero(t, state.InFlightTokens)
	assert.Zero(t, state.ContextSize())

	start := time.Now().Add(-2 * time.Minute)
	setRegistrySessionTimes(t, registry, "session-a", start, start)
	beforeSecond := time.Now()
	state = registry.GetState("session-a")
	afterSecond := time.Now()
	assert.GreaterOrEqual(t, state.Duration, beforeSecond.Sub(start))
	assert.LessOrEqual(t, state.Duration, afterSecond.Sub(start))
	assert.Equal(t, start, state.LastSeenAt)
}

func TestRegistryRecordsDispatchAndResponse(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "", "default/pod-a", 10)
	state := registry.GetState("session-a")
	assert.Equal(t, "session-a", state.SessionID)
	assert.Equal(t, "default/pod-a", state.Endpoint)
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Equal(t, int64(10), state.InFlightTokens)
	assert.Equal(t, int64(10), state.ContextSize())
	assert.Zero(t, state.CompletedRequests)

	registry.RecordResponse("session-a", "", true, TurnUsage{InputTokens: 12, OutputTokens: 34, TotalTokens: 46})
	state = registry.GetState("session-a")
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.InFlightTokens)
	assert.Equal(t, int64(1), state.CompletedRequests)
	assert.Equal(t, int64(12), state.TotalInputTokens)
	assert.Equal(t, int64(34), state.TotalOutputTokens)
	assert.Equal(t, int64(46), state.ContextTokens)
	assert.Equal(t, int64(46), state.ContextSize())
}

func TestRegistryRecordsAbnormalResponse(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "req-1", "default/pod-a", 150)
	registry.RecordResponse("session-a", "req-1", false, TurnUsage{InputTokens: 12, OutputTokens: 34})
	state := registry.GetState("session-a")
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.InFlightTokens)
	assert.Zero(t, state.CompletedRequests)
	assert.Zero(t, state.TotalInputTokens)
	assert.Zero(t, state.TotalOutputTokens)
	assert.Equal(t, int64(150), state.ContextTokens)

	registry.RecordResponse("session-a", "req-1", true, TurnUsage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3})
	state = registry.GetState("session-a")
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.InFlightTokens)
	assert.Zero(t, state.CompletedRequests)
	assert.Zero(t, state.TotalInputTokens)
	assert.Zero(t, state.TotalOutputTokens)
	assert.Equal(t, int64(150), state.ContextTokens)
}

func TestRegistrySessionsAreIsolated(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "", "", 0)
	other := registry.GetState("session-b")
	assert.Zero(t, other.TurnsTaken)
	assert.Zero(t, other.InFlightRequests)

	state := registry.GetState("session-a")
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Equal(t, 2, registrySessionCount(registry))
}

func TestRegistryEvictsIdleSessions(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "req-1", "default/pod-a", 50)
	ttl := time.Hour
	start := time.Now().Add(-2 * ttl)
	setRegistrySessionTimes(t, registry, "session-a", start, start)

	registry.EvictIdle(start.Add(ttl+time.Nanosecond), ttl)
	assert.Equal(t, 1, registrySessionCount(registry), "a session with an in-flight request must remain")

	registry.RecordResponse("session-a", "req-1", false, TurnUsage{InputTokens: 0, OutputTokens: 0})
	setRegistryLastResponseAt(t, registry, "session-a", start)
	registry.EvictIdle(start.Add(ttl), ttl)
	assert.Equal(t, 1, registrySessionCount(registry), "a session at the TTL boundary must remain")

	registry.EvictIdle(start.Add(ttl+time.Nanosecond), ttl)
	assert.Zero(t, registrySessionCount(registry))
	assert.Empty(t, registry.EndpointSessions("default/pod-a"))
	registry.mu.Lock()
	_, podStillIndexed := registry.byEndpoint["default/pod-a"]
	registry.mu.Unlock()
	assert.False(t, podStillIndexed, "evicting the last session on a pod must delete the empty inner map")

	state := registry.GetState("session-a")
	assert.Zero(t, state.TurnsTaken)
	assert.Zero(t, state.Duration)
	assert.Empty(t, state.Endpoint)
	assert.Zero(t, state.ContextTokens)
}

func TestRegistryPendingRequestIDMovesToNewerObservation(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.observe("session-a", "request-1")
	// The same request ID observed for another session moves its pending entry.
	registry.observe("session-b", "request-1")

	ttl := time.Hour
	start := time.Now().Add(-2 * ttl)
	setRegistrySessionTimes(t, registry, "session-a", start, start)
	setRegistrySessionTimes(t, registry, "session-b", start, start)
	registry.EvictIdle(time.Now(), ttl)
	assert.Equal(t, 1, registrySessionCount(registry), "only the session holding the pending entry must remain")

	registry.ReleasePending("request-1")
	registry.EvictIdle(time.Now(), ttl)
	assert.Zero(t, registrySessionCount(registry))
}

func TestRegistryEvictionCountsResponseActivity(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "", "", 0)
	ttl := time.Hour
	start := time.Now().Add(-2 * ttl)
	setRegistrySessionTimes(t, registry, "session-a", start, start)
	registry.RecordResponse("session-a", "", true, TurnUsage{})
	responseAt := start.Add(ttl)
	setRegistryLastResponseAt(t, registry, "session-a", responseAt)

	registry.EvictIdle(start.Add(ttl+time.Nanosecond), ttl)
	assert.Equal(t, 1, registrySessionCount(registry), "a response newer than the TTL must keep the session")

	registry.EvictIdle(responseAt.Add(ttl+time.Nanosecond), ttl)
	assert.Zero(t, registrySessionCount(registry))
}

func TestRegistryRecordsResponseTimeAndCachedTokens(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "", "", 0)
	registry.RecordDispatch("session-a", "", "", 0)

	before := time.Now()
	registry.RecordResponse("session-a", "", false, TurnUsage{InputTokens: 9, CachedTokens: 9})
	state := registry.GetState("session-a")
	assert.False(t, state.LastResponseAt.Before(before), "an abnormal end still records the response time")
	assert.Zero(t, state.TotalCachedTokens)

	registry.RecordResponse("session-a", "", true, TurnUsage{InputTokens: 10, OutputTokens: 2, CachedTokens: 8})
	state = registry.GetState("session-a")
	assert.Equal(t, int64(8), state.TotalCachedTokens)
	assert.Equal(t, int64(10), state.TotalInputTokens)
	assert.Equal(t, int64(12), state.ContextTokens)
}

func TestRegistryZeroTTLDisablesEviction(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.GetState("session-a")
	registry.EvictIdle(time.Now().Add(24*time.Hour), 0)
	assert.Equal(t, 1, registrySessionCount(registry))
}

func TestRegistryConcurrentUpdates(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	const requests = 100

	var wg sync.WaitGroup
	for range requests {
		wg.Add(1)
		go func() {
			defer wg.Done()
			registry.RecordDispatch("session-a", "", "default/pod-a", 10)
			registry.RecordResponse("session-a", "", true, TurnUsage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5})
		}()
	}
	wg.Wait()

	state := registry.GetState("session-a")
	assert.Equal(t, int64(requests), state.TurnsTaken)
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.InFlightTokens)
	assert.Equal(t, int64(requests), state.CompletedRequests)
	assert.Equal(t, int64(2*requests), state.TotalInputTokens)
	assert.Equal(t, int64(3*requests), state.TotalOutputTokens)
	assert.Equal(t, int64(5), state.ContextTokens)
}

func TestEstimatePromptTokensAndContextSize(t *testing.T) {
	t.Parallel()

	assert.Equal(t, int64(1), EstimatePromptTokens(-10))
	assert.Equal(t, int64(1), EstimatePromptTokens(0))
	assert.Equal(t, int64(1), EstimatePromptTokens(3))
	assert.Equal(t, int64(1), EstimatePromptTokens(4))
	assert.Equal(t, int64(100), EstimatePromptTokens(400))

	assert.Equal(t, int64(300), (SessionState{ContextTokens: 300, InFlightTokens: 200}).ContextSize())
	assert.Equal(t, int64(700), (SessionState{ContextTokens: 300, InFlightTokens: 700}).ContextSize())
}

func TestRegistryParallelInFlightSettlesIndependently(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "turn-1", "default/pod-a", 100)
	registry.RecordResponse("session-a", "turn-1", true, TurnUsage{TotalTokens: 300})

	// Two parallel requests overlap on the same session and settle out of order.
	registry.RecordDispatch("session-a", "big", "default/pod-a", 500)
	registry.RecordDispatch("session-a", "small", "default/pod-a", 100)

	state := registry.GetState("session-a")
	assert.Equal(t, int64(2), state.InFlightRequests)
	assert.Equal(t, int64(600), state.InFlightTokens)
	assert.Equal(t, int64(600), state.ContextSize())

	registry.RecordResponse("session-a", "small", false, TurnUsage{})
	state = registry.GetState("session-a")
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Equal(t, int64(500), state.InFlightTokens)
	assert.Equal(t, int64(300), state.ContextTokens, "small aborted turn must not shrink ContextTokens")
	assert.Equal(t, int64(500), state.ContextSize())

	registry.RecordResponse("session-a", "big", true, TurnUsage{TotalTokens: 520})
	state = registry.GetState("session-a")
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.InFlightTokens)
	assert.Equal(t, int64(520), state.ContextTokens)
	assert.Equal(t, int64(520), state.ContextSize())
}

func TestRegistryDuplicateDispatchDoesNotDoubleCharge(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "req-1", "default/pod-a", 120)
	// A duplicate dispatch for an already in-flight requestID must not
	// double-charge InFlightTokens or move the session's endpoint binding.
	registry.RecordDispatch("session-a", "req-1", "default/pod-b", 240)

	state := registry.GetState("session-a")
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Equal(t, int64(120), state.InFlightTokens)
	assert.Equal(t, "default/pod-a", state.Endpoint)
	assert.Empty(t, registry.EndpointSessions("default/pod-b"))
	require.Len(t, registry.EndpointSessions("default/pod-a"), 1)

	// An untracked response ID leaves the live in-flight request intact.
	registry.RecordResponse("session-a", "req-other", true, TurnUsage{TotalTokens: 999})
	state = registry.GetState("session-a")
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Equal(t, int64(120), state.InFlightTokens)

	registry.RecordResponse("session-a", "req-1", true, TurnUsage{TotalTokens: 200})
	state = registry.GetState("session-a")
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.InFlightTokens)
	assert.Equal(t, int64(200), state.ContextTokens)
}

func TestRegistryResponseDoesNotAffectOtherSession(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "req-from-a", "default/pod-a", 100)
	registry.RecordDispatch("session-b", "req-from-b", "default/pod-b", 200)

	// Settling session-b with session-a's requestID must not touch either session.
	registry.RecordResponse("session-b", "req-from-a", true, TurnUsage{TotalTokens: 999})
	stateA := registry.GetState("session-a")
	stateB := registry.GetState("session-b")
	assert.Equal(t, int64(1), stateA.InFlightRequests)
	assert.Equal(t, int64(100), stateA.InFlightTokens)
	assert.Zero(t, stateA.ContextTokens)
	assert.Equal(t, int64(1), stateB.InFlightRequests)
	assert.Equal(t, int64(200), stateB.InFlightTokens)
	assert.Zero(t, stateB.ContextTokens)

	// Each session's own requestID still settles normally.
	registry.RecordResponse("session-a", "req-from-a", true, TurnUsage{TotalTokens: 150})
	registry.RecordResponse("session-b", "req-from-b", true, TurnUsage{TotalTokens: 250})
	stateA = registry.GetState("session-a")
	stateB = registry.GetState("session-b")
	assert.Zero(t, stateA.InFlightRequests)
	assert.Zero(t, stateA.InFlightTokens)
	assert.Equal(t, int64(150), stateA.ContextTokens)
	assert.Zero(t, stateB.InFlightRequests)
	assert.Zero(t, stateB.InFlightTokens)
	assert.Equal(t, int64(250), stateB.ContextTokens)
}

func TestRegistryContextTokensFallbackOnEmptyUsage(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "req-1", "default/pod-a", 400)
	registry.RecordResponse("session-a", "req-1", true, TurnUsage{})
	state := registry.GetState("session-a")
	assert.Equal(t, int64(400), state.ContextTokens)

	// A smaller estimate on a turn without usage must not shrink ContextTokens.
	registry.RecordDispatch("session-a", "req-2", "default/pod-a", 150)
	registry.RecordResponse("session-a", "req-2", true, TurnUsage{})
	state = registry.GetState("session-a")
	assert.Equal(t, int64(400), state.ContextTokens)
}

func TestRegistryMovesSessionBetweenEndpointsAndRemovesEndpoint(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "req-1", "default/pod-a", 100)
	registry.RecordResponse("session-a", "req-1", true, TurnUsage{TotalTokens: 250})
	registry.RecordDispatch("session-b", "req-2", "default/pod-a", 80)

	assert.Len(t, registry.EndpointSessions("default/pod-a"), 2)

	// Move session-a from pod-a to pod-b.
	registry.RecordDispatch("session-a", "req-3", "default/pod-b", 300)
	assert.Equal(t, "default/pod-b", registry.GetState("session-a").Endpoint)

	podASessions := registry.EndpointSessions("default/pod-a")
	require.Len(t, podASessions, 1)
	assert.Equal(t, "session-b", podASessions[0].SessionID)

	podBSessions := registry.EndpointSessions("default/pod-b")
	require.Len(t, podBSessions, 1)
	assert.Equal(t, "session-a", podBSessions[0].SessionID)
	assert.Equal(t, int64(300), podBSessions[0].ContextSize())

	// Mutating the returned slice does not mutate registry state.
	podBSessions[0].ContextTokens = 9999
	assert.Equal(t, int64(250), registry.EndpointSessions("default/pod-b")[0].ContextTokens)

	// Removing pod-a clears session-b's Endpoint while preserving its history.
	registry.RemoveEndpoint("default/pod-a")
	assert.Empty(t, registry.EndpointSessions("default/pod-a"))
	stateB := registry.GetState("session-b")
	assert.Empty(t, stateB.Endpoint)
	assert.Equal(t, int64(1), stateB.TurnsTaken)
	assert.Equal(t, int64(80), stateB.InFlightTokens)
	assert.Len(t, registry.EndpointSessions("default/pod-b"), 1)
}
