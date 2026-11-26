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
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

func registrySessionCount(registry *SessionStateRegistry) int {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	return len(registry.sessions)
}

func registryBucketSessions(registry *SessionStateRegistry, endpoint string) []SessionState {
	registry.mu.Lock()
	bucket := registry.byEndpoint[endpoint]
	registry.mu.Unlock()
	if bucket == nil {
		return nil
	}
	return bucket.sessions()
}

func setRegistrySessionTimes(t *testing.T, registry *SessionStateRegistry, identity string, firstSeenAt, lastSeenAt time.Time) {
	t.Helper()
	registry.mu.Lock()
	defer registry.mu.Unlock()
	record, ok := registry.sessions[identity]
	require.True(t, ok)
	record.firstSeenAt = firstSeenAt
	record.state.LastSeenAt = lastSeenAt
	registry.syncRecordLocked(identity, record)
}

func setRegistryLastResponseAt(t *testing.T, registry *SessionStateRegistry, identity string, lastResponseAt time.Time) {
	t.Helper()
	registry.mu.Lock()
	defer registry.mu.Unlock()
	record, ok := registry.sessions[identity]
	require.True(t, ok)
	record.state.LastResponseAt = lastResponseAt
	registry.syncRecordLocked(identity, record)
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
	registry.RecordDispatch("session-a", "req-1", "default/pod-a", 10)
	state := registry.GetState("session-a")
	assert.Equal(t, "session-a", state.SessionID)
	assert.Equal(t, "default/pod-a", state.Endpoint)
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Equal(t, int64(10), state.InFlightTokens)
	assert.Equal(t, int64(10), state.ContextSize())
	assert.Zero(t, state.CompletedRequests)

	registry.RecordResponse("session-a", "req-1", true, TurnUsage{InputTokens: 12, OutputTokens: 34, TotalTokens: 46})
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

func TestRegistryIgnoresEmptyIdentityOrRequestID(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("", "req-1", "default/pod-a", 100)
	registry.RecordDispatch("session-a", "", "default/pod-a", 100)
	assert.Zero(t, registrySessionCount(registry))
	assert.Empty(t, registryBucketSessions(registry, "default/pod-a"))

	registry.RecordDispatch("session-a", "req-1", "default/pod-a", 100)
	registry.RecordResponse("", "req-1", true, TurnUsage{TotalTokens: 500})
	registry.RecordResponse("session-a", "", true, TurnUsage{TotalTokens: 500})
	state := registry.GetState("session-a")
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Equal(t, int64(100), state.InFlightTokens)
	assert.Zero(t, state.ContextTokens)
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
	registry.RecordDispatch("session-a", "req-1", "", 0)
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
	assert.Empty(t, registryBucketSessions(registry, "default/pod-a"))
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
	registry.RecordDispatch("session-a", "req-1", "", 0)
	ttl := time.Hour
	start := time.Now().Add(-2 * ttl)
	setRegistrySessionTimes(t, registry, "session-a", start, start)
	registry.RecordResponse("session-a", "req-1", true, TurnUsage{})
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
	registry.RecordDispatch("session-a", "req-1", "", 0)
	registry.RecordDispatch("session-a", "req-2", "", 0)

	before := time.Now()
	registry.RecordResponse("session-a", "req-1", false, TurnUsage{InputTokens: 9, CachedTokens: 9})
	state := registry.GetState("session-a")
	assert.False(t, state.LastResponseAt.Before(before), "an abnormal end still records the response time")
	assert.Zero(t, state.TotalCachedTokens)

	registry.RecordResponse("session-a", "req-2", true, TurnUsage{InputTokens: 10, OutputTokens: 2, CachedTokens: 8})
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
	for i := range requests {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			reqID := fmt.Sprintf("req-%d", idx)
			registry.RecordDispatch("session-a", reqID, "default/pod-a", 10)
			registry.RecordResponse("session-a", reqID, true, TurnUsage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5})
		}(i)
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
	assert.Empty(t, registryBucketSessions(registry, "default/pod-b"))
	require.Len(t, registryBucketSessions(registry, "default/pod-a"), 1)

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

func TestRegistryMovesSessionBetweenEndpointsAndUnregistersEndpoint(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	epA := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-a"}}, nil)
	epB := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-b"}}, nil)
	viewA := registry.RegisterEndpoint("default/pod-a", epA)
	viewB := registry.RegisterEndpoint("default/pod-b", epB)

	registry.RecordDispatch("session-a", "req-1", "default/pod-a", 100)
	registry.RecordResponse("session-a", "req-1", true, TurnUsage{TotalTokens: 250})
	registry.RecordDispatch("session-b", "req-2", "default/pod-a", 80)

	assert.Len(t, viewA.Sessions(), 2)

	// Move session-a from pod-a to pod-b.
	registry.RecordDispatch("session-a", "req-3", "default/pod-b", 300)
	assert.Equal(t, "default/pod-b", registry.GetState("session-a").Endpoint)

	podASessions := viewA.Sessions()
	require.Len(t, podASessions, 1)
	assert.Equal(t, "session-b", podASessions[0].SessionID)

	podBSessions := viewB.Sessions()
	require.Len(t, podBSessions, 1)
	assert.Equal(t, "session-a", podBSessions[0].SessionID)
	assert.Equal(t, int64(300), podBSessions[0].ContextSize())

	// Unregistering pod-a clears session-b's Endpoint while preserving its history.
	assert.True(t, registry.UnregisterEndpoint("default/pod-a", epA))
	assert.Empty(t, viewA.Sessions())
	stateB := registry.GetState("session-b")
	assert.Empty(t, stateB.Endpoint)
	assert.Equal(t, int64(1), stateB.TurnsTaken)
	assert.Equal(t, int64(80), stateB.InFlightTokens)
	assert.Len(t, viewB.Sessions(), 1)
}

func TestRegistryAbnormalResponseIgnoresPartialTotalTokens(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	registry.RecordDispatch("session-a", "req-1", "default/pod-a", 200)
	registry.RecordResponse("session-a", "req-1", true, TurnUsage{TotalTokens: 500})

	// An aborted stream carrying partial TotalTokens must not shrink ContextTokens.
	registry.RecordDispatch("session-a", "req-2", "default/pod-a", 150)
	registry.RecordResponse("session-a", "req-2", false, TurnUsage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15})
	state := registry.GetState("session-a")
	assert.Equal(t, int64(500), state.ContextTokens)
	assert.Equal(t, int64(1), state.CompletedRequests)

	// An aborted stream with a larger prompt estimate raises ContextTokens to the estimate.
	registry.RecordDispatch("session-a", "req-3", "default/pod-a", 700)
	registry.RecordResponse("session-a", "req-3", false, TurnUsage{TotalTokens: 20})
	state = registry.GetState("session-a")
	assert.Equal(t, int64(700), state.ContextTokens)
}

func TestRegistryEvictIdleAndRebindUpdateRegisteredViews(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	epA := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-a"}}, nil)
	epB := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-b"}}, nil)
	viewA := registry.RegisterEndpoint("default/pod-a", epA)
	viewB := registry.RegisterEndpoint("default/pod-b", epB)

	registry.RecordDispatch("s1", "r1", "default/pod-a", 100)
	registry.RecordResponse("s1", "r1", true, TurnUsage{TotalTokens: 300})
	registry.RecordDispatch("s2", "r2", "default/pod-a", 100)
	registry.RecordResponse("s2", "r2", true, TurnUsage{TotalTokens: 200})
	registry.RecordDispatch("s3", "r3", "default/pod-a", 100)
	registry.RecordResponse("s3", "r3", true, TurnUsage{TotalTokens: 150})

	now := time.Now()
	t1 := now.Add(-3 * time.Hour)
	t2 := now.Add(-2 * time.Hour)
	t3 := now.Add(-10 * time.Minute)
	setRegistrySessionTimes(t, registry, "s1", t1, t1)
	setRegistryLastResponseAt(t, registry, "s1", t1)
	setRegistrySessionTimes(t, registry, "s2", t2, t2)
	setRegistryLastResponseAt(t, registry, "s2", t2)
	setRegistrySessionTimes(t, registry, "s3", t3, t3)
	setRegistryLastResponseAt(t, registry, "s3", t3)

	assert.Equal(t, int64(650), viewA.TotalContextTokens())

	// Evicting s1 updates viewA's totals and lookup.
	registry.EvictIdle(now, 150*time.Minute)
	assert.Equal(t, int64(350), viewA.TotalContextTokens())
	assert.Len(t, viewA.Sessions(), 2)
	_, found := viewA.Session("s1")
	assert.False(t, found)

	// Rebinding s2 to pod-b updates both pod-a and pod-b views.
	registry.RecordDispatch("s2", "r2b", "default/pod-b", 250)
	registry.RecordResponse("s2", "r2b", true, TurnUsage{TotalTokens: 260})
	assert.Equal(t, int64(150), viewA.TotalContextTokens())
	assert.Equal(t, int64(260), viewB.TotalContextTokens())

	// Evicting the remaining session clears viewA's aggregates while keeping the registered view.
	registry.EvictIdle(now.Add(2*time.Hour), time.Hour)
	assert.Empty(t, viewA.Sessions())
	assert.Zero(t, viewA.TotalContextTokens())
}

func TestRegistryDispatchToUnregisteredEndpointKeepsExistingBinding(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	epA := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-a"}}, nil)
	viewA := registry.RegisterEndpoint("default/pod-a", epA)

	registry.RecordDispatch("s1", "r1", "default/pod-a", 100)
	registry.RecordResponse("s1", "r1", true, TurnUsage{TotalTokens: 200})

	registry.RecordDispatch("s1", "r2", "default/pod-unregistered", 300)
	state := registry.GetState("s1")
	assert.Equal(t, "default/pod-a", state.Endpoint)
	require.Len(t, viewA.Sessions(), 1)
	assert.Equal(t, int64(300), viewA.TotalContextTokens())
}

func TestRegistryConcurrentReadsAndLifecycle(t *testing.T) {
	t.Parallel()

	registry := &SessionStateRegistry{}
	epA := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-a"}}, nil)
	epB := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-b"}}, nil)
	viewA := registry.RegisterEndpoint("default/pod-a", epA)
	viewB := registry.RegisterEndpoint("default/pod-b", epB)
	var curViewB atomic.Pointer[EndpointSessions]
	curViewB.Store(&viewB)

	stop := make(chan struct{})
	var writerWG, readerWG sync.WaitGroup

	for w := range 4 {
		writerWG.Add(1)
		go func(worker int) {
			defer writerWG.Done()
			for i := range 100 {
				sid := fmt.Sprintf("session-%d", (worker*17+i)%12)
				reqID := fmt.Sprintf("req-%d-%d", worker, i)
				ep := "default/pod-a"
				if i%2 == 1 {
					ep = "default/pod-b"
				}
				obs := registry.observe(sid, reqID)
				_ = obs.state()
				if i%5 == 0 {
					registry.ReleasePending(reqID)
					continue
				}
				registry.RecordDispatch(sid, reqID, ep, 100+int64(i))
				registry.RecordResponse(sid, reqID, i%3 != 0, TurnUsage{
					InputTokens:  80,
					OutputTokens: 20,
					TotalTokens:  100 + int64(i),
				})
				if i%20 == 0 {
					registry.EvictIdle(time.Now().Add(2*time.Hour), time.Hour)
				}
			}
		}(w)
	}

	writerWG.Add(1)
	go func() {
		defer writerWG.Done()
		cur := epB
		for i := range 20 {
			next := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-b"}}, nil)
			v := registry.RegisterEndpoint("default/pod-b", next)
			curViewB.Store(&v)
			if i%3 == 0 {
				registry.UnregisterEndpoint("default/pod-b", cur)
			}
			cur = next
		}
	}()

	for range 4 {
		readerWG.Add(1)
		go func() {
			defer readerWG.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				for _, v := range []EndpointSessions{viewA, viewB, *curViewB.Load()} {
					_ = v.TotalContextTokens()
					_ = v.Sessions()
					_, _ = v.Session("session-0")
				}
			}
		}()
	}

	writerWG.Wait()
	close(stop)
	readerWG.Wait()
}
