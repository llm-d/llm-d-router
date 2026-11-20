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

func setRegistrySessionTimes(t *testing.T, registry *SessionStateRegistry, key SessionKey, firstSeenAt, lastSeenAt time.Time) {
	t.Helper()
	registry.mu.Lock()
	defer registry.mu.Unlock()
	record, ok := registry.sessions[key]
	require.True(t, ok)
	record.firstSeenAt = firstSeenAt
	record.state.LastSeenAt = lastSeenAt
}

func setRegistryLastResponseAt(t *testing.T, registry *SessionStateRegistry, key SessionKey, lastResponseAt time.Time) {
	t.Helper()
	registry.mu.Lock()
	defer registry.mu.Unlock()
	record, ok := registry.sessions[key]
	require.True(t, ok)
	record.state.LastResponseAt = lastResponseAt
}

func TestRegistryGetState(t *testing.T) {
	t.Parallel()

	key := SessionKey{SessionID: "session-a"}
	registry := &SessionStateRegistry{}
	beforeFirst := time.Now()
	state := registry.GetState(key)
	afterFirst := time.Now()
	assert.Equal(t, key, state.Key)
	assert.Zero(t, state.TurnsTaken)
	assert.Zero(t, state.Duration)
	assert.False(t, state.LastSeenAt.Before(beforeFirst))
	assert.False(t, state.LastSeenAt.After(afterFirst))
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.CompletedRequests)
	assert.Zero(t, state.TotalInputTokens)
	assert.Zero(t, state.TotalOutputTokens)

	start := time.Now().Add(-2 * time.Minute)
	setRegistrySessionTimes(t, registry, key, start, start)
	beforeSecond := time.Now()
	state = registry.GetState(key)
	afterSecond := time.Now()
	assert.Equal(t, key, state.Key)
	assert.GreaterOrEqual(t, state.Duration, beforeSecond.Sub(start))
	assert.LessOrEqual(t, state.Duration, afterSecond.Sub(start))
	assert.Equal(t, start, state.LastSeenAt)
}

func TestRegistryRecordsDispatchAndResponse(t *testing.T) {
	t.Parallel()

	key := SessionKey{SessionID: "session-a"}
	registry := &SessionStateRegistry{}
	registry.RecordDispatch(key, "")
	state := registry.GetState(key)
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Zero(t, state.CompletedRequests)

	registry.RecordResponse(key, true, TurnUsage{InputTokens: 12, OutputTokens: 34})
	state = registry.GetState(key)
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Zero(t, state.InFlightRequests)
	assert.Equal(t, int64(1), state.CompletedRequests)
	assert.Equal(t, int64(12), state.TotalInputTokens)
	assert.Equal(t, int64(34), state.TotalOutputTokens)
}

func TestRegistryRecordsAbnormalResponse(t *testing.T) {
	t.Parallel()

	key := SessionKey{SessionID: "session-a"}
	registry := &SessionStateRegistry{}
	registry.RecordDispatch(key, "")
	registry.RecordResponse(key, false, TurnUsage{InputTokens: 12, OutputTokens: 34})
	state := registry.GetState(key)
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.CompletedRequests)
	assert.Zero(t, state.TotalInputTokens)
	assert.Zero(t, state.TotalOutputTokens)

	registry.RecordResponse(key, true, TurnUsage{InputTokens: 1, OutputTokens: 2})
	state = registry.GetState(key)
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.CompletedRequests)
	assert.Zero(t, state.TotalInputTokens)
	assert.Zero(t, state.TotalOutputTokens)
}

func TestRegistrySessionsAreIsolated(t *testing.T) {
	t.Parallel()

	keyA := SessionKey{SessionID: "session-a"}
	keyB := SessionKey{SessionID: "session-b"}
	registry := &SessionStateRegistry{}
	registry.RecordDispatch(keyA, "")
	other := registry.GetState(keyB)
	assert.Equal(t, keyB, other.Key)
	assert.Zero(t, other.TurnsTaken)
	assert.Zero(t, other.InFlightRequests)

	state := registry.GetState(keyA)
	assert.Equal(t, keyA, state.Key)
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Equal(t, 2, registrySessionCount(registry))
}

func TestRegistryTenantsWithSameSessionIDAreIsolated(t *testing.T) {
	t.Parallel()

	t1 := SessionKey{TenantID: "t1", SessionID: "s"}
	t2 := SessionKey{TenantID: "t2", SessionID: "s"}
	noTenant := SessionKey{TenantID: "", SessionID: "s"}
	slashTenant := SessionKey{TenantID: "a/b", SessionID: "c"}
	slashSession := SessionKey{TenantID: "a", SessionID: "b/c"}

	registry := &SessionStateRegistry{}
	registry.RecordDispatch(t1, "")
	registry.RecordDispatch(t1, "")
	registry.RecordResponse(t1, true, TurnUsage{InputTokens: 10, OutputTokens: 20, CachedTokens: 5})

	registry.RecordDispatch(t2, "")
	registry.RecordDispatch(slashTenant, "")

	stateT1 := registry.GetState(t1)
	assert.Equal(t, t1, stateT1.Key)
	assert.Equal(t, int64(2), stateT1.TurnsTaken)
	assert.Equal(t, int64(1), stateT1.InFlightRequests)
	assert.Equal(t, int64(1), stateT1.CompletedRequests)
	assert.Equal(t, int64(10), stateT1.TotalInputTokens)
	assert.Equal(t, int64(20), stateT1.TotalOutputTokens)
	assert.Equal(t, int64(5), stateT1.TotalCachedTokens)

	stateT2 := registry.GetState(t2)
	assert.Equal(t, t2, stateT2.Key)
	assert.Equal(t, int64(1), stateT2.TurnsTaken)
	assert.Equal(t, int64(1), stateT2.InFlightRequests)
	assert.Zero(t, stateT2.CompletedRequests)
	assert.Zero(t, stateT2.TotalInputTokens)

	stateNoTenant := registry.GetState(noTenant)
	assert.Equal(t, noTenant, stateNoTenant.Key)
	assert.Zero(t, stateNoTenant.TurnsTaken)
	assert.Zero(t, stateNoTenant.InFlightRequests)

	stateSlashSession := registry.GetState(slashSession)
	assert.Equal(t, slashSession, stateSlashSession.Key)
	assert.Zero(t, stateSlashSession.TurnsTaken)

	assert.Equal(t, 5, registrySessionCount(registry))
}

func TestRegistryEvictsIdleSessionsPerTenant(t *testing.T) {
	t.Parallel()

	t1 := SessionKey{TenantID: "t1", SessionID: "s"}
	t2 := SessionKey{TenantID: "t2", SessionID: "s"}
	noTenant := SessionKey{TenantID: "", SessionID: "s"}

	registry := &SessionStateRegistry{}
	registry.observe(t1, "req-t1")
	registry.observe(t2, "")
	registry.observe(noTenant, "")

	ttl := time.Hour
	start := time.Now().Add(-2 * ttl)
	setRegistrySessionTimes(t, registry, t1, start, start)
	setRegistrySessionTimes(t, registry, t2, start, start)
	setRegistrySessionTimes(t, registry, noTenant, start, start)

	registry.EvictIdle(time.Now(), ttl)
	assert.Equal(t, 1, registrySessionCount(registry))
	assert.Equal(t, t1, registry.GetState(t1).Key)

	setRegistrySessionTimes(t, registry, t1, start, start)
	registry.ReleasePending("req-t1")
	registry.EvictIdle(time.Now(), ttl)
	assert.Zero(t, registrySessionCount(registry))
}

func TestRegistryEvictsIdleSessions(t *testing.T) {
	t.Parallel()

	key := SessionKey{SessionID: "session-a"}
	registry := &SessionStateRegistry{}
	registry.RecordDispatch(key, "")
	ttl := time.Hour
	start := time.Now().Add(-2 * ttl)
	setRegistrySessionTimes(t, registry, key, start, start)

	registry.EvictIdle(start.Add(ttl+time.Nanosecond), ttl)
	assert.Equal(t, 1, registrySessionCount(registry), "a session with an in-flight request must remain")

	registry.RecordResponse(key, false, TurnUsage{InputTokens: 0, OutputTokens: 0})
	setRegistryLastResponseAt(t, registry, key, start)
	registry.EvictIdle(start.Add(ttl), ttl)
	assert.Equal(t, 1, registrySessionCount(registry), "a session at the TTL boundary must remain")

	registry.EvictIdle(start.Add(ttl+time.Nanosecond), ttl)
	assert.Zero(t, registrySessionCount(registry))
	state := registry.GetState(key)
	assert.Zero(t, state.TurnsTaken)
	assert.Zero(t, state.Duration)
}

func TestRegistryPendingRequestIDMovesToNewerObservation(t *testing.T) {
	t.Parallel()

	keyA := SessionKey{SessionID: "session-a"}
	keyB := SessionKey{SessionID: "session-b"}
	registry := &SessionStateRegistry{}
	registry.observe(keyA, "request-1")
	// The same request ID observed for another session moves its pending entry.
	registry.observe(keyB, "request-1")

	ttl := time.Hour
	start := time.Now().Add(-2 * ttl)
	setRegistrySessionTimes(t, registry, keyA, start, start)
	setRegistrySessionTimes(t, registry, keyB, start, start)
	registry.EvictIdle(time.Now(), ttl)
	assert.Equal(t, 1, registrySessionCount(registry), "only the session holding the pending entry must remain")

	registry.ReleasePending("request-1")
	registry.EvictIdle(time.Now(), ttl)
	assert.Zero(t, registrySessionCount(registry))
}

func TestRegistryEvictionCountsResponseActivity(t *testing.T) {
	t.Parallel()

	key := SessionKey{SessionID: "session-a"}
	registry := &SessionStateRegistry{}
	registry.RecordDispatch(key, "")
	ttl := time.Hour
	start := time.Now().Add(-2 * ttl)
	setRegistrySessionTimes(t, registry, key, start, start)
	registry.RecordResponse(key, true, TurnUsage{})
	responseAt := start.Add(ttl)
	setRegistryLastResponseAt(t, registry, key, responseAt)

	registry.EvictIdle(start.Add(ttl+time.Nanosecond), ttl)
	assert.Equal(t, 1, registrySessionCount(registry), "a response newer than the TTL must keep the session")

	registry.EvictIdle(responseAt.Add(ttl+time.Nanosecond), ttl)
	assert.Zero(t, registrySessionCount(registry))
}

func TestRegistryRecordsResponseTimeAndCachedTokens(t *testing.T) {
	t.Parallel()

	key := SessionKey{SessionID: "session-a"}
	registry := &SessionStateRegistry{}
	registry.RecordDispatch(key, "")
	registry.RecordDispatch(key, "")

	before := time.Now()
	registry.RecordResponse(key, false, TurnUsage{InputTokens: 9, CachedTokens: 9})
	state := registry.GetState(key)
	assert.False(t, state.LastResponseAt.Before(before), "an abnormal end still records the response time")
	assert.Zero(t, state.TotalCachedTokens)

	registry.RecordResponse(key, true, TurnUsage{InputTokens: 10, OutputTokens: 2, CachedTokens: 8})
	state = registry.GetState(key)
	assert.Equal(t, int64(8), state.TotalCachedTokens)
	assert.Equal(t, int64(10), state.TotalInputTokens)
}

func TestRegistryZeroTTLDisablesEviction(t *testing.T) {
	t.Parallel()

	key := SessionKey{SessionID: "session-a"}
	registry := &SessionStateRegistry{}
	registry.GetState(key)
	registry.EvictIdle(time.Now().Add(24*time.Hour), 0)
	assert.Equal(t, 1, registrySessionCount(registry))
}

func TestRegistryConcurrentUpdates(t *testing.T) {
	t.Parallel()

	key := SessionKey{SessionID: "session-a"}
	registry := &SessionStateRegistry{}
	const requests = 100

	var wg sync.WaitGroup
	for range requests {
		wg.Add(1)
		go func() {
			defer wg.Done()
			registry.RecordDispatch(key, "")
			registry.RecordResponse(key, true, TurnUsage{InputTokens: 2, OutputTokens: 3})
		}()
	}
	wg.Wait()

	state := registry.GetState(key)
	assert.Equal(t, int64(requests), state.TurnsTaken)
	assert.Zero(t, state.InFlightRequests)
	assert.Equal(t, int64(requests), state.CompletedRequests)
	assert.Equal(t, int64(2*requests), state.TotalInputTokens)
	assert.Equal(t, int64(3*requests), state.TotalOutputTokens)
}
