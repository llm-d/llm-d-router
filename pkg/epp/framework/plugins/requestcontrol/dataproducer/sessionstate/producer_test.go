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
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

func newTestProducer(t *testing.T) *Producer {
	t.Helper()
	plg, err := Factory(SessionStateProducerType, nil, nil)
	require.NoError(t, err)
	producer, ok := plg.(*Producer)
	require.True(t, ok)
	return producer
}

func requestWithAgentIdentity(agentIdentity string) *fwksched.InferenceRequest {
	request := &fwksched.InferenceRequest{}
	request.PutAttribute(agentidentity.AgentIdentityKey, agentIdentity)
	return request
}

func resultWithProfiles(profileCount int) *fwksched.SchedulingResult {
	profiles := make(map[string]*fwksched.ProfileRunResult, profileCount)
	for i := range profileCount {
		profiles[string(rune('a'+i))] = &fwksched.ProfileRunResult{
			TargetEndpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, fwkdl.NewMetrics(), fwkdl.NewAttributes()),
			},
		}
	}
	return &fwksched.SchedulingResult{ProfileResults: profiles}
}

func sessionCount(producer *Producer) int {
	producer.mu.Lock()
	defer producer.mu.Unlock()
	return len(producer.sessions)
}

func setSessionTimes(t *testing.T, producer *Producer, identity string, firstSeenAt, lastSeenAt time.Time) {
	t.Helper()
	producer.mu.Lock()
	defer producer.mu.Unlock()
	record, ok := producer.sessions[identity]
	require.True(t, ok)
	record.firstSeenAt = firstSeenAt
	record.lastSeenAt = lastSeenAt
}

func TestFactoryAndProduces(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)

	assert.Equal(t, fwkplugin.TypedName{Type: SessionStateProducerType, Name: SessionStateProducerType}, producer.TypedName())
	expectedKey := attrsession.SessionStateDataKey.WithNonEmptyProducerName(SessionStateProducerType)
	produced, ok := producer.Produces()[expectedKey]
	require.True(t, ok)
	assert.IsType(t, attrsession.SessionState{}, produced)
	assert.Equal(t, defaultEvictionTTL, producer.evictionTTL)
	assert.Equal(t, defaultEvictionSweepInterval, producer.evictionSweepInterval)
}

func TestFactoryParameters(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		parameters      json.RawMessage
		wantTTL         time.Duration
		wantSweep       time.Duration
		wantErrContains string
	}{
		{
			name:       "custom durations",
			parameters: json.RawMessage(`{"evictionTtlSeconds":12.5,"evictionSweepSeconds":1.5}`),
			wantTTL:    12500 * time.Millisecond,
			wantSweep:  1500 * time.Millisecond,
		},
		{
			name:       "zero ttl disables eviction",
			parameters: json.RawMessage(`{"evictionTtlSeconds":0,"evictionSweepSeconds":10}`),
			wantTTL:    0,
			wantSweep:  10 * time.Second,
		},
		{
			name:            "negative ttl",
			parameters:      json.RawMessage(`{"evictionTtlSeconds":-1,"evictionSweepSeconds":10}`),
			wantErrContains: "evictionTtlSeconds must be >= 0",
		},
		{
			name:            "zero sweep",
			parameters:      json.RawMessage(`{"evictionSweepSeconds":0}`),
			wantErrContains: "evictionSweepSeconds must be > 0",
		},
		{
			name:            "negative sweep",
			parameters:      json.RawMessage(`{"evictionSweepSeconds":-1}`),
			wantErrContains: "evictionSweepSeconds must be > 0",
		},
		{
			name:            "unknown field",
			parameters:      json.RawMessage(`{"unknown":1}`),
			wantErrContains: "unknown field",
		},
		{
			name:            "invalid json",
			parameters:      json.RawMessage(`not-json`),
			wantErrContains: "invalid config",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			plg, err := Factory("custom", fwkplugin.StrictDecoder(test.parameters), nil)
			if test.wantErrContains != "" {
				require.ErrorContains(t, err, test.wantErrContains)
				return
			}
			require.NoError(t, err)
			producer, ok := plg.(*Producer)
			require.True(t, ok)
			assert.Equal(t, test.wantTTL, producer.evictionTTL)
			assert.Equal(t, test.wantSweep, producer.evictionSweepInterval)
		})
	}
}

func TestProduceWithoutAgentIdentityIsNoOp(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := &fwksched.InferenceRequest{}

	require.NoError(t, producer.Produce(context.Background(), nil, nil))
	require.NoError(t, producer.Produce(context.Background(), request, nil))
	require.NoError(t, producer.Produce(context.Background(), requestWithAgentIdentity(""), nil))

	assert.Empty(t, request.AttributeKeys())
	assert.Zero(t, sessionCount(producer))
}

func TestSessionIDAttributeAloneIsNoOp(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := &fwksched.InferenceRequest{}
	request.PutAttribute(attrsession.SessionIDDataKey, attrsession.SessionID("session-a"))

	require.NoError(t, producer.Produce(context.Background(), request, nil))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))

	_, published := attrsession.ReadSessionState(request)
	assert.False(t, published)
	assert.Zero(t, sessionCount(producer))
}

func TestProduceAndPreRequestTrackSessionHistory(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	first := requestWithAgentIdentity("session-a")

	beforeFirst := time.Now()
	require.NoError(t, producer.Produce(context.Background(), first, nil))
	afterFirst := time.Now()
	state, ok := attrsession.ReadSessionState(first)
	require.True(t, ok)
	assert.Equal(t, int64(0), state.TurnsTaken)
	assert.Zero(t, state.Duration)
	assert.False(t, state.LastSeenAt.Before(beforeFirst))
	assert.False(t, state.LastSeenAt.After(afterFirst))

	require.NoError(t, producer.PreRequest(context.Background(), first, resultWithProfiles(1)))

	start := time.Now().Add(-2 * time.Minute)
	setSessionTimes(t, producer, "session-a", start, start)
	second := requestWithAgentIdentity("session-a")
	beforeSecond := time.Now()
	require.NoError(t, producer.Produce(context.Background(), second, nil))
	afterSecond := time.Now()
	state, ok = attrsession.ReadSessionState(second)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.GreaterOrEqual(t, state.Duration, beforeSecond.Sub(start))
	assert.LessOrEqual(t, state.Duration, afterSecond.Sub(start))
	assert.Equal(t, start, state.LastSeenAt)
}

func TestPreRequestCountsMultipleProfilesOnce(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), request, nil))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(2)))

	next := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), next, nil))
	state, ok := attrsession.ReadSessionState(next)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.TurnsTaken)
}

func TestSessionsAreIsolated(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	first := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), first, nil))
	require.NoError(t, producer.PreRequest(context.Background(), first, resultWithProfiles(1)))

	other := requestWithAgentIdentity("session-b")
	require.NoError(t, producer.Produce(context.Background(), other, nil))
	otherState, ok := attrsession.ReadSessionState(other)
	require.True(t, ok)
	assert.Equal(t, int64(0), otherState.TurnsTaken)

	next := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), next, nil))
	state, ok := attrsession.ReadSessionState(next)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Equal(t, 2, sessionCount(producer))
}

func TestEvictIdleSession(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), request, nil))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))
	start := time.Now().Add(-2 * producer.evictionTTL)
	setSessionTimes(t, producer, "session-a", start, start)

	producer.evictIdle(start.Add(producer.evictionTTL))
	assert.Equal(t, 1, sessionCount(producer), "a session at the TTL boundary must remain")

	now := start.Add(producer.evictionTTL + time.Nanosecond)
	producer.evictIdle(now)
	assert.Zero(t, sessionCount(producer))

	next := requestWithAgentIdentity("session-a")
	before := time.Now()
	require.NoError(t, producer.Produce(context.Background(), next, nil))
	after := time.Now()
	state, ok := attrsession.ReadSessionState(next)
	require.True(t, ok)
	assert.Equal(t, int64(0), state.TurnsTaken)
	assert.Zero(t, state.Duration)
	assert.False(t, state.LastSeenAt.Before(before))
	assert.False(t, state.LastSeenAt.After(after))
}

func TestZeroTTLDisablesEviction(t *testing.T) {
	t.Parallel()

	plg, err := Factory(
		SessionStateProducerType,
		fwkplugin.StrictDecoder(json.RawMessage(`{"evictionTtlSeconds":0,"evictionSweepSeconds":1}`)),
		nil,
	)
	require.NoError(t, err)
	producer, ok := plg.(*Producer)
	require.True(t, ok)

	request := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), request, nil))

	producer.evictIdle(time.Now().Add(24 * time.Hour))
	assert.Equal(t, 1, sessionCount(producer))
}

func TestConcurrentDispatches(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	result := resultWithProfiles(1)
	const requests = 100

	var wg sync.WaitGroup
	for range requests {
		wg.Add(1)
		go func() {
			defer wg.Done()
			request := requestWithAgentIdentity("session-a")
			require.NoError(t, producer.Produce(context.Background(), request, nil))
			require.NoError(t, producer.PreRequest(context.Background(), request, result))
		}()
	}
	wg.Wait()

	next := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), next, nil))
	state, ok := attrsession.ReadSessionState(next)
	require.True(t, ok)
	assert.Equal(t, int64(requests), state.TurnsTaken)
}
