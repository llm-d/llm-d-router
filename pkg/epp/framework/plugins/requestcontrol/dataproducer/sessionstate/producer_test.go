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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
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

func finalResponse(cause fwkrc.TerminationCause, inputTokens, outputTokens int) *fwkrc.Response {
	return &fwkrc.Response{
		EndOfStream:      true,
		TerminationCause: cause,
		Usage: fwkrh.Usage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
		},
	}
}

func TestFactoryAndProduces(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)

	assert.Equal(t, fwkplugin.TypedName{Type: SessionStateProducerType, Name: SessionStateProducerType}, producer.TypedName())
	expectedKey := SessionStateDataKey.WithNonEmptyProducerName(SessionStateProducerType)
	produced, ok := producer.Produces()[expectedKey]
	require.True(t, ok)
	assert.IsType(t, SessionState{}, produced)
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
}

func TestSessionIDAttributeAloneIsNoOp(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := &fwksched.InferenceRequest{}
	request.PutAttribute(attrsession.SessionIDDataKey, attrsession.SessionID("session-a"))

	require.NoError(t, producer.Produce(context.Background(), request, nil))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))

	_, published := ReadSessionState(request)
	assert.False(t, published)
}

func TestProduceAndPreRequestTrackSessionHistory(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	first := requestWithAgentIdentity("session-a")

	beforeFirst := time.Now()
	require.NoError(t, producer.Produce(context.Background(), first, nil))
	afterFirst := time.Now()
	state, ok := ReadSessionState(first)
	require.True(t, ok)
	assert.Equal(t, int64(0), state.TurnsTaken)
	assert.Zero(t, state.Duration)
	assert.False(t, state.LastSeenAt.Before(beforeFirst))
	assert.False(t, state.LastSeenAt.After(afterFirst))
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.CompletedRequests)
	assert.Zero(t, state.TotalInputTokens)
	assert.Zero(t, state.TotalOutputTokens)
	firstState := state

	require.NoError(t, producer.PreRequest(context.Background(), first, resultWithProfiles(1)))

	second := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), second, nil))
	state, ok = ReadSessionState(second)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.GreaterOrEqual(t, state.Duration, time.Duration(0))
	assert.Equal(t, firstState.LastSeenAt, state.LastSeenAt)
}

func TestResponseBodyTracksCompletedRequestsAndTokens(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), request, nil))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))

	producer.ResponseBody(context.Background(), request, &fwkrc.Response{EndOfStream: false}, nil)
	inFlight := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), inFlight, nil))
	state, ok := ReadSessionState(inFlight)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Zero(t, state.CompletedRequests)

	producer.ResponseBody(context.Background(), request, finalResponse(fwkrc.TerminationCauseNatural, 12, 34), nil)
	completed := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), completed, nil))
	state, ok = ReadSessionState(completed)
	require.True(t, ok)
	assert.Zero(t, state.InFlightRequests)
	assert.Equal(t, int64(1), state.CompletedRequests)
	assert.Equal(t, int64(12), state.TotalInputTokens)
	assert.Equal(t, int64(34), state.TotalOutputTokens)
}

func TestAbnormalTerminationDoesNotCompleteRequest(t *testing.T) {
	t.Parallel()

	causes := []fwkrc.TerminationCause{
		fwkrc.TerminationCauseClientDisconnect,
		fwkrc.TerminationCauseEvicted,
		fwkrc.TerminationCauseError,
	}
	for _, cause := range causes {
		t.Run(string(cause), func(t *testing.T) {
			t.Parallel()
			producer := newTestProducer(t)
			request := requestWithAgentIdentity("session-a")
			require.NoError(t, producer.Produce(context.Background(), request, nil))
			require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))

			producer.ResponseBody(context.Background(), request, finalResponse(cause, 12, 34), nil)
			next := requestWithAgentIdentity("session-a")
			require.NoError(t, producer.Produce(context.Background(), next, nil))
			state, ok := ReadSessionState(next)
			require.True(t, ok)
			assert.Zero(t, state.InFlightRequests)
			assert.Zero(t, state.CompletedRequests)
			assert.Zero(t, state.TotalInputTokens)
			assert.Zero(t, state.TotalOutputTokens)
		})
	}
}

func TestNaturalCompletionWithoutUsage(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), request, nil))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))
	producer.ResponseBody(context.Background(), request, finalResponse(fwkrc.TerminationCauseNatural, 0, 0), nil)

	next := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), next, nil))
	state, ok := ReadSessionState(next)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.CompletedRequests)
	assert.Zero(t, state.TotalInputTokens)
	assert.Zero(t, state.TotalOutputTokens)
}

func TestResponseBodyWithoutTrackedDispatchIsNoOp(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), request, nil))

	producer.ResponseBody(context.Background(), nil, finalResponse(fwkrc.TerminationCauseNatural, 1, 2), nil)
	producer.ResponseBody(context.Background(), &fwksched.InferenceRequest{}, finalResponse(fwkrc.TerminationCauseNatural, 1, 2), nil)
	producer.ResponseBody(context.Background(), request, nil, nil)
	producer.ResponseBody(context.Background(), request, finalResponse(fwkrc.TerminationCauseNatural, 1, 2), nil)

	next := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), next, nil))
	state, ok := ReadSessionState(next)
	require.True(t, ok)
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.CompletedRequests)
	assert.Zero(t, state.TotalInputTokens)
	assert.Zero(t, state.TotalOutputTokens)
}

func TestPreRequestCountsMultipleProfilesOnce(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), request, nil))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(2)))

	next := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), next, nil))
	state, ok := ReadSessionState(next)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.TurnsTaken)
}
