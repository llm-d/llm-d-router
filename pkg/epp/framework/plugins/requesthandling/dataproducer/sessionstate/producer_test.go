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

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	eppdatalayer "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
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

func finalResponse(cause fwkrh.TerminationCause, inputTokens, outputTokens int) *fwkrh.Response {
	return &fwkrh.Response{
		EndOfStream:      true,
		TerminationCause: cause,
		Usage: fwkrequest.Usage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
		},
	}
}

func TestFactoryAndProduces(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)

	assert.Equal(t, fwkplugin.TypedName{Type: SessionStateProducerType, Name: SessionStateProducerType}, producer.TypedName())
	producedMap := producer.Produces()
	expectedSessionKey := SessionStateDataKey.WithNonEmptyProducerName(SessionStateProducerType)
	produced, ok := producedMap[expectedSessionKey]
	require.True(t, ok)
	assert.IsType(t, SessionState{}, produced)

	expectedEndpointKey := EndpointSessionsDataKey.WithNonEmptyProducerName(SessionStateProducerType)
	producedEndpoint, ok := producedMap[expectedEndpointKey]
	require.True(t, ok)
	assert.IsType(t, EndpointSessions{}, producedEndpoint)

	assert.Equal(t, defaultEvictionTTL, producer.evictionTTL)
	assert.Equal(t, defaultEvictionSweepInterval, producer.evictionSweepInterval)

	dependencies := producer.Consumes()
	consumed, ok := dependencies.Required[agentidentity.AgentIdentityKey]
	require.True(t, ok)
	assert.IsType(t, "", consumed)
	assert.Empty(t, dependencies.Optional)
}

func TestAgentIdentityIsARequiredDependency(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	handle.AddPlugin(producer.TypedName().Name, producer)

	err := eppdatalayer.CreateMissingDataProducers(
		context.Background(),
		map[string]string{},
		map[string]fwkplugin.FactoryFunc{},
		handle,
	)
	require.ErrorIs(t, err, eppdatalayer.ErrNoDefaultProducer)
	assert.ErrorContains(t, err, agentidentity.AgentIdentityKey.String())
	assert.ErrorContains(t, err, producer.TypedName().Name)
}

func TestAgentIdentitySatisfiesRequiredDependency(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	handle.AddPlugin(producer.TypedName().Name, producer)

	identityProvider, err := agentidentity.PluginFactory("custom-agent-identity", nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(identityProvider.TypedName().Name, identityProvider)

	require.NoError(t, eppdatalayer.CreateMissingDataProducers(
		context.Background(),
		map[string]string{},
		map[string]fwkplugin.FactoryFunc{},
		handle,
	))

	ordered, err := eppdatalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins())
	require.NoError(t, err)
	assert.Equal(t, []string{identityProvider.TypedName().String(), producer.TypedName().String()}, ordered)
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

func TestRequestHeaderWithoutAgentIdentityIsNoOp(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := &fwksched.InferenceRequest{}

	require.NoError(t, producer.RequestHeader(context.Background(), nil))
	require.NoError(t, producer.RequestHeader(context.Background(), request))
	require.NoError(t, producer.RequestHeader(context.Background(), requestWithAgentIdentity("")))

	assert.Empty(t, request.AttributeKeys())
}

func TestSessionIDAttributeAloneIsNoOp(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := &fwksched.InferenceRequest{}
	request.PutAttribute(attrsession.SessionIDDataKey, attrsession.SessionID("session-a"))

	require.NoError(t, producer.RequestHeader(context.Background(), request))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))

	_, published := ReadSessionState(request)
	assert.False(t, published)
}

func TestRequestHeaderAndPreRequestTrackSessionHistory(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	first := requestWithAgentIdentity("session-a")

	beforeFirst := time.Now()
	require.NoError(t, producer.RequestHeader(context.Background(), first))
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
	require.NoError(t, producer.RequestHeader(context.Background(), second))
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
	require.NoError(t, producer.RequestHeader(context.Background(), request))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))

	producer.ResponseBody(context.Background(), request, &fwkrh.Response{EndOfStream: false}, nil)
	inFlight := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.RequestHeader(context.Background(), inFlight))
	state, ok := ReadSessionState(inFlight)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Zero(t, state.CompletedRequests)

	producer.ResponseBody(context.Background(), request, finalResponse(fwkrh.TerminationCauseNatural, 12, 34), nil)
	completed := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.RequestHeader(context.Background(), completed))
	state, ok = ReadSessionState(completed)
	require.True(t, ok)
	assert.Zero(t, state.InFlightRequests)
	assert.Equal(t, int64(1), state.CompletedRequests)
	assert.Equal(t, int64(12), state.TotalInputTokens)
	assert.Equal(t, int64(34), state.TotalOutputTokens)
}

func TestAbnormalTerminationDoesNotCompleteRequest(t *testing.T) {
	t.Parallel()

	causes := []fwkrh.TerminationCause{
		fwkrh.TerminationCauseClientDisconnect,
		fwkrh.TerminationCauseEvicted,
		fwkrh.TerminationCauseError,
	}
	for _, cause := range causes {
		t.Run(string(cause), func(t *testing.T) {
			t.Parallel()
			producer := newTestProducer(t)
			request := requestWithAgentIdentity("session-a")
			require.NoError(t, producer.RequestHeader(context.Background(), request))
			require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))

			producer.ResponseBody(context.Background(), request, finalResponse(cause, 12, 34), nil)
			next := requestWithAgentIdentity("session-a")
			require.NoError(t, producer.RequestHeader(context.Background(), next))
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
	require.NoError(t, producer.RequestHeader(context.Background(), request))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(1)))
	producer.ResponseBody(context.Background(), request, finalResponse(fwkrh.TerminationCauseNatural, 0, 0), nil)

	next := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.RequestHeader(context.Background(), next))
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
	require.NoError(t, producer.RequestHeader(context.Background(), request))

	producer.ResponseBody(context.Background(), nil, finalResponse(fwkrh.TerminationCauseNatural, 1, 2), nil)
	producer.ResponseBody(context.Background(), &fwksched.InferenceRequest{}, finalResponse(fwkrh.TerminationCauseNatural, 1, 2), nil)
	producer.ResponseBody(context.Background(), request, nil, nil)
	producer.ResponseBody(context.Background(), request, finalResponse(fwkrh.TerminationCauseNatural, 1, 2), nil)

	next := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.RequestHeader(context.Background(), next))
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
	require.NoError(t, producer.RequestHeader(context.Background(), request))
	require.NoError(t, producer.PreRequest(context.Background(), request, resultWithProfiles(2)))

	next := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.RequestHeader(context.Background(), next))
	state, ok := ReadSessionState(next)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.TurnsTaken)
}

func TestRequestHeaderPublishesLiveState(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	queued := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.RequestHeader(context.Background(), queued))

	state, ok := ReadSessionState(queued)
	require.True(t, ok)
	assert.Zero(t, state.TurnsTaken)
	assert.Zero(t, state.InFlightRequests)

	// Another turn of the same session is dispatched and completes while the
	// first request is still queued.
	other := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.RequestHeader(context.Background(), other))
	require.NoError(t, producer.PreRequest(context.Background(), other, resultWithProfiles(1)))

	state, ok = ReadSessionState(queued)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.TurnsTaken)
	assert.Equal(t, int64(1), state.InFlightRequests)

	response := finalResponse(fwkrh.TerminationCauseNatural, 100, 7)
	response.Usage.PromptTokenDetails = &fwkrequest.PromptTokenDetails{CachedTokens: 64}
	before := time.Now()
	producer.ResponseBody(context.Background(), other, response, nil)

	state, ok = ReadSessionState(queued)
	require.True(t, ok)
	assert.Zero(t, state.InFlightRequests)
	assert.Equal(t, int64(1), state.CompletedRequests)
	assert.Equal(t, int64(100), state.TotalInputTokens)
	assert.Equal(t, int64(7), state.TotalOutputTokens)
	assert.Equal(t, int64(64), state.TotalCachedTokens)
	assert.False(t, state.LastResponseAt.Before(before))
}

func TestProduceIsNoOp(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	request := requestWithAgentIdentity("session-a")
	require.NoError(t, producer.Produce(context.Background(), request, nil))

	_, published := ReadSessionState(request)
	assert.False(t, published)
	assert.Zero(t, registrySessionCount(&producer.registry))
}

func TestPendingRequestKeepsSessionFromEviction(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		finish func(p *Producer, request *fwksched.InferenceRequest)
	}{
		{
			name: "dispatched",
			finish: func(p *Producer, request *fwksched.InferenceRequest) {
				require.NoError(t, p.PreRequest(context.Background(), request, resultWithProfiles(1)))
				p.ResponseBody(context.Background(), request, finalResponse(fwkrh.TerminationCauseNatural, 1, 1), nil)
			},
		},
		{
			name: "abandoned",
			finish: func(p *Producer, request *fwksched.InferenceRequest) {
				p.RequestAbandoned(context.Background(), request)
				// A second release of the same request must not affect other requests.
				p.RequestAbandoned(context.Background(), request)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			producer := newTestProducer(t)
			request := requestWithAgentIdentity("session-a")
			request.RequestID = "queued"
			require.NoError(t, producer.RequestHeader(context.Background(), request))

			// The request waits in a flow-control queue for longer than the TTL.
			ttl := time.Hour
			start := time.Now().Add(-2 * ttl)
			setRegistrySessionTimes(t, &producer.registry, "session-a", start, start)
			producer.registry.EvictIdle(time.Now(), ttl)
			require.Equal(t, 1, registrySessionCount(&producer.registry), "a session with a pending request must remain")

			test.finish(producer, request)
			setRegistrySessionTimes(t, &producer.registry, "session-a", start, start)
			setRegistryLastResponseAt(t, &producer.registry, "session-a", start)
			producer.registry.EvictIdle(time.Now(), ttl)
			assert.Zero(t, registrySessionCount(&producer.registry), "the session is evictable once the request leaves")
		})
	}
}

type sessionStateFairnessConsumer struct {
	fwkfcmocks.MockFairnessPolicy
}

func (c *sessionStateFairnessConsumer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Required: map[fwkplugin.DataKey]any{SessionStateDataKey: SessionState{}}}
}

func TestFairnessPolicyCanConsumeSessionState(t *testing.T) {
	t.Parallel()

	identityProvider, err := agentidentity.PluginFactory(agentidentity.PluginType, nil, nil)
	require.NoError(t, err)
	producer := newTestProducer(t)
	consumer := &sessionStateFairnessConsumer{fwkfcmocks.MockFairnessPolicy{TypedNameV: fwkplugin.TypedName{Type: "mock-fairness", Name: "fairness"}}}

	ordered, err := eppdatalayer.ValidateAndOrderDataDependencies([]fwkplugin.Plugin{consumer, producer, identityProvider})
	require.NoError(t, err)
	assert.Equal(t, []string{
		identityProvider.TypedName().String(),
		producer.TypedName().String(),
		consumer.TypedName().String(),
	}, ordered)
}

type recordingRegistrar struct {
	registrations []fwkdl.PendingRegistration
}

func (r *recordingRegistrar) Register(reg fwkdl.PendingRegistration) error {
	r.registrations = append(r.registrations, reg)
	return nil
}

func resultWithNamedEndpoint(name string) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "primary",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"primary": {
				TargetEndpoints: []fwksched.Endpoint{
					fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
						ID: types.NamespacedName{Namespace: "default", Name: name},
					}, fwkdl.NewMetrics(), fwkdl.NewAttributes()),
				},
			},
			"secondary": {
				TargetEndpoints: []fwksched.Endpoint{
					fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
						ID: types.NamespacedName{Namespace: "default", Name: "other-pod"},
					}, fwkdl.NewMetrics(), fwkdl.NewAttributes()),
				},
			},
		},
	}
}

func TestRegisterDependencies(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t)
	registrar := &recordingRegistrar{}
	require.NoError(t, producer.RegisterDependencies(registrar))
	require.Len(t, registrar.registrations, 1)

	reg := registrar.registrations[0]
	assert.Equal(t, sourcenotifications.EndpointNotificationSourceType, reg.SourceType)
	assert.Equal(t, producer.TypedName(), reg.Owner)
	assert.Same(t, producer, reg.Extractor)
	assert.NotNil(t, reg.DefaultSource)
}

func TestPreRequestBindsEndpointAndTracksFootprint(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	producer := newTestProducer(t)

	// Turn 1 dispatches on pod-a and finishes naturally with TotalTokens.
	req1 := requestWithAgentIdentity("session-a")
	req1.RequestID = "req-1"
	req1.RequestSizeBytes = 400 // 100 estimated tokens
	require.NoError(t, producer.RequestHeader(ctx, req1))
	require.NoError(t, producer.PreRequest(ctx, req1, resultWithNamedEndpoint("pod-a")))

	state, ok := ReadSessionState(req1)
	require.True(t, ok)
	assert.Equal(t, "session-a", state.SessionID)
	assert.Equal(t, "default/pod-a", state.Endpoint)
	assert.Equal(t, int64(100), state.InFlightTokens)
	assert.Equal(t, int64(100), state.ContextSize())

	resp1 := finalResponse(fwkrh.TerminationCauseNatural, 100, 150)
	resp1.Usage.TotalTokens = 250
	producer.ResponseBody(ctx, req1, resp1, nil)
	state, ok = ReadSessionState(req1)
	require.True(t, ok)
	assert.Zero(t, state.InFlightTokens)
	assert.Equal(t, int64(250), state.ContextTokens)
	assert.Equal(t, int64(250), state.ContextSize())

	// Two parallel turns run simultaneously; one aborts without leaking InFlightTokens.
	req2 := requestWithAgentIdentity("session-a")
	req2.RequestID = "req-2"
	req2.RequestSizeBytes = 2000 // 500 estimated tokens
	require.NoError(t, producer.RequestHeader(ctx, req2))
	require.NoError(t, producer.PreRequest(ctx, req2, resultWithNamedEndpoint("pod-a")))

	req3 := requestWithAgentIdentity("session-a")
	req3.RequestID = "req-3"
	req3.RequestSizeBytes = 800 // 200 estimated tokens
	require.NoError(t, producer.RequestHeader(ctx, req3))
	require.NoError(t, producer.PreRequest(ctx, req3, resultWithNamedEndpoint("pod-a")))

	state, ok = ReadSessionState(req2)
	require.True(t, ok)
	assert.Equal(t, int64(2), state.InFlightRequests)
	assert.Equal(t, int64(700), state.InFlightTokens)
	assert.Equal(t, int64(700), state.ContextSize())

	producer.ResponseBody(ctx, req3, finalResponse(fwkrh.TerminationCauseError, 10, 0), nil)
	state, ok = ReadSessionState(req2)
	require.True(t, ok)
	assert.Equal(t, int64(1), state.InFlightRequests)
	assert.Equal(t, int64(500), state.InFlightTokens)
	assert.Equal(t, int64(250), state.ContextTokens)

	// Moving to pod-b updates the binding and cleans pod-a's index.
	req4 := requestWithAgentIdentity("session-a")
	req4.RequestID = "req-4"
	req4.RequestSizeBytes = 400 // 100 estimated tokens
	require.NoError(t, producer.RequestHeader(ctx, req4))
	require.NoError(t, producer.PreRequest(ctx, req4, resultWithNamedEndpoint("pod-b")))

	assert.Empty(t, producer.registry.EndpointSessions("default/pod-a"))
	podB := producer.registry.EndpointSessions("default/pod-b")
	require.Len(t, podB, 1)
	assert.Equal(t, "session-a", podB[0].SessionID)
	assert.Equal(t, "default/pod-b", podB[0].Endpoint)

	producer.ResponseBody(ctx, req2, finalResponse(fwkrh.TerminationCauseClientDisconnect, 0, 0), nil)
	resp4 := finalResponse(fwkrh.TerminationCauseNatural, 300, 40)
	resp4.Usage.TotalTokens = 340
	producer.ResponseBody(ctx, req4, resp4, nil)
	state, ok = ReadSessionState(req4)
	require.True(t, ok)
	assert.Zero(t, state.InFlightRequests)
	assert.Zero(t, state.InFlightTokens)
	assert.Equal(t, int64(340), state.ContextTokens)
}

func TestExtractEndpointLifecycleAndStaleDelete(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	producer := newTestProducer(t)

	// Safe reads when endpoint or attributes are nil or absent.
	_, ok := ReadEndpointSessions(nil)
	assert.False(t, ok)
	_, ok = ReadEndpointSessions(&nilAttrsEndpoint{})
	assert.False(t, ok)

	podAID := types.NamespacedName{Namespace: "default", Name: "pod-a"}
	epA := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: podAID}, nil)
	_, ok = ReadEndpointSessions(epA)
	assert.False(t, ok)

	require.NoError(t, producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: epA}))
	sessions, ok := ReadEndpointSessions(epA)
	require.True(t, ok)
	assert.Empty(t, sessions)

	req := requestWithAgentIdentity("session-a")
	req.RequestID = "req-1"
	req.RequestSizeBytes = 400
	require.NoError(t, producer.RequestHeader(ctx, req))
	require.NoError(t, producer.PreRequest(ctx, req, resultWithNamedEndpoint("pod-a")))
	resp := finalResponse(fwkrh.TerminationCauseNatural, 100, 50)
	resp.Usage.TotalTokens = 150
	producer.ResponseBody(ctx, req, resp, nil)

	sessions, ok = ReadEndpointSessions(epA)
	require.True(t, ok)
	require.Len(t, sessions, 1)
	assert.Equal(t, "session-a", sessions[0].SessionID)
	assert.Equal(t, int64(150), sessions[0].ContextTokens)

	// EndpointSessions.Clone returns an independent slice copy.
	cloned := EndpointSessions(sessions).Clone().(EndpointSessions)
	cloned[0].ContextTokens = 9999
	sessionsAfter, ok := ReadEndpointSessions(epA)
	require.True(t, ok)
	assert.Equal(t, int64(150), sessionsAfter[0].ContextTokens)

	// Updating the same Endpoint object in place preserves existing bindings.
	require.NoError(t, producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: epA}))
	sessionsAfterInPlace, ok := ReadEndpointSessions(epA)
	require.True(t, ok)
	require.Len(t, sessionsAfterInPlace, 1)

	// Replacing pod-a with a new Endpoint object unbinds sessions from the old pod
	// while keeping their context history, and ignores the delayed delete for epA.
	epAReplaced := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: podAID}, nil)
	require.NoError(t, producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: epAReplaced}))

	oldSessionState, ok := ReadSessionState(req)
	require.True(t, ok)
	assert.Empty(t, oldSessionState.Endpoint, "replacement endpoint object must unbind sessions from the old pod")
	assert.Equal(t, int64(150), oldSessionState.ContextTokens)
	sessionsOnFreshReplacement, ok := ReadEndpointSessions(epAReplaced)
	require.True(t, ok)
	assert.Empty(t, sessionsOnFreshReplacement)

	reqReplacement := requestWithAgentIdentity("session-b")
	reqReplacement.RequestID = "req-2"
	reqReplacement.RequestSizeBytes = 800
	require.NoError(t, producer.RequestHeader(ctx, reqReplacement))
	require.NoError(t, producer.PreRequest(ctx, reqReplacement, resultWithNamedEndpoint("pod-a")))
	respReplacement := finalResponse(fwkrh.TerminationCauseNatural, 200, 40)
	respReplacement.Usage.TotalTokens = 240
	producer.ResponseBody(ctx, reqReplacement, respReplacement, nil)

	require.NoError(t, producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: epA}))
	sessionsAfterStale, ok := ReadEndpointSessions(epAReplaced)
	require.True(t, ok)
	require.Len(t, sessionsAfterStale, 1, "stale delete must not unbind sessions on replaced endpoint")
	assert.Equal(t, "session-b", sessionsAfterStale[0].SessionID)

	// Matching delete clears Endpoint while preserving ContextTokens.
	require.NoError(t, producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: epAReplaced}))
	sessionsAfterDelete, ok := ReadEndpointSessions(epAReplaced)
	require.True(t, ok)
	assert.Empty(t, sessionsAfterDelete)

	liveState, ok := ReadSessionState(reqReplacement)
	require.True(t, ok)
	assert.Empty(t, liveState.Endpoint)
	assert.Equal(t, int64(240), liveState.ContextTokens)
}

func TestExtractConcurrentReplaceAndStaleDelete(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	podID := types.NamespacedName{Namespace: "default", Name: "pod-a"}

	for range 200 {
		producer := newTestProducer(t)
		oldEp := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: podID}, nil)
		newEp := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: podID}, nil)
		require.NoError(t, producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: oldEp}))

		start := make(chan struct{})
		doneDelete := make(chan struct{})
		go func() {
			defer close(doneDelete)
			<-start
			_ = producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: oldEp})
		}()

		close(start)
		require.NoError(t, producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: newEp}))

		// Once newEp's EventAddOrUpdate returns, newEp is live for scheduling;
		// an overlapping or duplicate EventDelete for oldEp must not unbind it.
		req := requestWithAgentIdentity("session-new")
		req.RequestID = "req-new"
		req.RequestSizeBytes = 400
		require.NoError(t, producer.RequestHeader(ctx, req))
		require.NoError(t, producer.PreRequest(ctx, req, resultWithNamedEndpoint("pod-a")))
		<-doneDelete

		require.NoError(t, producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: oldEp}))
		sessions, ok := ReadEndpointSessions(newEp)
		require.True(t, ok)
		require.Len(t, sessions, 1, "concurrent or late delete of oldEp must never unbind newEp")
	}
}

type nilAttrsEndpoint struct{}

func (e *nilAttrsEndpoint) GetMetadata() *fwkdl.EndpointMetadata   { return nil }
func (e *nilAttrsEndpoint) UpdateMetadata(*fwkdl.EndpointMetadata) {}
func (e *nilAttrsEndpoint) GetAttributes() fwkdl.AttributeMap      { return nil }
func (e *nilAttrsEndpoint) GetMetrics() *fwkdl.Metrics             { return nil }
func (e *nilAttrsEndpoint) UpdateMetrics(*fwkdl.Metrics)           {}
func (e *nilAttrsEndpoint) String() string                         { return "" }

type multiHookSessionConsumer struct {
	fwkfcmocks.MockFairnessPolicy
	lastScoreState SessionState
	lastScoreEps   []SessionState
	lastSatEps     []SessionState
}

var (
	_ fwkfc.FairnessPolicy     = &multiHookSessionConsumer{}
	_ fwkfc.SaturationDetector = &multiHookSessionConsumer{}
	_ fwksched.Scorer          = &multiHookSessionConsumer{}
	_ fwkplugin.ConsumerPlugin = &multiHookSessionConsumer{}
)

func (c *multiHookSessionConsumer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			SessionStateDataKey:     SessionState{},
			EndpointSessionsDataKey: EndpointSessions{},
		},
	}
}

func (c *multiHookSessionConsumer) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

func (c *multiHookSessionConsumer) Saturation(_ context.Context, endpoints []fwkdl.Endpoint) float64 {
	c.lastSatEps = nil
	for _, ep := range endpoints {
		if sessions, ok := ReadEndpointSessions(ep); ok {
			c.lastSatEps = append(c.lastSatEps, sessions...)
		}
	}
	return 0
}

func (c *multiHookSessionConsumer) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	c.lastScoreState, _ = ReadSessionState(request)
	c.lastScoreEps = nil
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		if sessions, ok := ReadEndpointSessions(&scopedDLEndpoint{ep}); ok {
			c.lastScoreEps = append(c.lastScoreEps, sessions...)
		}
		scores[ep] = 1.0
	}
	return scores
}

// scopedDLEndpoint adapts a scoped scheduling.Endpoint to datalayer.Endpoint
// so tests can exercise ReadEndpointSessions directly on scoped endpoints.
type scopedDLEndpoint struct {
	fwksched.Endpoint
}

func (s *scopedDLEndpoint) UpdateMetadata(*fwkdl.EndpointMetadata) {}
func (s *scopedDLEndpoint) UpdateMetrics(*fwkdl.Metrics)           {}
func (s *scopedDLEndpoint) GetAttributes() fwkdl.AttributeMap      { return s.Endpoint }

func TestMultiHookConsumerAutoCreationDAGAndScope(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	handle := fwkplugin.NewEppHandle(ctx, nil)
	identityProvider, err := agentidentity.PluginFactory(agentidentity.PluginType, nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(identityProvider.TypedName().Name, identityProvider)

	consumer := &multiHookSessionConsumer{
		MockFairnessPolicy: fwkfcmocks.MockFairnessPolicy{
			TypedNameV: fwkplugin.TypedName{Type: "multi-hook", Name: "consumer"},
		},
	}
	handle.AddPlugin(consumer.TypedName().Name, consumer)

	defaultProducers := map[string]string{
		SessionStateDataKey.String():     SessionStateProducerType,
		EndpointSessionsDataKey.String(): SessionStateProducerType,
	}
	factories := map[string]fwkplugin.FactoryFunc{
		SessionStateProducerType: Factory,
	}
	require.NoError(t, eppdatalayer.CreateMissingDataProducers(ctx, defaultProducers, factories, handle))

	plg := handle.Plugin(SessionStateProducerType)
	require.NotNil(t, plg, "CreateMissingDataProducers must instantiate session-state-producer")
	producer, ok := plg.(*Producer)
	require.True(t, ok)

	ordered, err := eppdatalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins())
	require.NoError(t, err)
	assert.Equal(t, []string{
		identityProvider.TypedName().String(),
		producer.TypedName().String(),
		consumer.TypedName().String(),
	}, ordered)

	eppdatalayer.RegisterScopeSpecs(handle.GetAllPlugins())

	runtime := eppdatalayer.NewRuntime(0)
	require.NoError(t, producer.RegisterDependencies(runtime))
	require.NoError(t, runtime.Configure(nil, logr.Discard()))

	ep := runtime.NewEndpoint(ctx, &fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Namespace: "default", Name: "pod-a"},
	})
	require.NotNil(t, ep)

	req1 := requestWithAgentIdentity("session-a")
	req1.RequestID = "req-1"
	req1.RequestSizeBytes = 400
	require.NoError(t, producer.RequestHeader(ctx, req1))
	require.NoError(t, producer.PreRequest(ctx, req1, resultWithNamedEndpoint("pod-a")))
	resp1 := finalResponse(fwkrh.TerminationCauseNatural, 100, 20)
	resp1.Usage.TotalTokens = 120
	producer.ResponseBody(ctx, req1, resp1, nil)

	// Saturation reads ReadEndpointSessions on the datalayer Endpoint.
	_ = consumer.Saturation(ctx, []fwkdl.Endpoint{ep})
	require.Len(t, consumer.lastSatEps, 1)
	assert.Equal(t, "session-a", consumer.lastSatEps[0].SessionID)
	assert.Equal(t, int64(120), consumer.lastSatEps[0].ContextTokens)

	// A follow-up request reads both ReadSessionState and ReadEndpointSessions under ScopeRequest + ScopeEndpoints.
	req2 := requestWithAgentIdentity("session-a")
	req2.RequestID = "req-2"
	require.NoError(t, producer.RequestHeader(ctx, req2))
	schedEp := fwksched.NewEndpointFromSnapshot(ep.GetMetadata(), ep.GetMetrics(), ep.GetAttributes())

	scopedReq, scopedEps, violations := eppdatalayer.ScopeInvocation(
		logr.Discard(), "Scorer", consumer, req2, []fwksched.Endpoint{schedEp},
	)
	_ = consumer.Score(ctx, scopedReq, scopedEps)
	assert.NoError(t, violations.Write())
	assert.Equal(t, "session-a", consumer.lastScoreState.SessionID)
	assert.Equal(t, "default/pod-a", consumer.lastScoreState.Endpoint)
	assert.Equal(t, int64(120), consumer.lastScoreState.ContextTokens)
	require.Len(t, consumer.lastScoreEps, 1)
	assert.Equal(t, "session-a", consumer.lastScoreEps[0].SessionID)

	// An undeclared plugin must not read either attribute under ScopeRequest / ScopeEndpoints.
	undeclared := &fwkfcmocks.MockFairnessPolicy{
		TypedNameV: fwkplugin.TypedName{Type: "undeclared", Name: "undeclared"},
	}
	eppdatalayer.RegisterScopeSpecs([]fwkplugin.Plugin{undeclared})
	deniedReq, deniedEps, _ := eppdatalayer.ScopeInvocation(
		logr.Discard(), "Scorer", undeclared, req2, []fwksched.Endpoint{schedEp},
	)
	_, deniedSessionOK := ReadSessionState(deniedReq)
	assert.False(t, deniedSessionOK)
	_, deniedEndpointOK := ReadEndpointSessions(&scopedDLEndpoint{deniedEps[0]})
	assert.False(t, deniedEndpointOK)

	runtime.ReleaseEndpoint(ep)
	_ = consumer.Saturation(ctx, []fwkdl.Endpoint{ep})
	assert.Empty(t, consumer.lastSatEps)
}
