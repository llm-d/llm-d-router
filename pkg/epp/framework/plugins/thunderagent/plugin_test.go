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
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus/testutil"
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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/test/utils"
)

// Consumes requires agent identity plus both session-state-producer outputs,
// so configuration fails when agent-identity is missing and auto-creates
// session-state-producer when registered as default producer.
func TestRequiredDependenciesAndAutoCreation(t *testing.T) {
	ctx := context.Background()
	a := newTestLedger(testConfig()).agent
	consumed := a.Consumes().Required
	require.Contains(t, consumed, agentidentity.AgentIdentityKey)
	require.Contains(t, consumed, sessionstate.SessionStateDataKey)
	require.Contains(t, consumed, sessionstate.EndpointSessionsDataKey)

	// With session-state-producer provisioned but no agent-identity provider,
	// missing AgentIdentityKey fails CreateMissingDataProducers.
	handle := fwkplugin.NewEppHandle(ctx, nil)
	handle.AddPlugin(a.TypedName().Name, a)
	defaultProducers := map[string]string{
		sessionstate.SessionStateDataKey.String():     sessionstate.SessionStateProducerType,
		sessionstate.EndpointSessionsDataKey.String(): sessionstate.SessionStateProducerType,
	}
	factories := map[string]fwkplugin.FactoryFunc{
		sessionstate.SessionStateProducerType: sessionstate.Factory,
	}
	err := eppdatalayer.CreateMissingDataProducers(ctx, defaultProducers, factories, handle)
	require.ErrorIs(t, err, eppdatalayer.ErrNoDefaultProducer)
	require.ErrorContains(t, err, agentidentity.AgentIdentityKey.String())

	// Adding agent-identity lets CreateMissingDataProducers instantiate
	// session-state-producer and orders the DAG cleanly.
	idPlg, err := agentidentity.PluginFactory(agentidentity.PluginType, nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(idPlg.TypedName().Name, idPlg)

	require.NoError(t, eppdatalayer.CreateMissingDataProducers(ctx, defaultProducers, factories, handle))
	ordered, err := eppdatalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins())
	require.NoError(t, err)
	require.Equal(t, []string{
		idPlg.TypedName().String(),
		fwkplugin.TypedName{Type: sessionstate.SessionStateProducerType, Name: sessionstate.SessionStateProducerType}.String(),
		a.TypedName().String(),
	}, ordered)
}

// Factory applies defaults, accepts duration strings, and rejects unknown or
// invalid parameters.
func TestFactoryParameters(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	p, err := Factory(ThunderAgentPluginType, nil, utils.NewTestHandle(ctx))
	require.NoError(t, err)
	a := p.(*ThunderAgent)
	require.Equal(t, float64(4194304), a.fallbackCapacityTokens)
	require.Equal(t, 1.0, a.usableCapacityRatio)
	require.Equal(t, 5*time.Minute, a.forceAdmitTimeout)
	require.Equal(t, 30*time.Second, a.idleGracePeriod)

	raw := []byte(`{"fallbackCapacityTokens": 2048, "usableCapacityRatio": 0.5, "idleGracePeriod": "0s", "forceAdmitTimeout": "45m"}`)
	p, err = Factory(ThunderAgentPluginType, fwkplugin.StrictDecoder(raw), nil)
	require.NoError(t, err)
	a = p.(*ThunderAgent)
	require.Equal(t, float64(2048), a.fallbackCapacityTokens)
	require.Equal(t, 0.5, a.usableCapacityRatio)
	require.Zero(t, a.idleGracePeriod)
	require.Equal(t, 45*time.Minute, a.forceAdmitTimeout)

	partial := []byte(`{"forceAdmitTimeout": "30m"}`)
	p, err = Factory(ThunderAgentPluginType, fwkplugin.StrictDecoder(partial), nil)
	require.NoError(t, err)
	a = p.(*ThunderAgent)
	require.Equal(t, float64(4194304), a.fallbackCapacityTokens)
	require.Equal(t, 1.0, a.usableCapacityRatio)
	require.Equal(t, 30*time.Second, a.idleGracePeriod)
	require.Equal(t, 30*time.Minute, a.forceAdmitTimeout)

	for name, body := range map[string]string{
		"removed capacityTokens":       `{"capacityTokens": 1000}`,
		"removed utilThreshold":        `{"utilThreshold": 0.8}`,
		"removed idleLeaseSeconds":     `{"idleLeaseSeconds": 30}`,
		"removed headWaitStarvationMs": `{"headWaitStarvationMs": 1000}`,
		"removed evictionTtlSeconds":   `{"evictionTtlSeconds": 3600}`,
		"removed evictionSweepSeconds": `{"evictionSweepSeconds": 10}`,
		"invalid duration string":      `{"forceAdmitTimeout": "soon"}`,
		"null duration":                `{"idleGracePeriod": null}`,
		"negative duration":            `{"idleGracePeriod": "-1s"}`,
		"zero ratio":                   `{"usableCapacityRatio": 0}`,
		"zero capacity":                `{"fallbackCapacityTokens": 0}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := Factory(ThunderAgentPluginType, fwkplugin.StrictDecoder([]byte(body)), nil)
			require.Error(t, err)
		})
	}
}

// Factory starts the policy sweep for the handle's lifetime.
func TestFactoryStartsSweep(t *testing.T) {
	origInterval := sweepInterval
	sweepInterval = 10 * time.Millisecond
	t.Cleanup(func() { sweepInterval = origInterval })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	p, err := Factory(ThunderAgentPluginType, nil, utils.NewTestHandle(ctx))
	require.NoError(t, err)
	a := p.(*ThunderAgent)

	// Seed a stale endpoint directly; the background sweep removes it.
	a.mu.Lock()
	a.endpoints["default/pod-stale"] = &trackedEndpoint{
		updatedAt: time.Now().Add(-2 * endpointStaleAfter),
	}
	a.mu.Unlock()

	require.Eventually(t, func() bool {
		a.mu.Lock()
		defer a.mu.Unlock()
		return len(a.endpoints) == 0
	}, 5*time.Second, 10*time.Millisecond)
}

// DumpState reports counts and per-endpoint totals but no session ids.
func TestDumpState(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	l.runTurn("secret-session", "pod-a", 400, 300)
	l.startTurn("other", "pod-a", 800)
	l.runTurn("paused", "pod-a", 400, 200)
	l.pause("paused")
	l.prime(podA)

	raw, err := l.agent.DumpState()
	require.NoError(t, err)
	require.NotContains(t, string(raw), "secret-session")

	var got stateDump
	require.NoError(t, json.Unmarshal(raw, &got))
	require.Equal(t, stateDump{
		RunningSessions: 1,
		IdleSessions:    1,
		PausedSessions:  1,
		Endpoints: map[string]endpointDump{
			"default/pod-a": {WorkingSetTokens: 500, CapacityTokens: 1000},
		},
	}, got)
}

// TestEndToEndWithRealSessionStateProducer wires real agent-identity,
// auto-provisioned session-state-producer, datalayer Runtime + ScopeInvocation,
// and ThunderAgent across the full request and endpoint lifecycle.
func TestEndToEndWithRealSessionStateProducer(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	handle := utils.NewTestHandle(ctx)
	idRaw := json.NewDecoder(strings.NewReader(`{"additionalSessionHeaders": ["x-session-id"]}`))
	idPlg, err := agentidentity.PluginFactory(agentidentity.PluginType, idRaw, handle)
	require.NoError(t, err)
	idProvider := idPlg.(*agentidentity.Plugin)
	handle.AddPlugin(idProvider.TypedName().Name, idProvider)

	thunderRaw := json.NewDecoder(strings.NewReader(`{"fallbackCapacityTokens": 1000, "idleGracePeriod": "0s", "forceAdmitTimeout": "1m"}`))
	thunderPlg, err := Factory("thunder", thunderRaw, handle)
	require.NoError(t, err)
	agent := thunderPlg.(*ThunderAgent)
	handle.AddPlugin(agent.TypedName().Name, agent)

	defaultProducers := map[string]string{
		sessionstate.SessionStateDataKey.String():     sessionstate.SessionStateProducerType,
		sessionstate.EndpointSessionsDataKey.String(): sessionstate.SessionStateProducerType,
	}
	factories := map[string]fwkplugin.FactoryFunc{
		sessionstate.SessionStateProducerType: sessionstate.Factory,
	}
	require.NoError(t, eppdatalayer.CreateMissingDataProducers(ctx, defaultProducers, factories, handle))
	_, err = eppdatalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins())
	require.NoError(t, err)
	eppdatalayer.RegisterScopeSpecs(handle.GetAllPlugins())

	producer := handle.Plugin(sessionstate.SessionStateProducerType).(*sessionstate.Producer)
	runtime := eppdatalayer.NewRuntime(0)
	require.NoError(t, producer.RegisterDependencies(runtime))
	require.NoError(t, runtime.Configure(nil, logr.Discard()))

	epA := runtime.NewEndpoint(ctx, &fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-a"}})
	epB := runtime.NewEndpoint(ctx, &fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-b"}})
	require.NotNil(t, epA)
	require.NotNil(t, epB)

	reqCounter := 0
	newReq := func(session string, sizeBytes int) *fwksched.InferenceRequest {
		reqCounter++
		req := &fwksched.InferenceRequest{
			RequestID:        fmt.Sprintf("req-%d", reqCounter),
			Headers:          map[string]string{"x-session-id": session},
			RequestSizeBytes: sizeBytes,
		}
		require.NoError(t, idProvider.RequestHeader(ctx, req))
		require.NoError(t, producer.RequestHeader(ctx, req))
		return req
	}
	queueOf := func(session string, req *fwksched.InferenceRequest) *fwkfcmocks.MockFlowQueueAccessor {
		fcReq := fwkfcmocks.NewMockFlowControlRequest(uint64(req.RequestSizeBytes), req.RequestID, fwkfc.FlowKey{ID: session})
		fcReq.InferenceRequestV = req
		return &fwkfcmocks.MockFlowQueueAccessor{
			LenV:     1,
			FlowKeyV: fwkfc.FlowKey{ID: session},
			PeekV: &fwkfcmocks.MockQueueItemAccessor{
				EnqueueTimeV:     time.Now(),
				OriginalRequestV: fcReq,
			},
		}
	}
	liveEndpoints := []fwkdl.Endpoint{epA, epB}
	scoreAndPick := func(req *fwksched.InferenceRequest) fwksched.Endpoint {
		schedEps := make([]fwksched.Endpoint, 0, len(liveEndpoints))
		for _, ep := range liveEndpoints {
			schedEps = append(schedEps, fwksched.NewEndpointFromSnapshot(ep.GetMetadata(), ep.GetMetrics(), ep.GetAttributes()))
		}
		scopedReq, scopedEps, violations := eppdatalayer.ScopeInvocation(logr.Discard(), "Scorer", agent, req, schedEps)
		scores := agent.Score(ctx, scopedReq, scopedEps)
		require.NoError(t, violations.Write())
		bestIdx := -1
		for i, ep := range scopedEps {
			if scores == nil {
				bestIdx = i
				break
			}
			if bestIdx == -1 || scores[ep] > scores[scopedEps[bestIdx]] {
				bestIdx = i
			}
		}
		require.NotEqual(t, -1, bestIdx)
		return schedEps[bestIdx]
	}
	dispatch := func(req *fwksched.InferenceRequest, target fwksched.Endpoint, totalTokens int) {
		res := &fwksched.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{target}},
			},
		}
		require.NoError(t, producer.PreRequest(ctx, req, res))
		if totalTokens > 0 {
			producer.ResponseBody(ctx, req, &fwkrh.Response{
				EndOfStream:      true,
				TerminationCause: fwkrh.TerminationCauseNatural,
				Usage:            fwkrequest.Usage{TotalTokens: totalTokens},
			}, nil)
		}
	}

	// Turn 1 for s1 (1600 bytes -> 400 tokens): admits onto one pod and commits 400.
	_ = agent.Saturation(ctx, liveEndpoints)
	req1 := newReq("s1", 1600)
	q1 := queueOf("s1", req1)
	require.Equal(t, q1, pick(t, agent, q1))
	pod1 := scoreAndPick(req1)
	dispatch(req1, pod1, 400)
	firstPodName := pod1.GetMetadata().ID.String()
	secondPodName := "default/pod-b"
	if firstPodName == secondPodName {
		secondPodName = "default/pod-a"
	}

	// Turn 1 for s2 (3200 bytes -> 800 tokens): prefers the empty pod (1000 free > 600).
	_ = agent.Saturation(ctx, liveEndpoints)
	req2 := newReq("s2", 3200)
	q2 := queueOf("s2", req2)
	require.Equal(t, q2, pick(t, agent, q2))
	pod2 := scoreAndPick(req2)
	require.Equal(t, secondPodName, pod2.GetMetadata().ID.String())
	dispatch(req2, pod2, 800)

	// s3 needs 700 tokens: only firstPod (400 used, idleGracePeriod=0s) fits by pausing s1.
	_ = agent.Saturation(ctx, liveEndpoints)
	req3 := newReq("s3", 2800)
	q3 := queueOf("s3", req3)
	require.Equal(t, q3, pick(t, agent, q3))
	require.Equal(t, float64(1), testutil.ToFloat64(agent.metrics.pauses))
	pod3 := scoreAndPick(req3)
	require.Equal(t, firstPodName, pod3.GetMetadata().ID.String())
	dispatch(req3, pod3, 500) // s3 shrinks to 500 -> 500 free on firstPod

	// s1 (committed 400, pinned to firstPod) now fits back on firstPod (room 500 >= 400) and resumes.
	_ = agent.Saturation(ctx, liveEndpoints)
	req1b := newReq("s1", 1600)
	q1b := queueOf("s1", req1b)
	require.Equal(t, q1b, pick(t, agent, q1b))
	require.Equal(t, float64(1), testutil.ToFloat64(agent.metrics.resumes))
	pod1b := scoreAndPick(req1b)
	require.Equal(t, firstPodName, pod1b.GetMetadata().ID.String())
	dispatch(req1b, pod1b, 450)

	// Releasing epA clears bindings on pod-a; Saturation removes pod-a and s1/s3 re-enter on pod-b.
	runtime.ReleaseEndpoint(epA)
	liveEndpoints = []fwkdl.Endpoint{epB}
	_ = agent.Saturation(ctx, liveEndpoints)
	snap := agent.snapshot()
	require.NotContains(t, snap.endpoints, "default/pod-a")
	require.Contains(t, snap.endpoints, "default/pod-b")
}
