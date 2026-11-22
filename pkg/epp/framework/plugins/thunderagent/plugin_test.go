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
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	eppdatalayer "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/test/utils"
)

// Agent identity is a required input, so configuration loading fails without
// an identity provider.
func TestAgentIdentityIsARequiredDependency(t *testing.T) {
	a := newTestAgent(testConfig())
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	handle.AddPlugin(a.TypedName().Name, a)

	err := eppdatalayer.CreateMissingDataProducers(context.Background(),
		map[string]string{}, map[string]fwkplugin.FactoryFunc{}, handle)
	require.ErrorIs(t, err, eppdatalayer.ErrNoDefaultProducer)
	require.ErrorContains(t, err, agentidentity.AgentIdentityKey.String())
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
	require.Equal(t, 30*time.Second, a.mgr.idleGracePeriod)
	require.Equal(t, 10*time.Minute, a.mgr.ttl)

	raw := []byte(`{"fallbackCapacityTokens": 2048, "usableCapacityRatio": 0.5, "idleGracePeriod": "0s", "forceAdmitTimeout": "45m"}`)
	p, err = Factory(ThunderAgentPluginType, fwkplugin.StrictDecoder(raw), nil)
	require.NoError(t, err)
	a = p.(*ThunderAgent)
	require.Equal(t, float64(2048), a.fallbackCapacityTokens)
	require.Equal(t, 0.5, a.usableCapacityRatio)
	require.Zero(t, a.mgr.idleGracePeriod)
	require.Equal(t, 45*time.Minute, a.forceAdmitTimeout)
	require.Equal(t, 90*time.Minute, a.mgr.ttl)

	partial := []byte(`{"forceAdmitTimeout": "30m"}`)
	p, err = Factory(ThunderAgentPluginType, fwkplugin.StrictDecoder(partial), nil)
	require.NoError(t, err)
	a = p.(*ThunderAgent)
	require.Equal(t, float64(4194304), a.fallbackCapacityTokens)
	require.Equal(t, 1.0, a.usableCapacityRatio)
	require.Equal(t, 30*time.Second, a.mgr.idleGracePeriod)
	require.Equal(t, 30*time.Minute, a.forceAdmitTimeout)
	require.Equal(t, time.Hour, a.mgr.ttl)

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

// Factory starts the idle session sweep for the handle's lifetime.
func TestFactoryStartsSweep(t *testing.T) {
	origInterval := sweepInterval
	sweepInterval = 10 * time.Millisecond
	t.Cleanup(func() { sweepInterval = origInterval })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	p, err := Factory(ThunderAgentPluginType, nil, utils.NewTestHandle(ctx))
	require.NoError(t, err)
	a := p.(*ThunderAgent)

	runTurn(t, a, "s1", schedEndpoint("pod-a", 0, 0), 400, 300)
	a.mgr.mu.Lock()
	a.mgr.sessions["s1"].lastActivity = time.Now().Add(-2 * a.mgr.ttl)
	a.mgr.mu.Unlock()

	require.Eventually(t, func() bool {
		_, ok := sessionOf(a, "s1")
		return !ok
	}, 5*time.Second, 10*time.Millisecond)
}

// DumpState reports counts and per-endpoint totals but no session ids.
func TestDumpState(t *testing.T) {
	a := newTestAgent(testConfig())
	runTurn(t, a, "secret-session", schedEndpoint("pod-a", 0, 0), 400, 300)
	_ = startTurn(t, a, "other", schedEndpoint("pod-a", 0, 0), 800)
	runTurn(t, a, "paused", schedEndpoint("pod-a", 0, 0), 400, 200)
	pause(a, "paused")

	raw, err := a.DumpState()
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

// A delete event drops the endpoint and unbinds its sessions, which keep
// their footprint; add or update events leave the ledger alone.
func TestExtractEndpointDelete(t *testing.T) {
	a := newTestAgent(testConfig())
	runTurn(t, a, "s1", schedEndpoint("pod-a", 0, 0), 400, 300)
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "pod-a"}}, nil)

	require.NoError(t, a.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
	require.Equal(t, float64(300), endpointTokens(a, "default/pod-a"))

	require.NoError(t, a.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: ep}))
	require.Equal(t, float64(-1), endpointTokens(a, "default/pod-a"))
	s, ok := sessionOf(a, "s1")
	require.True(t, ok)
	require.Nil(t, s.endpoint)
	require.Equal(t, int64(300), s.committedTokens)
}

// Removing a pod through the datalayer runtime reaches the ledger.
func TestRuntimeReleaseEndpointRemovesEndpoint(t *testing.T) {
	a := newTestAgent(testConfig())
	runtime := eppdatalayer.NewRuntime(0)
	require.NoError(t, a.RegisterDependencies(runtime))
	require.NoError(t, runtime.Configure(nil, logr.Discard()))

	endpoint := runtime.NewEndpoint(context.Background(), &fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Namespace: "default", Name: "pod-a"},
	})
	require.NotNil(t, endpoint)
	runTurn(t, a, "s1", schedEndpoint("pod-a", 0, 0), 400, 300)

	runtime.ReleaseEndpoint(endpoint)
	require.Equal(t, float64(-1), endpointTokens(a, "default/pod-a"))
}

// PreRequest stores each turn's estimate under inflightEstimateKey for its
// later hooks. The data layer drops undeclared request-attribute writes and
// fails the request, so the key must be declared.
func TestProducesInflightEstimate(t *testing.T) {
	a := newTestAgent(testConfig())
	produced, ok := a.Produces()[inflightEstimateKey]
	require.True(t, ok)
	require.IsType(t, int64(0), produced)
}
