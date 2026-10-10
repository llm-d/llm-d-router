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
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	eppdatalayer "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
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

// Factory starts the idle session sweep for the handle's lifetime.
func TestFactoryStartsSweep(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	params := json.NewDecoder(strings.NewReader(`{"evictionTtlSeconds": 0.01, "evictionSweepSeconds": 0.01, "headWaitStarvationMs": 0}`))
	p, err := Factory("thunder", params, utils.NewTestHandle(ctx))
	require.NoError(t, err)
	a := p.(*ThunderAgent)

	runTurn(t, a, "s1", schedEndpoint("pod-a", 0, 0), 400, 300)
	require.Eventually(t, func() bool {
		_, ok := sessionOf(a, "s1")
		return !ok
	}, 5*time.Second, 10*time.Millisecond)
}

func TestFactoryRejectsInvalidSweepInterval(t *testing.T) {
	_, err := Factory("thunder", json.NewDecoder(strings.NewReader(`{"evictionSweepSeconds": 0}`)), nil)
	require.ErrorContains(t, err, "evictionSweepSeconds")
}

// DumpState reports counts and per-endpoint totals but no session ids.
func TestDumpState(t *testing.T) {
	a := newTestAgent(testConfig())
	runTurn(t, a, "secret-session", schedEndpoint("pod-a", 0, 0), 400, 300)
	_ = startTurn(t, a, "other", schedEndpoint("pod-a", 0, 0), 800)

	raw, err := a.DumpState()
	require.NoError(t, err)
	require.NotContains(t, string(raw), "secret-session")

	var got stateDump
	require.NoError(t, json.Unmarshal(raw, &got))
	require.Equal(t, stateDump{
		RunningSessions: 1,
		IdleSessions:    1,
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
