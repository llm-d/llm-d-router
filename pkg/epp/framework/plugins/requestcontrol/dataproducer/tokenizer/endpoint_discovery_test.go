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

package tokenizer

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/test/utils"
)

func discoveredEndpoint(name, address, port string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Namespace: "default", Name: name},
		Name:    name,
		Address: address,
		Port:    port,
	}, nil)
}

func TestDiscoveredEndpointPicker_RoundRobin(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(roundRobinLoadBalancerType)
	require.NoError(t, err)

	require.NoError(t, picker.Upsert(discoveredEndpoint("rank-b", "10.0.0.2", "8001").GetMetadata()))
	require.NoError(t, picker.Upsert(discoveredEndpoint("rank-a", "10.0.0.1", "8000").GetMetadata()))

	for _, want := range []string{
		"http://10.0.0.1:8000",
		"http://10.0.0.2:8001",
		"http://10.0.0.1:8000",
		"http://10.0.0.2:8001",
	} {
		got, pickErr := picker.Pick()
		require.NoError(t, pickErr)
		assert.Equal(t, want, got)
	}
}

func TestDiscoveredEndpointPicker_TracksUpdatesAndDeletes(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(roundRobinLoadBalancerType)
	require.NoError(t, err)

	meta := discoveredEndpoint("rank-a", "10.0.0.1", "8000").GetMetadata()
	require.NoError(t, picker.Upsert(meta))

	meta = meta.Clone()
	meta.Address = "10.0.0.9"
	meta.Port = "8200"
	require.NoError(t, picker.Upsert(meta))

	got, err := picker.Pick()
	require.NoError(t, err)
	assert.Equal(t, "http://10.0.0.9:8200", got)

	picker.Delete(meta)
	_, err = picker.Pick()
	require.ErrorContains(t, err, "no vLLM render endpoints discovered")
}

type firstEndpointLoadBalancer struct{}

func (firstEndpointLoadBalancer) Pick(endpoints []string) (string, error) {
	return endpoints[0], nil
}

func TestDiscoveredEndpointPicker_LoadBalancerIsPluggable(t *testing.T) {
	const loadBalancerType = "test-first"
	endpointLoadBalancerFactories[loadBalancerType] = func() endpointLoadBalancer {
		return firstEndpointLoadBalancer{}
	}
	t.Cleanup(func() { delete(endpointLoadBalancerFactories, loadBalancerType) })

	picker, err := newDiscoveredEndpointPicker(loadBalancerType)
	require.NoError(t, err)
	require.NoError(t, picker.Upsert(discoveredEndpoint("rank-a", "10.0.0.1", "8000").GetMetadata()))

	got, err := picker.Pick()
	require.NoError(t, err)
	assert.Equal(t, "http://10.0.0.1:8000", got)
}

func TestDiscoveredEndpointPicker_RejectsInvalidEndpoint(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(roundRobinLoadBalancerType)
	require.NoError(t, err)

	for _, meta := range []*fwkdl.EndpointMetadata{
		nil,
		{Address: "10.0.0.1", Port: "8000"},
		discoveredEndpoint("rank-a", "", "8000").GetMetadata(),
		discoveredEndpoint("rank-a", "10.0.0.1", "bad").GetMetadata(),
	} {
		require.Error(t, picker.Upsert(meta))
	}
}

func TestVLLMHTTPRenderer_DiscoveryRoundRobin(t *testing.T) {
	newRenderServer := func(tokenID uint32) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_ = json.NewEncoder(w).Encode([]renderResponse{{TokenIDs: []uint32{tokenID}}})
		}))
	}
	serverA := newRenderServer(1)
	t.Cleanup(serverA.Close)
	serverB := newRenderServer(2)
	t.Cleanup(serverB.Close)

	renderer, err := newVLLMHTTPRenderer(&vllmConfig{EndpointDiscovery: &endpointDiscoveryConfig{}}, testHTTPModel)
	require.NoError(t, err)
	picker := renderer.endpointPicker.(*discoveredEndpointPicker)
	for name, server := range map[string]*httptest.Server{"rank-a": serverA, "rank-b": serverB} {
		address := server.Listener.Addr().(*net.TCPAddr)
		require.NoError(t, picker.Upsert(discoveredEndpoint(name, address.IP.String(), strconv.Itoa(address.Port)).GetMetadata()))
	}

	for _, want := range []uint32{1, 2, 1, 2} {
		tokens, _, renderErr := renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
		require.NoError(t, renderErr)
		require.Equal(t, [][]uint32{{want}}, tokens)
	}
}

type recordingRegistrar struct {
	registrations []fwkdl.PendingRegistration
}

func (r *recordingRegistrar) Register(registration fwkdl.PendingRegistration) error {
	r.registrations = append(r.registrations, registration)
	return nil
}

func TestPlugin_DiscoveryRegistersAndTracksEndpointNotifications(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p, err := NewPlugin(ctx, "discovered-tokenizer", &tokenizerPluginConfig{
		ModelName: "model",
		VLLM:      &vllmConfig{EndpointDiscovery: &endpointDiscoveryConfig{}},
	})
	require.NoError(t, err)

	var _ fwkdl.Registrant = p
	var _ fwkdl.EndpointExtractor = p

	registrar := &recordingRegistrar{}
	require.NoError(t, p.RegisterDependencies(registrar))
	require.Len(t, registrar.registrations, 1)
	registration := registrar.registrations[0]
	assert.Equal(t, p.TypedName(), registration.Owner)
	assert.Equal(t, sourcenotifications.EndpointNotificationSourceType, registration.SourceType)
	assert.Same(t, p, registration.Extractor)
	require.NotNil(t, registration.DefaultSource)

	ep := discoveredEndpoint("rank-a", "10.0.0.1", "8000")
	require.NoError(t, p.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}))

	got, err := p.endpointDiscovery.Pick()
	require.NoError(t, err)
	assert.Equal(t, "http://10.0.0.1:8000", got)

	require.NoError(t, p.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: ep,
	}))
	_, err = p.endpointDiscovery.Pick()
	require.Error(t, err)
}

func TestPlugin_StaticURLDoesNotRegisterForEndpointNotifications(t *testing.T) {
	p, err := NewPlugin(utils.NewTestContext(t), "static-tokenizer", &tokenizerPluginConfig{
		ModelName: "model",
		VLLM:      &vllmConfig{URL: "http://localhost:8000"},
	})
	require.NoError(t, err)

	registrar := &recordingRegistrar{}
	require.NoError(t, p.RegisterDependencies(registrar))
	assert.Empty(t, registrar.registrations)
}

func TestPluginFactory_EndpointDiscoveryValidation(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{
			name: "accepts endpoint discovery",
			parameters: `{
				"modelName": "m",
				"vllm": {"endpointDiscovery": {"loadBalancer": {"type": "round-robin"}}}
			}`,
		},
		{
			name: "rejects URL with endpoint discovery",
			parameters: `{
				"modelName": "m",
				"vllm": {"url": "http://localhost:8000", "endpointDiscovery": {}}
			}`,
			wantErr: "only one of 'url' or 'endpointDiscovery'",
		},
		{
			name: "rejects unknown load balancer",
			parameters: `{
				"modelName": "m",
				"vllm": {"endpointDiscovery": {"loadBalancer": {"type": "random"}}}
			}`,
			wantErr: "unsupported load balancer",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			handle := plugin.NewEppHandle(ctx, nil)
			p, err := PluginFactory("test", plugin.StrictDecoder([]byte(tt.parameters)), handle)
			if tt.wantErr == "" {
				require.NoError(t, err)
				assert.NotNil(t, p)
				return
			}
			require.ErrorContains(t, err, tt.wantErr)
			assert.Nil(t, p)
		})
	}
}
