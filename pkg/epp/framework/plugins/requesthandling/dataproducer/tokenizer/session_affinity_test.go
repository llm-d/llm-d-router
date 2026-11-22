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
	"slices"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

func TestProduce_DiscoverySessionAffinity(t *testing.T) {
	warmupCtx, cancel := context.WithCancel(t.Context())
	cancel()
	p, err := NewPlugin(warmupCtx, "affinity", &tokenizerPluginConfig{
		ModelName: "m",
		VLLM: &vllmConfig{EndpointDiscovery: &endpointDiscoveryConfig{
			LoadBalancer: &loadBalancerConfig{Type: sessionAffinityLoadBalancerType},
		}},
	})
	require.NoError(t, err)
	picker := p.endpointDiscovery.picker
	var unavailable atomic.Uint32
	var failedCalls atomic.Int32
	for id := uint32(1); id <= 3; id++ {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Empty(t, r.Header.Get(sessionaffinity.DefaultSessionIDHeader))
			if unavailable.Load() == id {
				failedCalls.Add(1)
				http.Error(w, "unavailable", http.StatusServiceUnavailable)
				return
			}
			_ = json.NewEncoder(w).Encode([]renderResponse{{TokenIDs: []uint32{id}}})
		}))
		t.Cleanup(server.Close)
		host, port, err := net.SplitHostPort(server.Listener.Addr().String())
		require.NoError(t, err)
		require.NoError(t, picker.Upsert(discoveredEndpoint("rank-"+strconv.Itoa(int(id)), host, port).GetMetadata()))
	}

	render := func(model string, headers map[string]string) uint32 {
		t.Helper()
		req := &scheduling.InferenceRequest{
			TargetModel: model,
			Headers:     headers,
			Body: &fwkrequest.InferenceRequestBody{
				RawBody:     []byte(`{"prompt":"hello"}`),
				Completions: &fwkrequest.CompletionsRequest{Prompt: fwkrequest.Prompt{Raw: "hello"}},
			},
		}
		require.NoError(t, p.Produce(t.Context(), req, nil))
		require.NotNil(t, req.Body.TokenizedRequest)
		require.Len(t, req.Body.TokenizedRequest.Prompts, 1)
		tokens := req.Body.TokenizedRequest.Prompts[0].TokenIDs
		require.Len(t, tokens, 1)
		return tokens[0]
	}

	session := map[string]string{sessionaffinity.DefaultSessionIDHeader: "turns-42"}
	primary := render("m", session)
	require.Equal(t, primary, render("m", session))
	require.Equal(t, primary, render("m", map[string]string{sessionaffinity.DefaultSessionIDHeader: "  turns-42\t"}))
	modelTargets := make(map[uint32]bool)
	for i := range 24 {
		modelTargets[render("model-"+strconv.Itoa(i), session)] = true
	}
	require.Greater(t, len(modelTargets), 1, "target model scopes a session's renderer selection")
	for _, headers := range []map[string]string{nil, {sessionaffinity.DefaultSessionIDHeader: " \t "}} {
		targets := make(map[uint32]bool)
		for range 6 {
			targets[render("m", headers)] = true
		}
		assert.Len(t, targets, 3, "an unkeyed request rotates through every renderer")
	}
	unavailable.Store(primary)
	alternate := render("m", session)
	require.NotEqual(t, primary, alternate)
	require.Equal(t, alternate, render("m", session))
	require.EqualValues(t, 2, failedCalls.Load())
	unavailable.Store(0)
	require.Equal(t, primary, render("m", session))
}

func TestSessionAffinityLoadBalancer_MembershipChanges(t *testing.T) {
	balancer, err := newEndpointLoadBalancer(sessionAffinityLoadBalancerType)
	require.NoError(t, err)
	endpoints := []string{"http://a:8000", "http://b:8000", "http://c:8000"}
	reordered := slices.Clone(endpoints)
	slices.Reverse(reordered)
	pick := func(key string, urls []string, excluded map[string]struct{}) string {
		t.Helper()
		url, err := balancer.Pick(key, urls, excluded)
		require.NoError(t, err)
		require.Contains(t, urls, url)
		return url
	}
	winners := make(map[string]bool)
	for i := range 96 {
		key := "session-" + strconv.Itoa(i)
		primary := pick(key, endpoints, nil)
		winners[primary] = true
		require.Equal(t, primary, pick(key, reordered, nil))

		withoutC := pick(key, endpoints[:2], nil)
		if primary != endpoints[2] {
			require.Equal(t, primary, withoutC)
		}
		require.Equal(t, withoutC, pick(key, endpoints, map[string]struct{}{endpoints[2]: {}}))
	}
	require.Len(t, winners, len(endpoints))
	_, err = balancer.Pick("session", endpoints, map[string]struct{}{endpoints[0]: {}, endpoints[1]: {}, endpoints[2]: {}})
	require.ErrorIs(t, err, errNoRenderEndpoints)
}
