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
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

func TestDiscoveredModelLimits(t *testing.T) {
	var limit atomic.Int64
	limit.Store(265088)
	var bad atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/models", r.URL.Path)
		if bad.Load() {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "glm", "max_model_len": limit.Load()}}})
	}))
	defer server.Close()
	host, port, _ := net.SplitHostPort(server.Listener.Addr().String())
	p, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{DiscoverModelLimits: true, MinModelLen: 250000, ContextLimitLabel: "render-limit"})
	require.NoError(t, err)
	ep := discoveredEndpointWithRank("a", host, port, 0, map[string]string{"render-limit": "265088"})
	require.NoError(t, p.Upsert(ep.GetMetadata()))
	_, err = p.Pick()
	require.Error(t, err, "unknown capacity must not be selectable")
	p.refreshModelLimits(context.Background(), server.Client(), "glm")
	_, err = p.Pick()
	require.NoError(t, err)
	limit.Store(240000)
	p.refreshModelLimits(context.Background(), server.Client(), "glm")
	_, err = p.Pick()
	require.Error(t, err, "shrunk capacity must be excluded")
	limit.Store(300000)
	p.refreshModelLimits(context.Background(), server.Client(), "glm")
	_, err = p.Pick()
	require.NoError(t, err)
	bad.Store(true)
	p.refreshModelLimits(context.Background(), server.Client(), "glm")
	_, err = p.Pick()
	require.Error(t, err, "failed metadata refresh must not retain an unverified target")
	p.Delete(ep.GetMetadata())
	require.Empty(t, p.capabilities)
}

func TestModelLimitLabelValidation(t *testing.T) {
	for _, v := range []string{"", "0", "-1", "not-a-number"} {
		p, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{ContextLimitLabel: "limit", MinModelLen: 200000})
		require.NoError(t, err)
		ep := discoveredEndpointWithRank("a", "127.0.0.1", "8000", 0, map[string]string{"limit": v})
		require.Error(t, p.Upsert(ep.GetMetadata()))
	}
	p, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{ContextLimitLabel: "limit", MinModelLen: 300000})
	require.NoError(t, err)
	ep := discoveredEndpointWithRank("a", "127.0.0.1", "8000", 0, map[string]string{"limit": "265088"})
	require.NoError(t, p.Upsert(ep.GetMetadata()))
	_, err = p.Pick()
	require.Error(t, err)
}

func TestRenderOnlyBudgetPreservesPayload(t *testing.T) {
	for _, completions := range []bool{false, true} {
		for _, enabled := range []bool{false, true} {
			t.Run(fmt.Sprintf("completions=%t/prefillOnly=%t", completions, enabled), func(t *testing.T) {
				payload := fwkrh.PayloadMap{"max_tokens": 32000, "max_completion_tokens": 32000, "min_tokens": 5}
				path := chatRenderPath
				if completions {
					payload["prompt"] = "unchanged"
					path = completionsRenderPath
				} else {
					payload["messages"] = []any{map[string]any{"role": "user", "content": "unchanged"}}
				}
				before, err := json.Marshal(payload)
				require.NoError(t, err)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&body)) {
						return
					}
					assert.Equal(t, path, r.URL.Path)
					wantMax, wantMin := float64(32000), float64(5)
					if enabled {
						wantMax, wantMin = 1, 0
					}
					assert.Equal(t, wantMax, body["max_tokens"])
					assert.Equal(t, wantMax, body["max_completion_tokens"])
					assert.Equal(t, wantMin, body["min_tokens"])
					if completions {
						assert.Equal(t, "unchanged", body["prompt"])
					} else {
						assert.Equal(t, payload["messages"], body["messages"])
					}
					w.Header().Set("Content-Type", "application/json")
					if completions {
						_, _ = w.Write([]byte(`[{"token_ids":[1,2,3]}]`))
					} else {
						_, _ = w.Write([]byte(`{"token_ids":[1,2,3]}`))
					}
				}))
				defer server.Close()
				r, err := newVLLMHTTPRenderer(&vllmConfig{URL: server.URL, PrefillOnly: enabled}, "glm")
				require.NoError(t, err)
				if completions {
					_, _, err = r.Render(context.Background(), payload)
				} else {
					_, _, err = r.RenderChat(context.Background(), payload)
				}
				require.NoError(t, err)
				after, err := json.Marshal(payload)
				require.NoError(t, err)
				require.Equal(t, before, after)
			})
		}
	}
}

func TestModelLimitProbeDeadline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { <-r.Context().Done() }))
	defer server.Close()
	host, port, _ := net.SplitHostPort(server.Listener.Addr().String())
	p, _ := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{DiscoverModelLimits: true})
	require.NoError(t, p.Upsert(discoveredEndpoint("a", host, port).GetMetadata()))
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	p.refreshModelLimits(ctx, server.Client(), "glm")
	_, err := p.Pick()
	require.Error(t, err)
}

func TestRenderRejectsEmptyTokenIDs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == completionsRenderPath {
			_, _ = w.Write([]byte(`[{"token_ids":[]}]`))
		} else {
			_, _ = w.Write([]byte(`{"token_ids":[]}`))
		}
	}))
	defer server.Close()
	renderer, err := newVLLMHTTPRenderer(&vllmConfig{URL: server.URL}, "glm")
	require.NoError(t, err)
	_, _, err = renderer.RenderChat(context.Background(), fwkrh.PayloadMap{"messages": []any{}})
	require.Error(t, err)
	_, _, err = renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "test"})
	require.Error(t, err)
}

func TestModelLimitStaleProbeCannotResurrectDeletedEndpoint(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-release
		_, _ = w.Write([]byte(`{"data":[{"id":"glm","max_model_len":300000}]}`))
	}))
	defer server.Close()
	host, port, _ := net.SplitHostPort(server.Listener.Addr().String())
	p, _ := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{DiscoverModelLimits: true})
	ep := discoveredEndpoint("a", host, port)
	require.NoError(t, p.Upsert(ep.GetMetadata()))
	done := make(chan struct{})
	go func() { p.refreshModelLimits(context.Background(), server.Client(), "glm"); close(done) }()
	<-started
	p.Delete(ep.GetMetadata())
	require.NoError(t, p.Upsert(ep.GetMetadata()))
	close(release)
	<-done
	_, err := p.Pick()
	require.Error(t, err)
}

func TestReadModelLimitRejectsWrongModelAndInvalidMetadata(t *testing.T) {
	for _, body := range []string{`{"data":[{"id":"other","max_model_len":300000}]}`, `{"data":[{"id":"glm","max_model_len":0}]}`, `{"data":[{"id":"glm","max_model_len":"300000"}]}`, "invalid"} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte(body)) }))
		_, err := readModelLimit(context.Background(), server.Client(), server.URL, "glm")
		require.Error(t, err)
		server.Close()
	}
}

func TestModelLimitEligibility(t *testing.T) {
	now := time.Now()
	for _, tc := range []struct {
		name       string
		config     endpointDiscoveryConfig
		capability renderCapability
		want       bool
	}{
		{"defaults", endpointDiscoveryConfig{}, renderCapability{}, true},
		{"unknown", endpointDiscoveryConfig{DiscoverModelLimits: true}, renderCapability{}, false},
		{"fresh", endpointDiscoveryConfig{DiscoverModelLimits: true, MinModelLen: 100}, renderCapability{observed: 100, refreshed: now}, true},
		{"stale", endpointDiscoveryConfig{DiscoverModelLimits: true}, renderCapability{observed: 100, refreshed: now.Add(-modelLimitFreshness - time.Second)}, false},
		{"label bounds observed", endpointDiscoveryConfig{DiscoverModelLimits: true, MinModelLen: 100}, renderCapability{declared: 99, observed: 200, refreshed: now}, false},
		{"observed bounds label", endpointDiscoveryConfig{DiscoverModelLimits: true, MinModelLen: 100}, renderCapability{declared: 200, observed: 99, refreshed: now}, false},
		{"label only", endpointDiscoveryConfig{MinModelLen: 100, ContextLimitLabel: "limit"}, renderCapability{declared: 100}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := newDiscoveredEndpointPicker(&tc.config)
			require.NoError(t, err)
			require.Equal(t, tc.want, p.eligible(&tc.capability, now))
		})
	}
	for _, cfg := range []endpointDiscoveryConfig{{MinModelLen: -1}, {MinModelLen: 100}} {
		_, err := newDiscoveredEndpointPicker(&cfg)
		require.Error(t, err)
	}
}
