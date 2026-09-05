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
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	dlruntime "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/test/utils"
)

func discoveredEndpoint(name, address, port string) fwkdl.Endpoint {
	return discoveredEndpointWithRank(name, address, port, 0, nil)
}

func discoveredEndpointWithRank(name, address, port string, rank int, labels map[string]string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:        types.NamespacedName{Namespace: "default", Name: name},
		Name:      name,
		Address:   address,
		Port:      port,
		Labels:    labels,
		RankIndex: rank,
	}, nil)
}

func TestDiscoveredEndpointPicker_RoundRobinUsesInferencePortsByDefault(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
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
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
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

func TestDiscoveredEndpointPicker_ExclusionsPreserveCursor(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
	require.NoError(t, err)
	for _, name := range []string{"a", "b", "c"} {
		require.NoError(t, picker.Upsert(discoveredEndpoint(name, name, "8000").GetMetadata()))
	}
	first, err := picker.Pick()
	require.NoError(t, err)
	assert.Equal(t, "http://a:8000", first)
	second, err := picker.PickExcluding(map[string]struct{}{"http://a:8000": {}, "http://b:8000": {}})
	require.NoError(t, err)
	assert.Equal(t, "http://c:8000", second)
	_, err = picker.PickExcluding(map[string]struct{}{"http://a:8000": {}, "http://b:8000": {}, "http://c:8000": {}})
	require.ErrorIs(t, err, errNoRenderEndpoints)
	third, err := picker.Pick()
	require.NoError(t, err)
	assert.Equal(t, "http://a:8000", third)
}

type firstEndpointLoadBalancer struct{}

func (firstEndpointLoadBalancer) Pick(endpoints []string, excluded map[string]struct{}) (string, error) {
	for _, endpoint := range endpoints {
		if _, skip := excluded[endpoint]; !skip {
			return endpoint, nil
		}
	}
	return "", errNoRenderEndpoints
}

type endpointLoadBalancerFunc func(endpoints []string, excluded map[string]struct{}) (string, error)

func (f endpointLoadBalancerFunc) Pick(endpoints []string, excluded map[string]struct{}) (string, error) {
	return f(endpoints, excluded)
}

func TestDiscoveredEndpointPicker_LoadBalancerIsPluggable(t *testing.T) {
	const loadBalancerType = "test-first"
	endpointLoadBalancerFactories[loadBalancerType] = func() endpointLoadBalancer {
		return firstEndpointLoadBalancer{}
	}
	t.Cleanup(func() { delete(endpointLoadBalancerFactories, loadBalancerType) })

	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{
		LoadBalancer: &loadBalancerConfig{Type: loadBalancerType},
	})
	require.NoError(t, err)
	require.NoError(t, picker.Upsert(discoveredEndpoint("rank-a", "10.0.0.1", "8000").GetMetadata()))

	got, err := picker.Pick()
	require.NoError(t, err)
	assert.Equal(t, "http://10.0.0.1:8000", got)
}

func TestDiscoveredEndpointPicker_RejectsInvalidEndpoint(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
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

func TestDiscoveredEndpointPicker_PortRules(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{
		PortRules: []endpointPortRule{
			{
				Selector: metav1.LabelSelector{MatchLabels: map[string]string{"llm-d.ai/role": "prefill"}},
				BasePort: 8000,
			},
			{
				Selector: metav1.LabelSelector{MatchLabels: map[string]string{"llm-d.ai/role": "decode"}},
				BasePort: 8200,
			},
			{BasePort: 9000},
		},
	})
	require.NoError(t, err)

	require.NoError(t, picker.Upsert(discoveredEndpointWithRank(
		"prefill-rank", "10.0.0.1", "9002", 2, map[string]string{"llm-d.ai/role": "prefill"},
	).GetMetadata()))
	require.NoError(t, picker.Upsert(discoveredEndpointWithRank(
		"decode-rank", "10.0.0.2", "8003", 3, map[string]string{"llm-d.ai/role": "decode"},
	).GetMetadata()))
	require.NoError(t, picker.Upsert(discoveredEndpointWithRank("other-rank", "::1", "8001", 1, nil).GetMetadata()))

	for _, want := range []string{"http://10.0.0.2:8203", "http://[::1]:9001", "http://10.0.0.1:8002"} {
		got, pickErr := picker.Pick()
		require.NoError(t, pickErr)
		assert.Equal(t, want, got)
	}
}

func TestDiscoveredEndpointPicker_PortRulesRejectInvalidConfiguration(t *testing.T) {
	tests := []struct {
		name    string
		config  *endpointDiscoveryConfig
		wantErr string
	}{
		{
			name: "invalid selector",
			config: &endpointDiscoveryConfig{PortRules: []endpointPortRule{{
				Selector: metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key: "role", Operator: "not-an-operator",
				}}},
				BasePort: 8200,
			}}},
			wantErr: "invalid selector",
		},
		{
			name:    "invalid base port",
			config:  &endpointDiscoveryConfig{PortRules: []endpointPortRule{{BasePort: 0}}},
			wantErr: "invalid base port",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newDiscoveredEndpointPicker(tt.config)
			require.ErrorContains(t, err, tt.wantErr)
		})
	}
}

func TestDiscoveredEndpointPicker_PortRulesRejectUnmatchedAndInvalidRank(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{
		PortRules: []endpointPortRule{{
			Selector: metav1.LabelSelector{MatchLabels: map[string]string{"llm-d.ai/role": "decode"}},
			BasePort: 65535,
		}},
	})
	require.NoError(t, err)

	require.ErrorContains(t, picker.Upsert(discoveredEndpointWithRank(
		"prefill", "10.0.0.1", "8000", 0, map[string]string{"llm-d.ai/role": "prefill"},
	).GetMetadata()), "does not match any port rule")
	require.ErrorContains(t, picker.Upsert(discoveredEndpointWithRank(
		"negative", "10.0.0.1", "8000", -1, map[string]string{"llm-d.ai/role": "decode"},
	).GetMetadata()), "invalid rank index")
	require.ErrorContains(t, picker.Upsert(discoveredEndpointWithRank(
		"overflow", "10.0.0.1", "8000", 1, map[string]string{"llm-d.ai/role": "decode"},
	).GetMetadata()), "resolved render port")
}

func TestDiscoveredEndpointPicker_ReleasesLockBeforeLoadBalancing(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
	require.NoError(t, err)
	require.NoError(t, picker.Upsert(discoveredEndpoint("rank-a", "10.0.0.1", "8000").GetMetadata()))

	picker.loadBalancer = endpointLoadBalancerFunc(func(endpoints []string, _ map[string]struct{}) (string, error) {
		locked := picker.mu.TryLock()
		if locked {
			picker.mu.Unlock()
		}
		assert.True(t, locked, "picker lock must not cover the pluggable load balancer")
		return endpoints[0], nil
	})

	_, err = picker.Pick()
	require.NoError(t, err)
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

func TestVLLMHTTPRenderer_DiscoveryRetriesDifferentEndpoint(t *testing.T) {
	var failedCalls atomic.Int32
	failedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		failedCalls.Add(1)
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	t.Cleanup(failedServer.Close)

	var successfulCalls atomic.Int32
	successfulServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		successfulCalls.Add(1)
		_ = json.NewEncoder(w).Encode([]renderResponse{{TokenIDs: []uint32{42}}})
	}))
	t.Cleanup(successfulServer.Close)

	renderer, err := newVLLMHTTPRenderer(&vllmConfig{EndpointDiscovery: &endpointDiscoveryConfig{}}, testHTTPModel)
	require.NoError(t, err)
	picker := renderer.endpointPicker.(*discoveredEndpointPicker)
	for name, server := range map[string]*httptest.Server{"rank-a": failedServer, "rank-b": successfulServer} {
		address := server.Listener.Addr().(*net.TCPAddr)
		require.NoError(t, picker.Upsert(discoveredEndpoint(name, address.IP.String(), strconv.Itoa(address.Port)).GetMetadata()))
	}

	tokens, _, err := renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
	require.NoError(t, err)
	assert.Equal(t, [][]uint32{{42}}, tokens)
	assert.Equal(t, int32(1), failedCalls.Load())
	assert.Equal(t, int32(1), successfulCalls.Load())
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func TestVLLMHTTPRenderer_DiscoveryRetriesTransportFailureWithinRequestTimeout(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
	require.NoError(t, err)
	require.NoError(t, picker.Upsert(discoveredEndpoint("rank-a", "10.0.0.1", "8000").GetMetadata()))
	require.NoError(t, picker.Upsert(discoveredEndpoint("rank-b", "10.0.0.2", "8000").GetMetadata()))

	var deadlines []time.Time
	renderer := &vllmHTTPRenderer{
		client: &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			deadline, ok := request.Context().Deadline()
			require.True(t, ok)
			deadlines = append(deadlines, deadline)
			if request.URL.Host == "10.0.0.1:8000" {
				return nil, errors.New("connection refused")
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`[{"token_ids":[7]}]`)),
				Request:    request,
			}, nil
		})},
		endpointPicker: picker,
		modelName:      testHTTPModel,
		timeout:        time.Second,
	}

	tokens, _, err := renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
	require.NoError(t, err)
	assert.Equal(t, [][]uint32{{7}}, tokens)
	require.Len(t, deadlines, 2)
	assert.Equal(t, deadlines[0], deadlines[1], "retry must retain the original request deadline")
}

func TestVLLMHTTPRenderer_DiscoveryRetriesSlowEndpointWithAttemptTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
		require.NoError(t, err)
		require.NoError(t, picker.Upsert(discoveredEndpoint("rank-a", "10.0.0.1", "8000").GetMetadata()))
		require.NoError(t, picker.Upsert(discoveredEndpoint("rank-b", "10.0.0.2", "8000").GetMetadata()))
		var hosts []string
		start := time.Now()
		renderer := &vllmHTTPRenderer{
			endpointPicker: picker,
			timeout:        time.Second,
			attemptTimeout: 100 * time.Millisecond,
			client: &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
				hosts = append(hosts, req.URL.Host)
				if req.URL.Host == "10.0.0.1:8000" {
					<-req.Context().Done()
					return nil, req.Context().Err()
				}
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`[{"token_ids":[7]}]`))}, nil
			})},
		}
		tokens, _, err := renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
		require.NoError(t, err)
		assert.Equal(t, [][]uint32{{7}}, tokens)
		assert.Equal(t, []string{"10.0.0.1:8000", "10.0.0.2:8000"}, hosts)
		assert.Less(t, time.Since(start), time.Second)
	})
}

func TestVLLMHTTPRenderer_DiscoveryAttemptsAreBounded(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
	require.NoError(t, err)
	for _, name := range []string{"a", "b", "c"} {
		require.NoError(t, picker.Upsert(discoveredEndpoint(name, name, "8000").GetMetadata()))
	}
	var hosts []string
	renderer := &vllmHTTPRenderer{
		endpointPicker: picker,
		timeout:        time.Second,
		client: &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			hosts = append(hosts, req.URL.Host)
			return nil, errors.New("connection refused")
		})},
	}
	_, _, err = renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
	require.ErrorContains(t, err, "connection refused")
	require.Len(t, hosts, 2)
	assert.NotEqual(t, hosts[0], hosts[1])
}

func TestVLLMHTTPRenderer_DiscoveryRetriesDoNotStarveHealthyEndpoints(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
	require.NoError(t, err)
	for _, name := range []string{"a", "b", "c"} {
		require.NoError(t, picker.Upsert(discoveredEndpoint(name, name, "8000").GetMetadata()))
	}
	calls := map[string]int{}
	renderer := &vllmHTTPRenderer{
		endpointPicker: picker,
		timeout:        time.Second,
		client: &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			calls[req.URL.Hostname()]++
			if req.URL.Hostname() == "a" {
				return &http.Response{StatusCode: http.StatusServiceUnavailable, Body: io.NopCloser(strings.NewReader("unavailable"))}, nil
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`[{"token_ids":[7]}]`))}, nil
		})},
	}
	for range 12 {
		_, _, err := renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
		require.NoError(t, err)
	}
	assert.Equal(t, 6, calls["b"])
	assert.Equal(t, 6, calls["c"])
}

type failingRenderBody struct {
	err error
}

func (b failingRenderBody) Read([]byte) (int, error) { return 0, b.err }
func (b failingRenderBody) Close() error             { return nil }

func TestVLLMHTTPRenderer_DiscoveryRetriesResponseBodyNetworkError(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
	require.NoError(t, err)
	for _, name := range []string{"a", "b"} {
		require.NoError(t, picker.Upsert(discoveredEndpoint(name, name, "8000").GetMetadata()))
	}
	var hosts []string
	renderer := &vllmHTTPRenderer{
		endpointPicker: picker,
		timeout:        time.Second,
		client: &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			hosts = append(hosts, req.URL.Hostname())
			body := io.NopCloser(strings.NewReader(`[{"token_ids":[7]}]`))
			if req.URL.Hostname() == "a" {
				body = failingRenderBody{err: &net.OpError{Op: "read", Net: "tcp", Err: syscall.ECONNRESET}}
			}
			return &http.Response{StatusCode: http.StatusOK, Body: body}, nil
		})},
	}
	tokens, _, err := renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
	require.NoError(t, err)
	assert.Equal(t, [][]uint32{{7}}, tokens)
	assert.Equal(t, []string{"a", "b"}, hosts)
}

func TestVLLMHTTPRenderer_DiscoveryPreservesFullTimeoutByDefault(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
		require.NoError(t, err)
		for _, name := range []string{"a", "b"} {
			require.NoError(t, picker.Upsert(discoveredEndpoint(name, name, "8000").GetMetadata()))
		}
		calls := 0
		renderer := &vllmHTTPRenderer{
			endpointPicker: picker,
			timeout:        5 * time.Second,
			client: &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
				calls++
				select {
				case <-time.After(3 * time.Second):
					return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`[{"token_ids":[7]}]`))}, nil
				case <-req.Context().Done():
					return nil, req.Context().Err()
				}
			})},
		}
		tokens, _, err := renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
		require.NoError(t, err)
		assert.Equal(t, [][]uint32{{7}}, tokens)
		assert.Equal(t, 1, calls)
	})
}

func TestVLLMHTTPRenderer_DiscoveryAttemptTimeoutConfiguration(t *testing.T) {
	for _, value := range []string{"", "100ms", "0s", "-1s", "invalid"} {
		t.Run(value, func(t *testing.T) {
			renderer, err := newVLLMHTTPRenderer(&vllmConfig{
				EndpointDiscovery: &endpointDiscoveryConfig{AttemptTimeout: value},
			}, testHTTPModel)
			switch value {
			case "":
				require.NoError(t, err)
				assert.Zero(t, renderer.attemptTimeout)
			case "100ms":
				require.NoError(t, err)
				assert.Equal(t, 100*time.Millisecond, renderer.attemptTimeout)
			default:
				require.ErrorContains(t, err, "invalid 'endpointDiscovery.attemptTimeout'")
			}
		})
	}
}

func TestVLLMHTTPRenderer_DiscoveryHonorsCallerDeadline(t *testing.T) {
	for _, endpointCount := range []int{1, 2} {
		t.Run(strconv.Itoa(endpointCount), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
				require.NoError(t, err)
				for i := range endpointCount {
					name := strconv.Itoa(i)
					require.NoError(t, picker.Upsert(discoveredEndpoint(name, name, "8000").GetMetadata()))
				}
				calls := 0
				renderer := &vllmHTTPRenderer{
					endpointPicker: picker,
					timeout:        time.Second,
					client: &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
						calls++
						<-req.Context().Done()
						return nil, req.Context().Err()
					})},
				}
				ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
				defer cancel()
				start := time.Now()
				_, _, err = renderer.Render(ctx, fwkrh.PayloadMap{"prompt": "hello"})
				require.ErrorIs(t, err, context.DeadlineExceeded)
				assert.Equal(t, 1, calls, "caller deadline must stop retries")
				assert.Equal(t, 200*time.Millisecond, time.Since(start))
			})
		})
	}
}

func TestVLLMHTTPRenderer_DiscoveryDoesNotRetryDeterministicClientError(t *testing.T) {
	var clientErrorCalls atomic.Int32
	clientErrorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		clientErrorCalls.Add(1)
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	t.Cleanup(clientErrorServer.Close)

	var alternateCalls atomic.Int32
	alternateServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		alternateCalls.Add(1)
		_ = json.NewEncoder(w).Encode([]renderResponse{{TokenIDs: []uint32{42}}})
	}))
	t.Cleanup(alternateServer.Close)

	renderer, err := newVLLMHTTPRenderer(&vllmConfig{EndpointDiscovery: &endpointDiscoveryConfig{}}, testHTTPModel)
	require.NoError(t, err)
	picker := renderer.endpointPicker.(*discoveredEndpointPicker)
	for name, server := range map[string]*httptest.Server{"rank-a": clientErrorServer, "rank-b": alternateServer} {
		address := server.Listener.Addr().(*net.TCPAddr)
		require.NoError(t, picker.Upsert(discoveredEndpoint(name, address.IP.String(), strconv.Itoa(address.Port)).GetMetadata()))
	}

	_, _, err = renderer.Render(context.Background(), fwkrh.PayloadMap{"prompt": "hello"})
	require.ErrorContains(t, err, "status 400")
	assert.Equal(t, int32(1), clientErrorCalls.Load())
	assert.Zero(t, alternateCalls.Load())
}

func TestVLLMHTTPRenderer_RetryableStatus(t *testing.T) {
	for _, status := range []int{http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusServiceUnavailable} {
		assert.True(t, isRetryableRenderStatus(status), "status %d", status)
	}
	for _, status := range []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound, 600} {
		assert.False(t, isRetryableRenderStatus(status), "status %d", status)
	}
}

type errorEndpointPicker struct {
	err error
}

func (p errorEndpointPicker) Pick() (string, error) {
	return "", p.err
}

func TestVLLMHTTPRenderer_EndpointPickErrorHasContext(t *testing.T) {
	pickErr := errors.New("picker failed")
	renderer := &vllmHTTPRenderer{endpointPicker: errorEndpointPicker{err: pickErr}}

	err := renderer.postJSON(context.Background(), completionsRenderPath, map[string]any{}, time.Second, &renderResponse{})
	require.ErrorContains(t, err, "pick render endpoint")
	assert.ErrorIs(t, err, pickErr)
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

	registrar := &recordingRegistrar{}
	require.NoError(t, p.RegisterDependencies(registrar))
	require.Len(t, registrar.registrations, 1)
	registration := registrar.registrations[0]
	assert.Equal(t, p.TypedName(), registration.Owner)
	assert.Equal(t, sourcenotifications.EndpointNotificationSourceType, registration.SourceType)
	require.NotNil(t, registration.DefaultSource)
	handler, ok := registration.Extractor.(*endpointDiscoveryHandler)
	require.True(t, ok)

	ep := discoveredEndpoint("rank-a", "10.0.0.1", "8000")
	require.NoError(t, handler.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}))

	got, err := handler.picker.Pick()
	require.NoError(t, err)
	assert.Equal(t, "http://10.0.0.1:8000", got)

	require.NoError(t, handler.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: ep,
	}))
	_, err = handler.picker.Pick()
	require.Error(t, err)
}

func TestEndpointDiscoveryHandler_IgnoresStaleDelete(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{})
	require.NoError(t, err)
	handler := newEndpointDiscoveryHandler(plugin.TypedName{Type: PluginType, Name: "test"}, picker)
	oldEndpoint := discoveredEndpoint("rank-a", "10.0.0.1", "8000")
	replacement := discoveredEndpoint("rank-a", "10.0.0.2", "8000")

	require.NoError(t, handler.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: oldEndpoint}))
	require.NoError(t, handler.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: replacement}))
	require.NoError(t, handler.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: oldEndpoint}))

	got, err := picker.Pick()
	require.NoError(t, err)
	assert.Equal(t, "http://10.0.0.2:8000", got)

	require.NoError(t, handler.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: replacement}))
	_, err = picker.Pick()
	require.ErrorContains(t, err, "no vLLM render endpoints discovered")
}

func TestEndpointDiscoveryHandler_RemovesInvalidUpdate(t *testing.T) {
	picker, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{PortRules: []endpointPortRule{{
		Selector: metav1.LabelSelector{MatchLabels: map[string]string{"role": "decode"}}, BasePort: 8200,
	}}})
	require.NoError(t, err)
	handler := newEndpointDiscoveryHandler(plugin.TypedName{Type: PluginType, Name: "test"}, picker)
	ep := discoveredEndpointWithRank("a", "10.0.0.1", "8000", 0, map[string]string{"role": "decode"})
	require.NoError(t, handler.Extract(context.Background(), fwkdl.EndpointEvent{Endpoint: ep}))
	ep.UpdateMetadata(discoveredEndpoint("a", "10.0.0.1", "8000").GetMetadata())
	require.ErrorContains(t, handler.Extract(context.Background(), fwkdl.EndpointEvent{Endpoint: ep}), "does not match any port rule")
	_, err = picker.Pick()
	require.ErrorIs(t, err, errNoRenderEndpoints)
}

func TestPlugin_DiscoveryHandlersHavePerInstanceTypes(t *testing.T) {
	runtime := dlruntime.NewRuntime(0)
	registrations := make([]fwkdl.PendingRegistration, 0, 2)
	for _, name := range []string{"first", "second"} {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		p, err := NewPlugin(ctx, name, &tokenizerPluginConfig{
			ModelName: "model",
			VLLM:      &vllmConfig{EndpointDiscovery: &endpointDiscoveryConfig{}},
		})
		require.NoError(t, err)

		registrar := &recordingRegistrar{}
		require.NoError(t, p.RegisterDependencies(registrar))
		require.Len(t, registrar.registrations, 1)
		registrations = append(registrations, registrar.registrations[0])
		require.NoError(t, p.RegisterDependencies(runtime))
	}

	assert.NotEqual(t, registrations[0].Extractor.TypedName().Type, registrations[1].Extractor.TypedName().Type)
	ctx := utils.NewTestContext(t)
	require.NoError(t, runtime.Configure(nil, log.FromContext(ctx)))
	ep := runtime.NewEndpoint(ctx, discoveredEndpoint("a", "10.0.0.1", "8000").GetMetadata())
	require.NotNil(t, ep)
	for _, registration := range registrations {
		picker := registration.Extractor.(*endpointDiscoveryHandler).picker
		url, err := picker.Pick()
		require.NoError(t, err)
		assert.Equal(t, "http://10.0.0.1:8000", url)
	}
	runtime.ReleaseEndpoint(ep)
	for _, registration := range registrations {
		_, err := registration.Extractor.(*endpointDiscoveryHandler).picker.Pick()
		require.ErrorIs(t, err, errNoRenderEndpoints)
	}
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
				"vllm": {"endpointDiscovery": {
					"attemptTimeout": "1s",
					"portRules": [{"selector": {"matchLabels": {"llm-d.ai/role": "decode"}}, "basePort": 8200}],
					"loadBalancer": {"type": "round-robin"}
				}}
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
