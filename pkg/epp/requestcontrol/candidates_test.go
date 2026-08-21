/*
Copyright 2025 The Kubernetes Authors.

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

package requestcontrol

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

func TestDatastoreEndpointCandidates_LocateReturnsAllPods(t *testing.T) {
	t.Parallel()

	endpointA := makeMockEndpoint("pod-a", "10.0.0.1")
	endpointB := makeMockEndpoint("pod-b", "10.0.0.2")
	endpointC := makeMockEndpoint("pod-c", "10.0.0.3")
	allEndpoints := []fwkdl.Endpoint{endpointA, endpointB, endpointC}
	mockDS := &mockDatastore{pods: allEndpoints}

	candidates := NewDatastoreEndpointCandidates(mockDS)
	got := candidates.Locate(context.Background(), nil)

	gotIPs := make([]string, len(got))
	for idx, ep := range got {
		gotIPs[idx] = ep.GetMetadata().GetIPAddress()
	}
	assert.ElementsMatch(t, []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"}, gotIPs)
}

// --- CachedEndpointCandidates Tests ---

func TestCachedEndpointCandidates_CachingBehavior(t *testing.T) {
	t.Parallel()

	mockDelegate := &mockEndpointCandidates{
		result: []fwkdl.Endpoint{makeMockEndpoint("p1", "1.1.1.1")},
	}

	// Use a short TTL for testing.
	ttl := 20 * time.Millisecond
	cached := NewCachedEndpointCandidates(context.Background(), mockDelegate, ttl)

	// 1. First Call: Should hit delegate
	res1 := cached.Locate(context.Background(), nil)
	require.Len(t, res1, 1)
	assert.Equal(t, 1, mockDelegate.callCount(), "Expected delegate to be called on first access")

	// 2. Second Call (Immediate): Should hit cache
	res2 := cached.Locate(context.Background(), nil)
	require.Len(t, res2, 1)
	assert.Equal(t, 1, mockDelegate.callCount(), "Expected delegate NOT to be called again (cache hit)")

	// 3. Wait for Expiry
	time.Sleep(ttl * 2)

	// 4. Third Call (Expired): Should hit delegate again
	res3 := cached.Locate(context.Background(), nil)
	require.Len(t, res3, 1)
	assert.Equal(t, 2, mockDelegate.callCount(), "Expected delegate to be called after TTL expiry")
}

func TestCachedEndpointCandidates_IgnoresRequestMetadata(t *testing.T) {
	t.Parallel()

	// The Delegate no longer varies by request metadata, so the cache key is
	// the same regardless of the metadata handed to Locate.
	mockDelegate := &mockEndpointCandidates{
		result: []fwkdl.Endpoint{makeMockEndpoint("p1", "1.1.1.1")},
	}
	cached := NewCachedEndpointCandidates(context.Background(), mockDelegate, time.Minute)

	cached.Locate(context.Background(), nil)
	cached.Locate(context.Background(), map[string]any{"any": "thing"})
	cached.Locate(context.Background(), map[string]any{"other": "value"})

	assert.Equal(t, 1, mockDelegate.callCount(), "Cache key must not depend on request metadata")
}

func TestCachedEndpointCandidates_Concurrency_ThunderingHerd(t *testing.T) {
	t.Parallel()

	// Simulate a slow delegate to exacerbate race conditions.
	mockDelegate := &mockEndpointCandidates{
		delay: 10 * time.Millisecond,
		result: []fwkdl.Endpoint{
			makeMockEndpoint("p1", "1.1.1.1"),
		},
	}

	cached := NewCachedEndpointCandidates(context.Background(), mockDelegate, 100*time.Millisecond)

	concurrency := 50
	var wg sync.WaitGroup
	wg.Add(concurrency)

	start := make(chan struct{})

	// Spawn N routines trying to Locate simultaneously.
	for range concurrency {
		go func() {
			defer wg.Done()
			<-start // Synchronize start.
			res := cached.Locate(context.Background(), nil)
			assert.Len(t, res, 1)
		}()
	}

	close(start) // Release the hounds.
	wg.Wait()

	// Strict double-checked locking guarantees the delegate is called exactly once.
	assert.Equal(t, 1, mockDelegate.callCount(), "Delegate should be called exactly once despite concurrent access")
}

// --- Helpers & Mocks ---

// mockEndpointCandidates implements contracts.EndpointCandidates.
type mockEndpointCandidates struct {
	mu     sync.Mutex
	calls  int
	delay  time.Duration
	result []fwkdl.Endpoint
}

func (m *mockEndpointCandidates) Locate(ctx context.Context, _ map[string]any) []fwkdl.Endpoint {
	m.mu.Lock()
	m.calls++
	delay := m.delay
	result := m.result
	m.mu.Unlock()

	if delay > 0 {
		time.Sleep(delay)
	}
	return result
}

func (m *mockEndpointCandidates) callCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

func makeMockEndpoint(name, ip string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Namespace: "default", Name: name},
		Address: ip,
	}, nil)
}
