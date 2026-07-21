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

package eviction

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// --- Test helpers ---

func makeSchedulingResult() *scheduling.SchedulingResult {
	endpoint := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "8000"},
		nil, nil,
	)
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {
				TargetEndpoints: []scheduling.Endpoint{endpoint},
			},
		},
	}
}

func makeInferenceRequest(requestID string, priority int) *scheduling.InferenceRequest { //nolint:unparam
	return &scheduling.InferenceRequest{
		RequestID: requestID,
		Headers: map[string]string{
			reqcommon.RequestIDHeaderKey: requestID,
		},
		Objectives: scheduling.RequestObjectives{Priority: priority},
	}
}

// --- Tests ---

func TestRequestEvictor_PreRequest_CreatesEvictChannel(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &NoOpEvictor{})

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	evictCh := re.EvictionRegistry().Get("req-1")
	require.NotNil(t, evictCh, "EvictCh should be registered after PreRequest")

	assert.Equal(t, 1, re.queue.InFlightLen())
	assert.Equal(t, 1, re.queue.EvictableLen())
}

func TestRequestEvictor_ResponseBody_DeregistersEvictChannel(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &NoOpEvictor{})

	ctx := context.Background()
	request := makeInferenceRequest("req-1", -1)
	re.PreRequest(ctx, request, makeSchedulingResult())
	require.NotNil(t, re.EvictionRegistry().Get("req-1"))

	re.ResponseBody(ctx, request, &requestcontrol.Response{EndOfStream: true}, nil)

	assert.Nil(t, re.EvictionRegistry().Get("req-1"), "EvictCh should be deregistered after completion")
	assert.Equal(t, 0, re.queue.InFlightLen())
}

func TestRequestEvictor_EvictN_ClosesEvictChannel(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	evictCh := re.EvictionRegistry().Get("req-1")
	require.NotNil(t, evictCh)

	evicted, err := re.EvictN(ctx, 1, 0)
	require.NoError(t, err)
	require.Equal(t, []string{"req-1"}, evicted)

	select {
	case <-evictCh:
	default:
		t.Fatal("eviction channel should be closed after EvictN")
	}

	assert.Equal(t, errcommon.RequestDroppedReasonEvicted, re.EvictionRegistry().GetReason("req-1"),
		"EvictN should store the removal reason in the registry")
}

func TestRequestEvictor_EvictN_ReTracksOnFailure(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &failingEvictor{})

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	evicted, err := re.EvictN(ctx, 1, 0)
	require.NoError(t, err)
	assert.Empty(t, evicted)

	assert.Equal(t, 1, re.queue.EvictableLen())
}

func TestRequestEvictor_RaceBetweenEvictAndCompletion(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()

	requests := make([]*scheduling.InferenceRequest, 10)
	for i := range requests {
		requests[i] = makeInferenceRequest("req-"+string(rune('a'+i)), -1)
		re.PreRequest(ctx, requests[i], makeSchedulingResult())
	}

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for range 5 {
			_, _ = re.EvictN(ctx, 1, 0)
			time.Sleep(time.Millisecond)
		}
	}()

	go func() {
		defer wg.Done()
		for _, req := range requests {
			re.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
			time.Sleep(time.Millisecond)
		}
	}()

	wg.Wait()

	inFlight, evictable := re.Stats()
	assert.GreaterOrEqual(t, inFlight, 0)
	assert.GreaterOrEqual(t, evictable, 0)
	assert.GreaterOrEqual(t, inFlight, evictable)
}

func TestRequestEvictor_CtxCancellationTriggersCleanup(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx, cancel := context.WithCancel(context.Background())
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	// Verify tracked.
	assert.Equal(t, 1, re.queue.InFlightLen())
	assert.Equal(t, 1, re.queue.EvictableLen())
	assert.NotNil(t, re.EvictionRegistry().Get("req-1"))

	// Cancel the context — the goroutine in PreRequest should fire and call cleanupRequest.
	cancel()

	// Wait briefly for the goroutine to execute.
	assert.Eventually(t, func() bool {
		return re.queue.InFlightLen() == 0
	}, time.Second, 10*time.Millisecond, "InFlightLen should be 0 after context cancellation")

	assert.Equal(t, 0, re.queue.EvictableLen(), "EvictableLen should be 0 after context cancellation")
	assert.Nil(t, re.EvictionRegistry().Get("req-1"), "EvictionRegistry should be cleaned up after context cancellation")
}

func TestRequestEvictor_CleanupCallsEvictorCleanup(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	// Evict to create a sync.Once entry in the evictor.
	_, _ = re.EvictN(ctx, 1, 0)

	// Complete the request — this should call cleanupRequest which calls evictor.Cleanup.
	// After cleanup, the sync.Once entry for "req-1" should be removed.
	// We verify this indirectly: if Cleanup wasn't called, the sync.Once map would retain "req-1".
	// Re-track and re-evict with a new channel — if the old sync.Once is still there,
	// the new channel won't close (sync.Once already fired).
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	// The ResponseBody from the first eviction fires via defer in real code.
	// Simulate it here.
	re.ResponseBody(ctx, makeInferenceRequest("req-1", -1), &requestcontrol.Response{EndOfStream: true}, nil)

	// Re-track and evict again.
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())
	evictCh := re.EvictionRegistry().Get("req-1")
	require.NotNil(t, evictCh)

	evicted, err := re.EvictN(ctx, 1, 0)
	require.NoError(t, err)
	require.Len(t, evicted, 1)

	// New channel should be closed (Cleanup removed old sync.Once).
	select {
	case <-evictCh:
	default:
		t.Fatal("eviction channel should be closed — Cleanup should have removed stale sync.Once")
	}
}

func TestRequestEvictor_CleanupWorksWithNoOpEvictor(t *testing.T) {
	t.Parallel()
	// NoOpEvictor does not implement EvictorWithCleanup. cleanupRequest should not panic.
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &NoOpEvictor{})

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())
	re.ResponseBody(ctx, makeInferenceRequest("req-1", -1), &requestcontrol.Response{EndOfStream: true}, nil)

	assert.Equal(t, 0, re.queue.InFlightLen())
}

// failingEvictor always returns an error.
type failingEvictor struct{}

func (e *failingEvictor) Evict(_ context.Context, _ *flowcontrol.EvictionItem) error {
	return assert.AnError
}

func TestRequestEvictor_EvictionTerminatedListener_NotifiedOncePerEvictedRequest(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, NewImmediateResponseEvictor())

	var mu sync.Mutex
	notified := make(map[string]int)
	re.SetEvictionTerminatedListener(func(requestID string) {
		mu.Lock()
		defer mu.Unlock()
		notified[requestID]++
	})

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-evicted", -1), makeSchedulingResult())
	re.PreRequest(ctx, makeInferenceRequest("req-natural", -1), makeSchedulingResult())

	evicted, err := re.EvictN(ctx, 1, 0)
	require.NoError(t, err)
	require.Len(t, evicted, 1)
	evictedID := evicted[0]

	// Stream termination for the evicted request notifies; the idempotent second cleanup does not.
	re.ResponseBody(ctx, makeInferenceRequest(evictedID, -1), &requestcontrol.Response{EndOfStream: true}, nil)
	re.ResponseBody(ctx, makeInferenceRequest(evictedID, -1), &requestcontrol.Response{EndOfStream: true}, nil)

	// A naturally completing request must not notify.
	naturalID := "req-natural"
	if evictedID == naturalID {
		naturalID = "req-evicted"
	}
	re.ResponseBody(ctx, makeInferenceRequest(naturalID, -1), &requestcontrol.Response{EndOfStream: true}, nil)

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, map[string]int{evictedID: 1}, notified,
		"listener should fire exactly once, only for the evicted request")
}

func TestRequestEvictor_FailedEviction_DoesNotNotifyListener(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &failingEvictor{})

	var mu sync.Mutex
	var notifications []string
	re.SetEvictionTerminatedListener(func(requestID string) {
		mu.Lock()
		defer mu.Unlock()
		notifications = append(notifications, requestID)
	})

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	evicted, err := re.EvictN(ctx, 1, 0)
	require.NoError(t, err)
	assert.Empty(t, evicted, "failed eviction should evict nothing")

	// The request later completes naturally: still no notification.
	re.ResponseBody(ctx, makeInferenceRequest("req-1", -1), &requestcontrol.Response{EndOfStream: true}, nil)

	mu.Lock()
	defer mu.Unlock()
	assert.Empty(t, notifications, "a failed eviction must not mark the request as evicted")
}

func TestRequestEvictor_PeekVictimPriority(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &NoOpEvictor{})

	_, ok := re.PeekVictimPriority()
	assert.False(t, ok, "no victims when nothing is tracked")

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -2), makeSchedulingResult())

	priority, ok := re.PeekVictimPriority()
	require.True(t, ok)
	assert.Equal(t, -2, priority)
}

func TestRequestEvictor_EvictN_PriorityBound(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, NewImmediateResponseEvictor())

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-low", -5), makeSchedulingResult())
	re.PreRequest(ctx, makeInferenceRequest("req-high", -1), makeSchedulingResult())

	// Demand at priority -1: only victims strictly below -1 may be evicted, even though n allows
	// two. The -1 victim would be same-band churn against the demand.
	evicted, err := re.EvictN(ctx, 2, -1)
	require.NoError(t, err)
	assert.Equal(t, []string{"req-low"}, evicted, "only victims strictly below the bound may be evicted")

	inFlight, evictable := re.Stats()
	assert.Equal(t, 1, inFlight, "the at-bound victim must remain tracked")
	assert.Equal(t, 1, evictable, "the at-bound victim must remain evictable for future decisions")
}
