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

package internal

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func TestComputeDeadline_SLOHeader(t *testing.T) {
	t.Parallel()
	received := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	enqueued := received.Add(5 * time.Millisecond)

	item := &mocks.MockQueueItemAccessor{
		EnqueueTimeV:  enqueued,
		EffectiveTTLV: 60 * time.Second,
		OriginalRequestV: &mocks.MockFlowControlRequest{
			ReceivedTimestampV: received,
			InferenceRequestV: &scheduling.InferenceRequest{
				Headers: map[string]string{metadata.TTFTSLOHeaderKey: "500"},
			},
		},
	}

	deadline := computeDeadline(item)
	assert.Equal(t, received.Add(500*time.Millisecond), deadline)
}

func TestComputeDeadline_FallsBackToTTL(t *testing.T) {
	t.Parallel()
	enqueued := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	item := &mocks.MockQueueItemAccessor{
		EnqueueTimeV:  enqueued,
		EffectiveTTLV: 60 * time.Second,
		OriginalRequestV: &mocks.MockFlowControlRequest{
			InferenceRequestV: &scheduling.InferenceRequest{
				Headers: map[string]string{},
			},
		},
	}

	deadline := computeDeadline(item)
	assert.Equal(t, enqueued.Add(60*time.Second), deadline)
}

func TestComputeDeadline_InvalidSLOFallsBackToTTL(t *testing.T) {
	t.Parallel()
	received := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	enqueued := received.Add(5 * time.Millisecond)

	item := &mocks.MockQueueItemAccessor{
		EnqueueTimeV:  enqueued,
		EffectiveTTLV: 30 * time.Second,
		OriginalRequestV: &mocks.MockFlowControlRequest{
			ReceivedTimestampV: received,
			InferenceRequestV: &scheduling.InferenceRequest{
				Headers: map[string]string{metadata.TTFTSLOHeaderKey: "not-a-number"},
			},
		},
	}

	deadline := computeDeadline(item)
	assert.Equal(t, enqueued.Add(30*time.Second), deadline)
}

func TestComputeDeadline_NoDeadlineReturnsZero(t *testing.T) {
	t.Parallel()
	item := &mocks.MockQueueItemAccessor{
		EffectiveTTLV: 0,
		OriginalRequestV: &mocks.MockFlowControlRequest{
			InferenceRequestV: &scheduling.InferenceRequest{
				Headers: map[string]string{},
			},
		},
	}

	deadline := computeDeadline(item)
	assert.True(t, deadline.IsZero())
}

func TestComputeDeadline_NilRequest(t *testing.T) {
	t.Parallel()
	item := &mocks.MockQueueItemAccessor{
		EffectiveTTLV:    10 * time.Second,
		EnqueueTimeV:     time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		OriginalRequestV: nil,
	}

	deadline := computeDeadline(item)
	assert.Equal(t, item.EnqueueTimeV.Add(10*time.Second), deadline)
}

func TestComputeDeadline_NilHeaders(t *testing.T) {
	t.Parallel()
	item := &mocks.MockQueueItemAccessor{
		OriginalRequestV: &mocks.MockFlowControlRequest{
			InferenceRequestV: &scheduling.InferenceRequest{
				Headers: nil,
			},
		},
		EffectiveTTLV: 0,
	}

	deadline := computeDeadline(item)
	assert.True(t, deadline.IsZero())
}

// --- Integration tests: full processor eviction path ---

// captureHandler captures EvictionDemands for test assertions.
type captureHandler struct {
	mu      sync.Mutex
	demands []types.EvictionDemand
}

func (h *captureHandler) HandleEvictionDemand(_ context.Context, demand types.EvictionDemand) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.demands = append(h.demands, demand)
}

func (h *captureHandler) getDemands() []types.EvictionDemand {
	h.mu.Lock()
	defer h.mu.Unlock()
	result := make([]types.EvictionDemand, len(h.demands))
	copy(result, h.demands)
	return result
}

func TestRequestEvictionIfNeeded_FiresWhenDeadlineAtRisk(t *testing.T) {
	t.Parallel()
	handler := &captureHandler{}
	h := newTestHarnessWithEviction(t, handler, 10*time.Millisecond)

	highPriorityFlow := flowcontrol.FlowKey{ID: "flow-hp", Priority: 10}
	q := h.addQueue(highPriorityFlow)

	// Add an item with a tight SLO that has been queued for a while.
	received := h.clock.Now().Add(-200 * time.Millisecond)
	req := &mocks.MockFlowControlRequest{
		ByteSizeV:          100,
		IDV:                "req-urgent",
		FlowKeyV:           highPriorityFlow,
		ReceivedTimestampV: received,
		InferenceRequestV: &scheduling.InferenceRequest{
			Headers: map[string]string{metadata.TTFTSLOHeaderKey: "400"},
		},
	}
	item := NewItem(req, testTTL, h.clock.Now().Add(-200*time.Millisecond))
	require.NoError(t, q.Add(item))

	// Saturation above ceiling triggers HoL blocking.
	h.saturationDetector.SaturationFunc = func(_ context.Context, _ []fwkdl.Endpoint) float64 {
		return 1.1
	}

	// Advance clock so elapsed (200ms) × overloadRatio (1.1) = 220ms > remaining (200ms).
	h.processor.requestEvictionIfNeeded(h.ctx, highPriorityFlow.Priority, 1.1, 1.0)

	// Give the async goroutine time to fire.
	time.Sleep(20 * time.Millisecond)

	demands := handler.getDemands()
	require.Len(t, demands, 1)
	assert.Equal(t, highPriorityFlow.Priority, demands[0].BlockedPriority)
	assert.Equal(t, 1, demands[0].QueuedCount)
	assert.Equal(t, 1.1, demands[0].Saturation)
}

func TestRequestEvictionIfNeeded_SkipsWhenDeadlineHasBudget(t *testing.T) {
	t.Parallel()
	handler := &captureHandler{}
	h := newTestHarnessWithEviction(t, handler, 10*time.Millisecond)

	highPriorityFlow := flowcontrol.FlowKey{ID: "flow-hp", Priority: 10}
	q := h.addQueue(highPriorityFlow)

	// Item just enqueued with a generous SLO — plenty of budget remaining.
	req := &mocks.MockFlowControlRequest{
		ByteSizeV:          100,
		IDV:                "req-patient",
		FlowKeyV:           highPriorityFlow,
		ReceivedTimestampV: h.clock.Now(),
		InferenceRequestV: &scheduling.InferenceRequest{
			Headers: map[string]string{metadata.TTFTSLOHeaderKey: "5000"},
		},
	}
	item := NewItem(req, testTTL, h.clock.Now())
	require.NoError(t, q.Add(item))

	h.processor.requestEvictionIfNeeded(h.ctx, highPriorityFlow.Priority, 1.1, 1.0)

	time.Sleep(20 * time.Millisecond)

	demands := handler.getDemands()
	assert.Empty(t, demands, "should not evict when SLO deadline has ample budget")
}

func TestRequestEvictionIfNeeded_SkipsWhenQueueEmpty(t *testing.T) {
	t.Parallel()
	handler := &captureHandler{}
	h := newTestHarnessWithEviction(t, handler, 10*time.Millisecond)

	emptyFlow := flowcontrol.FlowKey{ID: "flow-empty", Priority: 10}
	h.addQueue(emptyFlow)

	h.processor.requestEvictionIfNeeded(h.ctx, emptyFlow.Priority, 1.5, 1.0)

	time.Sleep(20 * time.Millisecond)

	demands := handler.getDemands()
	assert.Empty(t, demands, "should not evict when queue is empty")
}

func TestRequestEvictionIfNeeded_Debounces(t *testing.T) {
	t.Parallel()
	handler := &captureHandler{}
	cooldown := 100 * time.Millisecond
	h := newTestHarnessWithEviction(t, handler, cooldown)

	flow := flowcontrol.FlowKey{ID: "flow-debounce", Priority: 10}
	q := h.addQueue(flow)

	received := h.clock.Now().Add(-300 * time.Millisecond)
	req := &mocks.MockFlowControlRequest{
		ByteSizeV:          100,
		IDV:                "req-debounce",
		FlowKeyV:           flow,
		ReceivedTimestampV: received,
		InferenceRequestV: &scheduling.InferenceRequest{
			Headers: map[string]string{metadata.TTFTSLOHeaderKey: "500"},
		},
	}
	item := NewItem(req, testTTL, h.clock.Now().Add(-300*time.Millisecond))
	require.NoError(t, q.Add(item))

	// First call fires.
	h.processor.requestEvictionIfNeeded(h.ctx, flow.Priority, 1.1, 1.0)
	time.Sleep(20 * time.Millisecond)
	assert.Len(t, handler.getDemands(), 1)

	// Second call within cooldown is debounced.
	h.clock.Step(50 * time.Millisecond)
	h.processor.requestEvictionIfNeeded(h.ctx, flow.Priority, 1.1, 1.0)
	time.Sleep(20 * time.Millisecond)
	assert.Len(t, handler.getDemands(), 1, "should debounce within cooldown")

	// After cooldown expires, fires again.
	h.clock.Step(60 * time.Millisecond)
	h.processor.requestEvictionIfNeeded(h.ctx, flow.Priority, 1.1, 1.0)
	time.Sleep(20 * time.Millisecond)
	assert.Len(t, handler.getDemands(), 2, "should fire again after cooldown")
}

func TestRequestEvictionIfNeeded_NilHandler(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t, testCleanupTick)

	flow := flowcontrol.FlowKey{ID: "flow-nil", Priority: 10}
	q := h.addQueue(flow)

	req := &mocks.MockFlowControlRequest{
		ByteSizeV: 100, IDV: "req-nil", FlowKeyV: flow,
		ReceivedTimestampV: h.clock.Now().Add(-300 * time.Millisecond),
		InferenceRequestV:  &scheduling.InferenceRequest{Headers: map[string]string{metadata.TTFTSLOHeaderKey: "400"}},
	}
	item := NewItem(req, testTTL, h.clock.Now().Add(-300*time.Millisecond))
	require.NoError(t, q.Add(item))

	// evictionHandler is nil — should not panic.
	assert.Nil(t, h.processor.evictionHandler)
	require.NotPanics(t, func() {
		h.processor.dispatchCycle(h.ctx)
	})
}

func TestRequestEvictionIfNeeded_SkipsWhenNoDeadline(t *testing.T) {
	t.Parallel()
	handler := &captureHandler{}
	h := newTestHarnessWithEviction(t, handler, 10*time.Millisecond)

	flow := flowcontrol.FlowKey{ID: "flow-no-deadline", Priority: 10}
	q := h.addQueue(flow)

	// Item with no SLO header and zero TTL — computeDeadline returns zero.
	req := &mocks.MockFlowControlRequest{
		ByteSizeV: 100, IDV: "req-no-deadline", FlowKeyV: flow,
		InferenceRequestV: &scheduling.InferenceRequest{Headers: map[string]string{}},
	}
	item := NewItem(req, 0, h.clock.Now())
	require.NoError(t, q.Add(item))

	h.processor.requestEvictionIfNeeded(h.ctx, flow.Priority, 1.5, 1.0)

	time.Sleep(20 * time.Millisecond)

	demands := handler.getDemands()
	assert.Empty(t, demands, "should not evict when no deadline signal exists")
}

func TestRequestEvictionIfNeeded_RecoverFromHandlerPanic(t *testing.T) {
	t.Parallel()
	panicHandler := &panicEvictionHandler{}
	h := newTestHarnessWithEviction(t, panicHandler, 10*time.Millisecond)

	flow := flowcontrol.FlowKey{ID: "flow-panic", Priority: 10}
	q := h.addQueue(flow)

	received := h.clock.Now().Add(-300 * time.Millisecond)
	req := &mocks.MockFlowControlRequest{
		ByteSizeV: 100, IDV: "req-panic", FlowKeyV: flow,
		ReceivedTimestampV: received,
		InferenceRequestV:  &scheduling.InferenceRequest{Headers: map[string]string{metadata.TTFTSLOHeaderKey: "400"}},
	}
	item := NewItem(req, testTTL, h.clock.Now().Add(-300*time.Millisecond))
	require.NoError(t, q.Add(item))

	require.NotPanics(t, func() {
		h.processor.requestEvictionIfNeeded(h.ctx, flow.Priority, 1.1, 1.0)
		time.Sleep(20 * time.Millisecond)
	})
}

// panicEvictionHandler panics on every call.
type panicEvictionHandler struct{}

func (h *panicEvictionHandler) HandleEvictionDemand(_ context.Context, _ types.EvictionDemand) {
	panic("test panic")
}

func TestDispatchCycle_CallsEvictionHandlerOnHoLBlocking(t *testing.T) {
	t.Parallel()
	handler := &captureHandler{}
	h := newTestHarnessWithEviction(t, handler, 10*time.Millisecond)

	flow := flowcontrol.FlowKey{ID: "flow-dispatch", Priority: 10}
	q := h.addQueue(flow)

	received := h.clock.Now().Add(-300 * time.Millisecond)
	req := &mocks.MockFlowControlRequest{
		ByteSizeV: 100, IDV: "req-dispatch", FlowKeyV: flow,
		ReceivedTimestampV: received,
		InferenceRequestV: &scheduling.InferenceRequest{
			Headers: map[string]string{metadata.TTFTSLOHeaderKey: "500"},
		},
	}
	item := NewItem(req, testTTL, h.clock.Now().Add(-300*time.Millisecond))
	require.NoError(t, q.Add(item))

	h.saturationDetector.SaturationFunc = func(_ context.Context, _ []fwkdl.Endpoint) float64 {
		return 1.1
	}

	// Call dispatchCycle directly — this tests the full path:
	// dispatchCycle → HoL blocking → evictionHandler != nil → requestEvictionIfNeeded → handler
	dispatched := h.processor.dispatchCycle(h.ctx)
	assert.False(t, dispatched, "should not dispatch when saturated")

	time.Sleep(20 * time.Millisecond)

	demands := handler.getDemands()
	require.Len(t, demands, 1, "dispatchCycle should trigger eviction handler via requestEvictionIfNeeded")
	assert.Equal(t, flow.Priority, demands[0].BlockedPriority)
}

// newTestHarnessWithEviction creates a test harness with eviction enabled.
func newTestHarnessWithEviction(t *testing.T, handler types.EvictionHandler, cooldown time.Duration) *testHarness {
	t.Helper()
	h := newTestHarness(t, testCleanupTick)
	h.processor.evictionHandler = handler
	h.processor.evictionCooldown = cooldown
	return h
}
