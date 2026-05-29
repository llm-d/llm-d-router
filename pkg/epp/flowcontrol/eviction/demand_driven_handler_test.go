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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
)

func TestDemandDrivenHandler_EvictsProportionally(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()
	for i := range 5 {
		re.PreRequest(ctx, makeInferenceRequest("req-"+string(rune('a'+i)), -1), makeSchedulingResult())
	}

	handler := NewDemandDrivenEvictionHandler(re)

	demand := types.EvictionDemand{
		BlockedPriority: 0,
		QueuedCount:     3,
		Saturation:      1.05,
		UsageLimit:      1.0,
	}

	handler.HandleEvictionDemand(ctx, demand)

	inFlight, evictable := re.Stats()
	assert.Equal(t, 2, inFlight, "should have 2 remaining in-flight (5 - 3 evicted)")
	assert.Equal(t, 2, evictable, "should have 2 remaining evictable")
}

func TestDemandDrivenHandler_CapsAtEvictableCount(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())
	re.PreRequest(ctx, makeInferenceRequest("req-2", -1), makeSchedulingResult())

	handler := NewDemandDrivenEvictionHandler(re)

	demand := types.EvictionDemand{
		BlockedPriority: 0,
		QueuedCount:     10,
		Saturation:      1.2,
		UsageLimit:      1.0,
	}

	handler.HandleEvictionDemand(ctx, demand)

	inFlight, evictable := re.Stats()
	assert.Equal(t, 0, inFlight, "should evict all 2 (capped at evictable)")
	assert.Equal(t, 0, evictable)
}

func TestDemandDrivenHandler_NoOpWhenNoEvictable(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &NoOpEvictor{})

	handler := NewDemandDrivenEvictionHandler(re)

	demand := types.EvictionDemand{
		BlockedPriority: 0,
		QueuedCount:     5,
		Saturation:      1.1,
		UsageLimit:      1.0,
	}

	handler.HandleEvictionDemand(context.Background(), demand)

	inFlight, evictable := re.Stats()
	assert.Equal(t, 0, inFlight)
	assert.Equal(t, 0, evictable)
}

func TestDemandDrivenHandler_SkipsOnCancelledContext(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	handler := NewDemandDrivenEvictionHandler(re)

	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	demand := types.EvictionDemand{
		BlockedPriority: 0,
		QueuedCount:     1,
		Saturation:      1.0,
		UsageLimit:      1.0,
	}

	handler.HandleEvictionDemand(cancelledCtx, demand)

	_, evictable := re.Stats()
	assert.Equal(t, 1, evictable, "should not evict when context is cancelled")
}

func TestDemandDrivenHandler_SkipsWhenEvictablePriorityNotLower(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	handler := NewDemandDrivenEvictionHandler(re)

	demand := types.EvictionDemand{
		BlockedPriority: -1,
		QueuedCount:     1,
		Saturation:      1.0,
		UsageLimit:      1.0,
	}

	handler.HandleEvictionDemand(ctx, demand)

	_, evictable := re.Stats()
	assert.Equal(t, 1, evictable, "should not evict when blocked priority is same as evictable priority")
}

func TestDemandDrivenHandler_EvictsWhenBlockedPriorityIsHigher(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -2), makeSchedulingResult())

	handler := NewDemandDrivenEvictionHandler(re)

	demand := types.EvictionDemand{
		BlockedPriority: -1,
		QueuedCount:     1,
		Saturation:      1.0,
		UsageLimit:      1.0,
	}

	handler.HandleEvictionDemand(ctx, demand)

	_, evictable := re.Stats()
	assert.Equal(t, 0, evictable, "should evict when blocked priority (-1) is higher than evictable (-2)")
}

func TestDemandDrivenHandler_HandlesEvictNError(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &failingEvictor{})

	ctx := context.Background()
	re.PreRequest(ctx, makeInferenceRequest("req-1", -1), makeSchedulingResult())

	handler := NewDemandDrivenEvictionHandler(re)

	demand := types.EvictionDemand{
		BlockedPriority: 0,
		QueuedCount:     1,
		Saturation:      1.0,
		UsageLimit:      1.0,
	}

	require.NotPanics(t, func() {
		handler.HandleEvictionDemand(ctx, demand)
	})
}
