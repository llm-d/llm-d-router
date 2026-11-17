/*
Copyright 2025 The Kubernetes Authors.
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

package roundrobin

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
)

const (
	flow1ID = "flow1"
	flow2ID = "flow2"
	flow3ID = "flow3"
)

var (
	flow1Key = flowcontrol.FlowKey{ID: flow1ID, Priority: 0}
	flow2Key = flowcontrol.FlowKey{ID: flow2ID, Priority: 0}
	flow3Key = flowcontrol.FlowKey{ID: flow3ID, Priority: 0}
)

func newMockBand(state any, queues ...flowcontrol.FlowQueueAccessor) *fwkfcmocks.MockPriorityBandAccessor {
	return &fwkfcmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		IterateQueuesFunc: func(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
			for _, q := range queues {
				if !callback(q) {
					return
				}
			}
		},
	}
}

func TestRoundRobin_Name(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("test-rr")
	assert.Equal(t, "test-rr", policy.TypedName().Name)
	assert.Equal(t, RoundRobinFairnessPolicyType, policy.TypedName().Type)
}

func TestRoundRobin_Pick_InvalidPolicyState(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()

	var typedNilCursor *roundRobinCursor
	for _, tc := range []struct {
		name  string
		state any
	}{
		{name: "nil state", state: nil},
		{name: "wrong state type", state: "not-a-cursor"},
		{name: "typed-nil cursor", state: typedNilCursor},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			selected, err := policy.Pick(ctx, newMockBand(tc.state))
			require.Error(t, err)
			assert.Nil(t, selected)
		})
	}
}

func TestRoundRobin_Pick_Logic(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()
	state := policy.NewState(ctx)

	// Unsorted input to verify FlowKey ordering.
	queue1 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key}
	queue2 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 2, FlowKeyV: flow2Key}
	queue3 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 3, FlowKeyV: flow3Key}
	mockBand := newMockBand(state, queue3, queue1, queue2)

	// Expected order is based on sorted FlowKeys: flow1, flow2, flow3
	expectedOrder := []string{flow1ID, flow2ID, flow3ID}

	// First cycle
	for i := range expectedOrder {
		selected, err := policy.Pick(ctx, mockBand)
		require.NoError(t, err, "Pick should not error on a valid band")
		require.NotNil(t, selected, "Pick should have selected a queue")
		assert.Equal(t, expectedOrder[i], selected.FlowKey().ID,
			"Cycle 1, selection %d should be %s", i+1, expectedOrder[i])
	}

	// Second cycle (wraps around)
	for i := range expectedOrder {
		selected, err := policy.Pick(ctx, mockBand)
		require.NoError(t, err, "Pick should not error on a valid band")
		require.NotNil(t, selected, "Pick should have selected a queue")
		assert.Equal(t, expectedOrder[i], selected.FlowKey().ID,
			"Cycle 2, selection %d should be %s", i+1, expectedOrder[i])
	}
}

func TestRoundRobin_Pick_SkipsEmptyQueues(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()
	state := policy.NewState(ctx)

	flowEmptyKey := flowcontrol.FlowKey{ID: "flowEmpty", Priority: 0}
	queue1 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key}
	queueEmpty := &fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowEmptyKey}
	queue3 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 3, FlowKeyV: flow3Key}
	mockBand := newMockBand(state, queue1, queueEmpty, nil, queue3)

	// Expected order: flow1, flow3, flow1, flow3, ...
	selected, err := policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error when skipping queues")
	require.NotNil(t, selected, "Pick should select the first non-empty queue")
	assert.Equal(t, flow1ID, selected.FlowKey().ID, "First selection should be flow1")

	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error when skipping queues")
	require.NotNil(t, selected, "Pick should select the second non-empty queue")
	assert.Equal(t, flow3ID, selected.FlowKey().ID, "Second selection should be flow3, skipping flowEmpty")

	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error when wrapping around")
	require.NotNil(t, selected, "Pick should wrap around and select a queue")
	assert.Equal(t, flow1ID, selected.FlowKey().ID, "Should wrap around and select flow1 again")
}

func TestRoundRobin_Pick_UsesIterateQueues(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()
	state := policy.NewState(ctx)

	flowEmptyKey := flowcontrol.FlowKey{ID: "flowEmpty", Priority: 0}
	queue1 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key}
	queue2 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 2, FlowKeyV: flow2Key}
	queue3 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 3, FlowKeyV: flow3Key}
	queueEmpty := &fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowEmptyKey}

	var flowKeysCalled, queueCalled bool
	mockBand := &fwkfcmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		FlowKeysFunc: func() []flowcontrol.FlowKey {
			flowKeysCalled = true
			return nil
		},
		QueueFunc: func(_ string) flowcontrol.FlowQueueAccessor {
			queueCalled = true
			return nil
		},
		IterateQueuesFunc: func(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
			for _, q := range []flowcontrol.FlowQueueAccessor{queue3, queueEmpty, queue1, queue2} {
				if !callback(q) {
					return
				}
			}
		},
	}

	expectedOrder := []string{flow1ID, flow2ID, flow3ID, flow1ID, flow2ID, flow3ID}
	for i, expectedID := range expectedOrder {
		selected, err := policy.Pick(ctx, mockBand)
		require.NoError(t, err, "Pick %d should not error", i+1)
		require.NotNil(t, selected, "Pick %d should select a queue", i+1)
		assert.Equal(t, expectedID, selected.FlowKey().ID, "Pick %d should select %s", i+1, expectedID)
	}
	assert.False(t, flowKeysCalled, "Pick must use IterateQueues and never call FlowKeys()")
	assert.False(t, queueCalled, "Pick must use IterateQueues and never call Queue()")
}

func TestRoundRobin_Pick_DrainedQueuePreservesCursorOrder(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()
	state := policy.NewState(ctx)

	queue1 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key}
	queue2 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow2Key}
	queue3 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow3Key}

	activeQueues := []flowcontrol.FlowQueueAccessor{queue1, queue2, queue3}
	mockBand := &fwkfcmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		IterateQueuesFunc: func(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
			for _, q := range activeQueues {
				if !callback(q) {
					return
				}
			}
		},
	}

	// Pick 1 selects flow1; Pick 2 selects flow2.
	selected, err := policy.Pick(ctx, mockBand)
	require.NoError(t, err)
	require.NotNil(t, selected)
	assert.Equal(t, flow1ID, selected.FlowKey().ID)

	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err)
	require.NotNil(t, selected)
	assert.Equal(t, flow2ID, selected.FlowKey().ID)

	// Simulate flow2 draining between picks so only flow1 and flow3 remain active.
	// The next pick advances to flow3 in sorted FlowKey order.
	activeQueues = []flowcontrol.FlowQueueAccessor{queue1, queue3}
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err)
	require.NotNil(t, selected)
	assert.Equal(t, flow3ID, selected.FlowKey().ID, "When lastSelected flow drains, Pick must advance to the next flow in sorted order")

	// Simulate flow3 draining while flow2 becomes active again (active: flow1, flow2).
	// The next pick wraps around to flow1.
	activeQueues = []flowcontrol.FlowQueueAccessor{queue1, queue2}
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err)
	require.NotNil(t, selected)
	assert.Equal(t, flow1ID, selected.FlowKey().ID, "When the last flow in sorted order drains, Pick must wrap around to the first active flow")
}

func TestRoundRobin_Pick_EmptyBandPreservesCursor(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()
	state := policy.NewState(ctx)

	queue1 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key}
	queue2 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow2Key}
	queue3 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow3Key}

	var activeQueues []flowcontrol.FlowQueueAccessor
	mockBand := &fwkfcmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		IterateQueuesFunc: func(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
			for _, q := range activeQueues {
				if !callback(q) {
					return
				}
			}
		},
	}

	// Empty tick before any flow has been selected.
	selected, err := policy.Pick(ctx, mockBand)
	require.NoError(t, err)
	assert.Nil(t, selected)

	// First burst selects flow1.
	activeQueues = []flowcontrol.FlowQueueAccessor{queue1, queue2, queue3}
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err)
	require.NotNil(t, selected)
	assert.Equal(t, flow1ID, selected.FlowKey().ID)

	// Idle tick with no active queues must not reset lastSelected.
	activeQueues = nil
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err)
	assert.Nil(t, selected)

	// Next burst resumes at flow2 rather than restarting at flow1.
	activeQueues = []flowcontrol.FlowQueueAccessor{queue1, queue2, queue3}
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err)
	require.NotNil(t, selected)
	assert.Equal(t, flow2ID, selected.FlowKey().ID)
}

func TestRoundRobin_Pick_HandlesDynamicFlows(t *testing.T) {
	t.Parallel()
	policy := newRoundRobin("")
	ctx := context.Background()
	state := policy.NewState(ctx)

	queue1 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key}
	queue2 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow2Key}
	queue3 := &fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow3Key}

	activeQueues := []flowcontrol.FlowQueueAccessor{queue1, queue2}
	mockBand := &fwkfcmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		IterateQueuesFunc: func(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
			for _, q := range activeQueues {
				if !callback(q) {
					return
				}
			}
		},
	}

	// First selection
	selected, err := policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error on initial selection")
	require.NotNil(t, selected, "Pick should select a queue initially")
	assert.Equal(t, flow1ID, selected.FlowKey().ID, "First selection should be flow1")

	// Add flow3
	activeQueues = []flowcontrol.FlowQueueAccessor{queue1, queue2, queue3}

	// Next selection should be flow2
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error after adding a flow")
	require.NotNil(t, selected, "Pick should select a queue after adding a flow")
	assert.Equal(t, flow2ID, selected.FlowKey().ID, "Next selection should be flow2")

	// Next selection should be flow3
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error on the third selection")
	require.NotNil(t, selected, "Pick should select the new flow")
	assert.Equal(t, flow3ID, selected.FlowKey().ID, "Next selection should be the new flow3")

	// Remove flow2
	activeQueues = []flowcontrol.FlowQueueAccessor{queue1, queue3}

	// Next selection should wrap around and pick flow1
	selected, err = policy.Pick(ctx, mockBand)
	require.NoError(t, err, "Pick should not error after removing a flow")
	require.NotNil(t, selected, "Pick should select a queue after removing a flow")
	assert.Equal(t, flow1ID, selected.FlowKey().ID, "Next selection should wrap around to flow1 after a removal")
}

func TestRoundRobin_Pick_Concurrency(t *testing.T) {
	t.Parallel()
	// Run this test multiple times to increase the chance of catching race conditions.
	for i := range 5 {
		t.Run(fmt.Sprintf("Iteration%d", i), func(t *testing.T) {
			t.Parallel()
			policy := newRoundRobin("")
			ctx := context.Background()
			state := policy.NewState(ctx) // Shared state for all goroutines

			// Setup: Three non-empty queues
			queues := []flowcontrol.FlowQueueAccessor{
				&fwkfcmocks.MockFlowQueueAccessor{LenV: 1, FlowKeyV: flow1Key},
				&fwkfcmocks.MockFlowQueueAccessor{LenV: 2, FlowKeyV: flow2Key},
				&fwkfcmocks.MockFlowQueueAccessor{LenV: 3, FlowKeyV: flow3Key},
			}
			numQueues := int64(len(queues))
			mockBand := newMockBand(state, queues...)

			var wg sync.WaitGroup
			numGoroutines := 10
			selectionsPerGoroutine := 30
			totalSelections := int64(numGoroutines * selectionsPerGoroutine)

			var selectionCounts sync.Map // Used like a concurrent map[string]*atomic.Int64

			wg.Add(numGoroutines)
			for range numGoroutines {
				go func() {
					defer wg.Done()
					for range selectionsPerGoroutine {
						selected, err := policy.Pick(ctx, mockBand)
						if err == nil && selected != nil {
							val, _ := selectionCounts.LoadOrStore(selected.FlowKey().ID, new(atomic.Int64))
							val.(*atomic.Int64).Add(1)
						}
					}
				}()
			}
			wg.Wait()

			var finalCount int64
			countsStr := ""
			selectionCounts.Range(func(key, value any) bool {
				count := value.(*atomic.Int64).Load()
				finalCount += count
				countsStr += fmt.Sprintf("%s: %d, ", key, count)

				// Check for reasonable distribution.
				// In a perfect world, each queue gets totalSelections / numQueues.
				// We allow for some variance due to scheduling, but it shouldn't be wildly off.
				// A simple check is that each queue gets at least a certain fraction of its expected share.
				expectedCount := totalSelections / numQueues
				minExpectedCount := expectedCount / 2 // Expect at least half of the ideal distribution
				assert.True(t, count > minExpectedCount,
					"Queue %s was selected only %d times, expected at least %d", key, count, minExpectedCount)
				return true
			})

			assert.Equal(t, totalSelections, finalCount, "Total selections should match the expected number")
			t.Logf("Selection distribution: %s", countsStr)
		})
	}
}
