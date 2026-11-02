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

package internal

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
)

func perFlowProcessor(t *testing.T, opts ...registry.ConfigOption) (*testHarness, *registry.FlowRegistry) {
	t.Helper()
	ordering, err := fcfs.FCFSOrderingPolicyFactory("test-ordering", nil, nil)
	require.NoError(t, err)
	fairness, err := globalstrict.GlobalStrictFairnessPolicyFactory("test-fairness", nil, nil)
	require.NoError(t, err)
	cfg, err := registry.NewConfig(registry.PriorityBandPolicyDefaults{
		OrderingPolicy: ordering.(flowcontrol.OrderingPolicy), FairnessPolicy: fairness.(flowcontrol.FairnessPolicy),
	}, opts...)
	require.NoError(t, err)
	reg := registry.NewFlowRegistry(cfg, logr.Discard())
	h := newTestHarness(t, testCleanupTick)
	h.processor.registry = reg
	t.Cleanup(h.processor.shutdown)
	return h, reg
}

func enqueuePerFlow(t *testing.T, h *testHarness, reg *registry.FlowRegistry, id string, key flowcontrol.FlowKey, size uint64) *FlowItem {
	t.Helper()
	item := NewItem(t.Context(), fwkfcmocks.NewMockFlowControlRequest(size, id, key), testTTL, h.clock.Now(), logr.Discard())
	require.NoError(t, reg.WithConnection(key, func(contracts.ActiveFlowConnection) error {
		h.processor.enqueue(item)
		return nil
	}))
	return item
}

func requireCapacityRejection(t *testing.T, item *FlowItem) {
	t.Helper()
	state := item.FinalState()
	require.NotNil(t, state, "a flow must not consume the rest of the band's capacity")
	require.Equal(t, types.QueueOutcomeRejectedCapacity, state.Outcome)
	require.ErrorIs(t, state.Err, types.ErrRejected)
	require.ErrorIs(t, state.Err, types.ErrQueueAtCapacity)
	require.Nil(t, item.Handle())
	item.Finalize(context.Canceled)
	require.Same(t, state, item.FinalState())
	require.Same(t, state, <-item.Done())
	_, open := <-item.Done()
	require.False(t, open, "finalization must be delivered exactly once")
}

func TestPerFlowCapacityBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		maxRequests, maxBytes uint64
		sizes                 []uint64
	}{
		{"count only", 2, 0, []uint64{4, 6}},
		{"bytes only", 0, 10, []uint64{4, 6, 0}},
		{"count binds independently", 2, 100, []uint64{4, 6}},
		{"bytes bind independently", 100, 10, []uint64{4, 6}},
		{"oversized request", 0, 10, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, reg := perFlowProcessor(t, registry.WithPriorityBand(&registry.PriorityBandConfig{
				Priority: testFlow.Priority, MaxRequests: 20, MaxBytes: 1000,
				MaxRequestsPerFlow: tc.maxRequests, MaxBytesPerFlow: tc.maxBytes,
			}))
			for i, size := range tc.sizes {
				item := enqueuePerFlow(t, h, reg, strconv.Itoa(i), testFlow, size)
				require.Nil(t, item.FinalState(), "reaching the limit must be allowed")
				require.NotNil(t, item.Handle())
			}
			size := uint64(1)
			if tc.sizes == nil {
				size = 11
			}
			requireCapacityRejection(t, enqueuePerFlow(t, h, reg, "overflow", testFlow, size))
			other := flowcontrol.FlowKey{ID: "flow-b", Priority: testFlow.Priority}
			require.Nil(t, enqueuePerFlow(t, h, reg, "other", other, 1).FinalState())
			stats, err := reg.CapacitySnapshot(testFlow.Priority)
			require.NoError(t, err)
			require.Equal(t, uint64(len(tc.sizes)+1), stats.Band.Len)
			require.Equal(t, stats.Band.Len, stats.Global.Len)
			require.Equal(t, stats.Band.ByteSize, stats.Global.ByteSize)
		})
	}
}

func TestPerFlowCapacityTemplates(t *testing.T) {
	h, reg := perFlowProcessor(t,
		registry.WithDefaultPriorityBand(&registry.PriorityBandConfig{MaxRequestsPerFlow: 2, MaxBytesPerFlow: 10}),
		registry.WithDefaultNegativePriorityBand(&registry.PriorityBandConfig{MaxRequestsPerFlow: 1, MaxBytesPerFlow: 5}),
		registry.WithPriorityBand(&registry.PriorityBandConfig{Priority: testFlow.Priority, MaxRequestsPerFlow: 3, MaxBytesPerFlow: 15}),
	)
	reg.ApplyDesiredPriorities(map[int]struct{}{7: {}, -7: {}, -8: {}})
	for _, tc := range []struct{ priority, count int }{{0, 2}, {7, 2}, {-7, 1}, {-8, 1}, {10, 3}} {
		key := flowcontrol.FlowKey{ID: testFlow.ID, Priority: tc.priority}
		for i := 0; i < tc.count; i++ {
			require.Nil(t, enqueuePerFlow(t, h, reg, fmt.Sprintf("%d-%d", tc.priority, i), key, 5).FinalState())
		}
		requireCapacityRejection(t, enqueuePerFlow(t, h, reg, fmt.Sprintf("overflow-%d", tc.priority), key, 1))
	}
}

func TestPerFlowCapacityDisabledAndAggregateLimits(t *testing.T) {
	for _, tc := range []struct {
		name string
		band registry.PriorityBandConfig
		opts []registry.ConfigOption
	}{
		{name: "unset", band: registry.PriorityBandConfig{MaxRequests: 3}},
		{name: "zero", band: registry.PriorityBandConfig{MaxRequests: 3, MaxRequestsPerFlow: 0, MaxBytesPerFlow: 0}},
		{name: "band count", band: registry.PriorityBandConfig{MaxRequests: 3, MaxRequestsPerFlow: 10}},
		{name: "band bytes", band: registry.PriorityBandConfig{MaxBytes: 30, MaxBytesPerFlow: 100}},
		{name: "global count", opts: []registry.ConfigOption{registry.WithMaxRequests(3)}, band: registry.PriorityBandConfig{MaxRequestsPerFlow: 10}},
		{name: "global bytes", opts: []registry.ConfigOption{registry.WithMaxBytes(30)}, band: registry.PriorityBandConfig{MaxBytesPerFlow: 100}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.band.Priority = testFlow.Priority
			h, reg := perFlowProcessor(t, append(tc.opts, registry.WithPriorityBand(&tc.band))...)
			key := testFlow
			if tc.name == "global count" || tc.name == "global bytes" {
				reg.ApplyDesiredPriorities(map[int]struct{}{10: {}, 11: {}})
			}
			for i := 0; i < 3; i++ {
				require.Nil(t, enqueuePerFlow(t, h, reg, strconv.Itoa(i), key, 10).FinalState())
			}
			if tc.name == "global count" || tc.name == "global bytes" {
				key.Priority = 11
			}
			requireCapacityRejection(t, enqueuePerFlow(t, h, reg, "overflow", key, 1))
		})
	}
}

func TestPerFlowCapacityReleased(t *testing.T) {
	for _, action := range []string{"dispatch", "cancel", "expire", "drain"} {
		t.Run(action, func(t *testing.T) {
			h, reg := perFlowProcessor(t, registry.WithPriorityBand(&registry.PriorityBandConfig{Priority: testFlow.Priority, MaxRequestsPerFlow: 1, MaxBytesPerFlow: 10}))
			reservations := 0
			h.saturationDetector.ReserveDispatchFunc = func(string) bool { reservations++; return true }
			first := enqueuePerFlow(t, h, reg, "first", testFlow, 10)
			require.Nil(t, first.FinalState())
			requireCapacityRejection(t, enqueuePerFlow(t, h, reg, "overflow", testFlow, 1))
			switch action {
			case "dispatch":
				require.Zero(t, reservations)
				require.NoError(t, h.processor.dispatchItem(first))
				require.Equal(t, types.QueueOutcomeDispatched, first.FinalState().Outcome)
				require.Equal(t, 1, reservations)
			case "cancel":
				first.Finalize(context.Canceled)
				requireCapacityRejection(t, enqueuePerFlow(t, h, reg, "unswept", testFlow, 1))
				h.processor.sweepFinalizedItems()
				require.Equal(t, types.QueueOutcomeEvictedContextCancelled, first.FinalState().Outcome)
			case "expire":
				h.clock.Step(testTTL)
				h.processor.sweepFinalizedItems()
				require.Equal(t, types.QueueOutcomeEvictedTTL, first.FinalState().Outcome)
			case "drain":
				h.processor.evictAll()
			}
			stats, err := reg.CapacitySnapshot(testFlow.Priority)
			require.NoError(t, err)
			require.Zero(t, stats.Band.Len)
			require.Zero(t, stats.Band.ByteSize)
			require.Zero(t, stats.Global.Len)
			require.Zero(t, stats.Global.ByteSize)
			require.Nil(t, enqueuePerFlow(t, h, reg, "replacement", testFlow, 10).FinalState())
		})
	}
}

func TestPerFlowCapacityNoEndpoints(t *testing.T) {
	h, reg := perFlowProcessor(t, registry.WithDefaultPriorityBand(&registry.PriorityBandConfig{MaxRequestsPerFlow: 1}))
	key := flowcontrol.FlowKey{ID: testFlow.ID, Priority: 0}
	require.Nil(t, enqueuePerFlow(t, h, reg, "first", key, 1).FinalState())
	h.processor.regime.Store(&regimeSample{empty: true})
	rejected := enqueuePerFlow(t, h, reg, "overflow", key, 1)
	state := rejected.FinalState()
	require.NotNil(t, state)
	require.Equal(t, types.QueueOutcomeRejectedNoEndpoints, state.Outcome)
	require.ErrorIs(t, state.Err, types.ErrRejected)
	require.ErrorIs(t, state.Err, types.ErrNoEndpoints)
	require.NotErrorIs(t, state.Err, types.ErrQueueAtCapacity)
	require.Zero(t, h.processor.dropCounts[types.QueueOutcomeRejectedCapacity].Load())
	require.Equal(t, uint64(1), h.processor.dropCounts[types.QueueOutcomeRejectedNoEndpoints].Load())
}

func TestPerFlowCapacityByteOverflow(t *testing.T) {
	h, reg := perFlowProcessor(t, registry.WithDefaultPriorityBand(&registry.PriorityBandConfig{MaxBytesPerFlow: 10}))
	key := flowcontrol.FlowKey{ID: testFlow.ID, Priority: 0}
	require.Nil(t, enqueuePerFlow(t, h, reg, "first", key, 1).FinalState())
	requireCapacityRejection(t, enqueuePerFlow(t, h, reg, "overflow", key, math.MaxUint64))
}
