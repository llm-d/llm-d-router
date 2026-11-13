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
	"slices"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
)

var fairFlowB = flowcontrol.FlowKey{ID: "flow-b", Priority: testFlow.Priority}

func fillFlow(t *testing.T, h *testHarness, reg *registry.FlowRegistry, key flowcontrol.FlowKey, n int, size uint64) []*FlowItem {
	t.Helper()
	items := make([]*FlowItem, n)
	for i := range items {
		items[i] = enqueuePerFlow(t, h, reg, fmt.Sprintf("%s-%d", key.ID, i), key, size)
		require.Nil(t, items[i].FinalState(), "the band must have room for the setup")
	}
	return items
}

func requireEvicted(t *testing.T, item *FlowItem) {
	t.Helper()
	state := item.FinalState()
	require.NotNil(t, state, "%s must be evicted", item.OriginalRequest().ID())
	require.Equal(t, types.QueueOutcomeEvictedFairAdmission, state.Outcome)
	require.ErrorIs(t, state.Err, types.ErrEvicted)
	require.ErrorIs(t, state.Err, types.ErrFairAdmissionEvicted)
	require.True(t, item.Handle().IsInvalidated(), "an evicted request must leave its queue")
}

func requireQueued(t *testing.T, items ...*FlowItem) {
	t.Helper()
	for _, item := range items {
		require.Nil(t, item.FinalState(), "%s must stay queued", item.OriginalRequest().ID())
		require.False(t, item.Handle().IsInvalidated(), "%s must stay in its queue", item.OriginalRequest().ID())
	}
}

func requireBand(t *testing.T, reg *registry.FlowRegistry, wantLen, wantBytes uint64) {
	t.Helper()
	stats, err := reg.CapacitySnapshot(testFlow.Priority)
	require.NoError(t, err)
	require.Equal(t, wantLen, stats.Band.Len, "band length")
	require.Equal(t, wantBytes, stats.Band.ByteSize, "band bytes")
	require.Equal(t, stats.Band.Len, stats.Global.Len)
	require.Equal(t, stats.Band.ByteSize, stats.Global.ByteSize)
}

// TestFairAdmissionMargin checks that an arrival at a full band evicts the last-added request of a longer flow only
// when that flow holds at least two more requests than the arrival's flow, so two flows cannot trade places.
func TestFairAdmissionMargin(t *testing.T) {
	for _, tc := range []struct {
		name          string
		heavy, light  int
		arrival       flowcontrol.FlowKey
		wantEvictLast bool
	}{
		{name: "two more", heavy: 3, light: 1, arrival: fairFlowB, wantEvictLast: true},
		{name: "one more", heavy: 3, light: 2, arrival: fairFlowB},
		{name: "equal", heavy: 2, light: 2, arrival: fairFlowB},
		{name: "arrival in the longest flow", heavy: 3, light: 1, arrival: testFlow},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, reg := perFlowProcessor(t, registry.WithPriorityBand(&registry.PriorityBandConfig{
				Priority: testFlow.Priority, MaxRequests: uint64(tc.heavy + tc.light), FairAdmission: true,
			}))
			heavy := fillFlow(t, h, reg, testFlow, tc.heavy, 1)
			light := fillFlow(t, h, reg, fairFlowB, tc.light, 1)
			arrival := enqueuePerFlow(t, h, reg, "arrival", tc.arrival, 1)
			if tc.wantEvictLast {
				requireQueued(t, arrival)
				requireEvicted(t, heavy[len(heavy)-1])
				requireQueued(t, heavy[:len(heavy)-1]...)
			} else {
				requireCapacityRejection(t, arrival, false)
				requireQueued(t, heavy...)
			}
			requireQueued(t, light...)
			requireBand(t, reg, uint64(tc.heavy+tc.light), uint64(tc.heavy+tc.light))
		})
	}
}

// TestFairAdmissionVictimIsLastAddedOfLongestFlow checks that each eviction takes the last-added request of the flow
// that holds the most requests at that moment, and that eviction leaves exact band accounting and an empty index.
func TestFairAdmissionVictimIsLastAddedOfLongestFlow(t *testing.T) {
	h, reg := perFlowProcessor(t, registry.WithPriorityBand(&registry.PriorityBandConfig{
		Priority: testFlow.Priority, MaxRequests: 8, FairAdmission: true,
	}))
	flowC := flowcontrol.FlowKey{ID: "flow-c", Priority: testFlow.Priority}
	a := fillFlow(t, h, reg, testFlow, 5, 10)
	c := fillFlow(t, h, reg, flowC, 3, 10)

	b0 := enqueuePerFlow(t, h, reg, "flow-b-0", fairFlowB, 7)
	requireEvicted(t, a[4])
	b1 := enqueuePerFlow(t, h, reg, "flow-b-1", fairFlowB, 7)
	requireEvicted(t, a[3])
	requireQueued(t, append(append(a[:3:3], c...), b0, b1)...)
	requireBand(t, reg, 8, 6*10+2*7)
	require.Equal(t, uint64(2), h.processor.dropCounts[types.QueueOutcomeEvictedFairAdmission].Load())

	h.processor.evictAll()
	requireBand(t, reg, 0, 0)
	require.Nil(t, reg.LongestQueue(testFlow.Priority), "an emptied band must leave nothing in the index")
}

// TestFairAdmissionStillRejects covers the limits that fair admission does not relieve: the arriving request is
// rejected as without it, and no queued request is touched.
func TestFairAdmissionStillRejects(t *testing.T) {
	for _, tc := range []struct {
		name        string
		band        registry.PriorityBandConfig
		opts        []registry.ConfigOption
		arrivalSize uint64
		noEndpoints bool
	}{
		{name: "disabled", band: registry.PriorityBandConfig{MaxRequests: 4}, arrivalSize: 1},
		{name: "band requests and band bytes", band: registry.PriorityBandConfig{MaxRequests: 4, MaxBytes: 4, FairAdmission: true}, arrivalSize: 1},
		{name: "band and global requests", opts: []registry.ConfigOption{registry.WithMaxRequests(4)}, band: registry.PriorityBandConfig{MaxRequests: 4, FairAdmission: true}, arrivalSize: 1},
		{name: "band requests and global bytes", opts: []registry.ConfigOption{registry.WithMaxBytes(4)}, band: registry.PriorityBandConfig{MaxRequests: 4, FairAdmission: true}, arrivalSize: 1},
		// The arrival's flow is also at its own byte limit, so one freed band slot would not admit it.
		{name: "band requests and own flow limit", band: registry.PriorityBandConfig{MaxRequests: 4, MaxBytesPerFlow: 4, FairAdmission: true}, arrivalSize: 5},
		{name: "no endpoints", band: registry.PriorityBandConfig{MaxRequests: 4, FairAdmission: true}, arrivalSize: 1, noEndpoints: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.band.Priority = testFlow.Priority
			h, reg := perFlowProcessor(t, append(tc.opts, registry.WithPriorityBand(&tc.band))...)
			heavy := fillFlow(t, h, reg, testFlow, 4, 1)
			if tc.noEndpoints {
				h.processor.regime.Store(&regimeSample{empty: true})
			}
			arrival := enqueuePerFlow(t, h, reg, "arrival", fairFlowB, tc.arrivalSize)
			if tc.noEndpoints {
				require.Equal(t, types.QueueOutcomeRejectedNoEndpoints, arrival.FinalState().Outcome)
			} else {
				requireCapacityRejection(t, arrival, false)
			}
			requireQueued(t, heavy...)
			require.Zero(t, h.processor.dropCounts[types.QueueOutcomeEvictedFairAdmission].Load())
		})
	}
}

// TestFairAdmissionDropsCancelledRequests checks that requests of the victim flow finalized while queued but not yet
// swept give up their slots with their own outcome, counted once, and that dropping any of them admits the arrival
// without evicting a live request.
func TestFairAdmissionDropsCancelledRequests(t *testing.T) {
	for _, tc := range []struct {
		name      string
		finalized []int
	}{
		{name: "last added", finalized: []int{3}},
		// The victim flow's only live request is its last-added one.
		{name: "all but the last added", finalized: []int{0, 1, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, reg := perFlowProcessor(t, registry.WithPriorityBand(&registry.PriorityBandConfig{
				Priority: testFlow.Priority, MaxRequests: 4, FairAdmission: true,
			}))
			a := fillFlow(t, h, reg, testFlow, 4, 1)
			for _, i := range tc.finalized {
				a[i].Finalize(context.Canceled)
			}

			requireQueued(t, enqueuePerFlow(t, h, reg, "flow-b-0", fairFlowB, 1))
			for i, item := range a {
				if slices.Contains(tc.finalized, i) {
					require.Equal(t, types.QueueOutcomeEvictedContextCancelled, item.FinalState().Outcome)
					require.True(t, item.Handle().IsInvalidated(), "a finalized request must leave its queue")
				} else {
					requireQueued(t, item)
				}
			}
			require.Equal(t, uint64(len(tc.finalized)), h.processor.dropCounts[types.QueueOutcomeEvictedContextCancelled].Load())
			require.Zero(t, h.processor.dropCounts[types.QueueOutcomeEvictedFairAdmission].Load())
			n := uint64(len(a) - len(tc.finalized) + 1)
			requireBand(t, reg, n, n)
		})
	}
}

// TestFairAdmissionDropsExpiredRequests checks that a victim request whose queue-wait budget ran out before the sweep
// saw it gives up its slot with the sweep's expiry outcome, counted once even after the next sweep, and that the
// arrival takes that slot without evicting a live request.
func TestFairAdmissionDropsExpiredRequests(t *testing.T) {
	h, reg := perFlowProcessor(t, registry.WithPriorityBand(&registry.PriorityBandConfig{
		Priority: testFlow.Priority, MaxRequests: 4, FairAdmission: true,
	}))
	expired := fillFlow(t, h, reg, testFlow, 1, 1)[0]
	h.clock.Step(testTTL / 2)
	live := fillFlow(t, h, reg, testFlow, 3, 1)
	h.clock.Step(testTTL / 2)

	requireQueued(t, enqueuePerFlow(t, h, reg, "flow-b-0", fairFlowB, 1))
	require.NotNil(t, expired.FinalState(), "the expired request must be finalized")
	require.Equal(t, types.QueueOutcomeEvictedTTL, expired.FinalState().Outcome)
	require.ErrorIs(t, expired.FinalState().Err, types.ErrTTLExpired)
	require.True(t, expired.Handle().IsInvalidated(), "the expired request must leave its queue")
	requireQueued(t, live...)

	h.processor.sweepFinalizedItems()
	requireQueued(t, live...)
	require.Equal(t, uint64(1), h.processor.dropCounts[types.QueueOutcomeEvictedTTL].Load())
	require.Zero(t, h.processor.dropCounts[types.QueueOutcomeEvictedFairAdmission].Load())
	requireBand(t, reg, 4, 4)
}

// TestFairAdmissionDropsStaleOncePerSweep checks that stale requests are dropped at most once per victim flow per
// cleanup sweep: a request cancelled after the drop keeps its slot until the sweep, and the first eviction after the
// sweep drops stale requests again.
func TestFairAdmissionDropsStaleOncePerSweep(t *testing.T) {
	h, reg := perFlowProcessor(t, registry.WithPriorityBand(&registry.PriorityBandConfig{
		Priority: testFlow.Priority, MaxRequests: 8, FairAdmission: true,
	}))
	a := fillFlow(t, h, reg, testFlow, 8, 1)
	// Each arrival comes from its own empty flow, so testFlow stays the victim.
	arrive := func(id string) {
		t.Helper()
		requireQueued(t, enqueuePerFlow(t, h, reg, id, flowcontrol.FlowKey{ID: id, Priority: testFlow.Priority}, 1))
	}

	for i := range 2 {
		dropped, unswept, evicted := a[2*i], a[2*i+1], a[7-i]
		dropped.Finalize(context.Canceled)
		arrive(fmt.Sprintf("drop-%d", i))
		require.True(t, dropped.Handle().IsInvalidated(), "the first arrival in a sweep interval must drop stale requests")
		requireQueued(t, evicted)

		unswept.Finalize(context.Canceled)
		arrive(fmt.Sprintf("evict-%d", i))
		requireEvicted(t, evicted)
		require.False(t, unswept.Handle().IsInvalidated(),
			"a second arrival in the same sweep interval must not drop stale requests")

		h.processor.sweepFinalizedItems()
		require.True(t, unswept.Handle().IsInvalidated(), "the sweep removes what the stale drop skipped")
		arrive(fmt.Sprintf("refill-%d", i))
	}
	requireQueued(t, a[4], a[5])
	require.Equal(t, uint64(4), h.processor.dropCounts[types.QueueOutcomeEvictedContextCancelled].Load())
	require.Equal(t, uint64(2), h.processor.dropCounts[types.QueueOutcomeEvictedFairAdmission].Load())
	requireBand(t, reg, 8, 8)
}

// hookedRegistry runs a hook inside the fair-admission victim read (LastAdded) or the victim queue's Remove, standing
// in for the cleanup sweep or a client cancel that lands while Run decides an eviction.
type hookedRegistry struct {
	*registry.FlowRegistry
	beforeLastAdded, beforeRemove func()
}

func (r *hookedRegistry) LongestQueue(priority int) contracts.ManagedQueue {
	if mq := r.FlowRegistry.LongestQueue(priority); mq != nil {
		return &hookedQueue{ManagedQueue: mq, r: r}
	}
	return nil
}

func (r *hookedRegistry) LastAdded(key flowcontrol.FlowKey) flowcontrol.QueueItemAccessor {
	if r.beforeLastAdded != nil {
		r.beforeLastAdded()
	}
	return r.FlowRegistry.LastAdded(key)
}

type hookedQueue struct {
	contracts.ManagedQueue
	r *hookedRegistry
}

func (q *hookedQueue) Remove(handle flowcontrol.QueueItemHandle) (flowcontrol.QueueItemAccessor, error) {
	if q.r.beforeRemove != nil {
		q.r.beforeRemove()
	}
	return q.ManagedQueue.Remove(handle)
}

// TestFairAdmissionRaces checks that a live request is evicted only when the band was still full and the arrival still
// waited at the recheck just before the eviction, when the sweep or a client cancel changes the state between the
// decision and the eviction.
func TestFairAdmissionRaces(t *testing.T) {
	flowC := flowcontrol.FlowKey{ID: "flow-c", Priority: testFlow.Priority}
	for _, tc := range []struct {
		name string
		// hook installs the race on the registry, given the victim flow's requests, the arrival, and a request of a
		// third flow that is queued without Run to refill the band.
		hook         func(h *testHarness, r *hookedRegistry, a []*FlowItem, arrival, refill *FlowItem, c contracts.ManagedQueue)
		wantAdmitted bool
		// wantLive lists the victim flow's requests that must stay queued.
		wantLive []int
	}{
		{
			name: "sweep frees a slot before the last-added read",
			hook: func(h *testHarness, r *hookedRegistry, a []*FlowItem, _, _ *FlowItem, _ contracts.ManagedQueue) {
				r.beforeLastAdded = func() {
					a[0].Finalize(context.Canceled)
					h.processor.sweepFinalizedItems()
				}
			},
			wantAdmitted: true, wantLive: []int{1, 2, 3},
		},
		{
			name: "sweep removes the victim before the eviction",
			hook: func(h *testHarness, r *hookedRegistry, a []*FlowItem, _, _ *FlowItem, _ contracts.ManagedQueue) {
				r.beforeRemove = func() {
					a[3].Finalize(context.Canceled)
					h.processor.sweepFinalizedItems()
				}
			},
			wantAdmitted: true, wantLive: []int{0, 1, 2},
		},
		{
			name: "sweep removes the victim and the band refills",
			hook: func(h *testHarness, r *hookedRegistry, a []*FlowItem, _, refill *FlowItem, c contracts.ManagedQueue) {
				r.beforeRemove = func() {
					a[3].Finalize(context.Canceled)
					h.processor.sweepFinalizedItems()
					require.NoError(t, c.Add(refill))
				}
			},
			wantLive: []int{0, 1, 2},
		},
		{
			name: "sweep empties the victim before the last-added read",
			hook: func(h *testHarness, r *hookedRegistry, a []*FlowItem, _, _ *FlowItem, _ contracts.ManagedQueue) {
				r.beforeLastAdded = func() {
					for _, item := range a {
						item.Finalize(context.Canceled)
					}
					h.processor.sweepFinalizedItems()
				}
			},
			wantAdmitted: true,
		},
		{
			name: "arrival is cancelled before the eviction",
			hook: func(_ *testHarness, r *hookedRegistry, _ []*FlowItem, arrival, _ *FlowItem, _ contracts.ManagedQueue) {
				r.beforeLastAdded = func() { arrival.Finalize(context.Canceled) }
			},
			wantLive: []int{0, 1, 2, 3},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, reg := perFlowProcessor(t, registry.WithPriorityBand(&registry.PriorityBandConfig{
				Priority: testFlow.Priority, MaxRequests: 5, FairAdmission: true,
			}))
			a := fillFlow(t, h, reg, testFlow, 4, 1)
			fillFlow(t, h, reg, flowC, 1, 1)
			cQueue, err := reg.ManagedQueue(flowC)
			require.NoError(t, err)
			newItem := func(id string, key flowcontrol.FlowKey) *FlowItem {
				return NewItem(t.Context(), fwkfcmocks.NewMockFlowControlRequest(1, id, key), testTTL, h.clock.Now(), logr.Discard())
			}
			refill, arrival := newItem("refill", flowC), newItem("arrival", fairFlowB)
			r := &hookedRegistry{FlowRegistry: reg}
			tc.hook(h, r, a, arrival, refill, cQueue)
			h.processor.registry = r
			require.NoError(t, reg.WithConnection(fairFlowB, func(contracts.ActiveFlowConnection) error {
				h.processor.enqueue(arrival)
				return nil
			}))

			if tc.wantAdmitted {
				requireQueued(t, arrival)
			} else {
				require.Nil(t, arrival.Handle(), "the arrival must not be admitted")
			}
			for i, item := range a {
				if slices.Contains(tc.wantLive, i) {
					requireQueued(t, item)
				} else {
					require.NotEqual(t, types.QueueOutcomeEvictedFairAdmission, item.FinalState().Outcome)
				}
			}
			require.Zero(t, h.processor.dropCounts[types.QueueOutcomeEvictedFairAdmission].Load(),
				"no live request may be evicted")
		})
	}
}
