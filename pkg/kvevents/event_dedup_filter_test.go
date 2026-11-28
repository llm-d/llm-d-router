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

package kvevents //nolint:testpackage // tests use the unexported eventDedupFilter

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func testScope(pod, tier string, group, dpRank int) blockScope {
	return blockScope{
		podIdentifier:    pod,
		deviceTier:       tier,
		groupIdx:         group,
		dataParallelRank: dpRank,
	}
}

func gpuScope(pod string) blockScope {
	return testScope(pod, "gpu", noGroupIdx, noDataParallelRank)
}

// forwarded returns the hashes filterRemove forwards to the index.
func forwarded(kept, _ []uint64) []uint64 {
	return kept
}

// TestEventDedupFilter_DuplicateStoreSuppressesFirstRemove is the core contract:
// two announcements of the same hashes (overlapping offloaded chunks) require
// two removes before the index eviction is forwarded.
func TestEventDedupFilter_DuplicateStoreSuppressesFirstRemove(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")
	hashes := []uint64{1, 2, 3}

	f.trackStore(s, hashes)
	f.trackStore(s, hashes) // sibling chunk re-announces the same constituent hashes

	assert.Empty(t, forwarded(f.filterRemove(s, hashes)),
		"first of two duplicate removes must be fully suppressed")
	assert.Equal(t, hashes, forwarded(f.filterRemove(s, hashes)),
		"second remove must forward every hash to the index")
}

// TestEventDedupFilter_AggregatesAcrossSources documents the pod-level intent:
// because the index identity is rank-agnostic on current main, stores that
// share a scope (e.g. different data-parallel ranks, both using the sentinel)
// aggregate into one count, so the block is only evicted once every reference
// is released.
func TestEventDedupFilter_AggregatesAcrossSources(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")

	f.trackStore(s, []uint64{7}) // source 1
	f.trackStore(s, []uint64{7}) // source 2, same scope

	assert.Empty(t, forwarded(f.filterRemove(s, []uint64{7})), "one reference still outstanding")
	assert.Equal(t, []uint64{7}, forwarded(f.filterRemove(s, []uint64{7})), "last reference released")
}

// TestEventDedupFilter_TierIndependence verifies a CPU store cannot mask a GPU
// remove (and vice versa): the index keeps gpu and cpu copies of a hash as
// distinct entries, so their reference counts must be independent.
func TestEventDedupFilter_TierIndependence(t *testing.T) {
	f := newEventDedupFilter()
	gpu := testScope("pod-a", "gpu", noGroupIdx, noDataParallelRank)
	cpu := testScope("pod-a", "cpu", noGroupIdx, noDataParallelRank)

	f.trackStore(gpu, []uint64{1})
	f.trackStore(cpu, []uint64{1})

	// A single cpu remove must forward (cpu count 1->0); if the counts were
	// shared this would be suppressed (2->1) and kept empty.
	assert.Equal(t, []uint64{1}, forwarded(f.filterRemove(cpu, []uint64{1})),
		"cpu remove must not be masked by the gpu store")
	// The gpu reference is still outstanding and forwards independently.
	assert.Equal(t, []uint64{1}, forwarded(f.filterRemove(gpu, []uint64{1})),
		"gpu remove must still forward after the cpu remove")
}

// TestEventDedupFilter_GroupIndependence verifies different KV-cache groups are
// reference-counted independently, mirroring distinct grouped PodEntries.
func TestEventDedupFilter_GroupIndependence(t *testing.T) {
	f := newEventDedupFilter()
	g0 := testScope("pod-a", "gpu", 0, noDataParallelRank)
	g1 := testScope("pod-a", "gpu", 1, noDataParallelRank)

	f.trackStore(g0, []uint64{1})
	f.trackStore(g1, []uint64{1})

	assert.Equal(t, []uint64{1}, forwarded(f.filterRemove(g1, []uint64{1})),
		"group 1 remove must be independent of the group 0 store")
}

// TestEventDedupFilter_DataParallelRankIndependence verifies that distinct
// dataParallelRank values keep independent reference counts within one
// podIdentifier bucket. Production callers pass noDataParallelRank because
// pod discovery encodes the rank in podIdentifier.
func TestEventDedupFilter_DataParallelRankIndependence(t *testing.T) {
	f := newEventDedupFilter()
	dp0 := testScope("pod-a", "gpu", noGroupIdx, 0)
	dp1 := testScope("pod-a", "gpu", noGroupIdx, 1)

	f.trackStore(dp0, []uint64{1})
	f.trackStore(dp1, []uint64{1})

	assert.Equal(t, []uint64{1}, forwarded(f.filterRemove(dp1, []uint64{1})),
		"rank 1 remove must be independent of the rank 0 store")
}

// TestEventDedupFilter_UnknownRemovePassesThrough verifies defensive
// pass-through for never-seen hashes and that the count never goes negative.
func TestEventDedupFilter_UnknownRemovePassesThrough(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")

	assert.Equal(t, []uint64{42}, forwarded(f.filterRemove(s, []uint64{42})),
		"unknown remove must pass through")
	assert.Equal(t, []uint64{42}, forwarded(f.filterRemove(s, []uint64{42})),
		"repeated unknown remove must keep passing through (no negative count)")

	// A store after underflow attempts still yields a clean single reference.
	f.trackStore(s, []uint64{42})
	assert.Equal(t, []uint64{42}, forwarded(f.filterRemove(s, []uint64{42})),
		"store must establish a fresh single reference")
}

// TestEventDedupFilter_ReportsUntrackedRemoves verifies that a forwarded hash
// is reported as untracked only when no store referenced it.
func TestEventDedupFilter_ReportsUntrackedRemoves(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")
	f.trackStore(s, []uint64{1, 2, 2})

	kept, untracked := f.filterRemove(s, []uint64{1, 1, 2, 3})
	assert.Equal(t, []uint64{1, 1, 3}, kept)
	assert.Equal(t, []uint64{1, 3}, untracked)
}

// TestEventDedupFilter_PartialForward verifies a mixed remove forwards only the
// hashes that are released or unknown, preserving input order.
func TestEventDedupFilter_PartialForward(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")

	f.trackStore(s, []uint64{1}) // hash 1: count 2 (still referenced after one remove)
	f.trackStore(s, []uint64{1})
	f.trackStore(s, []uint64{2}) // hash 2: count 1 (released by one remove)

	// hash 1 suppressed (2->1), hash 2 forwarded (1->0), hash 3 unknown forwarded.
	assert.Equal(t, []uint64{2, 3}, forwarded(f.filterRemove(s, []uint64{1, 2, 3})))
}

// TestEventDedupFilter_ClearResets verifies clear zeroes a pod's counts.
func TestEventDedupFilter_ClearResets(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")

	f.trackStore(s, []uint64{1})
	f.trackStore(s, []uint64{1}) // count 2

	f.clear("pod-a")

	// If clear had not reset the count, this remove would be suppressed (2->1).
	assert.Equal(t, []uint64{1}, forwarded(f.filterRemove(s, []uint64{1})),
		"clear must reset the count so the remove passes through")
}

// TestEventDedupFilter_ClearIsolatesPods verifies clearing one pod does not
// affect another pod's counts.
func TestEventDedupFilter_ClearIsolatesPods(t *testing.T) {
	f := newEventDedupFilter()
	a := gpuScope("pod-a")
	b := gpuScope("pod-b")

	f.trackStore(a, []uint64{1})
	f.trackStore(a, []uint64{1})
	f.trackStore(b, []uint64{1})
	f.trackStore(b, []uint64{1})

	f.clear("pod-a")

	assert.Equal(t, []uint64{1}, forwarded(f.filterRemove(a, []uint64{1})),
		"cleared pod-a count must be reset")
	assert.Empty(t, forwarded(f.filterRemove(b, []uint64{1})),
		"pod-b count must survive pod-a clear (first of two removes suppressed)")
}

// TestEventDedupFilter_ReleasedScopesAreDropped verifies the filter holds no
// state for a scope or pod once every reference in it is released.
func TestEventDedupFilter_ReleasedScopesAreDropped(t *testing.T) {
	f := newEventDedupFilter()
	gpu := gpuScope("pod-a")
	cpu := testScope("pod-a", "cpu", noGroupIdx, noDataParallelRank)

	f.trackStore(gpu, []uint64{1, 2})
	f.trackStore(cpu, []uint64{1})

	f.filterRemove(gpu, []uint64{1, 2})
	assert.NotContains(t, f.refs["pod-a"], gpu.podScope(), "released gpu scope must be dropped")
	assert.Contains(t, f.refs["pod-a"], cpu.podScope(), "cpu scope still holds a reference")

	f.filterRemove(cpu, []uint64{1})
	assert.Empty(t, f.refs, "pod with no references left must be dropped")
}

// TestEventDedupFilter_NilSafe verifies the nil-receiver guards so a Pool
// without a filter degrades to forwarding every event.
func TestEventDedupFilter_NilSafe(t *testing.T) {
	var f *eventDedupFilter
	s := gpuScope("pod-a")

	assert.NotPanics(t, func() { f.trackStore(s, []uint64{1}) })
	assert.NotPanics(t, func() { f.clear("pod-a") })
	assert.Equal(t, []uint64{1}, forwarded(f.filterRemove(s, []uint64{1})),
		"nil filter must forward removes unchanged")
}

func TestGroupIdxOrNoGroup(t *testing.T) {
	assert.Equal(t, noGroupIdx, groupIdxOrNoGroup(nil))
	g := 3
	assert.Equal(t, 3, groupIdxOrNoGroup(&g))
}
