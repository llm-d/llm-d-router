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

package kvblock_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	. "github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// createDenseIndexForTesting creates a new DenseMemoryIndex for testing.
func createDenseIndexForTesting(t *testing.T) Index {
	t.Helper()
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "16MiB"})
	require.NoError(t, err)
	return index
}

func TestDenseIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createDenseIndexForTesting)
}

func TestDenseIndexConfigValidation(t *testing.T) {
	_, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "not-a-size"})
	assert.Error(t, err)

	_, err = NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "0B"})
	assert.Error(t, err, "a non-positive budget must be rejected")

	_, err = NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB", PodCacheSize: 16})
	assert.Error(t, err, "a pod cap above 15 must be rejected")

	index, err := NewDenseMemoryIndex(nil)
	require.NoError(t, err)
	require.IsType(t, &DenseMemoryIndex{}, index)

	// NewIndex selects the dense backend from IndexConfig.
	selected, err := NewIndex(t.Context(), &IndexConfig{DenseMemoryConfig: DefaultDenseMemoryIndexConfig()})
	require.NoError(t, err)
	require.IsType(t, &DenseMemoryIndex{}, selected)
}

// TestDenseIndexClearRevivesOrdinal pins the dead-pod revival: after Clear,
// the pod's entries are invisible, a re-Add assigns a fresh ordinal, and the
// old ordinal stays dead.
func TestDenseIndexClearRevivesOrdinal(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB"})
	require.NoError(t, err)

	pod := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1, 2}, []PodEntry{pod}))

	before, ok := index.PodOrdinal("pod-a")
	require.True(t, ok)

	require.NoError(t, index.Clear(ctx, "pod-a"))
	hits, err := index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, hits[1])

	require.NoError(t, index.Add(ctx, nil, []BlockHash{3}, []PodEntry{pod}))
	after, ok := index.PodOrdinal("pod-a")
	require.True(t, ok)
	assert.NotEqual(t, before, after, "a cleared pod revives under a fresh ordinal")

	hits, err = index.Lookup(ctx, []BlockHash{3}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, hits[3], 1, "the revived pod's entries are visible")

	// The dead ordinal's entries stay filtered.
	hits, err = index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, hits[1])
}

// TestDenseIndexUngroupedGroupedCoexist pins full-PodEntry equality: a
// grouped and an ungrouped copy of one pod and tier coexist, and an eviction
// removes only the copy it names.
func TestDenseIndexUngroupedGroupedCoexist(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB"})
	require.NoError(t, err)

	plain := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	grouped := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu", HasGroup: true, GroupIdx: 1}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, []PodEntry{plain}))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, []PodEntry{grouped}))

	hits, err := index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.ElementsMatch(t, []PodEntry{plain, grouped}, hits[1])

	require.NoError(t, index.Evict(ctx, RequestKey, []BlockHash{1}, []PodEntry{plain}))
	hits, err = index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Equal(t, []PodEntry{grouped}, hits[1], "the grouped copy survives the ungrouped eviction")

	require.NoError(t, index.Evict(ctx, RequestKey, []BlockHash{1}, []PodEntry{grouped}))
	hits, err = index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Empty(t, hits[1])
}

// TestDenseIndexSpeculativeWord pins bit 8's identity: a speculative entry
// without a device tier round-trips, and its eviction clears it without
// touching any tier.
func TestDenseIndexSpeculativeWord(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB"})
	require.NoError(t, err)

	spec := PodEntry{PodIdentifier: "pod-a", Speculative: true}
	gpu := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, []PodEntry{spec, gpu}))

	hits, err := index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.ElementsMatch(t, []PodEntry{spec, gpu}, hits[1])

	require.NoError(t, index.Evict(ctx, RequestKey, []BlockHash{1}, []PodEntry{spec}))
	hits, err = index.Lookup(ctx, []BlockHash{1}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Equal(t, []PodEntry{gpu}, hits[1], "evicting the speculative copy leaves the tiered entry")
}

// TestDenseIndexSweepBoundsResidency pins the byte budget: with a budget far
// below the inserted volume, sampled eviction holds residency under the
// budget.
func TestDenseIndexSweepBoundsResidency(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "8KiB"})
	require.NoError(t, err)

	pod := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	for k := 1; k <= 2000; k++ {
		require.NoError(t, index.Add(ctx, nil, []BlockHash{BlockHash(k)}, []PodEntry{pod}))
	}

	stats := index.Stats()
	assert.Less(t, stats.ResidentKeys, 2000, "the sweep must evict under budget pressure")
	assert.Positive(t, stats.ResidentKeys)
	// A slot is 4 words; the sweep lands at or below the budget's key count.
	assert.LessOrEqual(t, stats.ResidentKeys, 8*1024/16, "residency must land at or below the budget")
}

// TestDenseWalkKeysContract exercises the dense walk against the KeyWalker
// contract: ordered visits, miss handling, group-duplicate folding, and
// dead-pod filtering.
func TestDenseWalkKeysContract(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB"})
	require.NoError(t, err)

	plain := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	grouped := PodEntry{PodIdentifier: "pod-b", DeviceTier: "gpu", HasGroup: true, GroupIdx: 1}
	other := PodEntry{PodIdentifier: "pod-b", DeviceTier: "gpu", HasGroup: true, GroupIdx: 2}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{10}, []PodEntry{plain, grouped, other}))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{30}, []PodEntry{plain}))

	type visit struct {
		pos     int
		found   bool
		entries []WalkEntry
	}
	var visits []visit
	err = index.WalkKeys(ctx, []BlockHash{10, 20, 30}, func(pos int, found bool, entries []WalkEntry) bool {
		visits = append(visits, visit{pos: pos, found: found, entries: append([]WalkEntry(nil), entries...)})
		return true
	})
	require.NoError(t, err)

	require.Len(t, visits, 3)
	assert.Equal(t, 0, visits[0].pos)
	assert.True(t, visits[0].found)
	require.Len(t, visits[0].entries, 2, "group duplicates of one pod and tier fold to one entry")
	podA, _ := index.PodOrdinal("pod-a")
	podB, _ := index.PodOrdinal("pod-b")
	gpu, _ := index.TierOrdinal("gpu")
	assert.ElementsMatch(t, []WalkEntry{
		{PodOrdinal: podA, TierOrdinal: gpu},
		{PodOrdinal: podB, TierOrdinal: gpu},
	}, visits[0].entries)
	assert.False(t, visits[1].found, "an absent key is a miss")
	assert.Empty(t, visits[1].entries)
	assert.True(t, visits[2].found)

	// Clearing a pod filters its entries from the walk without removing the
	// key: the slot stays, the entries go.
	require.NoError(t, index.Clear(ctx, "pod-a"))
	visits = nil
	err = index.WalkKeys(ctx, []BlockHash{10}, func(pos int, found bool, entries []WalkEntry) bool {
		visits = append(visits, visit{pos: pos, found: found, entries: append([]WalkEntry(nil), entries...)})
		return true
	})
	require.NoError(t, err)
	require.Len(t, visits, 1)
	assert.True(t, visits[0].found)
	assert.Len(t, visits[0].entries, 1, "a cleared pod's entries are filtered from the walk")
	assert.Equal(t, podB, visits[0].entries[0].PodOrdinal)
}

// TestDenseIndexGetRequestKey pins engine-to-request resolution through the
// dense backend's table.
func TestDenseIndexGetRequestKey(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "1MiB"})
	require.NoError(t, err)

	engineKey, requestKey := BlockHash(100), BlockHash(200)
	pod := PodEntry{PodIdentifier: "pod-a", DeviceTier: "gpu"}
	require.NoError(t, index.Add(ctx, []BlockHash{engineKey}, []BlockHash{requestKey}, []PodEntry{pod}))

	rk, err := index.GetRequestKey(ctx, engineKey)
	require.NoError(t, err)
	assert.Equal(t, requestKey, rk)
}
