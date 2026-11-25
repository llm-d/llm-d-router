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

package kvblock

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

func newStatsTestIndex(t *testing.T, size, podCacheSize int) *InMemoryIndex {
	t.Helper()
	idx, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: size, PodCacheSize: podCacheSize})
	require.NoError(t, err)
	return idx
}

// evictions reads a cause counter. The counters are process-global, so the
// tests assert deltas around the calls under test, never absolute values.
func evictions(cause string) float64 {
	return testutil.ToFloat64(metrics.IndexEvictions.WithLabelValues(cause))
}

// clearDurationCount reads IndexClearDuration's sample count.
func clearDurationCount(t *testing.T) uint64 {
	t.Helper()
	var m dto.Metric
	require.NoError(t, metrics.IndexClearDuration.(prometheus.Metric).Write(&m))
	return m.GetHistogram().GetSampleCount()
}

// TestInMemoryStatsLifecycle asserts the Stats snapshot and the residency,
// eviction, truncation, and clear metrics across Add, Evict, Clear, per-key
// truncation, and main-store capacity eviction.
func TestInMemoryStatsLifecycle(t *testing.T) {
	ctx := t.Context()
	podA := PodEntry{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"}
	podB := PodEntry{PodIdentifier: "10.0.0.2:8000", DeviceTier: "gpu"}
	podC := PodEntry{PodIdentifier: "10.0.0.3:8000", DeviceTier: "gpu"}

	t.Run("Add and Evict track keys, entries, and mappings", func(t *testing.T) {
		idx := newStatsTestIndex(t, 100, 10)
		require.NoError(t, idx.Add(ctx, []BlockHash{10}, []BlockHash{1, 2}, []PodEntry{podA, podB}))

		stats := idx.Stats()
		require.Equal(t, 2, stats.ResidentKeys)
		require.Equal(t, 4, int(idx.entries.Load()))
		require.Equal(t, 1, stats.EngineMappings)
		require.Equal(t, int64(0), stats.TruncatedEntries)

		emptyBefore := evictions("empty")
		require.NoError(t, idx.Evict(ctx, RequestKey, []BlockHash{1}, []PodEntry{podA, podB}))
		stats = idx.Stats()
		require.Equal(t, 1, stats.ResidentKeys)
		require.Equal(t, 2, int(idx.entries.Load()))
		// Emptying key 1 removed it and its engine mapping.
		require.Equal(t, emptyBefore+1, evictions("empty"))
	})

	t.Run("per-key cap truncation counts dropped pods", func(t *testing.T) {
		idx := newStatsTestIndex(t, 100, 2)
		require.NoError(t, idx.Add(ctx, []BlockHash{10}, []BlockHash{1}, []PodEntry{podA, podB}))
		truncatedBefore := testutil.ToFloat64(metrics.IndexTruncatedEntries)
		require.NoError(t, idx.Add(ctx, []BlockHash{10}, []BlockHash{1}, []PodEntry{podC}))

		stats := idx.Stats()
		require.Equal(t, 1, stats.ResidentKeys)
		require.Equal(t, 2, int(idx.entries.Load()))
		require.Equal(t, int64(1), stats.TruncatedEntries)
		require.Equal(t, truncatedBefore+1, testutil.ToFloat64(metrics.IndexTruncatedEntries))
	})

	t.Run("Clear removes matching entries and observes duration", func(t *testing.T) {
		idx := newStatsTestIndex(t, 100, 10)
		require.NoError(t, idx.Add(ctx, []BlockHash{10}, []BlockHash{1}, []PodEntry{podA, podB}))

		clearDur := clearDurationCount(t)
		clearBefore := evictions("clear")
		emptyBefore := evictions("empty")
		require.NoError(t, idx.Clear(ctx, podA.PodIdentifier))

		stats := idx.Stats()
		require.Equal(t, 1, stats.ResidentKeys)
		require.Equal(t, 1, int(idx.entries.Load()))
		require.Equal(t, clearDur+1, clearDurationCount(t))
		// podB remains, so the key survives and nothing is counted yet.
		require.Equal(t, clearBefore, evictions("clear"))

		// Clearing the last pod empties the key. The engine mapping is
		// intentionally left behind: it is LRU-bounded and self-heals.
		require.NoError(t, idx.Clear(ctx, podB.PodIdentifier))
		stats = idx.Stats()
		require.Equal(t, 0, stats.ResidentKeys)
		require.Equal(t, 0, int(idx.entries.Load()))
		require.Equal(t, 1, stats.EngineMappings)
		require.Equal(t, clearBefore+1, evictions("clear"))
		require.Equal(t, emptyBefore, evictions("empty"))
	})

	t.Run("main-store capacity eviction counts and decrements", func(t *testing.T) {
		idx := newStatsTestIndex(t, 2, 10)
		require.NoError(t, idx.Add(ctx, []BlockHash{10}, []BlockHash{1}, []PodEntry{podA}))
		require.NoError(t, idx.Add(ctx, []BlockHash{11}, []BlockHash{2}, []PodEntry{podB}))

		capacityBefore := evictions("capacity")
		require.NoError(t, idx.Add(ctx, []BlockHash{12}, []BlockHash{3}, []PodEntry{podC}))

		stats := idx.Stats()
		require.Equal(t, 2, stats.ResidentKeys)
		require.Equal(t, 2, int(idx.entries.Load()))
		require.Equal(t, capacityBefore+1, evictions("capacity"))
	})

	t.Run("gauges track residency", func(t *testing.T) {
		idx := newStatsTestIndex(t, 100, 10)
		require.NoError(t, idx.Add(ctx, []BlockHash{10}, []BlockHash{1}, []PodEntry{podA, podB}))

		stats := idx.Stats()
		require.Equal(t, float64(1), testutil.ToFloat64(metrics.IndexResidentKeys))
		require.Equal(t, float64(stats.ResidentBytes), testutil.ToFloat64(metrics.IndexResidentBytes))
	})
}
