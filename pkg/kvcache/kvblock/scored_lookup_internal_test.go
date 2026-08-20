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
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Both metric paths must report the same hit quantity: the legacy Lookup fold
// and the fused MatchedBlocks agree on a duplicate-tier plus prefix-gap
// fixture, where the per-appearance legacy fold would have reported 3.
func TestHitMetricPathsAgree(t *testing.T) {
	ctx := context.Background()
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)

	require.NoError(t, index.Add(ctx, nil, []BlockHash{1}, []PodEntry{
		{PodIdentifier: "pod-a", DeviceTier: "gpu"},
		{PodIdentifier: "pod-a", DeviceTier: "cpu"},
	}))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{2}, []PodEntry{
		{PodIdentifier: "pod-a", DeviceTier: "gpu"},
	}))
	// Key 3 is absent: the prefix chain breaks after two blocks.
	keys := []BlockHash{1, 2, 3}

	keyToPods, err := index.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	legacyHits := maxContiguousPodHits(keys, keyToPods)

	stats, err := index.ScoredLookup(ctx, keys, nil, map[string]float64{"gpu": 1.0, "cpu": 0.8})
	require.NoError(t, err)

	assert.Equal(t, 2, legacyHits)
	assert.Equal(t, legacyHits, stats["pod-a"].MatchedBlocks,
		"legacy and fused paths must feed the hit metrics the same quantity")
}

func TestScoredLookupColdScratchDoesNotTrackPodHistory(t *testing.T) {
	measure := func(churnPods int) uint64 {
		ctx := log.IntoContext(context.Background(), logr.Discard())
		index, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 128, PodCacheSize: 128})
		require.NoError(t, err)

		keys := []BlockHash{10, 20}
		for i := 0; i < 96; i++ {
			entry := []PodEntry{{PodIdentifier: fmt.Sprintf("pod-live-%d", i), DeviceTier: "gpu"}}
			require.NoError(t, index.Add(ctx, nil, keys, entry))
		}
		for i := 0; i < churnPods; i++ {
			pod := fmt.Sprintf("pod-gone-%d", i)
			require.NoError(t, index.Add(ctx, nil, []BlockHash{999}, []PodEntry{{PodIdentifier: pod, DeviceTier: "gpu"}}))
			require.NoError(t, index.Clear(ctx, pod))
		}

		scoredScratchPool = sync.Pool{New: func() any { return &scoredScratch{} }}
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		stats, err := index.ScoredLookup(ctx, keys, nil, map[string]float64{"gpu": 1})
		require.NoError(t, err)
		require.Len(t, stats, 96)
		runtime.ReadMemStats(&after)
		return after.TotalAlloc - before.TotalAlloc
	}

	measure(0) // warm lazy runtime and package initialization
	withoutChurn := measure(0)
	withChurn := measure(20_000)
	assert.Less(t, withChurn, withoutChurn+128*1024,
		"cold request scratch must scale with live candidates, not interned pod history")
}

// BenchmarkMaxContiguousPodHits measures the hit-metric fold on the legacy
// path's worst realistic shape: a full-hit lookup result at fleet scale.
func BenchmarkMaxContiguousPodHits(b *testing.B) {
	const numKeys, numPods = 3750, 96
	keys := make([]BlockHash, numKeys)
	keyToPods := make(map[BlockHash][]PodEntry, numKeys)
	for i := range keys {
		keys[i] = BlockHash(i + 1)
		entries := make([]PodEntry, numPods)
		for p := range entries {
			entries[p] = PodEntry{PodIdentifier: podNamesForBench[p], DeviceTier: "gpu"}
		}
		keyToPods[keys[i]] = entries
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := maxContiguousPodHits(keys, keyToPods); got != numKeys {
			b.Fatalf("unexpected result %d", got)
		}
	}
}

var podNamesForBench = func() []string {
	names := make([]string, 96)
	for i := range names {
		names[i] = fmt.Sprintf("10.0.%d.%d:8000", i/256, i%256)
	}
	return names
}()
