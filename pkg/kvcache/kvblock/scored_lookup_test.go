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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	. "github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

var scoredLookupTierWeights = map[string]float64{"gpu": 1.0, "cpu": 0.8}

func TestScoredLookup(t *testing.T) {
	podA := "pod-a"
	podB := "pod-b"
	keys := []BlockHash{10, 20, 30}

	tests := []struct {
		name        string
		entries     map[BlockHash][]PodEntry
		requestKeys []BlockHash
		podFilter   sets.Set[string]
		wantStats   map[string]PodMatchStats
	}{
		{
			name: "weighted chains and per-tier counts",
			entries: map[BlockHash][]PodEntry{
				10: {
					{PodIdentifier: podA, DeviceTier: "gpu"},
					{PodIdentifier: podA, DeviceTier: "cpu"},
					{PodIdentifier: podB, DeviceTier: "gpu"},
				},
				20: {
					{PodIdentifier: podA, DeviceTier: "gpu"},
					{PodIdentifier: podB, DeviceTier: "cpu"},
				},
				30: {
					{PodIdentifier: podA, DeviceTier: "cpu"},
				},
			},
			requestKeys: keys,
			wantStats: map[string]PodMatchStats{
				// Per block the highest tier weight counts: gpu, gpu, cpu.
				// gpu chain breaks at key 30, cpu chain at key 20.
				podA: {WeightedScore: 2.8, MatchedBlocks: 3, BlocksByTier: map[string]int{"gpu": 2, "cpu": 1}},
				// gpu@10 then cpu@20: any-tier chain spans both, no tier
				// survives past its own break.
				podB: {WeightedScore: 1.8, MatchedBlocks: 2, BlocksByTier: map[string]int{"gpu": 1}},
			},
		},
		{
			name: "chain breaks at missing key",
			entries: map[BlockHash][]PodEntry{
				10: {{PodIdentifier: podA, DeviceTier: "gpu"}},
				30: {{PodIdentifier: podA, DeviceTier: "gpu"}},
			},
			requestKeys: keys,
			wantStats: map[string]PodMatchStats{
				podA: {WeightedScore: 1.0, MatchedBlocks: 1, BlocksByTier: map[string]int{"gpu": 1}},
			},
		},
		{
			name: "missing first key yields no stats",
			entries: map[BlockHash][]PodEntry{
				20: {{PodIdentifier: podA, DeviceTier: "gpu"}},
			},
			requestKeys: keys,
			wantStats:   map[string]PodMatchStats{},
		},
		{
			name: "pod filter excludes other pods",
			entries: map[BlockHash][]PodEntry{
				10: {
					{PodIdentifier: podA, DeviceTier: "gpu"},
					{PodIdentifier: podB, DeviceTier: "gpu"},
				},
				20: {
					{PodIdentifier: podA, DeviceTier: "gpu"},
					{PodIdentifier: podB, DeviceTier: "gpu"},
				},
			},
			requestKeys: []BlockHash{10, 20},
			podFilter:   sets.New(podB),
			wantStats: map[string]PodMatchStats{
				podB: {WeightedScore: 2.0, MatchedBlocks: 2, BlocksByTier: map[string]int{"gpu": 2}},
			},
		},
		{
			name: "speculative entries count under the speculative tier",
			entries: map[BlockHash][]PodEntry{
				10: {{PodIdentifier: podA, Speculative: true}},
			},
			requestKeys: []BlockHash{10},
			wantStats: map[string]PodMatchStats{
				podA: {WeightedScore: 1.0, MatchedBlocks: 1, BlocksByTier: map[string]int{SpeculativeTier: 1}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := logging.NewTestLoggerIntoContext(t.Context())
			index, err := NewInMemoryIndex(nil)
			require.NoError(t, err)
			for key, entries := range tt.entries {
				require.NoError(t, index.Add(ctx, nil, []BlockHash{key}, entries))
			}

			stats, err := index.ScoredLookup(ctx, tt.requestKeys, tt.podFilter, scoredLookupTierWeights)
			require.NoError(t, err)
			require.Len(t, stats, len(tt.wantStats))
			for pod, want := range tt.wantStats {
				got, ok := stats[pod]
				require.True(t, ok, "missing pod %q", pod)
				assert.InDelta(t, want.WeightedScore, got.WeightedScore, 0.0001, "pod %q weighted score", pod)
				assert.Equal(t, want.MatchedBlocks, got.MatchedBlocks, "pod %q matched blocks", pod)
				assert.Equal(t, want.BlocksByTier, got.BlocksByTier, "pod %q blocks by tier", pod)
			}
		})
	}
}

func TestScoredLookupEmptyKeys(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)

	_, err = index.ScoredLookup(ctx, nil, nil, scoredLookupTierWeights)
	assert.Error(t, err)
}

func TestScoredLookupDoesNotRefreshIndexLRU(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 2, PodCacheSize: 1})
	require.NoError(t, err)

	entry := []PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}}
	require.NoError(t, index.Add(ctx, nil, []BlockHash{10}, entry))
	require.NoError(t, index.Add(ctx, nil, []BlockHash{20}, entry))

	stats, err := index.ScoredLookup(ctx, []BlockHash{10}, nil, scoredLookupTierWeights)
	require.NoError(t, err)
	require.Contains(t, stats, "pod-a")

	require.NoError(t, index.Add(ctx, nil, []BlockHash{30}, entry))

	stats, err = index.ScoredLookup(ctx, []BlockHash{10}, nil, scoredLookupTierWeights)
	require.NoError(t, err)
	assert.Empty(t, stats)
	stats, err = index.ScoredLookup(ctx, []BlockHash{20}, nil, scoredLookupTierWeights)
	require.NoError(t, err)
	assert.Contains(t, stats, "pod-a")
}

// Pod churn (interned ids from pods since cleared) must change neither
// results nor the candidate slot assignment.
func TestScoredLookupAfterPodChurn(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)

	entries := []PodEntry{{PodIdentifier: "pod-live", DeviceTier: "gpu"}}
	keys := []BlockHash{10, 20}
	require.NoError(t, index.Add(ctx, nil, keys, entries))

	churnKey := []BlockHash{999}
	for i := 0; i < 2000; i++ {
		pod := fmt.Sprintf("10.9.%d.%d:8000", i/256, i%256)
		churn := []PodEntry{{PodIdentifier: pod, DeviceTier: "gpu"}}
		require.NoError(t, index.Add(ctx, nil, churnKey, churn))
		require.NoError(t, index.Clear(ctx, pod))
	}

	stats, err := index.ScoredLookup(ctx, keys, nil, scoredLookupTierWeights)
	require.NoError(t, err)
	require.Len(t, stats, 1)
	got := stats["pod-live"]
	assert.InDelta(t, 2.0, got.WeightedScore, 0.0001)
	assert.Equal(t, 2, got.MatchedBlocks)
	assert.Equal(t, map[string]int{"gpu": 2}, got.BlocksByTier)
}

// More interned tiers than the chain bitmask represents must fall back to the
// legacy path via the sentinel, directly and through the wrapper chain.
func TestScoredLookupUnsupportedBeyond64Tiers(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)

	for i := 0; i < 65; i++ {
		entry := []PodEntry{{PodIdentifier: "pod-a", DeviceTier: fmt.Sprintf("tier-%d", i)}}
		require.NoError(t, index.Add(ctx, nil, []BlockHash{BlockHash(i + 1)}, entry))
	}

	_, err = index.ScoredLookup(ctx, []BlockHash{1}, nil, scoredLookupTierWeights)
	require.ErrorIs(t, err, ErrScoredLookupUnsupported)

	wrapped, ok := NewTracedIndex(NewInstrumentedIndex(index)).(ScoredLookupIndex)
	require.True(t, ok)
	_, err = wrapped.ScoredLookup(ctx, []BlockHash{1}, nil, scoredLookupTierWeights)
	require.ErrorIs(t, err, ErrScoredLookupUnsupported)
}

// Concurrent Adds that intern new tiers mid-walk must never yield impossible
// statistics: a matched pod always carries at least one per-tier count and a
// weight consistent with its configured tier.
func TestScoredLookupConcurrentNewTiers(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	index, err := NewInMemoryIndex(nil)
	require.NoError(t, err)

	weights := map[string]float64{}
	for i := 0; i < 60; i++ {
		weights[fmt.Sprintf("tier-%d", i)] = 5.0
	}

	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			entry := []PodEntry{{PodIdentifier: "pod-a", DeviceTier: fmt.Sprintf("tier-%d", i%60)}}
			_ = index.Add(ctx, nil, []BlockHash{10}, entry)
		}
	}()

	for i := 0; i < 300; i++ {
		stats, err := index.ScoredLookup(ctx, []BlockHash{10}, nil, weights)
		require.NoError(t, err)
		for pod, s := range stats {
			require.Positive(t, s.MatchedBlocks, "pod %q", pod)
			require.NotEmpty(t, s.BlocksByTier,
				"pod %q matched %d blocks but reports no tier", pod, s.MatchedBlocks)
			require.InDelta(t, 5.0, s.WeightedScore, 0.0001,
				"pod %q must score with the configured tier weight, not a stale default", pod)
		}
	}
	close(stop)
	<-done
}
