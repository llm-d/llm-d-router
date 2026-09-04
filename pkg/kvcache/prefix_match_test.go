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

package kvcache_test

import (
	"context"
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// newMatcher builds an Indexer over an in-memory index whose per-key pod
// cache is large enough that no fixture entry is evicted.
func newMatcher(t *testing.T, backends []*kvcache.KVCacheBackendConfig) (*kvcache.Indexer, kvblock.Index) {
	t.Helper()
	idx, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 12, PodCacheSize: 256})
	require.NoError(t, err)
	return kvcache.NewIndexerForTest(&mockTokenProcessor{}, idx, backends), idx
}

func assertPodMatches(t *testing.T, want, got map[string]kvcache.PodMatch) {
	t.Helper()
	require.Len(t, got, len(want))
	for pod, w := range want {
		g, ok := got[pod]
		require.True(t, ok, "missing pod %q", pod)
		assert.InDelta(t, w.WeightedScore, g.WeightedScore, 0.0001, "pod %q weighted score", pod)
		assert.Equal(t, w.MatchedBlocks, g.MatchedBlocks, "pod %q matched blocks", pod)
		assert.Equal(t, w.BlocksByTier, g.BlocksByTier, "pod %q blocks by tier", pod)
	}
}

func TestMatchBlockKeys(t *testing.T) {
	podA, podB := "pod-a", "pod-b"
	keys := []kvblock.BlockHash{10, 20, 30}

	tests := []struct {
		name        string
		entries     map[kvblock.BlockHash][]kvblock.PodEntry
		requestKeys []kvblock.BlockHash
		podFilter   sets.Set[string]
		want        map[string]kvcache.PodMatch
	}{
		{
			name: "weighted chains and per-tier counts",
			entries: map[kvblock.BlockHash][]kvblock.PodEntry{
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
			want: map[string]kvcache.PodMatch{
				// Per block the highest tier weight counts: gpu, gpu, cpu.
				// gpu chain breaks at key 30, cpu chain at key 20.
				podA: {WeightedScore: 2.8, MatchedBlocks: 3, BlocksByTier: map[string]int{"gpu": 2, "cpu": 1}},
				// gpu@10 then cpu@20: the any-tier chain spans both keys,
				// no tier survives past its own break.
				podB: {WeightedScore: 1.8, MatchedBlocks: 2, BlocksByTier: map[string]int{"gpu": 1}},
			},
		},
		{
			name: "chain breaks at missing key",
			entries: map[kvblock.BlockHash][]kvblock.PodEntry{
				10: {{PodIdentifier: podA, DeviceTier: "gpu"}},
				30: {{PodIdentifier: podA, DeviceTier: "gpu"}},
			},
			requestKeys: keys,
			want: map[string]kvcache.PodMatch{
				podA: {WeightedScore: 1.0, MatchedBlocks: 1, BlocksByTier: map[string]int{"gpu": 1}},
			},
		},
		{
			name: "missing first key yields no matches",
			entries: map[kvblock.BlockHash][]kvblock.PodEntry{
				20: {{PodIdentifier: podA, DeviceTier: "gpu"}},
			},
			requestKeys: keys,
			want:        map[string]kvcache.PodMatch{},
		},
		{
			name: "pod filter excludes other pods",
			entries: map[kvblock.BlockHash][]kvblock.PodEntry{
				10: {
					{PodIdentifier: podA, DeviceTier: "gpu"},
					{PodIdentifier: podB, DeviceTier: "gpu"},
				},
				20: {
					{PodIdentifier: podA, DeviceTier: "gpu"},
					{PodIdentifier: podB, DeviceTier: "gpu"},
				},
			},
			requestKeys: []kvblock.BlockHash{10, 20},
			podFilter:   sets.New(podB),
			want: map[string]kvcache.PodMatch{
				podB: {WeightedScore: 2.0, MatchedBlocks: 2, BlocksByTier: map[string]int{"gpu": 2}},
			},
		},
		{
			name: "unconfigured tier weighs the default",
			entries: map[kvblock.BlockHash][]kvblock.PodEntry{
				10: {{PodIdentifier: podA, DeviceTier: "disk"}},
			},
			requestKeys: []kvblock.BlockHash{10},
			want: map[string]kvcache.PodMatch{
				podA: {WeightedScore: 1.0, MatchedBlocks: 1, BlocksByTier: map[string]int{"disk": 1}},
			},
		},
		{
			name: "speculative entries count under the speculative tier",
			entries: map[kvblock.BlockHash][]kvblock.PodEntry{
				10: {{PodIdentifier: podA, Speculative: true}},
			},
			requestKeys: []kvblock.BlockHash{10},
			want: map[string]kvcache.PodMatch{
				podA: {WeightedScore: 1.0, MatchedBlocks: 1, BlocksByTier: map[string]int{kvcache.SpeculativeTier: 1}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := logging.NewTestLoggerIntoContext(t.Context())
			indexer, idx := newMatcher(t, kvcache.DefaultKVCacheBackendConfig())
			populateIndex(t, idx, tt.entries)

			got, err := indexer.MatchBlockKeys(ctx, tt.requestKeys, tt.podFilter)
			require.NoError(t, err)
			assertPodMatches(t, tt.want, got)
		})
	}
}

func TestMatchBlockKeysEmptyKeys(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	indexer, _ := newMatcher(t, kvcache.DefaultKVCacheBackendConfig())

	got, err := indexer.MatchBlockKeys(ctx, nil, nil)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Empty(t, got)
}

func TestMatchBlockKeysCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(logging.NewTestLoggerIntoContext(t.Context()))
	indexer, idx := newMatcher(t, kvcache.DefaultKVCacheBackendConfig())
	populateIndex(t, idx, map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: "pod-a", DeviceTier: "gpu"}},
	})
	cancel()

	_, err := indexer.MatchBlockKeys(ctx, []kvblock.BlockHash{10}, nil)
	require.ErrorIs(t, err, context.Canceled)
}

// A device tier reported under the speculative tier's name and speculative
// entries form one chain, as the per-tier counters always treated them.
func TestMatchBlockKeysDeviceTierNamedSpeculative(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	indexer, idx := newMatcher(t, kvcache.DefaultKVCacheBackendConfig())
	populateIndex(t, idx, map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: "pod-a", DeviceTier: kvcache.SpeculativeTier}},
		20: {{PodIdentifier: "pod-a", Speculative: true}},
		30: {{PodIdentifier: "pod-a", DeviceTier: kvcache.SpeculativeTier}},
	})

	got, err := indexer.MatchBlockKeys(ctx, []kvblock.BlockHash{10, 20, 30}, nil)
	require.NoError(t, err)
	assertPodMatches(t, map[string]kvcache.PodMatch{
		"pod-a": {WeightedScore: 3.0, MatchedBlocks: 3, BlocksByTier: map[string]int{kvcache.SpeculativeTier: 3}},
	}, got)
}

// lateCancelContext reports cancellation from its second poll after arm is
// called. The matcher polls once at its first checkpoint and once at
// completion, so the cancellation lands between them, as one arriving
// mid-walk would.
type lateCancelContext struct {
	context.Context
	armed bool
	polls int
}

func (c *lateCancelContext) arm() { c.armed = true }

func (c *lateCancelContext) Err() error {
	if !c.armed {
		return c.Context.Err()
	}
	c.polls++
	if c.polls > 1 {
		return context.Canceled
	}
	return nil
}

// armingIndex arms the request context once the index has been read, so the
// matcher folds the materialized entries under a context about to be
// cancelled.
type armingIndex struct {
	kvblock.Index
	ctx *lateCancelContext
}

func (a armingIndex) Lookup(ctx context.Context, keys []kvblock.BlockHash, podFilter sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
	result, err := a.Index.Lookup(ctx, keys, podFilter)
	a.ctx.arm()
	return result, err
}

// A request cancelled between the matcher's checkpoints, after its keys were
// read, reports the cancellation rather than a match.
func TestMatchBlockKeysCancelledBetweenCheckpoints(t *testing.T) {
	ctx := &lateCancelContext{Context: logging.NewTestLoggerIntoContext(t.Context())}
	idx, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 12, PodCacheSize: 256})
	require.NoError(t, err)
	populateIndex(t, idx, map[kvblock.BlockHash][]kvblock.PodEntry{
		10: {{PodIdentifier: "pod-a", DeviceTier: "gpu"}},
		20: {{PodIdentifier: "pod-a", DeviceTier: "gpu"}},
	})
	indexer := kvcache.NewIndexerForTest(&mockTokenProcessor{}, armingIndex{Index: idx, ctx: ctx}, kvcache.DefaultKVCacheBackendConfig())

	_, err = indexer.MatchBlockKeys(ctx, []kvblock.BlockHash{10, 20}, nil)
	require.ErrorIs(t, err, context.Canceled)
}

// Pod churn (index ordinals assigned to pods since cleared) must change
// neither the result nor its size.
func TestMatchBlockKeysAfterPodChurn(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	indexer, idx := newMatcher(t, kvcache.DefaultKVCacheBackendConfig())

	keys := []kvblock.BlockHash{10, 20}
	require.NoError(t, idx.Add(ctx, nil, keys, []kvblock.PodEntry{{PodIdentifier: "pod-live", DeviceTier: "gpu"}}))

	churnKey := []kvblock.BlockHash{999}
	for i := 0; i < 2000; i++ {
		pod := fmt.Sprintf("10.9.%d.%d:8000", i/256, i%256)
		require.NoError(t, idx.Add(ctx, nil, churnKey, []kvblock.PodEntry{{PodIdentifier: pod, DeviceTier: "gpu"}}))
		require.NoError(t, idx.Clear(ctx, pod))
	}

	got, err := indexer.MatchBlockKeys(ctx, keys, nil)
	require.NoError(t, err)
	assertPodMatches(t, map[string]kvcache.PodMatch{
		"pod-live": {WeightedScore: 2.0, MatchedBlocks: 2, BlocksByTier: map[string]int{"gpu": 2}},
	}, got)
}

// Tier count is unbounded: every tier keeps its own chain and its configured
// weight, however many the index has seen.
func TestMatchBlockKeysManyTiers(t *testing.T) {
	const numTiers = 65
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	indexer, idx := newMatcher(t, []*kvcache.KVCacheBackendConfig{{Name: "tier-64", Weight: 5.0}})

	entries := make([]kvblock.PodEntry, numTiers)
	wantByTier := make(map[string]int, numTiers)
	for i := range entries {
		tier := fmt.Sprintf("tier-%d", i)
		entries[i] = kvblock.PodEntry{PodIdentifier: "pod-a", DeviceTier: tier}
		wantByTier[tier] = 2
	}
	require.NoError(t, idx.Add(ctx, nil, []kvblock.BlockHash{1, 2}, entries))

	got, err := indexer.MatchBlockKeys(ctx, []kvblock.BlockHash{1, 2, 3}, nil)
	require.NoError(t, err)
	assertPodMatches(t, map[string]kvcache.PodMatch{
		"pod-a": {WeightedScore: 10.0, MatchedBlocks: 2, BlocksByTier: wantByTier},
	}, got)
}

// The matcher reproduces the three algorithms it replaces (the longest-prefix
// scorer and the producer's contiguous block counters, kept below as
// oracles) on random fixtures: filters, gaps, duplicate pod entries at a
// key, unconfigured tiers, and speculative entries.
func TestMatchBlockKeysMatchesLegacyAlgorithms(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	rng := rand.New(rand.NewSource(1))
	tiers := []string{"gpu", "cpu", "disk", kvcache.SpeculativeTier}
	backends := []*kvcache.KVCacheBackendConfig{{Name: "gpu", Weight: 1.0}, {Name: "cpu", Weight: 0.8}}
	weights := map[string]float64{"gpu": 1.0, "cpu": 0.8}

	for iter := 0; iter < 500; iter++ {
		numKeys, numPods := 1+rng.Intn(12), 1+rng.Intn(5)
		keys := make([]kvblock.BlockHash, numKeys)
		for i := range keys {
			keys[i] = kvblock.BlockHash(i + 1)
		}
		fixture := make(map[kvblock.BlockHash][]kvblock.PodEntry, numKeys)
		for _, key := range keys {
			for p := 0; p < numPods; p++ {
				if rng.Float64() >= 0.7 {
					continue
				}
				for n := 1 + rng.Intn(2); n > 0; n-- {
					entry := kvblock.PodEntry{PodIdentifier: fmt.Sprintf("pod-%d", p), DeviceTier: tiers[rng.Intn(len(tiers))]}
					if rng.Float64() < 0.15 {
						entry.Speculative = true
						entry.DeviceTier = ""
					}
					fixture[key] = append(fixture[key], entry)
				}
			}
		}
		filter := sets.New[string]()
		if rng.Intn(2) == 0 {
			for p := 0; p < numPods; p++ {
				if rng.Intn(2) == 0 {
					filter.Insert(fmt.Sprintf("pod-%d", p))
				}
			}
		}

		indexer, idx := newMatcher(t, backends)
		populateIndex(t, idx, fixture)

		got, err := indexer.MatchBlockKeys(ctx, keys, filter)
		require.NoError(t, err)
		keyToPods, err := idx.Lookup(ctx, keys, filter)
		require.NoError(t, err)

		want := map[string]kvcache.PodMatch{}
		for pod, score := range legacyLongestPrefixScore(keys, keyToPods, weights) {
			want[pod] = kvcache.PodMatch{
				WeightedScore: score,
				MatchedBlocks: legacyMatchedBlockCount(keys, keyToPods, pod),
				BlocksByTier:  legacyMatchedBlockCountByTier(keys, keyToPods, pod),
			}
		}
		assertPodMatches(t, want, got)
	}
}

// legacyLongestPrefixScore is the map-based longest-prefix scorer the matcher
// replaced.
func legacyLongestPrefixScore(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	weights map[string]float64,
) map[string]float64 {
	maxWeights := func(entries []kvblock.PodEntry) map[string]float64 {
		out := map[string]float64{}
		for _, e := range entries {
			w := 1.0
			if cw, ok := weights[e.DeviceTier]; ok {
				w = cw
			}
			if cur, ok := out[e.PodIdentifier]; !ok || w > cur {
				out[e.PodIdentifier] = w
			}
		}
		return out
	}
	scores := map[string]float64{}
	if len(keys) == 0 {
		return scores
	}
	active := map[string]struct{}{}
	for pod, w := range maxWeights(keyToPods[keys[0]]) {
		active[pod] = struct{}{}
		scores[pod] = w
	}
	for i := 1; i < len(keys) && len(active) > 0; i++ {
		cur := maxWeights(keyToPods[keys[i]])
		for pod := range active {
			if w, ok := cur[pod]; ok {
				scores[pod] += w
			} else {
				delete(active, pod)
			}
		}
	}
	return scores
}

// legacyMatchedBlockCount is the producer's contiguous block counter the
// matcher replaced.
func legacyMatchedBlockCount(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) int {
	count := 0
	for _, key := range keys {
		if !slices.ContainsFunc(keyToPods[key], func(e kvblock.PodEntry) bool { return e.PodIdentifier == podID }) {
			break
		}
		count++
	}
	return count
}

// legacyMatchedBlockCountByTier is the producer's per-tier contiguous block
// counter the matcher replaced.
func legacyMatchedBlockCountByTier(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) map[string]int {
	counts := map[string]int{}
	var alive sets.Set[string]
	for _, key := range keys {
		tiersAtKey := sets.New[string]()
		for _, e := range keyToPods[key] {
			if e.PodIdentifier != podID {
				continue
			}
			if e.Speculative {
				tiersAtKey.Insert(kvcache.SpeculativeTier)
			} else {
				tiersAtKey.Insert(e.DeviceTier)
			}
		}
		if alive == nil {
			alive = tiersAtKey
		} else {
			alive = alive.Intersection(tiersAtKey)
		}
		if alive.Len() == 0 {
			break
		}
		for tier := range alive {
			counts[tier]++
		}
	}
	return counts
}
