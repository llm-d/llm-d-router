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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// newSampledMatcher builds a walked matcher over an in-memory index whose
// Add and the matcher's query filter apply the same sampling modulus.
func newSampledMatcher(t *testing.T, modulus int) (*kvcache.Indexer, kvblock.Index) {
	t.Helper()
	idx, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{
		Size:          1 << 12,
		PodCacheSize:  256,
		SampleModulus: modulus,
	})
	require.NoError(t, err)
	return kvcache.NewSampledIndexerForTest(&mockTokenProcessor{}, idx, kvcache.DefaultKVCacheBackendConfig(), modulus), idx
}

func TestMatchBlockKeysSampled(t *testing.T) {
	podA, podB, podC := "pod-a", "pod-b", "pod-c"
	// Modulus 4 keeps keys whose low two bits are zero. Of the query keys
	// below the index holds and the match probes 4 (position 0), 8 (position
	// 3), and 12 (position 4).
	keys := []kvblock.BlockHash{4, 5, 6, 8, 12}
	entries := map[kvblock.BlockHash][]kvblock.PodEntry{
		4: {{PodIdentifier: podA, DeviceTier: "gpu"}, {PodIdentifier: podB, DeviceTier: "gpu"}, {PodIdentifier: podC, DeviceTier: "gpu"}},
		// 5 and 6 are unsampled: no entry stored, invisible to the match.
		6:  {{PodIdentifier: podA, DeviceTier: "gpu"}},
		8:  {{PodIdentifier: podA, DeviceTier: "gpu"}, {PodIdentifier: podB, DeviceTier: "gpu"}},
		12: {{PodIdentifier: podB, DeviceTier: "gpu"}},
	}

	ctx := logging.NewTestLoggerIntoContext(t.Context())
	walked, idx := newSampledMatcher(t, 4)
	populateIndex(t, idx, entries)

	want := map[string]kvcache.PodMatch{
		// Holds 4 and 8: matched runs to position 3, so 4 blocks. The
		// unsampled key 6 in between adds nothing.
		podA: {WeightedScore: 4.0, MatchedBlocks: 4, ConfirmedBlocks: 4, BlocksByTier: map[string]int{"gpu": 4}},
		// Holds 4, 8, and 12: matched runs to position 4, so 5 blocks.
		podB: {WeightedScore: 5.0, MatchedBlocks: 5, ConfirmedBlocks: 5, BlocksByTier: map[string]int{"gpu": 5}},
		// Holds only 4.
		podC: {WeightedScore: 1.0, MatchedBlocks: 1, ConfirmedBlocks: 1, BlocksByTier: map[string]int{"gpu": 1}},
	}

	got, err := walked.MatchBlockKeys(ctx, keys, nil)
	require.NoError(t, err)
	assertPodMatches(t, want, got)

	got, err = kvcache.NewSampledIndexerForTest(&mockTokenProcessor{}, nonWalkerIndex{idx},
		kvcache.DefaultKVCacheBackendConfig(), 4).MatchBlockKeys(ctx, keys, nil)
	require.NoError(t, err)
	assertPodMatches(t, want, got)
}

func TestMatchBlockKeysSampledNoSampledKeyInQuery(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	walked, _ := newSampledMatcher(t, 4)

	got, err := walked.MatchBlockKeys(ctx, []kvblock.BlockHash{5, 6, 7}, nil)
	require.NoError(t, err)
	assert.Empty(t, got)
}

// TestMatchBlockKeysSampledFromConfig runs the modulus through
// NewKVCacheIndexer's configuration path, as a deployment sets it.
func TestMatchBlockKeysSampledFromConfig(t *testing.T) {
	config, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)
	config.KVBlockIndexConfig = &kvblock.IndexConfig{
		InMemoryConfig: &kvblock.InMemoryIndexConfig{
			Size:          1 << 12,
			PodCacheSize:  256,
			SampleModulus: 4,
		},
	}
	indexer, err := kvcache.NewKVCacheIndexer(t.Context(), config, &mockTokenProcessor{})
	require.NoError(t, err)

	ctx := logging.NewTestLoggerIntoContext(t.Context())
	populateIndex(t, indexer.KVBlockIndex(), map[kvblock.BlockHash][]kvblock.PodEntry{
		4: {{PodIdentifier: "pod-a", DeviceTier: "gpu"}},
		8: {{PodIdentifier: "pod-a", DeviceTier: "gpu"}},
	})

	got, err := indexer.MatchBlockKeys(ctx, []kvblock.BlockHash{4, 5, 8}, nil)
	require.NoError(t, err)
	assertPodMatches(t, map[string]kvcache.PodMatch{
		"pod-a": {WeightedScore: 3.0, MatchedBlocks: 3, ConfirmedBlocks: 3, BlocksByTier: map[string]int{"gpu": 3}},
	}, got)
}

func TestSampledKey(t *testing.T) {
	assert.True(t, kvblock.SampledKey(8, 4))
	assert.False(t, kvblock.SampledKey(6, 4))
	// Modulus 0 and 1 keep every key.
	assert.True(t, kvblock.SampledKey(6, 0))
	assert.True(t, kvblock.SampledKey(6, 1))
}
