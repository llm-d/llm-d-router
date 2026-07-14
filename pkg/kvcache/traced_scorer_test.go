/*
Copyright 2025 The llm-d Authors.

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
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/require"
)

func TestNewTracedScorer(t *testing.T) {
	// Create a base scorer
	config := kvcache.DefaultKVBlockScorerConfig()
	baseScorer, err := kvcache.NewKVBlockScorer(config)
	require.NoError(t, err)

	// Wrap it with tracing
	tracedScorer := kvcache.NewTracedScorer(baseScorer)
	require.NotNil(t, tracedScorer)
}

func TestTracedScorerBehavior(t *testing.T) {
	// Create a base scorer
	config := kvcache.DefaultKVBlockScorerConfig()
	baseScorer, err := kvcache.NewKVBlockScorer(config)
	require.NoError(t, err)

	// Wrap it with tracing
	tracedScorer := kvcache.NewTracedScorer(baseScorer)

	// Test Strategy method
	strategy := tracedScorer.Strategy()
	require.Equal(t, kvcache.LongestPrefixMatch, strategy)

	// Test Score method with sample data
	keys := []kvblock.BlockHash{
		kvblock.BlockHash(1),
		kvblock.BlockHash(2),
		kvblock.BlockHash(3),
	}

	keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{
		kvblock.BlockHash(1): {
			{PodIdentifier: "pod1", DeviceTier: "gpu"},
			{PodIdentifier: "pod2", DeviceTier: "cpu"},
		},
		kvblock.BlockHash(2): {
			{PodIdentifier: "pod1", DeviceTier: "gpu"},
		},
		kvblock.BlockHash(3): {
			{PodIdentifier: "pod1", DeviceTier: "gpu"},
		},
	}

	scores, err := tracedScorer.Score(context.Background(), keys, keyToPods)
	require.NoError(t, err)
	require.NotNil(t, scores)

	// pod1 should have highest score (appears in all 3 keys)
	require.Greater(t, scores["pod1"], scores["pod2"])
}

func TestTracedScorerWithEmptyData(t *testing.T) {
	config := kvcache.DefaultKVBlockScorerConfig()
	baseScorer, err := kvcache.NewKVBlockScorer(config)
	require.NoError(t, err)

	tracedScorer := kvcache.NewTracedScorer(baseScorer)

	// Test with empty keys
	scores, err := tracedScorer.Score(context.Background(), []kvblock.BlockHash{}, map[kvblock.BlockHash][]kvblock.PodEntry{})
	require.NoError(t, err)
	require.Empty(t, scores)
}

func TestTracedScorerScoreDistribution(t *testing.T) {
	config := kvcache.DefaultKVBlockScorerConfig()
	baseScorer, err := kvcache.NewKVBlockScorer(config)
	require.NoError(t, err)

	tracedScorer := kvcache.NewTracedScorer(baseScorer)

	// Test with multiple pods and varying scores
	keys := []kvblock.BlockHash{
		kvblock.BlockHash(1),
		kvblock.BlockHash(2),
	}

	keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{
		kvblock.BlockHash(1): {
			{PodIdentifier: "pod1", DeviceTier: "gpu"},
			{PodIdentifier: "pod2", DeviceTier: "gpu"},
			{PodIdentifier: "pod3", DeviceTier: "cpu"},
		},
		kvblock.BlockHash(2): {
			{PodIdentifier: "pod1", DeviceTier: "gpu"},
		},
	}

	scores, err := tracedScorer.Score(context.Background(), keys, keyToPods)
	require.NoError(t, err)
	require.Len(t, scores, 3)

	// Verify pod1 has highest score
	require.Greater(t, scores["pod1"], scores["pod2"])
	require.Greater(t, scores["pod1"], scores["pod3"])
}
