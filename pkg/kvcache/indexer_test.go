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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// mockTokenProcessor implements kvblock.TokenProcessor for testing.
// It records the tokens it receives so tests can assert on them.
type mockTokenProcessor struct {
	blockKeys      []kvblock.BlockHash
	receivedTokens []uint32
}

func (m *mockTokenProcessor) TokensToKVBlockKeys(
	_ kvblock.BlockHash, tokens []uint32, _ string, _ []*kvblock.BlockExtraFeatures,
) ([]kvblock.BlockHash, error) {
	m.receivedTokens = tokens
	return m.blockKeys, nil
}

func (m *mockTokenProcessor) BlockSize() int {
	return 16
}

const (
	testModel = "test-model"
	testPodA  = "pod-a"
	testPodB  = "pod-b"
)

func u64ToBlockKeys(keys []uint64) []kvblock.BlockHash {
	out := make([]kvblock.BlockHash, len(keys))
	for i, k := range keys {
		out[i] = kvblock.BlockHash(k)
	}
	return out
}

// newTestIndexer creates an Indexer backed by an in-memory index and a
// LongestPrefixScorer using the project's default backend weights.
func newTestIndexer(t *testing.T, tp kvblock.TokenProcessor) *kvcache.Indexer {
	t.Helper()

	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	scorer, err := kvcache.NewKVBlockScorer(kvcache.DefaultKVBlockScorerConfig())
	require.NoError(t, err)

	return kvcache.NewIndexerForTest(tp, idx, scorer)
}

// populateIndex inserts block-key -> pod entries into the index.
func populateIndex(t *testing.T, idx kvblock.Index, entries map[kvblock.BlockHash][]kvblock.PodEntry) {
	t.Helper()
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	for key, pods := range entries {
		err := idx.Add(ctx, []kvblock.BlockHash{key}, []kvblock.BlockHash{key}, pods)
		require.NoError(t, err)
	}
}

// scoringTestCase defines a scenario exercised through ScoreTokens.
type scoringTestCase struct {
	name           string
	blockKeys      []uint64
	tokens         []uint32
	indexEntries   map[kvblock.BlockHash][]kvblock.PodEntry
	podIdentifiers []string
	wantScores     map[string]float64 // expected pod -> score (checked with InDelta)
	wantNil        bool               // if true, expect nil scores (not just empty)
}

var scoringTests = []scoringTestCase{
	{
		name:      "empty tokens",
		blockKeys: nil,
		tokens:    nil,
		wantNil:   true,
	},
	{
		name:       "no matching pods",
		blockKeys:  []uint64{100, 200, 300},
		tokens:     []uint32{1, 2, 3},
		wantScores: map[string]float64{},
	},
	{
		name:      "single pod full match",
		blockKeys: []uint64{10, 20, 30},
		tokens:    []uint32{1, 2, 3},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
			20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
			30: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		},
		wantScores: map[string]float64{testPodA: 3.0},
	},
	{
		name:      "multiple pods",
		blockKeys: []uint64{10, 20, 30},
		tokens:    []uint32{1, 2, 3},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
			20: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
			30: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
			},
		},
		wantScores: map[string]float64{testPodA: 3.0, testPodB: 2.0},
	},
	{
		name:      "mixed device tiers",
		blockKeys: []uint64{10, 20},
		tokens:    []uint32{1, 2},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
			20: {{PodIdentifier: testPodA, DeviceTier: "cpu"}},
		},
		wantScores: map[string]float64{testPodA: 1.8}, // gpu(1.0) + cpu(0.8)
	},
	{
		name:      "pod identifier filter",
		blockKeys: []uint64{10, 20},
		tokens:    []uint32{1, 2},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
			20: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
		},
		podIdentifiers: []string{testPodA},
		wantScores:     map[string]float64{testPodA: 2.0},
	},
	{
		name:      "prefix break",
		blockKeys: []uint64{10, 20, 30},
		tokens:    []uint32{1, 2, 3},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
			20: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				// testPodB missing => prefix breaks for podB
			},
			30: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
		},
		wantScores: map[string]float64{testPodA: 3.0, testPodB: 1.0},
	},
	{
		name:      "empty pod identifiers returns all",
		blockKeys: []uint64{10},
		tokens:    []uint32{1},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {
				{PodIdentifier: testPodA, DeviceTier: "gpu"},
				{PodIdentifier: testPodB, DeviceTier: "gpu"},
			},
		},
		podIdentifiers: []string{},
		wantScores:     map[string]float64{testPodA: 1.0, testPodB: 1.0},
	},
	{
		name:      "deterministic",
		blockKeys: []uint64{10, 20},
		tokens:    []uint32{42, 43},
		indexEntries: map[kvblock.BlockHash][]kvblock.PodEntry{
			10: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
			20: {{PodIdentifier: testPodA, DeviceTier: "gpu"}},
		},
		wantScores: map[string]float64{testPodA: 2.0},
	},
}

// assertScores verifies that the returned scores match expectations.
func assertScores(t *testing.T, tt *scoringTestCase, scores map[string]float64, err error) {
	t.Helper()
	require.NoError(t, err)

	if tt.wantNil {
		assert.Nil(t, scores, "expected nil scores")
		return
	}

	require.Len(t, scores, len(tt.wantScores), "unexpected number of scored pods")
	for pod, want := range tt.wantScores {
		require.Contains(t, scores, pod, "missing pod %q in scores", pod)
		assert.InDelta(t, want, scores[pod], 0.0001, "pod %q score mismatch", pod)
	}
}

func TestScoreTokens(t *testing.T) {
	for _, tt := range scoringTests {
		t.Run(tt.name, func(t *testing.T) {
			tp := &mockTokenProcessor{blockKeys: u64ToBlockKeys(tt.blockKeys)}
			indexer := newTestIndexer(t, tp)

			ctx := logging.NewTestLoggerIntoContext(context.Background())
			if tt.indexEntries != nil {
				populateIndex(t, indexer.KVBlockIndex(), tt.indexEntries)
			}

			scores, err := indexer.ScoreTokens(ctx, tt.tokens, testModel, tt.podIdentifiers, nil)
			assertScores(t, &tt, scores, err)
		})
	}
}
