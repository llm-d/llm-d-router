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

package kvblock_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	. "github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
)

// createInMemoryIndexForTesting creates a new InMemoryIndex for testing.
func createInMemoryIndexForTesting(t *testing.T) Index {
	t.Helper()
	cfg := DefaultInMemoryIndexConfig()
	cfg.PodCacheSize = 100 // for testConcurrentOperations
	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)
	return index
}

func TestInMemoryIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createInMemoryIndexForTesting)
}

func TestInMemoryIndexSize(t *testing.T) {
	// Test with small size to verify eviction
	cfg := &InMemoryIndexConfig{
		Size:         2, // Only 2 keys max
		PodCacheSize: 1, // Pod cache size doesn't matter for this test
	}

	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)

	ctx := t.Context()

	// Add first key
	key1 := Key{ModelName: "test-model", ChunkHash: 111}
	err = index.Add(ctx, []Key{key1}, []PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}})
	require.NoError(t, err)

	// Add second key
	key2 := Key{ModelName: "test-model", ChunkHash: 222}
	err = index.Add(ctx, []Key{key2}, []PodEntry{{PodIdentifier: "pod2", DeviceTier: "gpu"}})
	require.NoError(t, err)

	// Add third key - should evict the first one due to LRU
	key3 := Key{ModelName: "test-model", ChunkHash: 333}
	err = index.Add(ctx, []Key{key3}, []PodEntry{{PodIdentifier: "pod3", DeviceTier: "cpu"}})
	require.NoError(t, err)

	// Lookup should only return the last two keys
	podsPerKey, err := index.Lookup(ctx, []Key{key1, key2, key3}, nil)
	require.NoError(t, err)

	assert.Len(t, podsPerKey, 2) // Only key2 and key3 should be present
	assert.Len(t, podsPerKey[key2], 1)
	assert.Len(t, podsPerKey[key3], 1)
	assert.Contains(t, podsPerKey[key2], "pod2")
	assert.Contains(t, podsPerKey[key3], "pod3")
}

func TestInMemoryIndexPodCacheSize(t *testing.T) {
	// Test with small limits to verify enforcement
	cfg := &InMemoryIndexConfig{
		Size:         1, // Only 1 key max
		PodCacheSize: 2, // Only 2 pods per key
	}

	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)

	// Test PodCacheSize limit: add more pods than the limit for one key
	key := Key{ModelName: "test-model", ChunkHash: 111}
	pods := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "cpu"}, // This should evict pod1 due to LRU
	}

	ctx := t.Context()

	err = index.Add(ctx, []Key{key}, pods)
	require.NoError(t, err)

	// Lookup should only return 2 pods (pod2 and pod3), pod1 should be evicted
	podsPerKey, err := index.Lookup(ctx, []Key{key}, nil)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Len(t, podsPerKey[key], 2, "Should only have 2 pods due to PodCacheSize limit")
	assert.Contains(t, podsPerKey[key], "pod2")
	assert.Contains(t, podsPerKey[key], "pod3")
}
