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
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	. "github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
)

// testCommonIndexBehavior runs a comprehensive test suite for any Index implementation.
// indexFactory should return a fresh index instance for each test to ensure test isolation.
func testCommonIndexBehavior(t *testing.T, indexFactory func(t *testing.T) Index) {
	t.Helper()
	ctx := context.Background()

	t.Run("BasicAddAndLookup", func(t *testing.T) {
		index := indexFactory(t)
		testBasicAddAndLookup(t, ctx, index)
	})

	t.Run("DuplicatePodHandling", func(t *testing.T) {
		index := indexFactory(t)
		testDuplicatePodHandling(t, ctx, index)
	})

	t.Run("FilteredLookup", func(t *testing.T) {
		index := indexFactory(t)
		testFilteredLookup(t, ctx, index)
	})

	t.Run("EvictBasic", func(t *testing.T) {
		index := indexFactory(t)
		testEvictBasic(t, ctx, index)
	})

	t.Run("ConcurrentOperations", func(t *testing.T) {
		t.Skip("TODO: Re-enable this test after fixing https://github.com/llm-d/llm-d-kv-cache-manager/issues/104")
		index := indexFactory(t)
		testConcurrentOperations(t, ctx, index)
	})
}

// testBasicAddAndLookup tests basic Add and Lookup functionality.
func testBasicAddAndLookup(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	key := Key{ModelName: "test-model", ChunkHash: 12345}
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
	}

	// Add entries
	err := index.Add(ctx, []Key{key}, entries)
	require.NoError(t, err)

	// Lookup all entries
	podsPerKey, err := index.Lookup(ctx, []Key{key}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, key)
	assert.ElementsMatch(t, podsPerKey[key], []string{"pod1", "pod2"})
}

// testDuplicatePodHandling tests behavior when adding duplicate pod identifiers.
// The current implementation allows duplicate pod identifiers with different device tiers,
// treating them as separate entries in the index.
func testDuplicatePodHandling(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	key := Key{ModelName: "test-model", ChunkHash: 54321}

	// First batch of entries
	entries1 := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
	}

	err := index.Add(ctx, []Key{key}, entries1)
	require.NoError(t, err)

	// Second batch with one duplicate pod but different tier
	entries2 := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"}, // Same pod, same tier
		{PodIdentifier: "pod2", DeviceTier: "cpu"}, // Same pod, different tier
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}

	err = index.Add(ctx, []Key{key}, entries2)
	require.NoError(t, err)

	// Lookup and verify the behavior with duplicates
	// Note: The index currently preserves duplicate pod identifiers as separate entries
	podsPerKey, err := index.Lookup(ctx, []Key{key}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, key)

	// Should contain all pod entries, including duplicates with different tiers
	// Expected: pod1(gpu), pod2(gpu), pod2(cpu), pod3(gpu)
	expected := []string{"pod1", "pod2", "pod2", "pod3"}
	assert.ElementsMatch(t, podsPerKey[key], expected)
}

// testFilteredLookup tests lookup with pod identifier filtering.
// This verifies that the index can filter results based on specific pod identifiers.
func testFilteredLookup(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	key := Key{ModelName: "test-model", ChunkHash: 98765}
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}

	err := index.Add(ctx, []Key{key}, entries)
	require.NoError(t, err)

	// Lookup with filter - should only return pod1
	filterSet := sets.New("pod1")
	podsPerKey, err := index.Lookup(ctx, []Key{key}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, key)
	assert.Equal(t, []string{"pod1"}, podsPerKey[key])

	// Lookup with multiple filters
	filterSet = sets.New("pod1", "pod3")
	podsPerKey, err = index.Lookup(ctx, []Key{key}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.ElementsMatch(t, podsPerKey[key], []string{"pod1", "pod3"})

	// Lookup with non-existent pod filter should return empty result
	filterSet = sets.New("pod999")
	podsPerKey, err = index.Lookup(ctx, []Key{key}, filterSet)
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 0) // No matching pods found
}

// testEvictBasic tests basic eviction functionality.
// Verifies that specific pod entries can be removed from the index.
func testEvictBasic(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	key := Key{ModelName: "test-model", ChunkHash: 11111}
	entries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod2", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "gpu"},
	}

	// Add entries
	err := index.Add(ctx, []Key{key}, entries)
	require.NoError(t, err)

	// Evict specific pod entries (note: eviction is based on pod identifier only)
	evictEntries := []PodEntry{
		{PodIdentifier: "pod1", DeviceTier: "gpu"},
		{PodIdentifier: "pod3", DeviceTier: "cpu"}, // Device tier may differ from stored entry
	}

	err = index.Evict(ctx, key, evictEntries)
	require.NoError(t, err)

	// Verify that pod1 was evicted but pod2 and pod3 remain
	// Note: pod3 remains because eviction only matched pod identifier, not device tier
	podsPerKey, err := index.Lookup(ctx, []Key{key}, sets.Set[string]{})
	require.NoError(t, err)
	assert.Len(t, podsPerKey, 1)
	assert.Contains(t, podsPerKey, key)
	assert.ElementsMatch(t, []string{"pod2", "pod3"}, podsPerKey[key])
}

// testConcurrentOperations tests thread safety with concurrent operations.
func testConcurrentOperations(t *testing.T, ctx context.Context, index Index) {
	t.Helper()
	key := Key{ModelName: "test-model", ChunkHash: 1000}

	var wg sync.WaitGroup
	errChan := make(chan error, 30)

	// Run 100 goroutines doing concurrent operations
	for goroutineID := 0; goroutineID < 100; goroutineID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for operationIndex := 0; operationIndex < 10; operationIndex++ {
				switch operationIndex % 3 {
				case 0: // Add
					entries := []PodEntry{{PodIdentifier: fmt.Sprintf("pod-%d-%d", id, operationIndex), DeviceTier: "gpu"}}
					if err := index.Add(ctx, []Key{key}, entries); err != nil {
						errChan <- err
					}
				case 1: // Lookup
					podsPerKey, err := index.Lookup(ctx, []Key{key}, sets.Set[string]{})
					if err != nil {
						errChan <- err
					}
					assert.Contains(t, podsPerKey, key)
					assert.Contains(t, podsPerKey[key], fmt.Sprintf("pod-%d-%d", id, operationIndex-1))
				case 2: // Evict
					entries := []PodEntry{{PodIdentifier: fmt.Sprintf("pod-%d-%d", id, operationIndex-2), DeviceTier: "gpu"}}
					if err := index.Evict(ctx, key, entries); err != nil {
						errChan <- err
					}
					podsPerKey, err := index.Lookup(ctx, []Key{key}, sets.Set[string]{})
					if err != nil {
						errChan <- err
					}
					if _, ok := podsPerKey[key]; ok {
						assert.NotContains(t, podsPerKey[key], fmt.Sprintf("pod-%d-%d", id, operationIndex-2))
					}
				}
			}
		}(goroutineID)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		require.NoError(t, err)
	}

	// Verify index still works
	_, err := index.Lookup(ctx, []Key{key}, sets.Set[string]{})
	require.NoError(t, err)
}
