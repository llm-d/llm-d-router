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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	. "github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// createRedisIndexForTesting creates a new RedisIndex with a mock Redis server for testing.
func createCostAwareIndexForTesting(t *testing.T) Index {
	t.Helper()
	config := DefaultCostAwareMemoryIndexConfig()
	index, err := NewCostAwareMemoryIndex(config)
	require.NoError(t, err)
	return index
}

// TestRedisIndexBehavior tests the Redis index implementation using common test behaviors.
func TestCostAwareIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createCostAwareIndexForTesting)
}

func TestCostAwareIndexSize(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())

	// first key
	engineKey1 := BlockHash(32490241)
	requestKey1 := BlockHash(18986637)
	entry1 := PodEntry{PodIdentifier: "pod1", DeviceTier: "gpu"}

	costPodCache := &CostPodCache{}
	costPodCache.Add(entry1)
	cost := costPodCache.CalculateByteSize(requestKey1.String())

	// Test with small size to verify eviction
	cfg := DefaultCostAwareMemoryIndexConfig()
	cfg.Size = fmt.Sprintf("%d", cost+cost/2) // more than 1 key, less than 2 keys

	index, err := NewCostAwareMemoryIndex(cfg)
	require.NoError(t, err)

	err = index.Add(ctx, []BlockHash{engineKey1}, []BlockHash{requestKey1}, []PodEntry{entry1})
	assert.NoError(t, err)

	// Add second key
	engineKey2 := BlockHash(48712468)
	requestKey2 := BlockHash(87654321)
	err = index.Add(ctx, []BlockHash{engineKey2}, []BlockHash{requestKey2}, []PodEntry{{PodIdentifier: "pod2", DeviceTier: "gpu"}})
	require.NoError(t, err)

	// Add third key - should evict the first one due to LRU
	engineKey3 := BlockHash(96187092)
	requestKey3 := BlockHash(56789012)
	err = index.Add(ctx, []BlockHash{engineKey3}, []BlockHash{requestKey3}, []PodEntry{{PodIdentifier: "pod3", DeviceTier: "cpu"}})
	require.NoError(t, err)

	// Lookup should only return the last two keys
	podsPerKey, err := index.Lookup(ctx, []BlockHash{requestKey1, requestKey2, requestKey3}, nil)
	require.NoError(t, err)

	assert.Len(t, podsPerKey, 1) // Only requestKey3 should be present
	assert.Len(t, podsPerKey[requestKey3], 1)

	assert.Contains(t, podsPerKey[requestKey3], PodEntry{PodIdentifier: "pod3", DeviceTier: "cpu"})
}

func TestSizeHumanize(t *testing.T) {
	tests := []struct {
		size     string
		expected int64
	}{
		{"42 MB", 42 * 1000 * 1000},
		{"42M", 42 * 1000 * 1000},
		{"42Mi", 42 * 1024 * 1024},
		{"42", 42},
	}

	for _, tt := range tests {
		t.Run(tt.size, func(t *testing.T) {
			config := &CostAwareMemoryIndexConfig{Size: tt.size}
			index, err := NewCostAwareMemoryIndex(config)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, index.MaxCost())
		})
	}
}
