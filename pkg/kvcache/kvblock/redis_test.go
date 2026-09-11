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

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/require"

	. "github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// createRedisIndexForTesting creates a new RedisIndex with a mock Redis server for testing.
func createRedisIndexForTesting(t *testing.T) Index {
	t.Helper()
	server, err := miniredis.Run()
	require.NoError(t, err)

	// Store server reference for cleanup
	t.Cleanup(func() {
		server.Close()
	})

	redisConfig := &RedisIndexConfig{
		Address: server.Addr(),
	}
	index, err := NewRedisIndex(redisConfig)
	require.NoError(t, err)
	return index
}

// TestRedisIndexBehavior tests the Redis index implementation using common test behaviors.
func TestRedisIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createRedisIndexForTesting)
}

// TestRedisBatchEvictPreservesMappingAndPrunesNext verifies that the batched
// engine-key prune script advances to the next group after a group whose
// request key is retained by another pod's entry.
func TestRedisBatchEvictPreservesMappingAndPrunesNext(t *testing.T) {
	ctx := t.Context()
	index := createRedisIndexForTesting(t)
	pod := PodEntry{PodIdentifier: "target", DeviceTier: "gpu"}
	keeper := PodEntry{PodIdentifier: "keeper", DeviceTier: "gpu"}

	require.NoError(t, index.Add(ctx,
		[]BlockHash{11}, []BlockHash{21, 22}, []PodEntry{pod}))
	require.NoError(t, index.Add(ctx,
		[]BlockHash{12}, []BlockHash{23}, []PodEntry{pod}))
	require.NoError(t, index.Add(ctx,
		nil, []BlockHash{21}, []PodEntry{keeper}))

	require.NoError(t, index.Evict(ctx,
		EngineKey, []BlockHash{11, 12}, []PodEntry{pod}))

	result, err := index.Lookup(ctx, []BlockHash{21}, nil)
	require.NoError(t, err)
	require.Equal(t, []PodEntry{keeper}, result[21])

	for _, key := range []BlockHash{22, 23} {
		result, err := index.Lookup(ctx, []BlockHash{key}, nil)
		require.NoError(t, err)
		require.Empty(t, result[key])
	}

	_, err = index.GetRequestKey(ctx, 11)
	require.NoError(t, err)
	_, err = index.GetRequestKey(ctx, 12)
	require.Error(t, err)
}
