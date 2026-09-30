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
	"k8s.io/apimachinery/pkg/util/sets"

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

// TestRedisLookup_OneBadKeyDoesNotDiscardOtherMatches verifies that a single
// key with a Redis type conflict (for example a stale key of the wrong type
// sharing the request-key namespace) does not fail keys elsewhere in the
// same Lookup call.
func TestRedisLookup_OneBadKeyDoesNotDiscardOtherMatches(t *testing.T) {
	server, err := miniredis.Run()
	require.NoError(t, err)
	defer server.Close()

	index, err := NewRedisIndex(&RedisIndexConfig{Address: server.Addr()})
	require.NoError(t, err)

	goodKey := BlockHash(111)
	err = index.Add(t.Context(), nil, []BlockHash{goodKey},
		[]PodEntry{{PodIdentifier: "pod-a", DeviceTier: "gpu"}})
	require.NoError(t, err)

	// Request keys carry no namespace prefix, so a key of the wrong Redis
	// type can collide with one.
	badKey := BlockHash(222)
	require.NoError(t, server.Set(badKey.String(), "not-a-hash"))

	result, err := index.Lookup(t.Context(), []BlockHash{goodKey, badKey}, sets.Set[string]{})
	require.NoError(t, err)
	require.Contains(t, result, goodKey)
	require.NotContains(t, result, badKey)
}

// TestRedisLookup_ConnectionFailureReturnsError verifies that a Lookup call
// still reports an error when the pipeline round-trip fails outright, rather
// than silently returning an empty result.
func TestRedisLookup_ConnectionFailureReturnsError(t *testing.T) {
	server, err := miniredis.Run()
	require.NoError(t, err)

	index, err := NewRedisIndex(&RedisIndexConfig{Address: server.Addr()})
	require.NoError(t, err)

	server.Close()

	result, err := index.Lookup(t.Context(), []BlockHash{111, 222}, sets.Set[string]{})
	require.Error(t, err)
	require.Empty(t, result)
}
