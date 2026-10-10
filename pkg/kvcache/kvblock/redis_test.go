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

// TestRedisIndexEvictLookupFailure verifies that a lookup failure (e.g. lost
// connectivity) is propagated instead of being reported as a successful no-op,
// so callers and metrics do not treat a failed eviction as completed.
func TestRedisIndexEvictLookupFailure(t *testing.T) {
	server, err := miniredis.Run()
	require.NoError(t, err)

	index, err := NewRedisIndex(&RedisIndexConfig{Address: server.Addr()})
	require.NoError(t, err)

	server.Close()

	require.Error(t, index.Evict(t.Context(), EngineKey, []BlockHash{0xC1EA00F1}, []PodEntry{{}}))
}

// TestRedisIndexEvictMissingEngineKeyIsNoOp pins the intentional no-op when
// the engine key has no request-key mapping: eviction of an absent key is not
// an error and must not be counted as a failure.
func TestRedisIndexEvictMissingEngineKeyIsNoOp(t *testing.T) {
	index := createRedisIndexForTesting(t)

	require.NoError(t, index.Evict(t.Context(), EngineKey, []BlockHash{0xC1EA00F2}, []PodEntry{{}}))
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

// TestRedisLookup_SingleBadKeyIsNotAPipelineFailure verifies that a Lookup
// call for exactly one key -- the shape every single-block prompt uses --
// does not misclassify that key's own WRONGTYPE conflict as a pipeline
// round-trip failure. len(results) == 1 made every result error, which is
// exactly the shape a genuine connection failure also produces.
func TestRedisLookup_SingleBadKeyIsNotAPipelineFailure(t *testing.T) {
	server, err := miniredis.Run()
	require.NoError(t, err)
	defer server.Close()

	index, err := NewRedisIndex(&RedisIndexConfig{Address: server.Addr()})
	require.NoError(t, err)

	badKey := BlockHash(222)
	require.NoError(t, server.Set(badKey.String(), "not-a-hash"))

	result, err := index.Lookup(t.Context(), []BlockHash{badKey}, sets.Set[string]{})
	require.NoError(t, err)
	require.Empty(t, result)
}

// TestRedisLookup_AllKeysBadIsNotAPipelineFailure verifies that a Lookup
// call where every key hits a WRONGTYPE conflict reports the same outcome
// as a normal zero-block cache miss (no error, empty result), not a
// pipeline failure -- the same prefix-chain shape as
// TestRedisLookup_SingleBadKeyIsNotAPipelineFailure, just with more than
// one key all erroring the same way.
func TestRedisLookup_AllKeysBadIsNotAPipelineFailure(t *testing.T) {
	server, err := miniredis.Run()
	require.NoError(t, err)
	defer server.Close()

	index, err := NewRedisIndex(&RedisIndexConfig{Address: server.Addr()})
	require.NoError(t, err)

	badKey1, badKey2 := BlockHash(222), BlockHash(333)
	require.NoError(t, server.Set(badKey1.String(), "not-a-hash"))
	require.NoError(t, server.Set(badKey2.String(), "not-a-hash"))

	result, err := index.Lookup(t.Context(), []BlockHash{badKey1, badKey2}, sets.Set[string]{})
	require.NoError(t, err)
	require.Empty(t, result)
}
