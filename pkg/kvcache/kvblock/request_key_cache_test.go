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

package kvblock

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func requestKeysForShard(cache *requestKeyCache, shardIdx, count int) []BlockHash {
	keys := make([]BlockHash, 0, count)
	for key := BlockHash(1); len(keys) < count; key++ {
		if cache.shardIndex(key) == shardIdx {
			keys = append(keys, key)
		}
	}
	return keys
}

func TestRequestKeyCacheShardSelectionAndCapacity(t *testing.T) {
	tests := []struct {
		name       string
		size       int
		wantShards int
	}{
		{name: "minimum", size: 1, wantShards: 1},
		{name: "small", size: 2, wantShards: 1},
		{name: "below first split", size: 2*minRequestKeysPerShard - 1, wantShards: 1},
		{name: "first split", size: 2 * minRequestKeysPerShard, wantShards: 2},
		{name: "below second split", size: 4*minRequestKeysPerShard - 1, wantShards: 2},
		{name: "second split", size: 4 * minRequestKeysPerShard, wantShards: 4},
		{name: "below maximum split", size: maxRequestKeyCacheShards*minRequestKeysPerShard - 1, wantShards: maxRequestKeyCacheShards / 2},
		{name: "maximum", size: maxRequestKeyCacheShards * minRequestKeysPerShard, wantShards: maxRequestKeyCacheShards},
		{name: "maximum plus remainder", size: maxRequestKeyCacheShards*minRequestKeysPerShard + 17, wantShards: maxRequestKeyCacheShards},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache, err := newRequestKeyCache(tt.size)
			require.NoError(t, err)
			require.Len(t, cache.shards, tt.wantShards)

			total := 0
			minCapacity, maxCapacity := tt.size, 0
			for i := range cache.shards {
				capacity := cache.shards[i].capacity
				assert.Positive(t, capacity)
				total += capacity
				minCapacity = min(minCapacity, capacity)
				maxCapacity = max(maxCapacity, capacity)
			}
			assert.Equal(t, tt.size, total)
			assert.LessOrEqual(t, maxCapacity-minCapacity, 1)
		})
	}
}

func TestRequestKeyCacheRejectsInvalidConfiguration(t *testing.T) {
	_, err := newRequestKeyCache(0)
	assert.Error(t, err)
	_, err = newRequestKeyCache(-1)
	assert.Error(t, err)
	_, err = newRequestKeyCacheWithShards(8, 0)
	assert.Error(t, err)
	_, err = newRequestKeyCacheWithShards(8, 3)
	assert.Error(t, err)
	_, err = newRequestKeyCacheWithShards(2, 4)
	assert.Error(t, err)
	_, err = newRequestKeyCacheWithShards(1024, 2*maxRequestKeyCacheShards)
	assert.Error(t, err)
}

func TestRequestKeyCacheGetRefreshesWithinShard(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(8, 4)
	require.NoError(t, err)

	keys := requestKeysForShard(cache, 0, 3)
	first := &PodCache{}
	second := &PodCache{}
	cache.Add(keys[0], first)
	cache.Add(keys[1], second)

	value, found := cache.Get(keys[0])
	require.True(t, found)
	assert.Same(t, first, value)
	cache.Add(keys[2], &PodCache{})

	_, firstFound := cache.Peek(keys[0])
	_, secondFound := cache.Peek(keys[1])
	assert.True(t, firstFound)
	assert.False(t, secondFound)
}

func TestRequestKeyCacheShardQuotaCanEvictBelowGlobalCapacity(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(4, 2)
	require.NoError(t, err)

	keys := requestKeysForShard(cache, 0, 3)
	for _, key := range keys {
		cache.Add(key, &PodCache{})
	}

	keysInCache := cache.Keys()
	assert.Len(t, keysInCache, 2)
	assert.Less(t, len(keysInCache), 4)
}

func TestRequestKeyCacheReadOnlyOperationsDoNotRefresh(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(2, 1)
	require.NoError(t, err)

	cache.Add(10, &PodCache{})
	cache.Add(20, &PodCache{})
	_, found := cache.Peek(10)
	require.True(t, found)
	cache.Keys()
	cache.RangeKeys(func(BlockHash) bool { return true })
	_, found = cache.Get(99)
	assert.False(t, found)
	cache.Add(30, &PodCache{})

	_, oldestFound := cache.Peek(10)
	_, newerFound := cache.Peek(20)
	assert.False(t, oldestFound)
	assert.True(t, newerFound)
}

func TestRequestKeyCacheGetOrAddRefreshesExistingValue(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(2, 1)
	require.NoError(t, err)

	existing := &PodCache{}
	cache.Add(10, existing)
	cache.Add(20, &PodCache{})
	factoryCalled := false
	actual, loaded, evicted := cache.GetOrAdd(10, func() *PodCache {
		factoryCalled = true
		return &PodCache{}
	})
	assert.Same(t, existing, actual)
	assert.True(t, loaded)
	assert.False(t, evicted)
	assert.False(t, factoryCalled)

	cache.Add(30, &PodCache{})
	_, existingFound := cache.Peek(10)
	_, displacedFound := cache.Peek(20)
	assert.True(t, existingFound)
	assert.False(t, displacedFound)
}

func TestRequestKeyCacheGetOrAddCreatesAndReportsEviction(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(2, 1)
	require.NoError(t, err)

	cache.Add(10, &PodCache{})
	cache.Add(20, &PodCache{})
	created := &PodCache{}
	factoryCalls := 0
	actual, loaded, evicted := cache.GetOrAdd(30, func() *PodCache {
		factoryCalls++
		return created
	})

	assert.Same(t, created, actual)
	assert.False(t, loaded)
	assert.True(t, evicted)
	assert.Equal(t, 1, factoryCalls)
	_, oldestFound := cache.Peek(10)
	resident, createdFound := cache.Peek(30)
	assert.False(t, oldestFound)
	assert.True(t, createdFound)
	assert.Same(t, created, resident)
}

func TestRequestKeyCacheGetOrAddReturnsResidentValue(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(8, 4)
	require.NoError(t, err)

	const workers = 32
	values := make(chan *PodCache, workers)
	start := make(chan struct{})
	var factoryCalls atomic.Int32
	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		go func() {
			defer wg.Done()
			<-start
			actual, _, _ := cache.GetOrAdd(10, func() *PodCache {
				factoryCalls.Add(1)
				return &PodCache{}
			})
			values <- actual
		}()
	}
	close(start)
	wg.Wait()
	close(values)

	resident, found := cache.Peek(10)
	require.True(t, found)
	for value := range values {
		assert.Same(t, resident, value)
	}
	assert.Equal(t, int32(1), factoryCalls.Load())
}

func TestRequestKeyCacheKeysContainsEveryResidentOnce(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(16, 4)
	require.NoError(t, err)

	want := make(map[BlockHash]struct{})
	for shardIdx := range cache.shards {
		for _, key := range requestKeysForShard(cache, shardIdx, 3) {
			cache.Add(key, &PodCache{})
			want[key] = struct{}{}
		}
	}

	keys := cache.Keys()
	require.Len(t, keys, len(want))
	seen := make(map[BlockHash]struct{}, len(keys))
	for _, key := range keys {
		_, duplicate := seen[key]
		assert.False(t, duplicate, "duplicate key %d", key)
		seen[key] = struct{}{}
	}
	assert.Equal(t, want, seen)

	rangeSeen := make(map[BlockHash]struct{}, len(want))
	cache.RangeKeys(func(key BlockHash) bool {
		rangeSeen[key] = struct{}{}
		return true
	})
	assert.Equal(t, want, rangeSeen)

	visits := 0
	cache.RangeKeys(func(BlockHash) bool {
		visits++
		return false
	})
	assert.Equal(t, 1, visits)
}

func TestInMemoryIndexClearVisitsEveryRequestKeyCacheShard(t *testing.T) {
	ctx := log.IntoContext(t.Context(), logr.Discard())
	index, err := NewInMemoryIndex(&InMemoryIndexConfig{
		Size:         4 * minRequestKeysPerShard,
		PodCacheSize: 2,
	})
	require.NoError(t, err)
	require.Len(t, index.data.shards, 4)

	keys := make([]BlockHash, len(index.data.shards))
	for shardIdx := range index.data.shards {
		keys[shardIdx] = requestKeysForShard(index.data, shardIdx, 1)[0]
	}
	require.NoError(t, index.Add(ctx, nil, keys, []PodEntry{
		{PodIdentifier: "pod-clear", DeviceTier: "gpu"},
		{PodIdentifier: "pod-keep", DeviceTier: "gpu"},
	}))
	require.NoError(t, index.Clear(ctx, "pod-clear"))

	for _, key := range keys {
		podCache, found := index.data.Peek(key)
		require.True(t, found)
		assert.Empty(t, podCache.matching("pod-clear"))
		assert.Len(t, podCache.matching("pod-keep"), 1)
	}
}

func TestRequestKeyCacheIndependentShardLocks(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(8, 4)
	require.NoError(t, err)

	keyA := requestKeysForShard(cache, 0, 1)[0]
	keyB := requestKeysForShard(cache, 1, 1)[0]
	cache.Add(keyA, &PodCache{})
	cache.Add(keyB, &PodCache{})

	locked := make(chan struct{})
	release := make(chan struct{})
	holderDone := make(chan struct{})
	var releaseOnce sync.Once
	releaseLock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseLock)
	go func() {
		cache.shards[0].mu.Lock()
		close(locked)
		<-release
		cache.shards[0].mu.Unlock()
		close(holderDone)
	}()
	<-locked

	aDone := make(chan struct{})
	go func() {
		cache.Get(keyA)
		close(aDone)
	}()
	bDone := make(chan struct{})
	go func() {
		cache.Get(keyB)
		close(bDone)
	}()

	select {
	case <-bDone:
	case <-time.After(time.Second):
		t.Fatal("operation on an independent shard blocked")
	}
	select {
	case <-aDone:
		t.Fatal("operation on the locked shard completed")
	default:
	}
	releaseLock()

	select {
	case <-aDone:
	case <-time.After(time.Second):
		t.Fatal("operation did not resume after shard unlock")
	}
	<-holderDone
}

func TestRequestKeyCacheRangeKeysStopsBeforeLaterShard(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(8, 4)
	require.NoError(t, err)

	cache.Add(requestKeysForShard(cache, 0, 1)[0], &PodCache{})
	cache.Add(requestKeysForShard(cache, 1, 1)[0], &PodCache{})
	cache.shards[1].mu.Lock()
	var unlockOnce sync.Once
	unlock := func() { unlockOnce.Do(cache.shards[1].mu.Unlock) }
	t.Cleanup(unlock)

	done := make(chan struct{})
	visits := 0
	go func() {
		cache.RangeKeys(func(BlockHash) bool {
			visits++
			return false
		})
		close(done)
	}()

	select {
	case <-done:
		assert.Equal(t, 1, visits)
	case <-time.After(time.Second):
		unlock()
		<-done
		t.Fatal("range did not stop before the locked later shard")
	}
}

func TestRequestKeyCacheConcurrentOperations(t *testing.T) {
	cache, err := newRequestKeyCacheWithShards(1024, 16)
	require.NoError(t, err)

	const workers = 32
	var wg sync.WaitGroup
	wg.Add(workers)
	for worker := range workers {
		go func() {
			defer wg.Done()
			for i := range 200 {
				key := BlockHash(worker*200 + i + 1)
				cache.GetOrAdd(key, func() *PodCache { return &PodCache{} })
				cache.Get(key)
				cache.Peek(key)
				if i%3 == 0 {
					cache.Remove(key)
				}
				if i%50 == 0 {
					cache.Keys()
				}
			}
		}()
	}
	wg.Wait()
	assert.LessOrEqual(t, len(cache.Keys()), 1024)
}
