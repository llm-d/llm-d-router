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
	"errors"
	"hash/maphash"
	"sync"

	"github.com/hashicorp/golang-lru/v2/simplelru"
)

const (
	maxRequestKeyCacheShards = 64
	// Keep local quotas large enough that normal hash variance does not cause
	// meaningful capacity fragmentation.
	minRequestKeysPerShard = 1024
)

// requestKeyCache shards recency state so unrelated keys do not share an LRU
// lock. Capacity and recency are exact within a shard; the shard capacities
// sum to the configured cache size. A full shard evicts locally even when
// another shard has unused capacity.
type requestKeyCache struct {
	shards []requestKeyCacheShard
	mask   uint64
	seed   maphash.Seed
}

type requestKeyCacheShard struct {
	mu       sync.RWMutex
	data     *simplelru.LRU[BlockHash, *PodCache]
	capacity int
}

func newRequestKeyCache(size int) (*requestKeyCache, error) {
	if size <= 0 {
		return nil, errors.New("must provide a positive size")
	}
	shardCount := 1
	for shardCount < maxRequestKeyCacheShards && size/(shardCount*2) >= minRequestKeysPerShard {
		shardCount *= 2
	}
	return newRequestKeyCacheWithShards(size, shardCount)
}

func newRequestKeyCacheWithShards(size, shardCount int) (*requestKeyCache, error) {
	if size <= 0 {
		return nil, errors.New("must provide a positive size")
	}
	if shardCount <= 0 || shardCount > maxRequestKeyCacheShards || shardCount > size || shardCount&(shardCount-1) != 0 {
		return nil, errors.New("shard count must be a positive power of two no larger than the cache size or shard limit")
	}

	cache := &requestKeyCache{
		shards: make([]requestKeyCacheShard, shardCount),
		mask:   uint64(shardCount - 1),
		seed:   maphash.MakeSeed(),
	}
	baseCapacity := size / shardCount
	remainder := size % shardCount
	for i := range cache.shards {
		capacity := baseCapacity
		if i < remainder {
			capacity++
		}
		data, err := simplelru.NewLRU[BlockHash, *PodCache](capacity, nil)
		if err != nil {
			return nil, err
		}
		cache.shards[i] = requestKeyCacheShard{data: data, capacity: capacity}
	}
	return cache, nil
}

func (c *requestKeyCache) shardIndex(key BlockHash) int {
	if c.mask == 0 {
		return 0
	}
	// Rehash with a process-local seed so externally derived block hashes
	// cannot directly select a shard through their low bits.
	return int(maphash.Comparable(c.seed, key) & c.mask)
}

func (c *requestKeyCache) shard(key BlockHash) *requestKeyCacheShard {
	return &c.shards[c.shardIndex(key)]
}

func (c *requestKeyCache) Get(key BlockHash) (*PodCache, bool) {
	shard := c.shard(key)
	shard.mu.Lock()
	value, found := shard.data.Get(key)
	shard.mu.Unlock()
	return value, found
}

func (c *requestKeyCache) Peek(key BlockHash) (*PodCache, bool) {
	shard := c.shard(key)
	shard.mu.RLock()
	value, found := shard.data.Peek(key)
	shard.mu.RUnlock()
	return value, found
}

// GetOrAdd returns and refreshes the resident value or creates it atomically.
func (c *requestKeyCache) GetOrAdd(key BlockHash, newValue func() *PodCache) (value *PodCache, loaded, evicted bool) {
	shard := c.shard(key)
	shard.mu.Lock()
	if value, loaded = shard.data.Get(key); loaded {
		shard.mu.Unlock()
		return value, true, false
	}
	value = newValue()
	evicted = shard.data.Add(key, value)
	shard.mu.Unlock()
	return value, false, evicted
}

func (c *requestKeyCache) Add(key BlockHash, value *PodCache) bool {
	shard := c.shard(key)
	shard.mu.Lock()
	evicted := shard.data.Add(key, value)
	shard.mu.Unlock()
	return evicted
}

func (c *requestKeyCache) Remove(key BlockHash) bool {
	shard := c.shard(key)
	shard.mu.Lock()
	present := shard.data.Remove(key)
	shard.mu.Unlock()
	return present
}

// Keys returns separately timed per-shard snapshots. Global recency order is
// unspecified.
func (c *requestKeyCache) Keys() []BlockHash {
	total := 0
	for i := range c.shards {
		shard := &c.shards[i]
		shard.mu.RLock()
		total += shard.data.Len()
		shard.mu.RUnlock()
	}
	keys := make([]BlockHash, 0, total)
	for i := range c.shards {
		shard := &c.shards[i]
		shard.mu.RLock()
		keys = append(keys, shard.data.Keys()...)
		shard.mu.RUnlock()
	}
	return keys
}

// RangeKeys visits separately timed per-shard snapshots without retaining an
// all-shard key copy or holding a shard lock during yield. Returning false
// stops the walk.
func (c *requestKeyCache) RangeKeys(yield func(BlockHash) bool) {
	for i := range c.shards {
		shard := &c.shards[i]
		shard.mu.RLock()
		keys := shard.data.Keys()
		shard.mu.RUnlock()
		for _, key := range keys {
			if !yield(key) {
				return
			}
		}
	}
}
