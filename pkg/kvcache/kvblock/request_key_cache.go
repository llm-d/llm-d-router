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

	"github.com/hashicorp/golang-lru/v2/simplelru"
)

// requestKeyBatchSize amortizes LRU promotion while keeping event-writer lock
// wait bounded under concurrent scored lookups.
const requestKeyBatchSize = 16

// requestKeyCache is a thread-safe LRU for request-key entries. It exposes a
// batched read so scored lookups update prefix recency with bounded lock holds.
type requestKeyCache struct {
	mu   sync.RWMutex
	data *simplelru.LRU[BlockHash, *PodCache]
}

func newRequestKeyCache(size int) (*requestKeyCache, error) {
	data, err := simplelru.NewLRU[BlockHash, *PodCache](size, nil)
	if err != nil {
		return nil, err
	}
	return &requestKeyCache{data: data}, nil
}

func (c *requestKeyCache) Get(key BlockHash) (*PodCache, bool) {
	c.mu.Lock()
	value, found := c.data.Get(key)
	c.mu.Unlock()
	return value, found
}

func (c *requestKeyCache) Peek(key BlockHash) (*PodCache, bool) {
	c.mu.RLock()
	value, found := c.data.Peek(key)
	c.mu.RUnlock()
	return value, found
}

// GetPrefixBatch returns and refreshes at most one contiguous cache batch.
func (c *requestKeyCache) GetPrefixBatch(keys []BlockHash, dst []*PodCache) []*PodCache {
	clear(dst)
	dst = dst[:0]
	keys = keys[:min(len(keys), requestKeyBatchSize)]
	c.mu.Lock()
	for _, key := range keys {
		value, found := c.data.Get(key)
		if !found || value == nil {
			break
		}
		dst = append(dst, value)
	}
	c.mu.Unlock()
	return dst
}

func (c *requestKeyCache) Add(key BlockHash, value *PodCache) bool {
	c.mu.Lock()
	evicted := c.data.Add(key, value)
	c.mu.Unlock()
	return evicted
}

func (c *requestKeyCache) ContainsOrAdd(key BlockHash, value *PodCache) (bool, bool) {
	c.mu.Lock()
	if c.data.Contains(key) {
		c.mu.Unlock()
		return true, false
	}
	evicted := c.data.Add(key, value)
	c.mu.Unlock()
	return false, evicted
}

func (c *requestKeyCache) Remove(key BlockHash) bool {
	c.mu.Lock()
	present := c.data.Remove(key)
	c.mu.Unlock()
	return present
}

func (c *requestKeyCache) Keys() []BlockHash {
	c.mu.RLock()
	keys := c.data.Keys()
	c.mu.RUnlock()
	return keys
}
