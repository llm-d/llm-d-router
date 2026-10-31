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

package multimodal

import (
	"container/list"

	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

// podCache tracks the encoder-cache state for one pod. Entries with active
// request references are pinned. Only unreferenced entries participate in the
// least-recently-used reclaim order.
type podCache struct {
	capacity int
	used     int
	entries  map[string]*cacheEntry
	freeable *list.List
}

type cacheEntry struct {
	item            attrmm.MatchItem
	references      map[string]struct{}
	freeableElement *list.Element
}

func newPodCache(capacity int) *podCache {
	if capacity < 1 {
		capacity = 1
	}
	return &podCache{
		capacity: capacity,
		entries:  make(map[string]*cacheEntry),
		freeable: list.New(),
	}
}

// acquire pins an item for requestID. It returns false when the item cannot fit
// without reclaiming an entry that is still referenced by an active request.
func (c *podCache) acquire(requestID string, item attrmm.MatchItem) bool {
	if entry, ok := c.entries[item.Hash]; ok {
		if entry.freeableElement != nil {
			c.freeable.Remove(entry.freeableElement)
			entry.freeableElement = nil
		}
		entry.references[requestID] = struct{}{}
		return true
	}

	item.Size = normalizedItemSize(item.Size)
	if !c.makeRoom(item.Size) {
		return false
	}
	entry := &cacheEntry{
		item:       item,
		references: map[string]struct{}{requestID: {}},
	}
	c.entries[item.Hash] = entry
	c.used += item.Size
	return true
}

// release removes one request reference. Once the last reference is gone, the
// entry becomes the most recently used reclaimable item.
func (c *podCache) release(requestID, hash string) {
	entry, ok := c.entries[hash]
	if !ok {
		return
	}
	delete(entry.references, requestID)
	if len(entry.references) != 0 || entry.freeableElement != nil {
		return
	}
	entry.freeableElement = c.freeable.PushBack(hash)
}

// commit records an encoder output that completed without a prior cache hit.
// The resulting entry is immediately reclaimable because the encoder phase has
// finished by the time it is committed.
func (c *podCache) commit(item attrmm.MatchItem) bool {
	if entry, ok := c.entries[item.Hash]; ok {
		if len(entry.references) == 0 {
			if entry.freeableElement != nil {
				c.freeable.MoveToBack(entry.freeableElement)
			} else {
				entry.freeableElement = c.freeable.PushBack(item.Hash)
			}
		}
		return true
	}

	item.Size = normalizedItemSize(item.Size)
	if !c.makeRoom(item.Size) {
		return false
	}
	entry := &cacheEntry{item: item, references: make(map[string]struct{})}
	entry.freeableElement = c.freeable.PushBack(item.Hash)
	c.entries[item.Hash] = entry
	c.used += item.Size
	return true
}

func (c *podCache) makeRoom(size int) bool {
	if size > c.capacity {
		return false
	}
	needed := c.used + size - c.capacity
	if needed <= 0 {
		return true
	}

	reclaimable := 0
	for element := c.freeable.Front(); element != nil && reclaimable < needed; element = element.Next() {
		hash := element.Value.(string)
		reclaimable += c.entries[hash].item.Size
	}
	if reclaimable < needed {
		return false
	}

	for c.used+size > c.capacity {
		element := c.freeable.Front()
		hash := element.Value.(string)
		entry := c.entries[hash]
		c.freeable.Remove(element)
		delete(c.entries, hash)
		c.used -= entry.item.Size
	}
	return true
}

func (c *podCache) contains(hash string) bool {
	_, ok := c.entries[hash]
	return ok
}

func (c *podCache) len() int {
	return len(c.entries)
}

func (c *podCache) keys() []string {
	keys := make([]string, 0, len(c.entries))
	for hash := range c.entries {
		keys = append(keys, hash)
	}
	return keys
}

func normalizedItemSize(size int) int {
	if size < 1 {
		return 1
	}
	return size
}
