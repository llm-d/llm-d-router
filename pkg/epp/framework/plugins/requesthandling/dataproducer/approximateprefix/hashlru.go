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

package approximateprefix

// nodeNil is the prev/next sentinel. newHashLRU keeps capacity at or below it,
// so node indexes stay below it.
const nodeNil = ^uint32(0)

// hashNode is one cached block hash and its LRU links, which are indexes into
// hashLRU.nodes.
type hashNode struct {
	key        blockHash
	prev, next uint32
}

// hashLRU is one pod's block-hash set with LRU eviction. index maps a hash to
// its node. nodes grows with inserts up to capacity; after that, each new hash
// reuses the evicted node. head is the most recently added or refreshed hash;
// tail is the next eviction. Contains does not move a hit.
type hashLRU struct {
	index    map[blockHash]uint32
	nodes    []hashNode
	capacity int
	head     uint32
	tail     uint32
}

func newHashLRU(capacity int) *hashLRU {
	return &hashLRU{
		index:    make(map[blockHash]uint32),
		capacity: clampCapacity(capacity),
		head:     nodeNil,
		tail:     nodeNil,
	}
}

// clampCapacity bounds capacity to [0, nodeNil] so node indexes fit in uint32.
// The clamp runs in uint64 because nodeNil does not fit in a 32-bit int.
func clampCapacity(capacity int) int {
	return int(min(uint64(max(capacity, 0)), uint64(nodeNil))) // #nosec G115 -- bounded by capacity, an int.
}

func (c *hashLRU) Cap() int { return c.capacity }

// Resize sets the capacity. Growing, or shrinking to no fewer than the
// allocated nodes, only updates the limit: nodes are allocated lazily. Below
// that, it rebuilds with the capacity most recent hashes in their recency
// order. The rebuild keeps nodes dense, which Add relies on: once nodes reaches
// capacity, Add only reuses the tail node, so freeing nodes in place would
// strand them.
func (c *hashLRU) Resize(capacity int) {
	capacity = clampCapacity(capacity)
	if capacity >= len(c.nodes) {
		c.capacity = capacity
		return
	}
	keep := make([]blockHash, 0, capacity)
	for i := c.head; i != nodeNil && len(keep) < capacity; i = c.nodes[i].next {
		keep = append(keep, c.nodes[i].key)
	}
	rebuilt := newHashLRU(capacity)
	for j := len(keep) - 1; j >= 0; j-- { // oldest first, so the newest ends at head
		rebuilt.Add(keep[j])
	}
	*c = *rebuilt
}

func (c *hashLRU) Len() int { return len(c.index) }

func (c *hashLRU) Contains(key blockHash) bool {
	_, ok := c.index[key]
	return ok
}

// Add inserts key as most recent, or refreshes it if it is already present.
// At capacity the least recently added or refreshed key is evicted first.
func (c *hashLRU) Add(key blockHash) {
	if c.capacity <= 0 {
		return
	}
	if i, ok := c.index[key]; ok {
		c.moveFront(i)
		return
	}
	var i uint32
	if len(c.nodes) < c.capacity {
		i = uint32(len(c.nodes)) // #nosec G115 -- len(nodes) < capacity <= nodeNil.
		c.nodes = append(c.nodes, hashNode{})
	} else {
		i = c.tail
		c.unlink(i)
		delete(c.index, c.nodes[i].key)
	}
	c.nodes[i] = hashNode{key: key, prev: nodeNil, next: nodeNil}
	c.index[key] = i
	c.pushFront(i)
}

func (c *hashLRU) moveFront(i uint32) {
	if c.head == i {
		return
	}
	c.unlink(i)
	c.pushFront(i)
}

func (c *hashLRU) pushFront(i uint32) {
	c.nodes[i].prev = nodeNil
	c.nodes[i].next = c.head
	if c.head == nodeNil {
		c.tail = i
	} else {
		c.nodes[c.head].prev = i
	}
	c.head = i
}

func (c *hashLRU) unlink(i uint32) {
	prev, next := c.nodes[i].prev, c.nodes[i].next
	if prev == nodeNil {
		c.head = next
	} else {
		c.nodes[prev].next = next
	}
	if next == nodeNil {
		c.tail = prev
	} else {
		c.nodes[next].prev = prev
	}
}
