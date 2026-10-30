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

import "hash/maphash"

// idxNil is the prev/next sentinel. It fills 31 bits so it can sit beside the
// occupied flag in prev. Slot indexes therefore have to stay below 2^31.
const idxNil = uint32(0x7FFFFFFF)

// occupied is the high bit of prev. A block hash of zero is a real key, so an
// empty slot cannot be recognized by its key. The flag shares prev's word so
// the slot stays 16 bytes.
const occupied = uint32(1 << 31)

const maxHashLRUCapacity = int(idxNil) / 2

// hashSlot is one open-addressed entry. prev's high bit is occupied; its low
// 31 bits and next are LRU links. The zero slot is empty.
type hashSlot struct {
	key        blockHash
	prev, next uint32
}

// hashLRU is one pod's block-hash set with LRU eviction.
// slots is twice capacity so a lookup stops within a few entries, and the
// slice is allocated on the first Add. head is the most recently added or
// refreshed hash; tail is the next eviction. Contains does not move a hit.
// seed mixes the block hash before it selects a slot. Block hashes are public
// xxhash values, so an unseeded home would let a client place many hashes in
// one probe run.
type hashLRU struct {
	slots    []hashSlot
	seed     maphash.Seed
	capacity int
	n        int
	head     uint32
	tail     uint32
}

func newHashLRU(capacity int) *hashLRU {
	if capacity > maxHashLRUCapacity {
		capacity = defaultLRUCapacityPerServer
	}
	return &hashLRU{capacity: capacity, seed: maphash.MakeSeed(), head: idxNil, tail: idxNil}
}

func (c *hashLRU) Len() int { return c.n }

func (c *hashLRU) Contains(key blockHash) bool {
	_, ok := c.find(key)
	return ok
}

// Add inserts key as most recent, or refreshes it if it is already present.
// At capacity the least recently added or refreshed key is evicted first.
func (c *hashLRU) Add(key blockHash) {
	if c.capacity <= 0 {
		return
	}
	if c.slots == nil {
		c.slots = make([]hashSlot, c.capacity*2)
	}
	if i, ok := c.find(key); ok {
		c.moveFront(i)
		return
	}
	if c.n == c.capacity {
		c.evictOldest()
	}
	// find returns the first empty slot on a miss. Eviction slides the probe
	// cluster, so the empty slot is located again after it.
	i, _ := c.find(key)
	c.slots[i].key = key
	c.slots[i].prev = occupied | idxNil
	c.slots[i].next = idxNil
	c.pushFront(i)
	c.n++
}

func (c *hashLRU) find(key blockHash) (uint32, bool) {
	n := uint32(len(c.slots)) // #nosec G115 -- newHashLRU bounds slots below idxNil.
	if n == 0 {
		return 0, false
	}
	i := c.home(key)
	for range n {
		s := &c.slots[i]
		if s.prev&occupied == 0 {
			return i, false
		}
		if s.key == key {
			return i, true
		}
		i++
		if i == n {
			i = 0
		}
	}
	return 0, false
}

func (c *hashLRU) home(key blockHash) uint32 {
	n := uint64(len(c.slots))
	if n == 0 {
		return 0
	}
	// #nosec G115 -- the modulus is len(slots), bounded by newHashLRU.
	return uint32(maphash.Comparable(c.seed, key) % n)
}

func (c *hashLRU) evictOldest() {
	i := c.tail
	c.unlink(i)
	c.removeSlot(i)
	c.n--
}

func (c *hashLRU) moveFront(i uint32) {
	if c.head == i {
		return
	}
	c.unlink(i)
	c.slots[i].prev = occupied | idxNil
	c.slots[i].next = idxNil
	c.pushFront(i)
}

func (c *hashLRU) pushFront(i uint32) {
	c.slots[i].next = c.head
	if c.head == idxNil {
		c.tail = i
	} else {
		c.slots[c.head].prev = occupied | i
	}
	c.head = i
}

func (c *hashLRU) unlink(i uint32) {
	prev := c.slots[i].prev &^ occupied
	next := c.slots[i].next
	if prev == idxNil {
		c.head = next
	} else {
		c.slots[prev].next = next
	}
	if next == idxNil {
		c.tail = prev
	} else {
		c.slots[next].prev = occupied | prev
	}
}

// removeSlot deletes the slot and closes its probe cluster by sliding later
// entries back into the hole. Each slide updates LRU links, which name slots
// by index. Closing the cluster keeps every lookup on a path that ends at an
// empty slot.
func (c *hashLRU) removeSlot(hole uint32) {
	n := uint32(len(c.slots)) // #nosec G115 -- newHashLRU bounds slots below idxNil.
	c.slots[hole] = hashSlot{}
	i := hole
	for range n {
		i++
		if i == n {
			i = 0
		}
		if c.slots[i].prev&occupied == 0 {
			return
		}
		if !onProbe(c.home(c.slots[i].key), hole, i) {
			continue
		}
		c.relocate(i, hole)
		hole = i
	}
}

func (c *hashLRU) relocate(from, to uint32) {
	c.slots[to] = c.slots[from]
	c.slots[from] = hashSlot{}
	prev := c.slots[to].prev &^ occupied
	next := c.slots[to].next
	if c.head == from {
		c.head = to
	} else if prev != idxNil {
		c.slots[prev].next = to
	}
	if c.tail == from {
		c.tail = to
	} else if next != idxNil {
		c.slots[next].prev = occupied | to
	}
}

// onProbe reports whether hole lies on the linear probe that placed the entry
// now at i. home is where that probe started. i itself is not included.
func onProbe(home, hole, i uint32) bool {
	if home <= i {
		return hole >= home && hole < i
	}
	return hole >= home || hole < i
}
