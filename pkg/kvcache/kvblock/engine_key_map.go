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

import "sync/atomic"

// engineKeyMap holds the engine-to-request key mappings with value-type
// slots: each mapping's first request key sits inline in its map slot and
// longer chains continue into a shared arena, so a mapping costs no heap
// object of its own. The structure is internally lock-free; every caller
// holds the enclosing index's lock across all of its methods, except len,
// which reads an atomic count and is safe without it: the in-memory index
// reports residency gauges after releasing its lock.
//
// Preserved from the previous lru.Cache-backed map: get returns the last
// element of the request-key chain and refreshes recency; peek resolves
// without refreshing; removal is the caller's decision; the count cap
// evicts by last-visit recency. Recency is a coarse stamp rather than an
// ordered list, so eviction under the cap is approximate; the cap sits
// far above what the container's memory holds, so the approximation is
// unobservable in the default configuration.
type engineKeyMap struct {
	// slots maps an engine key to its request-key chain.
	slots map[BlockHash]engineSlot
	// arena holds the chain elements past each slot's inline first key,
	// addressed by handle (handle h is arena[h-1]; 0 ends a chain). Nodes
	// are recycled through free.
	arena []arenaNode
	free  []uint32
	// limit caps len(slots); a non-positive value is unlimited.
	limit int
	// clock is the last-visit stamp source, incremented per refresh.
	clock uint32
	// count is the live mapping count, maintained atomically so len can be
	// called without the enclosing index's lock.
	count atomic.Int64
}

// engineSlot is one mapping's map value. Chains of one request key (the
// 1:1 and many:1 geometries) use only first and next stays 0.
type engineSlot struct {
	first BlockHash
	next  uint32
	stamp uint32
}

// arenaNode is one request-key chain element past the slot's inline first.
type arenaNode struct {
	key  BlockHash
	next uint32
}

// newEngineKeyMap returns a map capped at limit mappings. The caller's
// construction path validates limit; a non-positive value means unlimited.
func newEngineKeyMap(limit int) *engineKeyMap {
	return &engineKeyMap{
		slots: make(map[BlockHash]engineSlot),
		limit: limit,
	}
}

// evictSampleSize is how many slots the count-cap eviction inspects for a
// cold one. Map iteration order is randomized, which makes the inspection
// a sample rather than a scan.
const evictSampleSize = 16

// set stores rks as ek's chain, replacing any prior chain.
func (m *engineKeyMap) set(ek BlockHash, rks []BlockHash) {
	if len(rks) == 0 {
		return
	}
	old, exists := m.slots[ek]
	if exists {
		m.freeChain(old.next)
	} else {
		m.count.Add(1)
		if m.limit > 0 && len(m.slots) >= m.limit {
			m.evictOne()
		}
	}
	// Link the arena nodes tail-first so each node points at the element
	// after it.
	next := uint32(0)
	for i := len(rks) - 1; i >= 1; i-- {
		h := m.alloc(rks[i])
		m.arena[h-1].next = next
		next = h
	}
	m.slots[ek] = engineSlot{first: rks[0], next: next, stamp: m.tick()}
}

// get returns the chain's last request key, refreshing the slot's stamp.
// The last element is what parent-hash resolution needs: the highest-index
// request key the engine key covers.
func (m *engineKeyMap) get(ek BlockHash) (BlockHash, bool) {
	slot, ok := m.slots[ek]
	if !ok {
		return EmptyBlockHash, false
	}
	slot.stamp = m.tick()
	m.slots[ek] = slot
	last := slot.first
	for h := slot.next; h != 0; h = m.arena[h-1].next {
		last = m.arena[h-1].key
	}
	return last, true
}

// peek returns ek's full chain without refreshing recency. The returned
// slice is a copy: the caller reads it after releasing its lock, and set
// and remove recycle the nodes it would alias.
func (m *engineKeyMap) peek(ek BlockHash) ([]BlockHash, bool) {
	slot, ok := m.slots[ek]
	if !ok {
		return nil, false
	}
	rks := []BlockHash{slot.first}
	for h := slot.next; h != 0; h = m.arena[h-1].next {
		rks = append(rks, m.arena[h-1].key)
	}
	return rks, true
}

// remove deletes ek's mapping.
func (m *engineKeyMap) remove(ek BlockHash) {
	if slot, ok := m.slots[ek]; ok {
		m.freeChain(slot.next)
		delete(m.slots, ek)
		m.count.Add(-1)
	}
}

// contains reports whether ek is mapped.
func (m *engineKeyMap) contains(ek BlockHash) bool {
	_, ok := m.slots[ek]
	return ok
}

// len returns the mapping count. It is the one method safe to call without
// the enclosing index's lock.
func (m *engineKeyMap) len() int {
	return int(m.count.Load())
}

// tick advances and returns the stamp clock. The uint32 wraps after 2^32
// refreshes; a wrapped stamp can misorder one eviction sweep at the count
// cap, which the default configuration cannot reach.
func (m *engineKeyMap) tick() uint32 {
	m.clock++
	return m.clock
}

// evictOne removes the coldest-stamped slot among a sample.
func (m *engineKeyMap) evictOne() {
	var victim BlockHash
	var coldest uint32
	found := false
	n := 0
	for ek, slot := range m.slots {
		if !found || slot.stamp < coldest {
			victim, coldest, found = ek, slot.stamp, true
		}
		n++
		if n >= evictSampleSize {
			break
		}
	}
	if found {
		m.freeChain(m.slots[victim].next)
		delete(m.slots, victim)
		m.count.Add(-1)
	}
}

// alloc returns a handle to a node holding key, drawing on the free list
// first. Handles are uint32, so the arena holds at most 2^32-1 nodes;
// only 1:many geometries (engine block larger than the canonical block)
// allocate any.
func (m *engineKeyMap) alloc(key BlockHash) uint32 {
	if n := len(m.free); n > 0 {
		h := m.free[n-1]
		m.free = m.free[:n-1]
		m.arena[h-1] = arenaNode{key: key}
		return h
	}
	m.arena = append(m.arena, arenaNode{key: key})
	return uint32(len(m.arena)) //#nosec G115 -- bounded by addressable memory well below 2^32
}

// freeChain returns h's chain to the free list.
func (m *engineKeyMap) freeChain(h uint32) {
	for h != 0 {
		next := m.arena[h-1].next
		m.arena[h-1] = arenaNode{}
		m.free = append(m.free, h)
		h = next
	}
}
