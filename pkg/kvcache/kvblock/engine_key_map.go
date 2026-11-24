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
// slots: an open-addressed table (linear probing, backward-shift deletion,
// doubling growth, the dense table's hash and load bound) whose slot carries
// the mapping's first request key inline, and longer chains continue into a
// shared arena, so a mapping costs no heap object of its own. The structure
// is internally lock-free; every caller holds the enclosing index's lock
// across all of its methods, except len, which reads an atomic count and is
// safe without it: the in-memory index reports residency gauges after
// releasing its lock.
//
// Preserved from the lru.Cache-backed map: get returns the last element of
// the request-key chain and refreshes recency; peek resolves without
// refreshing; removal is the caller's decision; the count cap evicts by
// last-visit recency. Recency is a coarse stamp rather than an ordered
// list, so eviction under the cap is approximate; the cap sits far above
// what the container's memory holds, so the approximation is unobservable
// in the default configuration.
type engineKeyMap struct {
	// slots is the open-addressed mapping table. A slot is empty exactly
	// when its stamp is zero: stamps are clock ticks, nonzero for every
	// occupied slot, and deletion zeroes the whole slot.
	slots []engineSlot
	mask  int
	used  int
	// cursor is the next table index the count-cap eviction and the engine
	// sweep sample from, rotating for coverage.
	cursor int
	// arena holds the chain elements past each slot's inline first key,
	// addressed by handle (handle h is arena[h-1]; 0 ends a chain). Nodes
	// are recycled through free.
	arena []arenaNode
	free  []uint32
	// limit caps the mapping count; a non-positive value is unlimited.
	limit int
	// clock is the last-visit stamp source, incremented per refresh.
	clock uint32
	// count is the live mapping count, maintained atomically so len can be
	// called without the enclosing index's lock.
	count atomic.Int64
}

// engineSlot is one mapping's table entry. Chains of one request key (the
// 1:1 and many:1 geometries) use only first and next stays 0.
type engineSlot struct {
	key   BlockHash
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
	m := &engineKeyMap{
		slots: make([]engineSlot, tableInitialSlots),
		limit: limit,
	}
	m.mask = len(m.slots) - 1
	return m
}

// evictSampleSize is how many slots the count-cap eviction inspects for a
// cold one. The inspection walks from the rotating cursor, so it samples a
// contiguous stretch of the table rather than a random one.
const evictSampleSize = 16

// set stores rks as ek's chain, replacing any prior chain.
func (m *engineKeyMap) set(ek BlockHash, rks []BlockHash) {
	if len(rks) == 0 {
		return
	}
	i, ok := m.find(ek)
	if !ok {
		if m.limit > 0 && m.used >= m.limit {
			m.evictOne()
		}
		i = m.slotForInsert(ek)
		m.count.Add(1)
	} else {
		m.freeChain(m.slots[i].next)
	}
	// Link the arena nodes tail-first so each node points at the element
	// after it.
	next := uint32(0)
	for j := len(rks) - 1; j >= 1; j-- {
		h := m.alloc(rks[j])
		m.arena[h-1].next = next
		next = h
	}
	m.slots[i].key = ek
	m.slots[i].first = rks[0]
	m.slots[i].next = next
	m.slots[i].stamp = m.tick()
}

// find returns the index of ek's slot, and whether it is present.
func (m *engineKeyMap) find(ek BlockHash) (int, bool) {
	i := int(tableHash(ek)) & m.mask //#nosec G115 -- the hash is masked to the slot count
	for m.slots[i].stamp != 0 {
		if m.slots[i].key == ek {
			return i, true
		}
		i = (i + 1) & m.mask
	}
	return 0, false
}

// slotForInsert returns the index of ek's slot, growing the table first
// when inserting would cross the load bound. The returned slot is occupied
// if ek was present, and empty (with key unset) when it is new.
func (m *engineKeyMap) slotForInsert(ek BlockHash) int {
	if (m.used+1)*tableMaxLoadDiv > len(m.slots)*tableMaxLoadTimes {
		m.grow()
	}
	i := int(tableHash(ek)) & m.mask //#nosec G115 -- the hash is masked to the slot count
	for m.slots[i].stamp != 0 {
		if m.slots[i].key == ek {
			return i
		}
		i = (i + 1) & m.mask
	}
	m.used++
	m.slots[i].key = ek
	return i
}

// deleteAt removes the slot at i and backward-shifts the probe cluster so
// no tombstone is left: every slot after i whose probe path crosses i moves
// into the hole, and the chain's tail is emptied.
func (m *engineKeyMap) deleteAt(i int) {
	m.used--
	j := (i + 1) & m.mask
	for m.slots[j].stamp != 0 {
		home := int(tableHash(m.slots[j].key)) & m.mask //#nosec G115 -- the hash is masked to the slot count
		if (j-i)&m.mask <= (j-home)&m.mask {
			m.slots[i] = m.slots[j]
			i = j
		}
		j = (j + 1) & m.mask
	}
	m.slots[i] = engineSlot{}
}

// grow doubles the slot array and rehashes every occupied slot. The old
// array becomes garbage, so a growth step transiently holds both arrays.
func (m *engineKeyMap) grow() {
	old := m.slots
	m.slots = make([]engineSlot, len(old)*2)
	m.mask = len(m.slots) - 1
	m.used = 0
	for i := range old {
		if old[i].stamp == 0 {
			continue
		}
		j := int(tableHash(old[i].key)) & m.mask //#nosec G115 -- the hash is masked to the slot count
		for m.slots[j].stamp != 0 {
			j = (j + 1) & m.mask
		}
		m.slots[j] = old[i]
		m.used++
	}
}

// get returns the chain's last request key, refreshing the slot's stamp.
// The last element is what parent-hash resolution needs: the highest-index
// request key the engine key covers.
func (m *engineKeyMap) get(ek BlockHash) (BlockHash, bool) {
	i, ok := m.find(ek)
	if !ok {
		return EmptyBlockHash, false
	}
	m.slots[i].stamp = m.tick()
	last := m.slots[i].first
	for h := m.slots[i].next; h != 0; h = m.arena[h-1].next {
		last = m.arena[h-1].key
	}
	return last, true
}

// peek returns ek's full chain without refreshing recency. The returned
// slice is a copy: the caller reads it after releasing its lock, and set
// and remove recycle the nodes it would alias.
func (m *engineKeyMap) peek(ek BlockHash) ([]BlockHash, bool) {
	i, ok := m.find(ek)
	if !ok {
		return nil, false
	}
	rks := []BlockHash{m.slots[i].first}
	for h := m.slots[i].next; h != 0; h = m.arena[h-1].next {
		rks = append(rks, m.arena[h-1].key)
	}
	return rks, true
}

// remove deletes ek's mapping.
func (m *engineKeyMap) remove(ek BlockHash) {
	if i, ok := m.find(ek); ok {
		m.freeChain(m.slots[i].next)
		m.deleteAt(i)
		m.count.Add(-1)
	}
}

// contains reports whether ek is mapped.
func (m *engineKeyMap) contains(ek BlockHash) bool {
	_, ok := m.find(ek)
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

// engineMappingSample names one sampled mapping for eviction decisions.
type engineMappingSample struct {
	key   BlockHash
	stamp uint32
}

// sampleMappings returns up to n mappings from a rotating cursor over the
// table, advancing the cursor past the scanned stretch.
func (m *engineKeyMap) sampleMappings(n int) []engineMappingSample {
	out := make([]engineMappingSample, 0, min(n, m.used))
	tableLen := len(m.slots)
	scanned := 0
	for len(out) < n && scanned < tableLen {
		i := (m.cursor + scanned) % tableLen
		scanned++
		if m.slots[i].stamp != 0 {
			out = append(out, engineMappingSample{key: m.slots[i].key, stamp: m.slots[i].stamp})
		}
	}
	m.cursor = (m.cursor + scanned) % tableLen
	return out
}

// evictOne removes the coldest-stamped slot among a sample.
func (m *engineKeyMap) evictOne() {
	sample := m.sampleMappings(evictSampleSize)
	if len(sample) == 0 {
		return
	}
	victim := sample[0]
	for _, s := range sample[1:] {
		if s.stamp < victim.stamp {
			victim = s
		}
	}
	if i, ok := m.find(victim.key); ok {
		m.freeChain(m.slots[i].next)
		m.deleteAt(i)
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
