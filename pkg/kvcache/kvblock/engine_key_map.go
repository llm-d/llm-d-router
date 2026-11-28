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

// engineKeyMap holds the engine-to-request key mappings on a plain Go map:
// each mapping's first request key and recency stamp sit inline in its map
// slot, and a mapping with more than one request key (the 1:many geometry)
// keeps its remaining keys in a side map. The structure is internally
// lock-free; every caller holds the enclosing index's lock across all of
// its methods, except len, which reads an atomic count and is safe without
// it: the in-memory index reports residency gauges after releasing its
// lock.
//
// Preserved from the previous lru.Cache-backed map: get returns the last
// element of the request-key chain and refreshes recency; peek resolves
// without refreshing; removal is the caller's decision; the count cap
// evicts by last-visit recency. Recency is a coarse stamp rather than an
// ordered list, so eviction under the cap is approximate; the cap sits
// far above what the container's memory holds, so the approximation is
// unobservable in the default configuration.
type engineKeyMap struct {
	// slots maps an engine key to its first request key and stamp.
	slots map[BlockHash]engineSlot
	// multi holds the request keys past the slot's inline first, for
	// mappings with more than one request key. It stays nil in the 1:1 and
	// many:1 geometries.
	multi map[BlockHash][]BlockHash
	// limit caps the mapping count; a non-positive value is unlimited.
	limit int
	// clock is the last-visit stamp source, incremented per refresh.
	clock uint32
	// count is the live mapping count, maintained atomically so len can be
	// called without the enclosing index's lock.
	count atomic.Int64
}

// engineSlot is one mapping's map value.
type engineSlot struct {
	first BlockHash
	stamp uint32
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
	if _, exists := m.slots[ek]; exists {
		delete(m.multi, ek)
	} else {
		m.count.Add(1)
		if m.limit > 0 && len(m.slots) >= m.limit {
			m.evictOne()
		}
	}
	if len(rks) > 1 {
		if m.multi == nil {
			m.multi = make(map[BlockHash][]BlockHash)
		}
		m.multi[ek] = append([]BlockHash(nil), rks[1:]...)
	}
	m.slots[ek] = engineSlot{first: rks[0], stamp: m.tick()}
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
	if tail, ok := m.multi[ek]; ok {
		return tail[len(tail)-1], true
	}
	return slot.first, true
}

// peek returns ek's full chain without refreshing recency. The returned
// slice is a copy: the caller reads it after releasing its lock, and set
// and remove replace the tail it would alias.
func (m *engineKeyMap) peek(ek BlockHash) ([]BlockHash, bool) {
	slot, ok := m.slots[ek]
	if !ok {
		return nil, false
	}
	tail := m.multi[ek]
	rks := make([]BlockHash, 0, len(tail)+1)
	rks = append(rks, slot.first)
	rks = append(rks, tail...)
	return rks, true
}

// remove deletes ek's mapping.
func (m *engineKeyMap) remove(ek BlockHash) {
	if _, ok := m.slots[ek]; ok {
		delete(m.slots, ek)
		delete(m.multi, ek)
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
		delete(m.slots, victim)
		delete(m.multi, victim)
		m.count.Add(-1)
	}
}
