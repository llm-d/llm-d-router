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

package kvblock

import (
	"errors"
	"math"
	"math/rand/v2"
	"sync"
)

const (
	engineKeyTableMinIndex = 16
	noEntry                = int32(-1)
)

// engineKeyTable maps engine keys to their request keys, bounded by entry
// count with least-recently-used eviction. It holds no pointer per entry: at
// SGLang page size 1 every cached token is an engine key, and a pointer-based
// LRU costs about 140 B per key.
//
// Entries live densely in a slab linked into the recency list by int32
// position. index is an open-addressing table with linear probing that holds
// slab position + 1, so 0 marks an empty slot; removal uses backward-shift
// deletion, so there are no tombstones. An engine key with more than one
// request key also has its full list in multi.
type engineKeyTable struct {
	mu      sync.Mutex
	size    int
	seed    uint64
	index   []int32
	entries []engineKeyEntry
	// head is the most recently used entry, tail the least.
	head, tail int32
	multi      map[BlockHash][]BlockHash
}

type engineKeyEntry struct {
	engineKey  BlockHash
	requestKey BlockHash // last request key
	prev, next int32
}

func newEngineKeyTable(size int) (*engineKeyTable, error) {
	if size <= 0 || size > math.MaxInt32 {
		return nil, errors.New("engine key table size must be in [1, MaxInt32]")
	}
	return &engineKeyTable{
		size:  size,
		seed:  rand.Uint64(), //#nosec G404 -- spreads keys across slots, not a secret
		head:  noEntry,
		tail:  noEntry,
		multi: make(map[BlockHash][]BlockHash),
	}, nil
}

// Add maps engineKey to requestKeys and marks it most recently used,
// evicting the least recently used key when the table is full. The table
// retains requestKeys. An empty requestKeys removes engineKey.
func (t *engineKeyTable) Add(engineKey BlockHash, requestKeys []BlockHash) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(requestKeys) == 0 {
		t.removeLocked(engineKey)
		return
	}
	if _, pos := t.find(engineKey); pos != noEntry {
		t.entries[pos].requestKey = requestKeys[len(requestKeys)-1]
		t.setMulti(engineKey, requestKeys)
		t.moveToFront(pos)
		return
	}

	if len(t.entries) == t.size {
		t.removeLocked(t.entries[t.tail].engineKey)
	}
	if (len(t.entries)+1)*4 > len(t.index)*3 {
		t.growIndex()
	}
	if len(t.entries) == cap(t.entries) {
		grown := make([]engineKeyEntry, len(t.entries), min(max(2*cap(t.entries), engineKeyTableMinIndex), t.size))
		copy(grown, t.entries)
		t.entries = grown
	}

	pos := int32(len(t.entries)) //#nosec G115 -- len(entries) <= size <= MaxInt32
	t.entries = append(t.entries, engineKeyEntry{
		engineKey:  engineKey,
		requestKey: requestKeys[len(requestKeys)-1],
		prev:       noEntry,
		next:       noEntry,
	})
	slot, _ := t.find(engineKey)
	t.index[slot] = pos + 1
	t.pushFront(pos)
	t.setMulti(engineKey, requestKeys)
}

// Get returns the last of engineKey's request keys and marks it most
// recently used.
func (t *engineKeyTable) Get(engineKey BlockHash) (BlockHash, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	_, pos := t.find(engineKey)
	if pos == noEntry {
		return EmptyBlockHash, false
	}
	t.moveToFront(pos)
	return t.entries[pos].requestKey, true
}

// Peek returns engineKey's request keys without touching recency.
func (t *engineKeyTable) Peek(engineKey BlockHash) ([]BlockHash, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	_, pos := t.find(engineKey)
	if pos == noEntry {
		return nil, false
	}
	if rks, ok := t.multi[engineKey]; ok {
		return rks, true
	}
	return []BlockHash{t.entries[pos].requestKey}, true
}

// Remove deletes engineKey and reports whether it was present.
func (t *engineKeyTable) Remove(engineKey BlockHash) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.removeLocked(engineKey)
}

func (t *engineKeyTable) home(engineKey BlockHash) int {
	// fmix64 finalizer from MurmurHash3, keyed by a per-table seed.
	x := uint64(engineKey) ^ t.seed
	x ^= x >> 33
	x *= 0xff51afd7ed558ccd
	x ^= x >> 33
	x *= 0xc4ceb9fe1a85ec53
	x ^= x >> 33
	return int(x & uint64(len(t.index)-1)) //#nosec G115 -- masked to a slot index
}

// find returns engineKey's slot and slab position, or the empty slot where it
// would go and noEntry. With no index allocated it returns -1, noEntry.
func (t *engineKeyTable) find(engineKey BlockHash) (int, int32) {
	if len(t.index) == 0 {
		return -1, noEntry
	}
	mask := len(t.index) - 1
	for slot := t.home(engineKey); ; slot = (slot + 1) & mask {
		v := t.index[slot]
		if v == 0 {
			return slot, noEntry
		}
		if t.entries[v-1].engineKey == engineKey {
			return slot, v - 1
		}
	}
}

func (t *engineKeyTable) growIndex() {
	t.index = make([]int32, max(2*len(t.index), engineKeyTableMinIndex))
	mask := len(t.index) - 1
	for i := range t.entries {
		slot := t.home(t.entries[i].engineKey)
		for t.index[slot] != 0 {
			slot = (slot + 1) & mask
		}
		t.index[slot] = int32(i) + 1 //#nosec G115 -- i < len(entries) <= MaxInt32
	}
}

func (t *engineKeyTable) setMulti(engineKey BlockHash, requestKeys []BlockHash) {
	if len(requestKeys) > 1 {
		t.multi[engineKey] = requestKeys
	} else if len(t.multi) > 0 {
		delete(t.multi, engineKey)
	}
}

func (t *engineKeyTable) removeLocked(engineKey BlockHash) bool {
	slot, pos := t.find(engineKey)
	if pos == noEntry {
		return false
	}
	t.deleteSlot(slot)
	t.unlink(pos)
	if len(t.multi) > 0 {
		delete(t.multi, engineKey)
	}

	// Keep the slab dense: move the last entry into the freed position.
	last := int32(len(t.entries) - 1) //#nosec G115 -- len(entries) <= MaxInt32
	if pos != last {
		moved := t.entries[last]
		t.entries[pos] = moved
		if moved.prev != noEntry {
			t.entries[moved.prev].next = pos
		} else {
			t.head = pos
		}
		if moved.next != noEntry {
			t.entries[moved.next].prev = pos
		} else {
			t.tail = pos
		}
		movedSlot, _ := t.find(moved.engineKey)
		t.index[movedSlot] = pos + 1
	}
	t.entries = t.entries[:last]
	return true
}

// deleteSlot empties slot and shifts later members of its probe run back so
// every remaining key stays reachable from its home slot.
func (t *engineKeyTable) deleteSlot(slot int) {
	mask := len(t.index) - 1
	for next := (slot + 1) & mask; t.index[next] != 0; next = (next + 1) & mask {
		home := t.home(t.entries[t.index[next]-1].engineKey)
		// The key at next may fill slot unless its home lies in (slot, next].
		if (next-home)&mask >= (next-slot)&mask {
			t.index[slot] = t.index[next]
			slot = next
		}
	}
	t.index[slot] = 0
}

func (t *engineKeyTable) pushFront(pos int32) {
	t.entries[pos].prev = noEntry
	t.entries[pos].next = t.head
	if t.head != noEntry {
		t.entries[t.head].prev = pos
	}
	t.head = pos
	if t.tail == noEntry {
		t.tail = pos
	}
}

func (t *engineKeyTable) unlink(pos int32) {
	e := &t.entries[pos]
	if e.prev != noEntry {
		t.entries[e.prev].next = e.next
	} else {
		t.head = e.next
	}
	if e.next != noEntry {
		t.entries[e.next].prev = e.prev
	} else {
		t.tail = e.prev
	}
}

func (t *engineKeyTable) moveToFront(pos int32) {
	if t.head == pos {
		return
	}
	t.unlink(pos)
	t.pushFront(pos)
}
