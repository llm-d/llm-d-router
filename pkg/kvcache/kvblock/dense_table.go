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

// The open-addressed table replacing the Go map of stage 1b: a power-of-two
// array of value-type slots with linear probing. Slot addressability is the
// point - it lets a reader record read evidence in a slot with one atomic OR
// and no write acquisition, which a Go map value cannot offer. Deletion is
// backward-shift (no tombstones), so the load factor tracks live keys only;
// the table grows by doubling when the load factor crosses tableMaxLoad,
// where the unsuccessful-search cost of linear probing is still bounded.

// tableMaxLoadTimes and tableMaxLoadDiv bound the load factor at 7/10. The
// unsuccessful-search probe count of linear probing grows as
// (1 + 1/(1-a)^2)/2, which is about 6 at this bound.
const (
	tableMaxLoadTimes = 7
	tableMaxLoadDiv   = 10
	// tableInitialSlots is the starting capacity; every doubling is a
	// wholesale rehash, so the table starts small and grows on demand.
	tableInitialSlots = 1 << 12
)

// tableSlot is one addressed table entry. A slot is empty exactly when its
// stamp is zero: stamps are wall seconds, nonzero for every occupied slot,
// and deletion zeroes the whole slot. readBits holds the lock-free read
// evidence: readers set it with an atomic OR while holding at most the read
// lock, and the sweep consumes it under the write lock with atomic ops, so
// the word is always accessed atomically.
type tableSlot struct {
	key  BlockHash
	slot keySlot
	// readBits is nonzero while the key was read since the sweep last
	// sampled its slot.
	readBits uint32
}

// denseTable is the open-addressed slot array. All methods assume the
// caller holds the index's lock; only readBits is safe for unlocked atomic
// access, by readers.
type denseTable struct {
	slots []tableSlot
	mask  int
	used  int
}

func newDenseTable() *denseTable {
	t := &denseTable{slots: make([]tableSlot, tableInitialSlots)}
	t.mask = len(t.slots) - 1
	return t
}

// tableHash mixes a block hash into a probe position. Block hashes are
// chained hash values, already close to uniform; the mix breaks any
// alignment with the power-of-two mask that low-bit patterns would cause.
func tableHash(key BlockHash) uint64 {
	x := uint64(key)
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// home returns key's probe start index.
func (t *denseTable) home(key BlockHash) int {
	return int(tableHash(key)) & t.mask //#nosec G115 -- the hash is masked to the slot count
}

// empty reports whether the slot at i is unoccupied.
func (t *denseTable) empty(i int) bool { return t.slots[i].slot.stamp == 0 }

// find returns the index of key's slot, and whether it is present.
func (t *denseTable) find(key BlockHash) (int, bool) {
	i := t.home(key)
	for !t.empty(i) {
		if t.slots[i].key == key {
			return i, true
		}
		i = (i + 1) & t.mask
	}
	return 0, false
}

// slotForInsert returns the index of key's slot, growing the table first
// when inserting would cross the load bound. The returned slot is occupied
// if key was present, and empty (with key unset) when it is new.
func (t *denseTable) slotForInsert(key BlockHash) int {
	if (t.used+1)*tableMaxLoadDiv > len(t.slots)*tableMaxLoadTimes {
		t.grow()
	}
	i := t.home(key)
	for !t.empty(i) {
		if t.slots[i].key == key {
			return i
		}
		i = (i + 1) & t.mask
	}
	t.used++
	t.slots[i].key = key
	return i
}

// deleteAt removes the slot at i and backward-shifts the probe cluster so
// no tombstone is left: every slot after i whose probe path crosses i moves
// into the hole, and the chain's tail is emptied.
func (t *denseTable) deleteAt(i int) {
	t.used--
	j := (i + 1) & t.mask
	for !t.empty(j) {
		home := t.home(t.slots[j].key)
		// The slot at j sits (j-home) mod m steps from its home. It can
		// fill the hole at i, (j-i) mod m steps back, exactly when its
		// probe distance covers the move.
		if (j-i)&t.mask <= (j-home)&t.mask {
			t.slots[i] = t.slots[j]
			i = j
		}
		j = (j + 1) & t.mask
	}
	t.slots[i] = tableSlot{}
}

// grow doubles the slot array and rehashes every occupied slot. The old
// array becomes garbage, so a growth step transiently holds both arrays.
func (t *denseTable) grow() {
	old := t.slots
	t.slots = make([]tableSlot, len(old)*2)
	t.mask = len(t.slots) - 1
	t.used = 0
	for i := range old {
		if old[i].slot.stamp == 0 {
			continue
		}
		j := t.home(old[i].key)
		for !t.empty(j) {
			j = (j + 1) & t.mask
		}
		t.slots[j] = old[i]
		t.used++
	}
}
