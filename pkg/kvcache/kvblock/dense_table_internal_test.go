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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fillTable inserts keys 1..n with the given stamp and returns the table.
func fillTable(t *testing.T, n int, stamp uint32) *denseTable {
	t.Helper()
	tbl := newDenseTable(tableInitialSlots)
	for k := 1; k <= n; k++ {
		i := tbl.slotForInsert(BlockHash(k))
		tbl.slots[i].slot.stamp = stamp
	}
	return tbl
}

// assertTableInvariant checks every occupied slot is findable exactly once
// and the load bound holds.
func assertTableInvariant(t *testing.T, tbl *denseTable) {
	t.Helper()
	seen := map[BlockHash]int{}
	for i := range tbl.slots {
		if tbl.empty(i) {
			continue
		}
		key := tbl.slots[i].key
		assert.NotEqual(t, BlockHash(0), tbl.slots[i].slot.stamp,
			"occupied slot %d must carry a stamp", i)
		j, ok := tbl.find(key)
		require.True(t, ok, "occupied key %d must be findable", key)
		assert.Equal(t, i, j, "find must locate key %d at its stored slot", key)
		assert.NotContains(t, seen, key, "key %d stored twice", key)
		seen[key] = i
	}
	assert.Equal(t, len(seen), tbl.used, "used must count occupied slots")
	assert.LessOrEqual(t, tbl.used*tableMaxLoadDiv, len(tbl.slots)*tableMaxLoadTimes,
		"load factor must stay at or under the bound")
}

func TestDenseTableInsertFindDelete(t *testing.T) {
	tbl := fillTable(t, 100, 1)
	assertTableInvariant(t, tbl)
	assert.Equal(t, 100, tbl.used)

	// Deleting keys in scattered order keeps every survivor findable;
	// backward-shift deletion must repair each probe cluster.
	for _, k := range []int{50, 1, 100, 37, 38, 36, 99, 2} {
		i, ok := tbl.find(BlockHash(k))
		require.True(t, ok)
		tbl.deleteAt(i)
		_, ok = tbl.find(BlockHash(k))
		assert.False(t, ok, "deleted key %d must be gone", k)
		assertTableInvariant(t, tbl)
	}
	assert.Equal(t, 92, tbl.used)

	// A deleted key can be reinserted and found again.
	i := tbl.slotForInsert(BlockHash(50))
	tbl.slots[i].slot.stamp = 2
	assertTableInvariant(t, tbl)
}

func TestDenseTableGrowthPreservesKeys(t *testing.T) {
	// Insert enough keys to cross several growth steps.
	const n = 5000
	tbl := fillTable(t, n, 1)
	assertTableInvariant(t, tbl)
	assert.Greater(t, len(tbl.slots), tableInitialSlots, "the table must have grown")
	for k := 1; k <= n; k++ {
		_, ok := tbl.find(BlockHash(k))
		assert.True(t, ok, "key %d must survive growth", k)
	}
	// The load bound holds after growth, so the next insertion does not
	// grow again.
	before := len(tbl.slots)
	tbl.slotForInsert(BlockHash(n + 1))
	assert.Equal(t, before, len(tbl.slots))
}

func TestDenseTableZeroKeyIsStorable(t *testing.T) {
	// The empty marker is the stamp, not the key: BlockHash(0) is a legal
	// key that must round-trip.
	tbl := newDenseTable(tableInitialSlots)
	i := tbl.slotForInsert(0)
	tbl.slots[i].slot.stamp = 1
	j, ok := tbl.find(0)
	require.True(t, ok)
	assert.Equal(t, i, j)
	tbl.deleteAt(j)
	_, ok = tbl.find(0)
	assert.False(t, ok)
}

func TestDenseTableBackwardShiftAtHighLoad(t *testing.T) {
	// A 16-slot table at 11 keys sits just under the growth bound, forcing
	// long probe clusters whose deletion repairs can cross the wraparound.
	tbl := &denseTable{slots: make([]tableSlot, 16)}
	tbl.mask = 15
	const n = 11
	for k := 1; k <= n; k++ {
		i := tbl.slotForInsert(BlockHash(k))
		tbl.slots[i].slot.stamp = 1
	}
	assertTableInvariant(t, tbl)

	for k := 1; k <= n; k++ {
		if i, ok := tbl.find(BlockHash(k)); ok {
			tbl.deleteAt(i)
		}
		assertTableInvariant(t, tbl)
	}
	assert.Equal(t, 0, tbl.used)
}
