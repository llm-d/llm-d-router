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
	"math"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/hashicorp/golang-lru/v2/simplelru"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewEngineKeyTableRejectsInvalidSize(t *testing.T) {
	for _, size := range []int{0, -1, math.MaxInt32 + 1} {
		_, err := newEngineKeyTable(size)
		assert.Error(t, err, "size %d", size)
	}
}

func TestEngineKeyTableLookup(t *testing.T) {
	tests := []struct {
		name        string
		requestKeys []BlockHash
	}{
		{name: "one_request_key", requestKeys: []BlockHash{7}},
		{name: "many_request_keys", requestKeys: []BlockHash{7, 8, 9}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			table, err := newEngineKeyTable(8)
			require.NoError(t, err)
			table.Add(42, tc.requestKeys)

			last, ok := table.Get(42)
			require.True(t, ok)
			assert.Equal(t, tc.requestKeys[len(tc.requestKeys)-1], last)

			all, ok := table.Peek(42)
			require.True(t, ok)
			assert.Equal(t, tc.requestKeys, all)
		})
	}
}

func TestEngineKeyTableMissingKey(t *testing.T) {
	table, err := newEngineKeyTable(8)
	require.NoError(t, err)
	table.Add(1, []BlockHash{10})

	_, ok := table.Get(2)
	assert.False(t, ok)
	_, ok = table.Peek(2)
	assert.False(t, ok)
	assert.False(t, table.Remove(2))
}

func TestEngineKeyTableZeroIsAValidKey(t *testing.T) {
	table, err := newEngineKeyTable(8)
	require.NoError(t, err)
	_, ok := table.Get(0)
	require.False(t, ok)

	table.Add(0, []BlockHash{10})
	got, ok := table.Get(0)
	require.True(t, ok)
	assert.Equal(t, BlockHash(10), got)
	assert.True(t, table.Remove(0))
	_, ok = table.Get(0)
	assert.False(t, ok)
}

func TestEngineKeyTableAddReplacesRequestKeys(t *testing.T) {
	tests := []struct {
		name          string
		first, second []BlockHash
	}{
		{name: "one_to_one", first: []BlockHash{10}, second: []BlockHash{11}},
		{name: "one_to_many", first: []BlockHash{10}, second: []BlockHash{11, 12}},
		{name: "many_to_one", first: []BlockHash{10, 11}, second: []BlockHash{12}},
		{name: "many_to_many", first: []BlockHash{10, 11}, second: []BlockHash{12, 13, 14}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			table, err := newEngineKeyTable(8)
			require.NoError(t, err)
			table.Add(1, tc.first)
			table.Add(1, tc.second)

			all, ok := table.Peek(1)
			require.True(t, ok)
			assert.Equal(t, tc.second, all)
			last, ok := table.Get(1)
			require.True(t, ok)
			assert.Equal(t, tc.second[len(tc.second)-1], last)
		})
	}
}

func TestEngineKeyTableRemove(t *testing.T) {
	tests := []struct {
		name        string
		requestKeys []BlockHash
	}{
		{name: "one_request_key", requestKeys: []BlockHash{10}},
		{name: "many_request_keys", requestKeys: []BlockHash{10, 11}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			table, err := newEngineKeyTable(8)
			require.NoError(t, err)
			table.Add(1, tc.requestKeys)
			table.Add(2, []BlockHash{20})

			assert.True(t, table.Remove(1))
			assert.False(t, table.Remove(1))
			_, ok := table.Peek(1)
			assert.False(t, ok)
			got, ok := table.Get(2)
			require.True(t, ok)
			assert.Equal(t, BlockHash(20), got)
		})
	}
}

// TestEngineKeyTableEvictsLeastRecentlyUsed fills a two-entry table, applies
// one access to key 1, then adds key 3, and checks which key was evicted.
func TestEngineKeyTableEvictsLeastRecentlyUsed(t *testing.T) {
	const (
		touchNone = iota
		touchGet
		touchPeek
		touchAdd
		touchRemove
	)
	tests := []struct {
		name               string
		touch              int
		wantKey1, wantKey2 bool
	}{
		{name: "oldest_is_evicted", touch: touchNone, wantKey1: false, wantKey2: true},
		{name: "get_promotes", touch: touchGet, wantKey1: true, wantKey2: false},
		{name: "peek_does_not_promote", touch: touchPeek, wantKey1: false, wantKey2: true},
		{name: "re_add_promotes", touch: touchAdd, wantKey1: true, wantKey2: false},
		{name: "remove_frees_capacity", touch: touchRemove, wantKey1: false, wantKey2: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			table, err := newEngineKeyTable(2)
			require.NoError(t, err)
			table.Add(1, []BlockHash{10})
			table.Add(2, []BlockHash{20})

			switch tc.touch {
			case touchGet:
				_, ok := table.Get(1)
				require.True(t, ok)
			case touchPeek:
				_, ok := table.Peek(1)
				require.True(t, ok)
			case touchAdd:
				table.Add(1, []BlockHash{10})
			case touchRemove:
				require.True(t, table.Remove(1))
			}

			table.Add(3, []BlockHash{30})

			_, ok := table.Peek(1)
			assert.Equal(t, tc.wantKey1, ok, "key 1")
			_, ok = table.Peek(2)
			assert.Equal(t, tc.wantKey2, ok, "key 2")
			_, ok = table.Peek(3)
			assert.True(t, ok, "key 3")
		})
	}
}

// TestEngineKeyTableGrowsPastInitialCapacity adds far more keys than any
// small initial allocation holds, then removes every other key so deletion
// runs across long probe sequences.
func TestEngineKeyTableGrowsPastInitialCapacity(t *testing.T) {
	const n = 100_000
	table, err := newEngineKeyTable(1 << 20)
	require.NoError(t, err)
	key := func(i int) BlockHash { return BlockHash(uint64(i) * 0x9E3779B97F4A7C15) } //#nosec G115 -- test data
	for i := range n {
		table.Add(key(i), []BlockHash{BlockHash(i)}) //#nosec G115 -- test data
	}
	requireEngineKeyTableInvariants(t, table)
	for i := 0; i < n; i += 2 {
		require.True(t, table.Remove(key(i)), "remove %d", i)
	}
	requireEngineKeyTableInvariants(t, table)
	for i := range n {
		got, ok := table.Get(key(i))
		if i%2 == 0 {
			require.False(t, ok, "key %d should be removed", i)
			continue
		}
		require.True(t, ok, "key %d", i)
		require.Equal(t, BlockHash(i), got, "key %d", i) //#nosec G115 -- test data
	}
}

// requireEngineKeyTableInvariants checks the table's internal structure:
// the recency list, the index, and multi must describe the same live entries.
// It uses plain comparisons because the model tests call it after every op.
func requireEngineKeyTableInvariants(t *testing.T, table *engineKeyTable) {
	t.Helper()
	table.mu.Lock()
	defer table.mu.Unlock()

	n := len(table.entries)
	if cap(table.entries) > table.size {
		t.Fatalf("slab capacity %d exceeds size %d", cap(table.entries), table.size)
	}

	// The list visits every entry once, front to back, with matching back links.
	prev := noEntry
	visited := 0
	for pos := table.head; pos != noEntry; pos = table.entries[pos].next {
		if visited == n {
			t.Fatalf("recency list is longer than the %d entries", n)
		}
		if pos < 0 || int(pos) >= n {
			t.Fatalf("list position %d out of range [0, %d)", pos, n)
		}
		if table.entries[pos].prev != prev {
			t.Fatalf("back link of position %d is %d, want %d", pos, table.entries[pos].prev, prev)
		}
		prev = pos
		visited++
	}
	if visited != n || table.tail != prev {
		t.Fatalf("list visits %d of %d entries and ends at %d, tail is %d", visited, n, prev, table.tail)
	}

	// Every index slot names a distinct live entry, and every entry is found
	// from its home slot.
	if n*4 > len(table.index)*3 {
		t.Fatalf("%d entries exceed the load factor of %d slots", n, len(table.index))
	}
	seen := make([]bool, n)
	occupied := 0
	for slot, v := range table.index {
		if v == 0 {
			continue
		}
		if int(v-1) >= n || seen[v-1] {
			t.Fatalf("slot %d names position %d, which is out of range or named twice", slot, v-1)
		}
		seen[v-1] = true
		occupied++
	}
	if occupied != n {
		t.Fatalf("%d occupied slots for %d entries", occupied, n)
	}
	for pos, e := range table.entries {
		if _, found := table.find(e.engineKey); int(found) != pos {
			t.Fatalf("find(%d) = %d, want %d", e.engineKey, found, pos)
		}
	}

	// multi holds only live keys with more than one request key, whose last
	// request key matches the entry.
	for ek, rks := range table.multi {
		_, pos := table.find(ek)
		if pos == noEntry || len(rks) < 2 || rks[len(rks)-1] != table.entries[pos].requestKey {
			t.Fatalf("multi entry %d = %v does not match a live multi-request-key entry", ek, rks)
		}
	}
}

const (
	opAdd = iota
	opGet
	opPeek
	opRemove
	numOps
)

type modelOp struct {
	kind        int
	key         BlockHash
	requestKeys []BlockHash
}

// collidingKey spreads k into the high bits so every key shares its low bits
// and keys share probe sequences under any hash that reads the low bits.
func collidingKey(k int) BlockHash {
	return BlockHash(uint64(k) << 40) //#nosec G115 -- test data, small non-negative ints
}

// requireMatchesLRUModel applies ops to an engineKeyTable and to simplelru,
// which backs the request-key store, and requires identical results after
// every op and for every key in [0, keySpace) at the end. It checks the
// table's internal structure every checkEvery ops and at the end.
func requireMatchesLRUModel(t *testing.T, size, keySpace, checkEvery int, ops []modelOp) {
	t.Helper()
	table, err := newEngineKeyTable(size)
	require.NoError(t, err)
	model, err := simplelru.NewLRU[BlockHash, []BlockHash](size, nil)
	require.NoError(t, err)

	for i, op := range ops {
		switch op.kind {
		case opAdd:
			table.Add(op.key, op.requestKeys)
			model.Add(op.key, op.requestKeys)
		case opGet:
			got, ok := table.Get(op.key)
			want, wantOK := model.Get(op.key)
			require.Equal(t, wantOK, ok, "op %d: Get(%d) presence", i, op.key)
			if wantOK {
				require.Equal(t, want[len(want)-1], got, "op %d: Get(%d)", i, op.key)
			}
		case opPeek:
			got, ok := table.Peek(op.key)
			want, wantOK := model.Peek(op.key)
			require.Equal(t, wantOK, ok, "op %d: Peek(%d) presence", i, op.key)
			require.Equal(t, want, got, "op %d: Peek(%d)", i, op.key)
		case opRemove:
			require.Equal(t, model.Remove(op.key), table.Remove(op.key), "op %d: Remove(%d)", i, op.key)
		}
		if i%checkEvery == 0 {
			requireEngineKeyTableInvariants(t, table)
		}
	}
	requireEngineKeyTableInvariants(t, table)
	for k := range keySpace {
		got, ok := table.Peek(collidingKey(k))
		want, wantOK := model.Peek(collidingKey(k))
		require.Equal(t, wantOK, ok, "final Peek(%d) presence", collidingKey(k))
		require.Equal(t, want, got, "final Peek(%d)", collidingKey(k))
	}
}

// TestEngineKeyTableMatchesLRUModel runs random operation sequences against
// the table and simplelru. A key space larger than the size keeps the table
// at its cap, so eviction, removal, and re-insertion interleave.
func TestEngineKeyTableMatchesLRUModel(t *testing.T) {
	tests := []struct {
		name            string
		size, keySpace  int
		ops, checkEvery int
	}{
		{name: "at_cap", size: 64, keySpace: 200, ops: 200_000, checkEvery: 7},
		// Starts empty and fills past any small initial allocation, so growth
		// happens before and while the table evicts at its cap.
		{name: "grows_then_evicts", size: 4096, keySpace: 12_000, ops: 200_000, checkEvery: 997},
		{name: "size_one", size: 1, keySpace: 4, ops: 10_000, checkEvery: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2)) //#nosec G404 -- test data
			ops := make([]modelOp, tc.ops)
			for i := range ops {
				ops[i] = modelOp{kind: rng.IntN(numOps), key: collidingKey(rng.IntN(tc.keySpace))}
				if ops[i].kind == opAdd {
					ops[i].requestKeys = make([]BlockHash, 1+rng.IntN(3))
					for j := range ops[i].requestKeys {
						ops[i].requestKeys[j] = BlockHash(rng.Uint64())
					}
				}
			}
			requireMatchesLRUModel(t, tc.size, tc.keySpace, tc.checkEvery, ops)
		})
	}
}

// FuzzEngineKeyTableMatchesLRUModel decodes the input into a table size and
// an operation sequence and checks it against simplelru. The first byte sets
// the size in [1, 8]; each following byte pair is one op on one of 16 keys.
func FuzzEngineKeyTableMatchesLRUModel(f *testing.F) {
	const keySpace = 16
	f.Add([]byte{1, 0, 1, 0, 2, 0, 3, 1, 1, 2, 2, 3, 1})
	f.Add([]byte{0, 0, 0, 0, 1, 1, 0, 3, 0})
	f.Add([]byte{7, 4, 0, 8, 1, 12, 2, 0, 3, 1, 4, 2, 5, 3, 6, 0, 7, 1, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}
		size := 1 + int(data[0]%8)
		var ops []modelOp
		for i := 1; i+1 < len(data); i += 2 {
			op := modelOp{kind: int(data[i] % numOps), key: collidingKey(int(data[i+1] % keySpace))}
			if op.kind == opAdd {
				op.requestKeys = make([]BlockHash, 1+int(data[i]>>2)%3)
				for j := range op.requestKeys {
					op.requestKeys[j] = BlockHash(uint64(i)<<8 | uint64(j)) //#nosec G115 -- test data, small non-negative ints
				}
			}
			ops = append(ops, op)
		}
		requireMatchesLRUModel(t, size, keySpace, 1, ops)
	})
}

func TestEngineKeyTableReAddAtCapDoesNotEvict(t *testing.T) {
	table, err := newEngineKeyTable(2)
	require.NoError(t, err)
	table.Add(1, []BlockHash{10})
	table.Add(2, []BlockHash{20})

	table.Add(1, []BlockHash{11, 12})

	got, ok := table.Peek(1)
	require.True(t, ok, "key 1")
	assert.Equal(t, []BlockHash{11, 12}, got)
	got, ok = table.Peek(2)
	require.True(t, ok, "key 2")
	assert.Equal(t, []BlockHash{20}, got)
}

// TestEngineKeyTableConcurrentAccess runs mixed operations from several
// goroutines; it checks for races under -race and that the table stays usable.
func TestEngineKeyTableConcurrentAccess(t *testing.T) {
	const (
		workers = 8
		ops     = 5_000
	)
	table, err := newEngineKeyTable(256)
	require.NoError(t, err)

	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(uint64(w), 0)) //#nosec G404 G115 -- test data
			for range ops {
				ek := BlockHash(rng.IntN(512)) //#nosec G115 -- test data
				switch rng.IntN(4) {
				case 0:
					table.Add(ek, []BlockHash{ek + 1})
				case 1:
					if got, ok := table.Get(ek); ok && got != ek+1 {
						t.Errorf("Get(%d) = %d, want %d", ek, got, ek+1)
					}
				case 2:
					table.Peek(ek)
				case 3:
					table.Remove(ek)
				}
			}
		}()
	}
	wg.Wait()

	table.Add(1, []BlockHash{2})
	got, ok := table.Get(1)
	require.True(t, ok)
	assert.Equal(t, BlockHash(2), got)
}
