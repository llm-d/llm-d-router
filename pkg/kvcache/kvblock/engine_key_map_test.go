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
	"os"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setGetPeek checks the chain shape one geometry produces: 1:1 and many:1
// map many engine keys onto single request keys, 1:many maps one engine key
// onto a chain. get must return the chain's last element, peek the whole
// chain.
func TestEngineKeyMapChainShapes(t *testing.T) {
	req := func(n int) []BlockHash {
		keys := make([]BlockHash, n)
		for i := range keys {
			keys[i] = BlockHash(i + 1)
		}
		return keys
	}

	tests := []struct {
		name     string
		engine   []BlockHash
		requests []BlockHash
	}{
		{name: "1:1", engine: []BlockHash{100, 101, 102}, requests: req(3)},
		{name: "many:1", engine: []BlockHash{100, 101, 102, 103}, requests: req(1)},
		{name: "1:many", engine: []BlockHash{100}, requests: req(4)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := newEngineKeyMap(0)
			mappings := engineToRequestMapping(tt.engine, tt.requests)
			for ek, rks := range mappings {
				m.set(ek, rks)
			}

			for ek, rks := range mappings {
				last, ok := m.get(ek)
				require.True(t, ok, "engine key %d must resolve", ek)
				assert.Equal(t, rks[len(rks)-1], last, "get must return the chain's last element")

				chain, ok := m.peek(ek)
				require.True(t, ok)
				assert.Equal(t, rks, chain, "peek must return the whole chain in order")
			}
			assert.Equal(t, len(mappings), m.len())
		})
	}
}

// A set of an already-mapped engine key replaces the chain: the old chain's
// arena nodes are not part of the new chain, and the mapping count does not
// grow.
func TestEngineKeyMapSetReplacesChain(t *testing.T) {
	m := newEngineKeyMap(0)
	m.set(1, []BlockHash{10, 11, 12})
	arenaBefore := len(m.arena)
	require.Equal(t, 2, arenaBefore, "a 3-key chain holds 2 nodes past the inline first")

	m.set(1, []BlockHash{20})
	chain, ok := m.peek(1)
	require.True(t, ok)
	assert.Equal(t, []BlockHash{20}, chain)
	assert.Equal(t, 1, m.len())
	assert.Len(t, m.free, arenaBefore, "the replaced chain's nodes must return to the free list")
}

// get refreshes the last-visit stamp and peek does not, which is what the
// count-cap eviction orders by.
func TestEngineKeyMapStampSemantics(t *testing.T) {
	m := newEngineKeyMap(0)
	m.set(1, []BlockHash{10})
	m.set(2, []BlockHash{20})
	i1, ok := m.find(1)
	require.True(t, ok)
	stamp1 := m.slots[i1].stamp

	_, ok = m.peek(1)
	require.True(t, ok)
	assert.Equal(t, stamp1, m.slots[i1].stamp, "peek must not refresh the stamp")

	_, ok = m.get(1)
	require.True(t, ok)
	assert.Greater(t, m.slots[i1].stamp, stamp1, "get must refresh the stamp")
}

// The count cap evicts the coldest-stamped mapping. The sample covers the
// whole map at this size, so the choice is deterministic.
func TestEngineKeyMapCountCapEvictsColdest(t *testing.T) {
	m := newEngineKeyMap(3)
	m.set(1, []BlockHash{10}) // stamp 1
	m.set(2, []BlockHash{20}) // stamp 2
	m.set(3, []BlockHash{30}) // stamp 3
	_, ok := m.get(1)         // stamp 4: key 1 is now the most recent
	require.True(t, ok)

	m.set(4, []BlockHash{40}) // at the cap, evicts the coldest: key 2

	assert.Equal(t, 3, m.len())
	assert.True(t, m.contains(1), "the refreshed key must survive")
	assert.True(t, m.contains(3))
	assert.True(t, m.contains(4))
	assert.False(t, m.contains(2), "the coldest-stamped key must be evicted")
}

// remove deletes the mapping and recycles its chain; an absent key is a
// no-op.
func TestEngineKeyMapRemove(t *testing.T) {
	m := newEngineKeyMap(0)
	m.set(1, []BlockHash{10, 11})
	arenaBefore := len(m.arena)

	m.remove(1)
	assert.False(t, m.contains(1))
	assert.Equal(t, 0, m.len())
	assert.Len(t, m.free, arenaBefore)

	m.remove(999) // no-op
	assert.Equal(t, 0, m.len())
}

// Chains longer than one request key keep resolving after earlier elements
// were freed by a replacement, so recycled arena nodes cannot alias a live
// chain.
func TestEngineKeyMapRecycledNodesDoNotAlias(t *testing.T) {
	m := newEngineKeyMap(0)
	m.set(1, []BlockHash{10, 11, 12})
	m.set(2, []BlockHash{20, 21, 22})

	m.set(1, []BlockHash{30}) // frees 1's two nodes
	m.set(3, []BlockHash{40, 41, 42})

	chain, ok := m.peek(2)
	require.True(t, ok)
	assert.Equal(t, []BlockHash{20, 21, 22}, chain)
	chain, ok = m.peek(3)
	require.True(t, ok)
	assert.Equal(t, []BlockHash{40, 41, 42}, chain)
	last, ok := m.get(3)
	require.True(t, ok)
	assert.Equal(t, BlockHash(42), last)
}

// A mapping whose set carried no request keys is not stored, matching the
// previous map's absence of empty-slice values.
func TestEngineKeyMapSetEmptyIsNoop(t *testing.T) {
	m := newEngineKeyMap(0)
	m.set(1, nil)
	assert.Equal(t, 0, m.len())
	_, ok := m.get(1)
	assert.False(t, ok)
}

// readHeap returns live heap stats after a forced collection.
func readHeap(t *testing.T) runtime.MemStats {
	t.Helper()
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return ms
}

// TestEngineKeyMapResidency reports the structure's steady-state bytes per
// mapping at the 1:1 geometry (no arena nodes). Runs only under
// KVINDEXER_SCALE_BENCH=1 alongside the other operating-point benches.
func TestEngineKeyMapResidency(t *testing.T) {
	if os.Getenv("KVINDEXER_SCALE_BENCH") != "1" {
		t.Skip("set KVINDEXER_SCALE_BENCH=1 to run the residency checks")
	}
	const mappings = 1 << 20
	before := readHeap(t)
	m := newEngineKeyMap(0)
	for i := 1; i <= mappings; i++ {
		m.set(BlockHash(i), []BlockHash{BlockHash(i)})
	}
	after := readHeap(t)
	runtime.KeepAlive(m)
	t.Logf("engine key map: %.1f B/mapping over %d mappings",
		float64(after.HeapAlloc-before.HeapAlloc)/float64(mappings), mappings)
}
