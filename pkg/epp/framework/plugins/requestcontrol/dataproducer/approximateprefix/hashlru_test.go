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

import (
	"math/rand/v2"
	"strconv"
	"testing"
	"unsafe"

	"github.com/hashicorp/golang-lru/v2/simplelru"
)

func TestHashSlotIs16Bytes(t *testing.T) {
	if got := unsafe.Sizeof(hashSlot{}); got != 16 {
		t.Fatalf("hashSlot is %d bytes", got)
	}
}

func TestHashLRUAllocatesOnFirstAdd(t *testing.T) {
	c := newHashLRU(4)
	if c.slots != nil {
		t.Fatal("slots allocated before the first add")
	}
	if c.Contains(0) || c.Len() != 0 {
		t.Fatalf("empty cache len %d contains zero %v", c.Len(), c.Contains(0))
	}
	c.Add(0)
	if len(c.slots) != 8 {
		t.Fatalf("table length %d, want 8", len(c.slots))
	}
	if !c.Contains(0) || c.Len() != 1 {
		t.Fatalf("after add len %d contains zero %v", c.Len(), c.Contains(0))
	}
}

func TestHashLRUZeroCapacity(t *testing.T) {
	c := newHashLRU(0)
	c.Add(1)
	if c.Contains(1) || c.Len() != 0 {
		t.Fatalf("len %d contains %v", c.Len(), c.Contains(1))
	}
}

// TestHashLRUMatchesSimpleLRU checks membership and eviction order against
// simplelru for the same sequence of adds and lookups, including a block hash
// of zero and hashes that wrap the probe.
func TestHashLRUMatchesSimpleLRU(t *testing.T) {
	for _, capacity := range []int{1, 2, 3, 7, 32} {
		t.Run(strconv.Itoa(capacity), func(t *testing.T) {
			ref, err := simplelru.NewLRU[blockHash, struct{}](capacity, nil)
			if err != nil {
				t.Fatal(err)
			}
			got := newHashLRU(capacity)
			rng := rand.New(rand.NewPCG(1, uint64(capacity)))
			const space = 64
			for op := range 8000 {
				key := blockHash(rng.Uint64N(space))
				switch rng.IntN(8) {
				case 0:
					key = 0
				case 1:
					key = blockHash(rng.Uint64())
				}
				if rng.IntN(5) == 0 {
					if ref.Contains(key) != got.Contains(key) {
						t.Fatalf("op %d Contains(%d) ref %v got %v", op, key, ref.Contains(key), got.Contains(key))
					}
					assertSameOldest(t, op, ref, got)
					continue
				}
				ref.Add(key, struct{}{})
				got.Add(key)
				if ref.Len() != got.Len() {
					t.Fatalf("op %d len ref %d got %d", op, ref.Len(), got.Len())
				}
				assertSameOldest(t, op, ref, got)
				for i := range space {
					k := blockHash(i)
					if ref.Contains(k) != got.Contains(k) {
						t.Fatalf("op %d membership of %d ref %v got %v", op, k, ref.Contains(k), got.Contains(k))
					}
				}
				if key > space && ref.Contains(key) != got.Contains(key) {
					t.Fatalf("op %d membership of wide key %d", op, key)
				}
			}
		})
	}
}

func assertSameOldest(t *testing.T, op int, ref *simplelru.LRU[blockHash, struct{}], got *hashLRU) {
	t.Helper()
	want, _, wantOK := ref.GetOldest()
	have, haveOK := oldestKey(got)
	if wantOK != haveOK || want != have {
		t.Fatalf("op %d oldest ref %d (%v) got %d (%v)", op, want, wantOK, have, haveOK)
	}
}

func oldestKey(c *hashLRU) (blockHash, bool) {
	if c.n == 0 || c.tail == idxNil {
		return 0, false
	}
	return c.slots[c.tail].key, true
}

func BenchmarkHashLRUContainsHit(b *testing.B) {
	for _, impl := range []struct {
		name string
		new  func() func(blockHash) bool
	}{
		{
			name: "simplelru",
			new: func() func(blockHash) bool {
				lru, err := simplelru.NewLRU[blockHash, struct{}](benchmarkHashLRUCapacity, nil)
				if err != nil {
					b.Fatal(err)
				}
				for i := range benchmarkHashLRUCapacity {
					lru.Add(benchmarkHashKey(i), struct{}{})
				}
				return lru.Contains
			},
		},
		{
			name: "compact",
			new: func() func(blockHash) bool {
				lru := newHashLRU(benchmarkHashLRUCapacity)
				for i := range benchmarkHashLRUCapacity {
					lru.Add(benchmarkHashKey(i))
				}
				return lru.Contains
			},
		},
	} {
		b.Run(impl.name, func(b *testing.B) {
			contains := impl.new()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if !contains(benchmarkHashKey(i % benchmarkHashLRUCapacity)) {
					b.Fatal("cache miss")
				}
			}
		})
	}
}

func BenchmarkHashLRUReplaceAfterFull(b *testing.B) {
	for _, impl := range []struct {
		name string
		new  func() func(blockHash)
	}{
		{
			name: "simplelru",
			new: func() func(blockHash) {
				lru, err := simplelru.NewLRU[blockHash, struct{}](benchmarkHashLRUCapacity, nil)
				if err != nil {
					b.Fatal(err)
				}
				for i := range benchmarkHashLRUCapacity {
					lru.Add(benchmarkHashKey(i), struct{}{})
				}
				return func(key blockHash) { lru.Add(key, struct{}{}) }
			},
		},
		{
			name: "compact",
			new: func() func(blockHash) {
				lru := newHashLRU(benchmarkHashLRUCapacity)
				for i := range benchmarkHashLRUCapacity {
					lru.Add(benchmarkHashKey(i))
				}
				return lru.Add
			},
		},
	} {
		b.Run(impl.name, func(b *testing.B) {
			add := impl.new()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				add(benchmarkHashKey(benchmarkHashLRUCapacity + i))
			}
		})
	}
}

const benchmarkHashLRUCapacity = 31250

func benchmarkHashKey(i int) blockHash {
	return blockHash(uint64(i+1) * 0x9E3779B97F4A7C15)
}
