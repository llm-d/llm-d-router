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
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"strconv"
	"testing"
	"unsafe"

	"github.com/hashicorp/golang-lru/v2/simplelru"
)

func TestHashNodeIs16Bytes(t *testing.T) {
	if got := unsafe.Sizeof(hashNode{}); got != 16 {
		t.Fatalf("hashNode is %d bytes", got)
	}
}

func TestHashLRUGrowsWithInserts(t *testing.T) {
	c := newHashLRU(1 << 30)
	c.Add(1)
	c.Add(2)
	if len(c.nodes) != 2 || c.Len() != 2 {
		t.Fatalf("nodes %d len %d, want 2", len(c.nodes), c.Len())
	}
}

func TestHashLRUCapsCapacityAtNodeIndexRange(t *testing.T) {
	want := int(min(uint64(math.MaxInt), uint64(nodeNil)))
	c := newHashLRU(math.MaxInt)
	if c.capacity != want {
		t.Fatalf("capacity %d, want %d", c.capacity, want)
	}
	c.Add(1)
	if !c.Contains(1) {
		t.Fatal("add into a capped cache missed")
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
// of zero.
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
	if c.tail == nodeNil {
		return 0, false
	}
	return c.nodes[c.tail].key, true
}

const benchmarkHashLRUCapacity = 31250

func benchmarkHashKey(i int) blockHash {
	return blockHash(uint64(i+1) * 0x9E3779B97F4A7C15)
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
			name: "hashLRU",
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
			name: "hashLRU",
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

// BenchmarkHashLRURetainedHeap reports the heap one full cache retains.
// MiB-per-100-endpoints is that measurement times 100, since each endpoint
// owns a separate cache.
func BenchmarkHashLRURetainedHeap(b *testing.B) {
	const endpoints = 100
	for _, capacity := range []int{benchmarkHashLRUCapacity, 72000} {
		b.Run(strconv.Itoa(capacity), func(b *testing.B) {
			keys := make([]blockHash, capacity)
			for i := range keys {
				keys[i] = benchmarkHashKey(i)
			}
			measure := func(b *testing.B, fill func() any) {
				var sum, n float64
				for b.Loop() {
					var before, after runtime.MemStats
					runtime.GC()
					runtime.ReadMemStats(&before)
					kept := fill()
					runtime.GC()
					runtime.ReadMemStats(&after)
					sum += float64(int64(after.HeapAlloc) - int64(before.HeapAlloc))
					n++
					runtime.KeepAlive(kept)
				}
				retained := sum / n
				b.ReportMetric(retained/float64(capacity), "bytes-per-hash")
				b.ReportMetric(retained/(1<<20), "MiB-per-endpoint")
				b.ReportMetric(retained*endpoints/(1<<20), "MiB-per-100-endpoints")
			}
			b.Run("simplelru", func(b *testing.B) {
				measure(b, func() any {
					lru, err := simplelru.NewLRU[blockHash, struct{}](capacity, nil)
					if err != nil {
						b.Fatal(err)
					}
					for _, key := range keys {
						lru.Add(key, struct{}{})
					}
					return lru
				})
			})
			b.Run("hashLRU", func(b *testing.B) {
				measure(b, func() any {
					lru := newHashLRU(capacity)
					for _, key := range keys {
						lru.Add(key)
					}
					return lru
				})
			})
		})
	}
}

// keysByRecency returns the cached hashes from most to least recent.
func keysByRecency(c *hashLRU) []blockHash {
	var keys []blockHash
	for i := c.head; i != nodeNil; i = c.nodes[i].next {
		keys = append(keys, c.nodes[i].key)
	}
	return keys
}

func TestHashLRUResizeGrowKeepsEntries(t *testing.T) {
	c := newHashLRU(3)
	for k := range blockHash(5) {
		c.Add(k) // 2, 3, 4 remain
	}
	c.Resize(5)
	c.Add(5)
	c.Add(6)
	if got, want := keysByRecency(c), []blockHash{6, 5, 4, 3, 2}; !slices.Equal(got, want) {
		t.Fatalf("after grow: %v, want %v", got, want)
	}
	c.Add(7)
	if c.Contains(2) || c.Len() != 5 {
		t.Fatalf("grown cache should evict its oldest entry at the new capacity: len %d contains(2) %v", c.Len(), c.Contains(2))
	}
}

func TestHashLRUResizeShrinkKeepsMostRecent(t *testing.T) {
	c := newHashLRU(5)
	for k := range blockHash(5) {
		c.Add(k)
	}
	c.Add(1) // refresh: recency is 1, 4, 3, 2, 0
	c.Resize(3)
	if got, want := keysByRecency(c), []blockHash{1, 4, 3}; !slices.Equal(got, want) {
		t.Fatalf("after shrink: %v, want %v", got, want)
	}
	if len(c.nodes) != 3 {
		t.Fatalf("shrink should compact nodes: %d, want 3", len(c.nodes))
	}
	c.Add(9)
	if got, want := keysByRecency(c), []blockHash{9, 1, 4}; !slices.Equal(got, want) {
		t.Fatalf("add after shrink: %v, want %v", got, want)
	}
}

func TestHashLRUResizeShrinkAboveLenOnlyLowersLimit(t *testing.T) {
	c := newHashLRU(100)
	c.Add(1)
	c.Add(2)
	c.Resize(3)
	if c.Cap() != 3 || c.Len() != 2 {
		t.Fatalf("cap %d len %d, want 3 and 2", c.Cap(), c.Len())
	}
	c.Add(3)
	c.Add(4)
	if got, want := keysByRecency(c), []blockHash{4, 3, 2}; !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
