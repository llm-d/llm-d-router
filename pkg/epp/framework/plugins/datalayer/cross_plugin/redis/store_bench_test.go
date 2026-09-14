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

package redis

import (
	"context"
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"
)

// These benchmarks run against miniredis, an in-process Redis simulator. Any
// case that touches the server therefore measures our code plus a simulator,
// with no network and no real Redis engine. The results are useful for relative
// comparisons and allocation counts, not production latency estimates.
//
// The cached Get benchmarks never reach the server after warm-up. They measure
// the store lookup performed once per scored endpoint per request.

func newBenchStore(b *testing.B, replicas int) (*RedisStateStore, *miniredis.Miniredis) {
	b.Helper()
	server, err := miniredis.Run()
	if err != nil {
		b.Fatalf("miniredis: %v", err)
	}
	b.Cleanup(server.Close)

	client := goredis.NewClient(&goredis.Options{Addr: server.Addr()})
	b.Cleanup(func() { _ = client.Close() })

	store := &RedisStateStore{
		replicaID: testReplicaID,
		client:    client,
		ttl:       testTTL,
	}

	now := time.Now()
	for i := 1; i < replicas; i++ {
		data, err := encodeStamped(i, now)
		if err != nil {
			b.Fatalf("encode: %v", err)
		}
		server.HSet(string(testStateKey)+":"+testEndpointID, fmt.Sprintf("epp-%d", i), string(data))
	}
	return store, server
}

func warm(b *testing.B, store *RedisStateStore) {
	b.Helper()
	if err := store.Set(context.Background(), testStateKey, testEndpointID, 0, sumInts); err != nil {
		b.Fatalf("warm-up Set: %v", err)
	}
	if _, ok, err := store.Get(context.Background(), testStateKey, testEndpointID); err != nil || !ok {
		b.Fatalf("warm-up Get: ok=%v err=%v", ok, err)
	}
}

func BenchmarkGetCached(b *testing.B) {
	for _, replicas := range []int{1, 3, 8, 16} {
		b.Run(fmt.Sprintf("replicas=%d", replicas), func(b *testing.B) {
			store, _ := newBenchStore(b, replicas)
			warm(b, store)
			ctx := context.Background()

			b.ReportAllocs()
			b.ResetTimer()
			var result any
			for range b.N {
				v, ok, err := store.Get(ctx, testStateKey, testEndpointID)
				if err != nil {
					b.Fatal(err)
				}
				if !ok {
					b.Fatal("cached value disappeared")
				}
				result = v
			}
			runtime.KeepAlive(result)
		})
	}
}

func BenchmarkGetCachedParallel(b *testing.B) {
	store, _ := newBenchStore(b, 3)
	warm(b, store)

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		var result any
		for pb.Next() {
			v, ok, err := store.Get(ctx, testStateKey, testEndpointID)
			if err != nil {
				b.Error(err)
				return
			}
			if !ok {
				b.Error("cached value disappeared")
				return
			}
			result = v
		}
		runtime.KeepAlive(result)
	})
}

func BenchmarkSet(b *testing.B) {
	for _, replicas := range []int{1, 3, 8, 16} {
		b.Run(fmt.Sprintf("replicas=%d", replicas), func(b *testing.B) {
			store, _ := newBenchStore(b, replicas)
			ctx := context.Background()

			b.ReportAllocs()
			b.ResetTimer()
			for i := range b.N {
				if err := store.Set(ctx, testStateKey, testEndpointID, i, sumInts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkGetOrSet(b *testing.B) {
	b.Run("contended", func(b *testing.B) {
		store, _ := newBenchStore(b, 0)
		ctx := context.Background()
		if _, existed, err := store.GetOrSet(ctx, testStateKey, "req", "winner"); err != nil {
			b.Fatal(err)
		} else if existed {
			b.Fatal("coordination key already existed")
		}

		b.ReportAllocs()
		b.ResetTimer()
		var result any
		for range b.N {
			v, existed, err := store.GetOrSet(ctx, testStateKey, "req", "loser")
			if err != nil {
				b.Fatal(err)
			}
			if !existed {
				b.Fatal("coordination key disappeared")
			}
			result = v
		}
		runtime.KeepAlive(result)
	})

	b.Run("uncontended", func(b *testing.B) {
		store, _ := newBenchStore(b, 0)
		ctx := context.Background()

		b.ReportAllocs()
		b.ResetTimer()
		var result any
		for i := range b.N {
			v, existed, err := store.GetOrSet(ctx, testStateKey, fmt.Sprintf("req-%d", i), "candidate")
			if err != nil {
				b.Fatal(err)
			}
			if existed {
				b.Fatal("fresh coordination key already existed")
			}
			result = v
		}
		runtime.KeepAlive(result)
	})
}
