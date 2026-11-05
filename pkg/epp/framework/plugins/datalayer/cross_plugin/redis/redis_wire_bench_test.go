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
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// benchPromptHashes is one 128k-token prompt at blockSizeTokens 64.
const benchPromptHashes = 2000

// benchWindowHashes are window sizes in block hashes. 32_000 is just above the
// default maxSyncedTokens budget, 2_000_000 tokens / 64 tokens per block.
var benchWindowHashes = []int{32_000, 72_000, 150_000, 300_000, 500_000}

type countingConn struct {
	net.Conn
	bytes *atomic.Int64
}

func (c countingConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	c.bytes.Add(int64(n))
	return n, err
}

func (c countingConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	c.bytes.Add(int64(n))
	return n, err
}

func newCountingSyncer(b *testing.B, server *miniredis.Miniredis, replicaID string, incremental bool, bytes *atomic.Int64) *RedisSyncer {
	b.Helper()
	client := goredis.NewClient(&goredis.Options{
		Addr: server.Addr(),
		Dialer: func(ctx context.Context, network, addr string) (net.Conn, error) {
			conn, err := (&net.Dialer{}).DialContext(ctx, network, addr)
			if err != nil {
				return nil, err
			}
			return countingConn{Conn: conn, bytes: bytes}, nil
		},
	})
	client.AddHook(&miniredisHExpireCompatibilityHook{})
	b.Cleanup(func() { _ = client.Close() })
	return &RedisSyncer{replicaID: replicaID, client: client, stateTTL: testStateTTL, incremental: incremental}
}

// benchWindow is a full window of benchPromptHashes-hash prompts whose hashes
// start at seed.
func benchWindow(hashes int, seed uint64) *attrprefix.PrefixHashSet {
	prompts := make([][]uint64, hashes/benchPromptHashes)
	for i := range prompts {
		prompts[i] = benchPrompt(seed + uint64(i*benchPromptHashes))
	}
	return &attrprefix.PrefixHashSet{Prompts: prompts}
}

// benchPrompt returns uniformly distributed hashes, as block hashes are, so
// gob's variable-length integers cost what they cost in production.
func benchPrompt(seed uint64) []uint64 {
	prompt := make([]uint64, benchPromptHashes)
	for j := range prompt {
		z := seed + uint64(j) + 0x9e3779b97f4a7c15
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		prompt[j] = z ^ (z >> 31)
	}
	return prompt
}

// slide drops the oldest prompt and appends a new one, as the recent window
// does when a request arrives at a full window.
func slide(w *attrprefix.PrefixHashSet, seed uint64) *attrprefix.PrefixHashSet {
	prompts := make([][]uint64, 0, len(w.Prompts))
	prompts = append(prompts, w.Prompts[1:]...)
	prompts = append(prompts, benchPrompt(seed))
	return &attrprefix.PrefixHashSet{Base: w.Base + 1, Prompts: prompts}
}

func benchSpec(windows []*attrprefix.PrefixHashSet, i int) fwkdl.CrossReplicaSpec {
	return fwkdl.CrossReplicaSpec{
		StateKey:  "approx-prefix:bench",
		Read:      func(string) fwkdl.Cloneable { return windows[i] },
		Aggregate: func(values []any) any { return values[0] },
	}
}

// BenchmarkSyncerWireBytesPerTick reports client wire bytes, CPU, and
// allocations for one publish tick of two replicas sharing one endpoint, with
// full and incremental window replication. Bytes scale linearly with the
// number of endpoints.
func BenchmarkSyncerWireBytesPerTick(b *testing.B) {
	for _, mode := range []string{windowReplicationFull, windowReplicationIncremental} {
		for _, hashes := range benchWindowHashes {
			for _, change := range []string{"unchanged", "one-128k-token-prompt"} {
				b.Run(fmt.Sprintf("mode=%s/hashes=%d/%s", mode, hashes, change), func(b *testing.B) {
					server := miniredis.RunT(b)
					var bytes atomic.Int64
					incremental := mode == windowReplicationIncremental
					syncers := []*RedisSyncer{
						newCountingSyncer(b, server, "epp-a", incremental, &bytes),
						newCountingSyncer(b, server, "epp-b", incremental, &bytes),
					}
					windows := []*attrprefix.PrefixHashSet{benchWindow(hashes, 0), benchWindow(hashes, 1<<40)}
					tick := func() {
						for i, s := range syncers {
							if err := s.Set(context.Background(), benchSpec(windows, i), testEndpointID); err != nil {
								b.Fatal(err)
							}
						}
					}
					tick()
					tick()
					seed := uint64(1 << 50)
					bytes.Store(0)
					b.ReportAllocs()
					for b.Loop() {
						if change != "unchanged" {
							for i := range windows {
								seed += benchPromptHashes
								windows[i] = slide(windows[i], seed)
							}
						}
						tick()
					}
					b.ReportMetric(float64(bytes.Load())/float64(b.N), "wireB/tick")
				})
			}
		}
	}
}

// BenchmarkSyncerColdWindow reports two ticks starting from an empty Redis.
// The replicas publish one after the other, so the first tick carries both
// windows out and one window back, and the second tick is the first where
// each replica can see the other. full mode sends the windows again on that
// second tick. incremental mode sends only what the lagging replica has not
// read.
func BenchmarkSyncerColdWindow(b *testing.B) {
	for _, mode := range []string{windowReplicationFull, windowReplicationIncremental} {
		for _, hashes := range benchWindowHashes {
			b.Run(fmt.Sprintf("mode=%s/hashes=%d", mode, hashes), func(b *testing.B) {
				server := miniredis.RunT(b)
				var bytes atomic.Int64
				incremental := mode == windowReplicationIncremental
				clients := []*goredis.Client{
					newCountingClient(b, server, &bytes),
					newCountingClient(b, server, &bytes),
				}
				windows := []*attrprefix.PrefixHashSet{benchWindow(hashes, 0), benchWindow(hashes, 1<<40)}
				var first, second atomic.Int64
				b.ReportAllocs()
				for b.Loop() {
					server.FlushAll()
					syncers := make([]*RedisSyncer, len(clients))
					for i, client := range clients {
						syncers[i] = &RedisSyncer{
							replicaID:     fmt.Sprintf("epp-%d", i),
							client:        client,
							stateTTL:      testStateTTL,
							incremental:   incremental,
							publishBudget: budgetOrNil(incremental),
							readBudget:    budgetOrNil(incremental),
						}
					}
					tick := func() {
						for i, s := range syncers {
							if err := s.Set(context.Background(), benchSpec(windows, i), testEndpointID); err != nil {
								b.Fatal(err)
							}
						}
					}
					bytes.Store(0)
					tick()
					first.Add(bytes.Load())
					bytes.Store(0)
					tick()
					second.Add(bytes.Load())
				}
				b.ReportMetric(float64(first.Load())/float64(b.N), "tick1B")
				b.ReportMetric(float64(second.Load())/float64(b.N), "tick2B")
			})
		}
	}
}

func budgetOrNil(incremental bool) *byteBudget {
	if !incremental {
		return nil
	}
	return newByteBudget(defaultMaxBytesPerSecond)
}

func newCountingClient(b *testing.B, server *miniredis.Miniredis, bytes *atomic.Int64) *goredis.Client {
	b.Helper()
	client := goredis.NewClient(&goredis.Options{
		Addr: server.Addr(),
		Dialer: func(ctx context.Context, network, addr string) (net.Conn, error) {
			conn, err := (&net.Dialer{}).DialContext(ctx, network, addr)
			if err != nil {
				return nil, err
			}
			return countingConn{Conn: conn, bytes: bytes}, nil
		},
	})
	client.AddHook(&miniredisHExpireCompatibilityHook{})
	b.Cleanup(func() { _ = client.Close() })
	return client
}

// BenchmarkSyncerColdCatchUp reports how a replica with full windows on every
// endpoint and a peer that has read nothing converge through SetBatch: ticks
// needed at a 50ms interval, total wire bytes, and the largest tick, under the
// default maxBytesPerSecond.
func BenchmarkSyncerColdCatchUp(b *testing.B) {
	const (
		interval  = 50 * time.Millisecond
		endpoints = 20
		hashes    = 500_000
	)
	for _, mode := range []string{windowReplicationFull, windowReplicationIncremental} {
		b.Run(fmt.Sprintf("mode=%s/endpoints=%d/hashes=%d", mode, endpoints, hashes), func(b *testing.B) {
			ids := make([]string, endpoints)
			windows := map[string]*attrprefix.PrefixHashSet{}
			for i := range ids {
				ids[i] = fmt.Sprintf("ns/pod-%d", i)
				windows[ids[i]] = benchWindow(hashes, uint64(i)<<40)
			}
			empty := &attrprefix.PrefixHashSet{}
			specs := []fwkdl.CrossReplicaSpec{
				{StateKey: "approx-prefix:bench", Read: func(id string) fwkdl.Cloneable { return windows[id] }, Aggregate: func(v []any) any { return v[0] }},
				{StateKey: "approx-prefix:bench", Read: func(string) fwkdl.Cloneable { return empty }, Aggregate: func(v []any) any { return v[0] }},
			}
			for b.Loop() {
				server := miniredis.RunT(b)
				var bytes atomic.Int64
				incremental := mode == windowReplicationIncremental
				a := newCountingSyncer(b, server, "epp-a", incremental, &bytes)
				peer := newCountingSyncer(b, server, "epp-b", incremental, &bytes)
				now := time.Unix(0, 0)
				clock := func() time.Time { return now }
				a.publishBudget = &byteBudget{rate: defaultMaxBytesPerSecond, tokens: defaultMaxBytesPerSecond, last: now, now: clock}
				peer.readBudget = &byteBudget{rate: defaultMaxBytesPerSecond, tokens: defaultMaxBytesPerSecond, last: now, now: clock}
				var ticks, largest int64
				for !caughtUp(peer, ids, windows) {
					before := bytes.Load()
					for i, s := range []*RedisSyncer{a, peer} {
						if errs := s.SetBatch(context.Background(), specs[i], ids, time.Minute); len(errs) > 0 {
							b.Fatal(errs)
						}
					}
					ticks++
					largest = max(largest, bytes.Load()-before)
					now = now.Add(interval)
				}
				b.ReportMetric(float64(ticks), "ticks")
				b.ReportMetric(float64(bytes.Load()), "wireB")
				b.ReportMetric(float64(largest), "maxWireB/tick")
			}
		})
	}
}

func caughtUp(peer *RedisSyncer, ids []string, windows map[string]*attrprefix.PrefixHashSet) bool {
	for _, id := range ids {
		v, ok := peer.cache.Load(peer.hashKey("approx-prefix:bench", id))
		if !ok || len(v.(*aggregateCacheEntry).value.(*attrprefix.PrefixHashSet).Prompts) != len(windows[id].Prompts) {
			return false
		}
	}
	return true
}
