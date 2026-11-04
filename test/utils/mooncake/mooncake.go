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

// Package mooncake turns the Mooncake FAST'25 arXiv trace into per-request
// block-key workloads for prefix indexer benchmarks. The parameters follow
// NVIDIA Dynamo's mooncake indexer benchmark, so results compare with its
// published block_ops/s figures.
//
// The trace is not vendored. Download mooncake_trace.jsonl from
// https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/arxiv-trace
// and set TraceEnv to its path; benchmarks skip when it is unset.
package mooncake

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/hashicorp/golang-lru/v2/simplelru"
)

// TraceEnv names the environment variable holding the trace path.
const TraceEnv = "MOONCAKE_TRACE_PATH"

const (
	// Workers is the number of simulated inference workers.
	Workers = 128
	// WorkerBlocks is each worker's KV cache capacity in blocks.
	WorkerBlocks = 16384

	// expandFactor is the blocks per trace hash ID: a trace block of 512
	// tokens at 128 tokens per block, times Dynamo's length factor of 4.
	expandFactor = 16
	// dupFactor replays the records this many times under distinct hash
	// chains, standing in for distinct users.
	dupFactor = 20
	// maxRecords caps the trace records read.
	maxRecords = 1024
)

// Turn is one request of the replay.
type Turn struct {
	// Worker is the index of the worker the request is routed to.
	Worker int
	// Query is the request's block keys in prefix order.
	Query []uint64
	// Stored is the keys the worker had to store to serve the request.
	Stored []uint64
	// Removed is the keys the worker evicted to make room for Stored.
	Removed []uint64
	// Warm is the subset of Stored still resident when the replay ends.
	Warm []uint64
}

// Replay is the trace expanded into turns.
type Replay struct {
	Turns []Turn
	// QueryBlocks is the mean len(Query) per turn.
	QueryBlocks float64
	// EventBlocks is the mean len(Query)+len(Stored)+len(Removed) per turn,
	// the block count Dynamo divides by when queries run with KV events.
	EventBlocks float64
}

var loadOnce = sync.OnceValues(func() (*Replay, error) {
	return Build(os.Getenv(TraceEnv))
})

// Load returns the replay of the trace at TraceEnv, built once per process,
// and skips tb when TraceEnv is unset.
func Load(tb testing.TB) *Replay {
	tb.Helper()
	if os.Getenv(TraceEnv) == "" {
		tb.Skipf("%s is not set", TraceEnv)
	}
	r, err := loadOnce()
	if err != nil {
		tb.Fatal(err)
	}
	return r
}

// Build reads the trace at path and expands it into turns.
//
// Each record's hash IDs expand into chained block keys, so two requests
// share key i exactly when they share hash IDs 0..i/expandFactor. A request
// goes to a worker chosen by a hash of its first three quarters of hash IDs.
// Each worker's cache is simulated as an LRU of WorkerBlocks keys, touched in
// prefix order, to derive Stored, Removed and Warm.
func Build(path string) (*Replay, error) {
	records, err := readRecords(path)
	if err != nil {
		return nil, err
	}

	var removed []uint64
	caches := make([]*simplelru.LRU[uint64, struct{}], Workers)
	for w := range caches {
		caches[w], err = simplelru.NewLRU(WorkerBlocks, func(key uint64, _ struct{}) {
			removed = append(removed, key)
		})
		if err != nil {
			return nil, err
		}
	}

	turns := make([]Turn, 0, len(records)*dupFactor)
	var queryBlocks, eventBlocks int
	for dup := range uint64(dupFactor) {
		for i, ids := range records {
			t := Turn{Worker: worker(dup, uint64(i), ids), Query: keys(dup, ids)}
			removed = nil
			cache := caches[t.Worker]
			for _, k := range t.Query {
				if _, ok := cache.Get(k); !ok {
					t.Stored = append(t.Stored, k)
					cache.Add(k, struct{}{})
				}
			}
			t.Removed = removed
			queryBlocks += len(t.Query)
			eventBlocks += len(t.Query) + len(t.Stored) + len(t.Removed)
			turns = append(turns, t)
		}
	}
	for i := range turns {
		t := &turns[i]
		for _, k := range t.Stored {
			if caches[t.Worker].Contains(k) {
				t.Warm = append(t.Warm, k)
			}
		}
	}
	return &Replay{
		Turns:       turns,
		QueryBlocks: float64(queryBlocks) / float64(len(turns)),
		EventBlocks: float64(eventBlocks) / float64(len(turns)),
	}, nil
}

func readRecords(path string) ([][]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var records [][]uint64
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for len(records) < maxRecords && scanner.Scan() {
		var rec struct {
			HashIDs []uint64 `json:"hash_ids"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if len(rec.HashIDs) > 0 {
			records = append(records, rec.HashIDs)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("%s: no records with hash_ids", path)
	}
	return records, nil
}

// keys expands hash IDs into chained block keys, seeded per duplicate so
// duplicates never share a key.
func keys(dup uint64, ids []uint64) []uint64 {
	out := make([]uint64, 0, len(ids)*expandFactor)
	h := mix((dup + 1) * 0x9e3779b97f4a7c15)
	for _, id := range ids {
		for sub := range uint64(expandFactor) {
			h = mix(h ^ mix(((id+1)*1000003)^(sub+1)))
			if h == 0 {
				h = 1
			}
			out = append(out, h)
		}
	}
	return out
}

// worker picks a worker from the first three quarters of the hash IDs, or
// from the record index when those are all zero, the trace's shared root.
func worker(dup, record uint64, ids []uint64) int {
	session := uint64(42)
	nonZero := false
	for _, id := range ids[:max(1, len(ids)*3/4)] {
		nonZero = nonZero || id != 0
		session = mix(session ^ (id + 1))
	}
	if !nonZero {
		session = mix(session ^ (record + 1))
	}
	return int(mix(session^mix(dup+1)) % Workers)
}

// Convert copies keys into a slice of an indexer's block hash type.
func Convert[T ~uint64](keys []uint64) []T {
	out := make([]T, len(keys))
	for i, k := range keys {
		out[i] = T(k)
	}
	return out
}

// mix is the SplitMix64 finalizer.
func mix(x uint64) uint64 {
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}
