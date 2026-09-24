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

package kvevents_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

func loadSnapshotChunks(tb testing.TB) (names []string, all [][][]byte, total int64) {
	pattern := os.Getenv("KV_SNAPSHOT_FILES")
	if pattern == "" {
		tb.Skip("KV_SNAPSHOT_FILES not set")
	}
	files, err := filepath.Glob(pattern)
	if err != nil || len(files) == 0 {
		tb.Fatalf("no snapshot files match %q: %v", pattern, err)
	}
	for _, file := range files {
		raw, err := os.ReadFile(file)
		if err != nil {
			tb.Fatal(err)
		}
		var frames []any
		if err := msgpack.Unmarshal(raw, &frames); err != nil {
			tb.Fatal(err)
		}
		chunks := make([][]byte, 0, len(frames))
		for i, frame := range frames[2:] {
			chunk, ok := frame.([]byte)
			if !ok {
				tb.Fatalf("%s: frame %d is %T, not bytes", file, i+2, frame)
			}
			chunks = append(chunks, chunk)
		}
		names = append(names, filepath.Base(file))
		all = append(all, chunks)
		total += int64(len(raw))
	}
	return names, all, total
}

// TestSnapshotReplayStagedMatchesDirect checks, on captured engine snapshots,
// that staged replay leaves the shared index exactly as direct replay does.
func TestSnapshotReplayStagedMatchesDirect(t *testing.T) {
	names, all, _ := loadSnapshotChunks(t)
	ctx := context.Background()
	tokens, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	adapter := engineadapter.NewVLLMAdapter()
	adapter.SnapshotMode = true
	for i, chunks := range all {
		indexes := [2]kvblock.Index{}
		owned := [2][]kvblock.BlockHash{}
		for mode, staged := range []bool{false, true} {
			if indexes[mode], err = kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 10_000_000, PodCacheSize: 1 << 20}); err != nil {
				t.Fatal(err)
			}
			if owned[mode], err = kvevents.ReplaySnapshotForTest(ctx, indexes[mode], tokens, adapter,
				"kv@10.0.0.1:8000@zai-org/GLM-5.3", "pod#snapshot-1", chunks, staged); err != nil {
				t.Fatal(err)
			}
		}
		if len(owned[0]) != len(owned[1]) {
			t.Fatalf("%s: direct owns %d keys, staged %d", names[i], len(owned[0]), len(owned[1]))
		}
		entries := 0
		for _, key := range owned[0] {
			want, err := indexes[0].Lookup(ctx, []kvblock.BlockHash{key}, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := indexes[1].Lookup(ctx, []kvblock.BlockHash{key}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !assert.ElementsMatch(t, want[key], got[key], "%s key %d", names[i], key) {
				t.FailNow()
			}
			entries += len(want[key])
		}
		t.Logf("%s: %d keys, %d entries identical", names[i], len(owned[0]), entries)
	}
}

// BenchmarkSnapshotRecovery replays every captured snapshot into one shared
// index with a bounded number of concurrent replays, as a cold router does.
func BenchmarkSnapshotRecovery(b *testing.B) {
	names, all, total := loadSnapshotChunks(b)
	tokens, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: 64})
	if err != nil {
		b.Fatal(err)
	}
	for _, staged := range []bool{false, true} {
		for _, workers := range []int{1, 4, 8, 16} {
			b.Run(fmt.Sprintf("staged-%t/workers-%d", staged, workers), func(b *testing.B) {
				ctx := context.Background()
				b.SetBytes(total)
				for b.Loop() {
					index, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 10_000_000, PodCacheSize: 1 << 20})
					if err != nil {
						b.Fatal(err)
					}
					jobs := make(chan int)
					var wg sync.WaitGroup
					errs := make(chan error, len(all))
					for range workers {
						wg.Add(1)
						go func() {
							defer wg.Done()
							adapter := engineadapter.NewVLLMAdapter()
							adapter.SnapshotMode = true
							for i := range jobs {
								if _, err := kvevents.ReplaySnapshotForTest(ctx, index, tokens, adapter,
									"kv@10.0.0.1:8000@zai-org/GLM-5.3", names[i]+"#snapshot-1", all[i], staged); err != nil {
									errs <- err
								}
							}
						}()
					}
					for i := range all {
						jobs <- i
					}
					close(jobs)
					wg.Wait()
					close(errs)
					for err := range errs {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkSnapshotReplay replays captured engine snapshots
// (msgpack [seq, stream_id, chunk...]) from KV_SNAPSHOT_FILES (a glob).
func BenchmarkSnapshotReplay(b *testing.B) {
	pattern := os.Getenv("KV_SNAPSHOT_FILES")
	if pattern == "" {
		b.Skip("KV_SNAPSHOT_FILES not set")
	}
	files, err := filepath.Glob(pattern)
	if err != nil || len(files) == 0 {
		b.Fatalf("no snapshot files match %q: %v", pattern, err)
	}
	for _, file := range files {
		raw, err := os.ReadFile(file)
		if err != nil {
			b.Fatal(err)
		}
		var frames []any
		if err := msgpack.Unmarshal(raw, &frames); err != nil {
			b.Fatal(err)
		}
		chunks := make([][]byte, 0, len(frames))
		for _, frame := range frames[2:] {
			chunks = append(chunks, frame.([]byte))
		}
		b.Run(filepath.Base(file), func(b *testing.B) {
			ctx := context.Background()
			tokens, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: 64})
			if err != nil {
				b.Fatal(err)
			}
			adapter := engineadapter.NewVLLMAdapter()
			adapter.SnapshotMode = true
			b.SetBytes(int64(len(raw)))
			b.ReportAllocs()
			for b.Loop() {
				index, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 10_000_000, PodCacheSize: 1 << 20})
				if err != nil {
					b.Fatal(err)
				}
				keys, err := kvevents.ReplaySnapshotForTest(ctx, index, tokens, adapter,
					"kv@10.0.0.1:8000@zai-org/GLM-5.3", "pod#snapshot-1", chunks, true)
				if err != nil {
					b.Fatal(err)
				}
				b.ReportMetric(float64(len(keys)), "entries")
			}
		})
	}
}
