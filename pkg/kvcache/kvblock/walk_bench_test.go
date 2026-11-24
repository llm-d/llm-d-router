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

package kvblock_test

import (
	"context"
	"testing"

	. "github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// BenchmarkWalkDeepPromptPD reports the walk cost of a deep prompt in the
// disaggregated shape: both the prefill and decode pods hold every block,
// each in its gpu and cpu tiers. 3125 keys is a 100K-token prompt at block
// size 32.
func BenchmarkWalkDeepPromptPD(b *testing.B) {
	const numKeys = 3125
	entries := []PodEntry{
		{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu"},
		{PodIdentifier: "10.0.0.1:8000", DeviceTier: "cpu"},
		{PodIdentifier: "10.0.0.2:8000", DeviceTier: "gpu"},
		{PodIdentifier: "10.0.0.2:8000", DeviceTier: "cpu"},
	}
	keys := make([]BlockHash, numKeys)
	for i := range keys {
		keys[i] = BlockHash(uint64(i)*0x9E3779B97F4A7C15 + 1)
	}
	ctx := context.Background()

	dense, err := NewDenseMemoryIndex(&DenseMemoryIndexConfig{Size: "64MiB"})
	if err != nil {
		b.Fatal(err)
	}
	if err := dense.Add(ctx, nil, keys, entries); err != nil {
		b.Fatal(err)
	}
	inMemory, err := NewInMemoryIndex(&InMemoryIndexConfig{Size: 1 << 16, PodCacheSize: 8})
	if err != nil {
		b.Fatal(err)
	}
	if err := inMemory.Add(ctx, nil, keys, entries); err != nil {
		b.Fatal(err)
	}

	seen := 0
	visit := func(_ int, found bool, es []WalkEntry) bool {
		if found {
			seen += len(es)
		}
		return true
	}
	b.Run("dense", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := dense.WalkKeys(ctx, keys, visit); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("in-memory", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := inMemory.WalkKeys(ctx, keys, visit); err != nil {
				b.Fatal(err)
			}
		}
	})
	if seen < 4*numKeys {
		b.Fatalf("visited %d entries, want at least %d", seen, 4*numKeys)
	}
}
