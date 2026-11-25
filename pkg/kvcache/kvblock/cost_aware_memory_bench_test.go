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
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// BenchmarkCostAwareMemoryIndexLookup measures a full-hit lookup with trace
// logging disabled, across fleet sizes.
func BenchmarkCostAwareMemoryIndexLookup(b *testing.B) {
	const numKeys = 256
	for _, numPods := range []int{8, 96} {
		b.Run(fmt.Sprintf("keys=%d/pods=%d", numKeys, numPods), func(b *testing.B) {
			idx, err := kvblock.NewCostAwareMemoryIndex(kvblock.DefaultCostAwareMemoryIndexConfig())
			if err != nil {
				b.Fatal(err)
			}
			keys := make([]kvblock.BlockHash, numKeys)
			for i := range keys {
				keys[i] = kvblock.BlockHash(uint64(i) + 1)
			}
			entries := make([]kvblock.PodEntry, numPods)
			for p := range entries {
				entries[p] = kvblock.PodEntry{PodIdentifier: fmt.Sprintf("10.0.%d.%d:8000", p/256, p%256), DeviceTier: "gpu"}
			}
			ctx := context.Background()
			if err := idx.Add(ctx, nil, keys, entries); err != nil {
				b.Fatal(err)
			}
			res, err := idx.Lookup(ctx, keys, sets.New[string]())
			if err != nil || len(res) != numKeys {
				b.Fatalf("index not populated: %d keys, err %v", len(res), err)
			}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := idx.Lookup(ctx, keys, sets.New[string]()); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
