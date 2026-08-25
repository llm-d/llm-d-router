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
	"fmt"
	"testing"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	. "github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

func BenchmarkInMemoryIndexLookup(b *testing.B) {
	for _, tc := range []struct {
		blocks  int
		entries int
	}{
		{blocks: 225, entries: 2},
		{blocks: 225, entries: 10},
		{blocks: 1034, entries: 2},
	} {
		b.Run(fmt.Sprintf("blocks=%d/entries=%d", tc.blocks, tc.entries), func(b *testing.B) {
			ctx := log.IntoContext(b.Context(), logr.Discard())
			index, err := NewInMemoryIndex(&InMemoryIndexConfig{
				Size:         tc.blocks + 1,
				PodCacheSize: tc.entries,
			})
			if err != nil {
				b.Fatal(err)
			}

			keys := make([]BlockHash, tc.blocks)
			for i := range keys {
				keys[i] = BlockHash(i + 1)
			}
			entries := make([]PodEntry, tc.entries)
			for i := range entries {
				entries[i] = PodEntry{
					PodIdentifier: fmt.Sprintf("pod-%d", i),
					DeviceTier:    "gpu",
				}
			}
			if err := index.Add(ctx, nil, keys, entries); err != nil {
				b.Fatal(err)
			}

			b.Run("serial", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := index.Lookup(ctx, keys, nil); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("parallel", func(b *testing.B) {
				b.ReportAllocs()
				b.RunParallel(func(pb *testing.PB) {
					for pb.Next() {
						if _, err := index.Lookup(ctx, keys, nil); err != nil {
							b.Error(err)
							return
						}
					}
				})
			})
		})
	}
}
