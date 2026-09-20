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

package kvevents

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"k8s.io/apimachinery/pkg/util/sets"
)

func BenchmarkSnapshotMatch192Publishers(b *testing.B) {
	ctx := context.Background()
	tokens, err := kvblock.NewChunkedTokenDatabase(nil)
	if err != nil {
		b.Fatal(err)
	}
	cfg, err := kvcache.NewDefaultConfig()
	if err != nil {
		b.Fatal(err)
	}
	cfg.KVBlockIndexConfig.InMemoryConfig.PodCacheSize = snapshotEntriesPerKey
	matcher, err := kvcache.NewKVCacheIndexer(ctx, cfg, tokens)
	if err != nil {
		b.Fatal(err)
	}
	keys := make([]kvblock.BlockHash, 32)
	for i := range keys {
		keys[i] = kvblock.BlockHash(i + 1)
	}
	entries := make([]kvblock.PodEntry, 192)
	aggregate, err := kvblock.NewInMemoryIndex(cfg.KVBlockIndexConfig.InMemoryConfig)
	if err != nil {
		b.Fatal(err)
	}
	manager := &SnapshotManager{
		matcher:     matcher.WithIndex(aggregate),
		index:       aggregate,
		subscribers: make(map[string]*snapshotSubscriber, len(entries)),
	}
	filter := sets.New[string]()
	for i := range entries {
		pod := fmt.Sprintf("pod-%03d", i)
		entries[i] = kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu"}
		filter.Insert(pod)
		generation := pod + "#snapshot-1"
		if err := aggregate.Add(ctx, nil, keys, []kvblock.PodEntry{{PodIdentifier: generation, DeviceTier: "gpu"}}); err != nil {
			b.Fatal(err)
		}
		subscriber := &snapshotSubscriber{sourceEndpoint: pod, activeID: generation}
		subscriber.lastReceive.Store(time.Now().UnixNano())
		manager.subscribers[pod] = subscriber
	}
	if err := matcher.KVBlockIndex().Add(ctx, nil, keys, entries); err != nil {
		b.Fatal(err)
	}
	if matches, err := manager.MatchBlockKeys(ctx, keys, filter); err != nil {
		b.Fatal(err)
	} else if len(matches) != len(entries) {
		b.Fatalf("snapshot matcher retained %d of %d publishers", len(matches), len(entries))
	}
	if matches, err := matcher.MatchBlockKeys(ctx, keys, filter); err != nil {
		b.Fatal(err)
	} else if len(matches) != len(entries) {
		b.Fatalf("ordinary matcher retained %d of %d publishers", len(matches), len(entries))
	}

	b.Run("shared-index", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := matcher.MatchBlockKeys(ctx, keys, filter); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("snapshot-generations", func(b *testing.B) {
		now := time.Now().UnixNano()
		for _, subscriber := range manager.subscribers {
			subscriber.lastReceive.Store(now)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			if _, err := manager.MatchBlockKeys(ctx, keys, filter); err != nil {
				b.Fatal(err)
			}
		}
	})
}
