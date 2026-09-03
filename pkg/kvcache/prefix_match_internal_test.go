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

package kvcache

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// Cold request state must scale with the live candidates of a request, not
// with the ordinals the index has assigned to pods since cleared.
func TestMatchBlockKeysColdStateDoesNotTrackPodHistory(t *testing.T) {
	measure := func(churnPods int) uint64 {
		ctx := log.IntoContext(context.Background(), logr.Discard())
		index, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 128, PodCacheSize: 128})
		require.NoError(t, err)
		indexer := newIndexer(nil, index, DefaultKVCacheBackendConfig(), true)

		keys := []kvblock.BlockHash{10, 20}
		for i := 0; i < 96; i++ {
			entry := []kvblock.PodEntry{{PodIdentifier: fmt.Sprintf("pod-live-%d", i), DeviceTier: "gpu"}}
			require.NoError(t, index.Add(ctx, nil, keys, entry))
		}
		for i := 0; i < churnPods; i++ {
			pod := fmt.Sprintf("pod-gone-%d", i)
			require.NoError(t, index.Add(ctx, nil, []kvblock.BlockHash{999}, []kvblock.PodEntry{{PodIdentifier: pod, DeviceTier: "gpu"}}))
			require.NoError(t, index.Clear(ctx, pod))
		}

		accumulatorPool = sync.Pool{New: func() any { return &prefixAccumulator{} }}
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		matches, err := indexer.MatchBlockKeys(ctx, keys, nil)
		require.NoError(t, err)
		require.Len(t, matches, 96)
		runtime.ReadMemStats(&after)
		return after.TotalAlloc - before.TotalAlloc
	}

	measure(0) // warm lazy runtime and package initialization
	withoutChurn := measure(0)
	withChurn := measure(20_000)
	assert.Less(t, withChurn, withoutChurn+128*1024,
		"cold request state must scale with live candidates, not with pod ordinal history")
}

// Request state must not scale with the tier ordinals the index has assigned
// either: a live tier interned after thousands of others costs the same as
// the first.
func TestMatchBlockKeysColdStateDoesNotTrackTierHistory(t *testing.T) {
	measure := func(churnTiers int) uint64 {
		ctx := log.IntoContext(context.Background(), logr.Discard())
		index, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 128, PodCacheSize: 128})
		require.NoError(t, err)
		indexer := newIndexer(nil, index, DefaultKVCacheBackendConfig(), true)

		for i := 0; i < churnTiers; i++ {
			entry := []kvblock.PodEntry{{PodIdentifier: "pod-gone", DeviceTier: fmt.Sprintf("tier-%d", i)}}
			require.NoError(t, index.Add(ctx, nil, []kvblock.BlockHash{999}, entry))
		}
		require.NoError(t, index.Clear(ctx, "pod-gone"))
		keys := []kvblock.BlockHash{10, 20}
		for i := 0; i < 8; i++ {
			entry := []kvblock.PodEntry{{PodIdentifier: fmt.Sprintf("pod-live-%d", i), DeviceTier: "live-tier"}}
			require.NoError(t, index.Add(ctx, nil, keys, entry))
		}

		accumulatorPool = sync.Pool{New: func() any { return &prefixAccumulator{} }}
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		matches, err := indexer.MatchBlockKeys(ctx, keys, nil)
		require.NoError(t, err)
		require.Len(t, matches, 8)
		runtime.ReadMemStats(&after)
		return after.TotalAlloc - before.TotalAlloc
	}

	measure(0)
	withoutChurn := measure(0)
	withChurn := measure(3000)
	assert.Less(t, withChurn, withoutChurn+16*1024,
		"request state must scale with the tiers a request sees, not with tier ordinal history")
}
