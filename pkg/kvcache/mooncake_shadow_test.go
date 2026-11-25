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

package kvcache_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
	"github.com/llm-d/llm-d-router/test/utils/mooncake"
)

// TestMooncakeShadowReplay is an offline stand-in for the production shadow
// run that gates the dense backend: the Mooncake trace is replayed once,
// sequentially, through a shadow index with the in-memory backend as
// primary and the dense backend as shadow, comparing every read. Both
// backends get the identical op stream and generous budgets, so the
// measured divergence rate is the backends' semantic disagreement floor,
// not budget-eviction noise. The dense backend structurally caps each key
// at 15 pods, so the podCap=128 run measures that cap's blast radius on
// this trace, whose shared prefix keys are held by up to mooncake.Workers
// pods; the podCap=15 run matches the cap and isolates the semantic floor.
// Set mooncake.TraceEnv to run it.
func TestMooncakeShadowReplay(t *testing.T) {
	for _, podCap := range []int{15, 128} {
		t.Run(fmt.Sprintf("podCap=%d", podCap), func(t *testing.T) {
			r := mooncake.Load(t)
			ctx := context.Background()

			primary, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 23, PodCacheSize: podCap})
			if err != nil {
				t.Fatal(err)
			}
			shadow, err := kvblock.NewDenseMemoryIndex(&kvblock.DenseMemoryIndexConfig{Size: "2GiB"})
			if err != nil {
				t.Fatal(err)
			}
			idx := kvblock.NewShadowIndex(primary, shadow, 1)

			workers := make([][]kvblock.PodEntry, mooncake.Workers)
			for w := range workers {
				workers[w] = []kvblock.PodEntry{{PodIdentifier: fmt.Sprintf("10.0.0.%d:8000", w), DeviceTier: "gpu"}}
			}

			comparedBefore := testutil.ToFloat64(metrics.IndexShadowCompared)
			divergentBefore := testutil.ToFloat64(metrics.IndexShadowDivergent)
			errorsBefore := testutil.ToFloat64(metrics.IndexShadowErrors)

			for _, turn := range r.Turns {
				entries := workers[turn.Worker]
				if warm := mooncake.Convert[kvblock.BlockHash](turn.Warm); len(warm) > 0 {
					if err := idx.Add(ctx, warm, warm, entries); err != nil {
						t.Fatal(err)
					}
				}
				query := mooncake.Convert[kvblock.BlockHash](turn.Query)
				if _, err := idx.Lookup(ctx, query, nil); err != nil {
					t.Fatal(err)
				}
				if stored := mooncake.Convert[kvblock.BlockHash](turn.Stored); len(stored) > 0 {
					if err := idx.Add(ctx, stored, stored, entries); err != nil {
						t.Fatal(err)
					}
				}
				if removed := mooncake.Convert[kvblock.BlockHash](turn.Removed); len(removed) > 0 {
					if err := idx.Evict(ctx, kvblock.EngineKey, removed, entries); err != nil {
						t.Fatal(err)
					}
				}
			}

			compared := testutil.ToFloat64(metrics.IndexShadowCompared) - comparedBefore
			divergent := testutil.ToFloat64(metrics.IndexShadowDivergent) - divergentBefore
			shadowErrors := testutil.ToFloat64(metrics.IndexShadowErrors) - errorsBefore
			t.Logf("compared=%d divergent=%d shadowErrors=%d rate=%.6f",
				int64(compared), int64(divergent), int64(shadowErrors), divergent/compared)

			if shadowErrors > 0 {
				t.Errorf("shadow write errors: %d", int64(shadowErrors))
			}
		})
	}
}
