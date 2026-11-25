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
	"sync"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock/workload"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
)

// newShadowPair builds a decorator over an in-memory primary and a dense
// shadow with roomy budgets, so no backend's own eviction diverges the views.
func newShadowPair(t *testing.T, sampleEvery int) (kvblock.Index, kvblock.Index, kvblock.Index) {
	t.Helper()
	primary, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 24, PodCacheSize: 15})
	require.NoError(t, err)
	shadow, err := kvblock.NewDenseMemoryIndex(&kvblock.DenseMemoryIndexConfig{Size: "1GiB", PodCacheSize: 15})
	require.NoError(t, err)
	return kvblock.NewShadowIndex(primary, shadow, sampleEvery), primary, shadow
}

// applyOps drives a workload stream's index operations into idx. MatchOp
// queries run through WalkKeys when the index is a walker (the production
// read path) and through Lookup otherwise.
func applyOps(t *testing.T, idx kvblock.Index, p workload.Profile, maxOps int) {
	t.Helper()
	ctx := context.Background()
	walker, isWalker := idx.(kvblock.KeyWalker)
	g := workload.New(p)
	for n := 0; n < maxOps; n++ {
		op, ok := g.Next()
		if !ok {
			return
		}
		switch op.Kind {
		case workload.AddOp:
			require.NoError(t, idx.Add(ctx, op.EngineKeys, op.RequestKeys, op.Entries))
		case workload.EvictOp:
			keys := op.RequestKeys
			if op.KeyType == kvblock.EngineKey {
				keys = op.EngineKeys
			}
			require.NoError(t, idx.Evict(ctx, op.KeyType, keys, op.Entries))
		case workload.ClearOp:
			require.NoError(t, idx.Clear(ctx, op.Pod))
		case workload.MatchOp:
			if isWalker {
				require.NoError(t, walker.WalkKeys(ctx, op.Query, func(_ int, found bool, _ []kvblock.WalkEntry) bool {
					return found
				}))
			} else {
				_, err := idx.Lookup(ctx, op.Query, nil)
				require.NoError(t, err)
			}
		}
	}
}

func TestShadowIndexFansOutAndCompares(t *testing.T) {
	idx, primary, shadow := newShadowPair(t, 1)
	ctx := context.Background()

	entries := []kvblock.PodEntry{
		{PodIdentifier: "pod-0", DeviceTier: "gpu"},
		{PodIdentifier: "pod-1", DeviceTier: "cpu"},
	}
	keys := []kvblock.BlockHash{1, 2, 3}
	require.NoError(t, idx.Add(ctx, keys, keys, entries))

	comparedBefore := testutil.ToFloat64(metrics.IndexShadowCompared)
	divergentBefore := testutil.ToFloat64(metrics.IndexShadowDivergent)
	result, err := idx.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	assert.Len(t, result, 3)

	// Both backends hold the written entries.
	for _, backend := range []kvblock.Index{primary, shadow} {
		held, err := backend.Lookup(ctx, keys, nil)
		require.NoError(t, err)
		assert.Len(t, held, 3)
	}

	assert.InDelta(t, comparedBefore+3, testutil.ToFloat64(metrics.IndexShadowCompared), 1e-9,
		"every sampled lookup compares each requested key")
	assert.InDelta(t, divergentBefore, testutil.ToFloat64(metrics.IndexShadowDivergent), 1e-9,
		"identically driven backends do not diverge")
}

func TestShadowIndexDetectsDivergence(t *testing.T) {
	idx, _, shadow := newShadowPair(t, 1)
	ctx := context.Background()

	entries := []kvblock.PodEntry{{PodIdentifier: "pod-0", DeviceTier: "gpu"}}
	keys := []kvblock.BlockHash{1, 2, 3, 4}
	require.NoError(t, idx.Add(ctx, keys, keys, entries))

	// Drop one key from the shadow behind the decorator's back.
	require.NoError(t, shadow.Evict(ctx, kvblock.RequestKey, keys[2:3], entries))

	_, err := idx.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	assert.InDelta(t, 1, testutil.ToFloat64(metrics.IndexShadowDivergent), 1e-9,
		"a key the shadow lost is counted as divergent")
}

func TestShadowIndexCountsShadowErrors(t *testing.T) {
	primary, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 20})
	require.NoError(t, err)
	idx := kvblock.NewShadowIndex(primary, &failingIndex{err: assert.AnError}, 1)
	ctx := context.Background()

	errorsBefore := testutil.ToFloat64(metrics.IndexShadowErrors)
	entries := []kvblock.PodEntry{{PodIdentifier: "pod-0", DeviceTier: "gpu"}}
	keys := []kvblock.BlockHash{1}

	// The primary's success is the caller's result; the shadow's failure is
	// only counted.
	require.NoError(t, idx.Add(ctx, keys, keys, entries))
	require.NoError(t, idx.Clear(ctx, "pod-0"))
	assert.InDelta(t, errorsBefore+2, testutil.ToFloat64(metrics.IndexShadowErrors), 1e-9)
}

func TestShadowIndexSampling(t *testing.T) {
	idx, _, _ := newShadowPair(t, 4)
	ctx := context.Background()

	entries := []kvblock.PodEntry{{PodIdentifier: "pod-0", DeviceTier: "gpu"}}
	keys := []kvblock.BlockHash{1, 2}
	require.NoError(t, idx.Add(ctx, keys, keys, entries))

	comparedBefore := testutil.ToFloat64(metrics.IndexShadowCompared)
	const reads = 8
	for i := 0; i < reads; i++ {
		_, err := idx.Lookup(ctx, keys, nil)
		require.NoError(t, err)
	}
	assert.InDelta(t, comparedBefore+float64(reads/4*len(keys)), testutil.ToFloat64(metrics.IndexShadowCompared), 1e-9,
		"every fourth read compares")
}

// TestShadowIndexWorkloadZeroDivergence is the local shadow gate: an
// identically driven in-memory primary and dense shadow stay at zero
// divergence over a full workload stream, with every read sampled.
func TestShadowIndexWorkloadZeroDivergence(t *testing.T) {
	idx, _, _ := newShadowPair(t, 1)

	divergentBefore := testutil.ToFloat64(metrics.IndexShadowDivergent)
	errorsBefore := testutil.ToFloat64(metrics.IndexShadowErrors)
	comparedBefore := testutil.ToFloat64(metrics.IndexShadowCompared)

	applyOps(t, idx, workload.Agentic(8, 2, 128), 20000)

	assert.InDelta(t, 0, testutil.ToFloat64(metrics.IndexShadowDivergent)-divergentBefore, 1e-9,
		"the dense backend must agree with the in-memory backend on every key")
	assert.InDelta(t, 0, testutil.ToFloat64(metrics.IndexShadowErrors)-errorsBefore, 1e-9,
		"the shadow's writes must all succeed")
	assert.Greater(t, testutil.ToFloat64(metrics.IndexShadowCompared), comparedBefore,
		"the stream must have exercised the comparison")
}

// TestShadowIndexConcurrent exercises fan-out and sampled comparison under
// concurrent readers and writers. Divergence is legal here (a write fanned
// to the primary but not yet the shadow is visible to a compare), so the
// test asserts progress, not zero divergence.
func TestShadowIndexConcurrent(t *testing.T) {
	idx, _, _ := newShadowPair(t, 2)
	ctx := context.Background()

	entries := []kvblock.PodEntry{{PodIdentifier: "pod-0", DeviceTier: "gpu"}}
	query := []kvblock.BlockHash{1, 2, 3, 4}
	require.NoError(t, idx.Add(ctx, query, query, entries))

	stop := make(chan struct{})
	var wg sync.WaitGroup
	for w := 0; w < 2; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			base := uint64(w) << 40
			keys := make([]kvblock.BlockHash, 8)
			for i := uint64(0); ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				for j := range keys {
					keys[j] = kvblock.BlockHash(base + i*8 + uint64(j) + 1)
				}
				if err := idx.Add(ctx, keys, keys, entries); err != nil {
					return
				}
			}
		}(w)
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		walker := idx.(kvblock.KeyWalker)
		for {
			select {
			case <-stop:
				return
			default:
			}
			_ = walker.WalkKeys(ctx, query, func(_ int, found bool, _ []kvblock.WalkEntry) bool {
				return found
			})
		}
	}()

	comparedBefore := testutil.ToFloat64(metrics.IndexShadowCompared)
	for i := 0; i < 100; i++ {
		_, err := idx.Lookup(ctx, query, nil)
		require.NoError(t, err)
	}
	close(stop)
	wg.Wait()
	assert.Greater(t, testutil.ToFloat64(metrics.IndexShadowCompared), comparedBefore)
}
