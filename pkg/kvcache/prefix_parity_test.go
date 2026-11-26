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
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock/workload"
)

// parityEpsilonPerBlock is the WeightedScore tolerance per matched block
// between two read paths or backends replaying the same stream.
const parityEpsilonPerBlock = 1e-9

// parityRunner drives one Index through the production decorator chain
// (TracedIndex over InstrumentedIndex) and offers both read paths over the
// same state: the walked matcher and the materialized-lookup matcher.
type parityRunner struct {
	idx      kvblock.Index
	walked   *kvcache.Indexer
	material *kvcache.Indexer
}

func newParityRunner(t *testing.T, cfg *kvblock.InMemoryIndexConfig) *parityRunner {
	t.Helper()
	raw, err := kvblock.NewInMemoryIndex(cfg)
	require.NoError(t, err)
	wrapped := kvblock.NewTracedIndex(kvblock.NewInstrumentedIndex(raw))
	backends := kvcache.DefaultKVCacheBackendConfig()
	return &parityRunner{
		idx:      wrapped,
		walked:   kvcache.NewIndexerForTest(&mockTokenProcessor{}, wrapped, backends),
		material: kvcache.NewIndexerForTest(&mockTokenProcessor{}, nonWalkerIndex{wrapped}, backends),
	}
}

// replay applies ops in order. At every MatchOp both read paths run over
// the same index state and their results must agree exactly (integer and
// map fields) and within parityEpsilonPerBlock per block (WeightedScore).
func (r *parityRunner) replay(t *testing.T, ops []workload.Op) {
	t.Helper()
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	for i, op := range ops {
		switch op.Kind {
		case workload.AddOp:
			require.NoErrorf(t, r.idx.Add(ctx, op.EngineKeys, op.RequestKeys, op.Entries), "op %d", i)
		case workload.EvictOp:
			keys := op.RequestKeys
			if op.KeyType == kvblock.EngineKey {
				keys = op.EngineKeys
			}
			require.NoErrorf(t, r.idx.Evict(ctx, op.KeyType, keys, op.Entries), "op %d", i)
		case workload.ClearOp:
			require.NoErrorf(t, r.idx.Clear(ctx, op.Pod), "op %d", i)
		case workload.MatchOp:
			filter := sets.New(op.Filter...)
			walked, err := r.walked.MatchBlockKeys(ctx, op.Query, filter)
			require.NoErrorf(t, err, "walked match at op %d", i)
			material, err := r.material.MatchBlockKeys(ctx, op.Query, filter)
			require.NoErrorf(t, err, "materialized match at op %d", i)
			assertParityMatches(t, i, walked, material)
		}
	}
}

// assertParityMatches compares two match maps under the Tier A contract.
func assertParityMatches(t *testing.T, opIdx int, want, got map[string]kvcache.PodMatch) {
	t.Helper()
	require.Lenf(t, got, len(want), "pod set at op %d", opIdx)
	for pod, w := range want {
		g, ok := got[pod]
		require.Truef(t, ok, "pod %q missing at op %d", pod, opIdx)
		tol := parityEpsilonPerBlock * math.Max(1, float64(w.MatchedBlocks))
		assert.InDeltaf(t, w.WeightedScore, g.WeightedScore, tol, "pod %q score at op %d", pod, opIdx)
		assert.Equalf(t, w.MatchedBlocks, g.MatchedBlocks, "pod %q matched at op %d", pod, opIdx)
		assert.Equalf(t, w.ConfirmedBlocks, g.ConfirmedBlocks, "pod %q confirmed at op %d", pod, opIdx)
		assert.Equalf(t, w.BlocksByTier, g.BlocksByTier, "pod %q tiers at op %d", pod, opIdx)
	}
}

// TestPrefixParityGeneratorStreams replays the two design workloads (plus
// sub-canonical, group-heavy, and speculative-heavy variants) through both
// read paths: every match must agree exactly.
func TestPrefixParityGeneratorStreams(t *testing.T) {
	profiles := []workload.Profile{
		workload.Agentic(8, 2, 16),
		workload.Interactive(16, 1, 4),
	}
	sub := workload.Agentic(4, 1, 8)
	sub.EngineRatio = 4
	profiles = append(profiles, sub)
	groups := workload.Interactive(8, 3, 6)
	groups.GroupChance = 0.5
	profiles = append(profiles, groups)
	specs := workload.Agentic(8, 2, 12)
	specs.SpecChance = 0.8
	profiles = append(profiles, specs)

	for _, p := range profiles {
		for seed := int64(1); seed <= 5; seed++ {
			p.Seed = seed
			t.Run(fmt.Sprintf("%s/seed=%d", p.Name, seed), func(t *testing.T) {
				r := newParityRunner(t, &kvblock.InMemoryIndexConfig{Size: 1 << 16, PodCacheSize: 256})
				var ops []workload.Op
				g := workload.New(p)
				for {
					op, ok := g.Next()
					if !ok {
						break
					}
					ops = append(ops, op)
				}
				r.replay(t, ops)
			})
		}
	}
}

// parityScenario is one scripted replay with its own index configuration.
type parityScenario struct {
	name string
	cfg  *kvblock.InMemoryIndexConfig
	ops  func() []workload.Op
}

func addOp(engineKeys, requestKeys []kvblock.BlockHash, entries ...kvblock.PodEntry) workload.Op {
	return workload.Op{Kind: workload.AddOp, EngineKeys: engineKeys, RequestKeys: requestKeys, Entries: entries}
}

func evictRequestOp(key kvblock.BlockHash, entries ...kvblock.PodEntry) workload.Op {
	return workload.Op{Kind: workload.EvictOp, KeyType: kvblock.RequestKey, RequestKeys: []kvblock.BlockHash{key}, Entries: entries}
}

func evictEngineOp(key kvblock.BlockHash, entries ...kvblock.PodEntry) workload.Op {
	return workload.Op{Kind: workload.EvictOp, KeyType: kvblock.EngineKey, EngineKeys: []kvblock.BlockHash{key}, Entries: entries}
}

func clearOp(pod string) workload.Op {
	return workload.Op{Kind: workload.ClearOp, Pod: pod}
}

func matchOp(keys ...kvblock.BlockHash) workload.Op {
	return workload.Op{Kind: workload.MatchOp, Query: keys}
}

// filteredMatchOp carries a pod filter restricting the match; a nil filter
// matches every pod.
func filteredMatchOp(filter []string, keys ...kvblock.BlockHash) workload.Op {
	return workload.Op{Kind: workload.MatchOp, Query: keys, Filter: filter}
}

func entry(pod, tier string) kvblock.PodEntry {
	return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: tier}
}

func grouped(pod, tier string, group int) kvblock.PodEntry {
	return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: tier, HasGroup: true, GroupIdx: kvblock.GroupID(group)}
}

// TestPrefixParityScenarios pins the scripted parity cases: each edge the
// design calls out gets a deterministic replay through both read paths.
func TestPrefixParityScenarios(t *testing.T) {
	large := &kvblock.InMemoryIndexConfig{Size: 1 << 12, PodCacheSize: 256}
	tight := &kvblock.InMemoryIndexConfig{Size: 1 << 12, PodCacheSize: 4}
	keys := func(n int) []kvblock.BlockHash {
		out := make([]kvblock.BlockHash, n)
		for i := range out {
			out[i] = kvblock.BlockHash(10 + 10*i)
		}
		return out
	}
	scenarios := []parityScenario{
		{
			name: "duplicate removes",
			cfg:  large,
			ops: func() []workload.Op {
				k := keys(3)
				return []workload.Op{
					addOp(k, k, entry("pod-a", "gpu"), entry("pod-b", "gpu")),
					evictRequestOp(k[1], entry("pod-a", "gpu")),
					evictRequestOp(k[1], entry("pod-a", "gpu")), // duplicate remove
					matchOp(k...),
					evictRequestOp(k[1], entry("pod-a", "gpu"), entry("pod-b", "gpu")),
					matchOp(k...),
				}
			},
		},
		{
			name: "tier-only offload",
			cfg:  large,
			ops: func() []workload.Op {
				k := keys(3)
				return []workload.Op{
					addOp(k, k, entry("pod-a", "gpu")),
					addOp(k, k, entry("pod-a", "cpu")), // offload keeps the gpu copy
					matchOp(k...),
					evictRequestOp(k[0], entry("pod-a", "gpu")), // wire removes the gpu copy only
					matchOp(k...),
					evictRequestOp(k[0], entry("pod-a", "cpu")),
					matchOp(k...),
				}
			},
		},
		{
			name: "clear then re-add same prefix",
			cfg:  large,
			ops: func() []workload.Op {
				k := keys(5)
				return []workload.Op{
					addOp(k, k, entry("pod-a", "gpu"), entry("pod-b", "gpu")),
					matchOp(k...),
					clearOp("pod-a"), // AllBlocksCleared; the pod keeps serving
					matchOp(k...),
					addOp(k, k, entry("pod-a", "gpu")),
					matchOp(k...),
				}
			},
		},
		{
			name: "pod filter subset",
			cfg:  large,
			ops: func() []workload.Op {
				k := keys(4)
				return []workload.Op{
					addOp(k, k, entry("pod-a", "gpu"), entry("pod-b", "gpu"), entry("pod-c", "cpu")),
					filteredMatchOp([]string{"pod-a", "pod-c"}, k...),
					filteredMatchOp([]string{"pod-b"}, k...),
					filteredMatchOp([]string{"pod-a", "pod-z"}, k...), // names a pod the index never held
					filteredMatchOp(nil, k...),                        // empty filter matches every pod
					evictRequestOp(k[1], entry("pod-b", "gpu")),
					filteredMatchOp([]string{"pod-b"}, k...), // filtered to the evicted pod
				}
			},
		},
		{
			name: "mid-prefix eviction breaks the chain",
			cfg:  large,
			ops: func() []workload.Op {
				k := keys(5)
				return []workload.Op{
					addOp(k, k, entry("pod-a", "gpu")),
					evictRequestOp(k[2], entry("pod-a", "gpu")),
					matchOp(k...),
					addOp([]kvblock.BlockHash{k[2]}, []kvblock.BlockHash{k[2]}, entry("pod-a", "gpu")),
					matchOp(k...),
				}
			},
		},
		{
			name: "per-key cap truncation",
			cfg:  tight, // PodCacheSize 4: the least-recently-announced pod drops
			ops: func() []workload.Op {
				k := keys(2)
				var ops []workload.Op
				for p := 0; p < 8; p++ {
					ops = append(ops, addOp(k, k, entry(fmt.Sprintf("pod-%d", p), "gpu")))
				}
				ops = append(ops, matchOp(k...))
				return ops
			},
		},
		{
			name: "group-tagged and speculative entries coexist",
			cfg:  large,
			ops: func() []workload.Op {
				k := keys(4)
				return []workload.Op{
					addOp(k, k, grouped("pod-a", "gpu", 1), entry("pod-a", "gpu")),
					addOp(nil, []kvblock.BlockHash{k[2]}, kvblock.PodEntry{PodIdentifier: "pod-b", Speculative: true}),
					matchOp(k...),
					evictRequestOp(k[0], grouped("pod-a", "gpu", 1)),
					matchOp(k...),
					evictRequestOp(k[0], entry("pod-a", "gpu")),
					matchOp(k...),
				}
			},
		},
		{
			name: "sub-canonical geometry",
			cfg:  large,
			ops: func() []workload.Op {
				rk := keys(2)
				ek := []kvblock.BlockHash{100, 101, 102, 103, 110, 111, 112, 113}
				return []workload.Op{
					addOp(ek, rk, entry("pod-a", "gpu")),
					matchOp(rk...),
					evictEngineOp(ek[2], entry("pod-a", "gpu")),
					matchOp(rk...),
				}
			},
		},
		{
			name: "speculative-named tier collision",
			cfg:  large,
			ops: func() []workload.Op {
				k := keys(3)
				return []workload.Op{
					addOp(k, k, entry("pod-a", "speculative"), kvblock.PodEntry{PodIdentifier: "pod-a", Speculative: true}),
					matchOp(k...),
					addOp(k, k, entry("pod-b", "speculative")),
					matchOp(k...),
				}
			},
		},
		{
			name: "clear re-add parent resolution",
			cfg:  large,
			ops: func() []workload.Op {
				k := keys(4)
				return []workload.Op{
					addOp(k, k, entry("pod-a", "gpu")),
					clearOp("pod-a"),
					addOp(k, k, entry("pod-a", "gpu")),
					matchOp(k...),
					// The stale-then-rebuilt engine mapping must resolve to the
					// rebuilt request keys.
					evictEngineOp(k[3], entry("pod-a", "gpu")),
					matchOp(k...),
				}
			},
		},
		{
			name: "identical consecutive keys fold runs",
			cfg:  large,
			ops: func() []workload.Op {
				k := keys(6)
				return []workload.Op{
					addOp(k, k, entry("pod-a", "gpu"), entry("pod-b", "cpu")),
					matchOp(k[:3]...), matchOp(k[:4]...), matchOp(k...),
					// A differing middle key ends the run and folds deferred
					// increments.
					evictRequestOp(k[3], entry("pod-b", "cpu")),
					matchOp(k...),
				}
			},
		},
	}

	for _, sc := range scenarios {
		t.Run(sc.name, func(t *testing.T) {
			newParityRunner(t, sc.cfg).replay(t, sc.ops())
		})
	}
}

// TestPrefixParityFuzz builds random op scripts from fixed seeds over a
// small key and pod space, with a tight per-key cap so truncation fires,
// matches at times restricted to a random pod subset, and requires both
// read paths to agree at every match.
func TestPrefixParityFuzz(t *testing.T) {
	const (
		numKeys = 16
		numPods = 6
		tiers   = 3
	)
	podNames := []string{"pod-a", "pod-b", "pod-c", "pod-d", "pod-e", "pod-f"}
	tierNames := []string{"gpu", "cpu", "storage"}

	randomEntry := func(rng *rand.Rand) kvblock.PodEntry {
		e := kvblock.PodEntry{
			PodIdentifier: podNames[rng.Intn(numPods)],
			DeviceTier:    tierNames[rng.Intn(tiers)],
		}
		switch rng.Intn(8) {
		case 0:
			e.Speculative = true
			e.DeviceTier = ""
		case 1:
			e.HasGroup = true
			e.GroupIdx = kvblock.GroupID(1 + rng.Intn(3))
		}
		return e
	}

	for seed := int64(1); seed <= 20; seed++ {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))
			key := func(i int) kvblock.BlockHash { return kvblock.BlockHash(10 + 10*i) }
			var ops []workload.Op
			for n := 0; n < 200; n++ {
				switch rng.Intn(10) {
				case 0, 1, 2, 3: // add
					nk := 1 + rng.Intn(3)
					rks := make([]kvblock.BlockHash, nk)
					for i := range rks {
						rks[i] = key(rng.Intn(numKeys))
					}
					entries := make([]kvblock.PodEntry, 1+rng.Intn(4))
					for i := range entries {
						entries[i] = randomEntry(rng)
					}
					var eks []kvblock.BlockHash
					if rng.Intn(2) == 0 {
						eks = rks
					}
					ops = append(ops, addOp(eks, rks, entries...))
				case 4, 5: // evict a random (key, entry) pair
					entries := make([]kvblock.PodEntry, 1+rng.Intn(2))
					for i := range entries {
						entries[i] = randomEntry(rng)
					}
					if rng.Intn(4) == 0 {
						ops = append(ops, evictEngineOp(key(rng.Intn(numKeys)), entries...))
					} else {
						ops = append(ops, evictRequestOp(key(rng.Intn(numKeys)), entries...))
					}
				case 6: // clear
					ops = append(ops, clearOp(podNames[rng.Intn(numPods)]))
				default: // match a random prefix, at times through a pod filter
					from := rng.Intn(numKeys)
					to := from + 1 + rng.Intn(numKeys-from)
					q := make([]kvblock.BlockHash, 0, to-from)
					for i := from; i < to; i++ {
						q = append(q, key(i))
					}
					var filter []string
					if rng.Intn(3) == 0 {
						for _, i := range rng.Perm(numPods)[:1+rng.Intn(numPods)] {
							filter = append(filter, podNames[i])
						}
					}
					ops = append(ops, filteredMatchOp(filter, q...))
				}
			}
			newParityRunner(t, &kvblock.InMemoryIndexConfig{Size: 1 << 12, PodCacheSize: 3}).replay(t, ops)
		})
	}
}
