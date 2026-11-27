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
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

const (
	hintPodA = "10.0.0.1:8080"
	hintPodB = "10.0.0.2:8080"
)

// Chains follow the matcher's rules: candidates are the pods hinted for the
// first key, and a chain ends at the first key its pod is not hinted for.
func TestSpeculativeIndexMatchChains(t *testing.T) {
	s := kvcache.NewSpeculativeIndex(time.Minute)
	s.Add([]kvblock.BlockHash{1, 2, 3}, hintPodA)
	s.Add([]kvblock.BlockHash{1, 2}, hintPodB)
	s.Add([]kvblock.BlockHash{9}, hintPodA)

	assert.Equal(t, map[string]int{hintPodA: 3, hintPodB: 2}, s.Match([]kvblock.BlockHash{1, 2, 3, 4}, nil))
	assert.Equal(t, map[string]int{hintPodA: 2, hintPodB: 1}, s.Match([]kvblock.BlockHash{2, 3}, nil))
	assert.Equal(t, map[string]int{hintPodA: 1}, s.Match([]kvblock.BlockHash{9, 5}, nil))
	assert.Nil(t, s.Match([]kvblock.BlockHash{5, 1}, nil))
	assert.Nil(t, s.Match(nil, nil))
	assert.Equal(t, 4, s.Len())
}

func TestSpeculativeIndexMatchPodFilter(t *testing.T) {
	s := kvcache.NewSpeculativeIndex(time.Minute)
	keys := []kvblock.BlockHash{1, 2, 3}
	s.Add(keys, hintPodA)
	s.Add(keys, hintPodB)

	assert.Equal(t, map[string]int{hintPodB: 3}, s.Match(keys, sets.New(hintPodB)))
	assert.Nil(t, s.Match(keys, sets.New("10.0.0.3:8080")))
}

// Hints expire per pod: a pod hinted later outlives one hinted earlier for
// the same blocks, and a renewed hint lives a full TTL from the renewal.
// Rotations drop a hint only after it has expired.
func TestSpeculativeIndexExpiry(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := kvcache.NewSpeculativeIndex(2 * time.Second)
		go s.Run(t.Context()) // rotates at 2s, 4s, 6s
		keys := []kvblock.BlockHash{1, 2}
		s.Add(keys, hintPodA) // expires at 2s
		time.Sleep(time.Second)
		s.Add(keys, hintPodB)     // expires at 3s
		s.Add(keys[:1], hintPodA) // renewed, expires at 3s

		time.Sleep(1500 * time.Millisecond) // 2.5s
		assert.Equal(t, map[string]int{hintPodA: 1, hintPodB: 2}, s.Match(keys, nil))

		s.Add(keys[:1], hintPodB) // expires at 4.5s
		time.Sleep(time.Second)   // 3.5s
		assert.Equal(t, map[string]int{hintPodB: 1}, s.Match(keys, nil))
		time.Sleep(900 * time.Millisecond) // 4.4s, past the rotation at 4s
		assert.Equal(t, map[string]int{hintPodB: 1}, s.Match(keys, nil))

		time.Sleep(200 * time.Millisecond) // 4.6s
		assert.Nil(t, s.Match(keys, nil))

		time.Sleep(2 * time.Second) // 6.6s, past the rotation at 6s
		synctest.Wait()
		assert.Equal(t, 0, s.Len())
	})
}

// Add, Match and the rotation run concurrently. Run with -race.
func TestSpeculativeIndexConcurrentUse(t *testing.T) {
	s := kvcache.NewSpeculativeIndex(time.Minute)
	keys := []kvblock.BlockHash{1, 2, 3, 4}
	const pods = 8
	var wg sync.WaitGroup
	for i := range pods {
		wg.Go(func() {
			pod := fmt.Sprintf("10.0.0.%d:8080", i)
			for range 50 {
				s.Add(keys, pod)
				s.Match(keys, nil)
				s.Rotate()
			}
			s.Add(keys, pod)
		})
	}
	wg.Wait()

	// A hint added before another worker's last two rotations is dropped, so
	// only the pods hinted since then remain. Each is hinted for every key.
	got := s.Match(keys, nil)
	require.NotEmpty(t, got)
	for pod, n := range got {
		assert.Equal(t, len(keys), n, pod)
	}
}

// hintMatcher builds a walking matcher and a materializing one over the same
// index, both reading speculative.
func hintMatcher(t *testing.T, backends []*kvcache.KVCacheBackendConfig, speculative *kvcache.SpeculativeIndex,
) (walked, materialized *kvcache.Indexer, idx kvblock.Index) {
	t.Helper()
	walked, idx = newMatcher(t, backends)
	walked.SetSpeculativeIndex(speculative)
	materialized = newMaterializedMatcher(idx, backends)
	materialized.SetSpeculativeIndex(speculative)
	return walked, materialized, idx
}

func TestMatchBlockKeysSpeculativeHints(t *testing.T) {
	const podA, podB, podC = "pod-a", "pod-b", "pod-c"
	gpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu"} }
	cpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "cpu"} }
	storage := func(pod string) kvblock.PodEntry {
		return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "storage"}
	}
	halfSpeculative := append(kvcache.DefaultKVCacheBackendConfig(),
		&kvcache.KVCacheBackendConfig{Name: kvcache.SpeculativeTier, Weight: 0.5})
	keys := []kvblock.BlockHash{10, 20, 30, 40, 50, 60}

	tests := []struct {
		name     string
		backends []*kvcache.KVCacheBackendConfig
		index    map[kvblock.BlockHash][]kvblock.PodEntry
		hints    map[string]int // pod -> leading keys hinted
		filter   sets.Set[string]
		want     map[string]kvcache.PodMatch
	}{
		{
			name:     "a hint alone matches a prefix the index does not hold",
			backends: kvcache.DefaultKVCacheBackendConfig(),
			hints:    map[string]int{podA: 4},
			want: map[string]kvcache.PodMatch{
				podA: {WeightedScore: 4, MatchedBlocks: 4, BlocksByTier: map[string]int{kvcache.SpeculativeTier: 4}},
			},
		},
		{
			name:     "a hint extends a confirmed chain and ends the confirmed count",
			backends: kvcache.DefaultKVCacheBackendConfig(),
			index:    map[kvblock.BlockHash][]kvblock.PodEntry{10: {gpu(podA), gpu(podB)}, 20: {gpu(podA), gpu(podB)}},
			hints:    map[string]int{podA: 5},
			want: map[string]kvcache.PodMatch{
				podA: {WeightedScore: 5, MatchedBlocks: 5, ConfirmedBlocks: 2,
					BlocksByTier: map[string]int{"gpu": 2, kvcache.SpeculativeTier: 5}},
				podB: {WeightedScore: 2, MatchedBlocks: 2, ConfirmedBlocks: 2, BlocksByTier: map[string]int{"gpu": 2}},
			},
		},
		{
			name:     "a hinted block scores the higher of its confirmed and speculative weights",
			backends: kvcache.DefaultKVCacheBackendConfig(),
			index: map[kvblock.BlockHash][]kvblock.PodEntry{
				10: {cpu(podA)}, 20: {cpu(podA)}, 30: {cpu(podA)}, 40: {cpu(podA)},
			},
			hints: map[string]int{podA: 2},
			want: map[string]kvcache.PodMatch{
				// Two blocks at the speculative 1.0, two at cpu's 0.8.
				podA: {WeightedScore: 3.6, MatchedBlocks: 4, ConfirmedBlocks: 4,
					BlocksByTier: map[string]int{"cpu": 4, kvcache.SpeculativeTier: 2}},
			},
		},
		{
			name:     "a hint ending inside a run of identical keys closes its tier chain there",
			backends: halfSpeculative,
			index: map[kvblock.BlockHash][]kvblock.PodEntry{
				10: {gpu(podA), gpu(podB)}, 20: {gpu(podA), gpu(podB)}, 30: {gpu(podA), gpu(podB)},
				40: {gpu(podA), gpu(podB)}, 50: {gpu(podA), gpu(podB)}, 60: {gpu(podA), gpu(podB)},
			},
			hints: map[string]int{podA: 3},
			want: map[string]kvcache.PodMatch{
				podA: {WeightedScore: 6, MatchedBlocks: 6, ConfirmedBlocks: 6,
					BlocksByTier: map[string]int{"gpu": 6, kvcache.SpeculativeTier: 3}},
				podB: {WeightedScore: 6, MatchedBlocks: 6, ConfirmedBlocks: 6, BlocksByTier: map[string]int{"gpu": 6}},
			},
		},
		{
			name:     "a hint keeps a tail-only tier from scoring under it",
			backends: tailOnlyBackends(),
			index: map[kvblock.BlockHash][]kvblock.PodEntry{
				10: {storage(podA), storage(podB)}, 20: {storage(podA), storage(podB)},
				30: {storage(podA), storage(podB)}, 40: {storage(podA), storage(podB)},
				50: {storage(podA), storage(podB)}, 60: {storage(podA), storage(podB)},
			},
			hints: map[string]int{podA: 4},
			want: map[string]kvcache.PodMatch{
				// Storage scores only past pod-a's hinted chain.
				podA: {WeightedScore: 4 + 2*0.6, MatchedBlocks: 6, ConfirmedBlocks: 6,
					BlocksByTier: map[string]int{"storage": 6, kvcache.SpeculativeTier: 4}},
				podB: {WeightedScore: 2 * 0.6, MatchedBlocks: 6, ConfirmedBlocks: 6,
					BlocksByTier: map[string]int{"storage": 6}},
			},
		},
		{
			name:     "entries past the first key join a pod hinted there without one",
			backends: halfSpeculative,
			index: map[kvblock.BlockHash][]kvblock.PodEntry{
				10: {gpu(podB)}, 20: {gpu(podA), gpu(podB)}, 30: {gpu(podA)}, 40: {gpu(podA)},
			},
			hints: map[string]int{podA: 2},
			want: map[string]kvcache.PodMatch{
				// pod-a: 0.5 at the hinted first key, then gpu's 1.0 for three.
				podA: {WeightedScore: 3.5, MatchedBlocks: 4, BlocksByTier: map[string]int{kvcache.SpeculativeTier: 2}},
				podB: {WeightedScore: 2, MatchedBlocks: 2, ConfirmedBlocks: 2, BlocksByTier: map[string]int{"gpu": 2}},
			},
		},
		{
			name:     "the pod filter drops a hinted pod",
			backends: kvcache.DefaultKVCacheBackendConfig(),
			index:    map[kvblock.BlockHash][]kvblock.PodEntry{10: {gpu(podB)}},
			hints:    map[string]int{podA: 3, podC: 2},
			filter:   sets.New(podB, podC),
			want: map[string]kvcache.PodMatch{
				podB: {WeightedScore: 1, MatchedBlocks: 1, ConfirmedBlocks: 1, BlocksByTier: map[string]int{"gpu": 1}},
				podC: {WeightedScore: 2, MatchedBlocks: 2, BlocksByTier: map[string]int{kvcache.SpeculativeTier: 2}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := logging.NewTestLoggerIntoContext(t.Context())
			speculative := kvcache.NewSpeculativeIndex(time.Minute)
			for pod, length := range tt.hints {
				speculative.Add(keys[:length], pod)
			}
			walked, materialized, idx := hintMatcher(t, tt.backends, speculative)
			populateIndex(t, idx, tt.index)

			got, err := walked.MatchBlockKeys(ctx, keys, tt.filter)
			require.NoError(t, err)
			assertPodMatches(t, tt.want, got)

			got, err = materialized.MatchBlockKeys(ctx, keys, tt.filter)
			require.NoError(t, err)
			assertPodMatches(t, tt.want, got)
		})
	}
}

// A hint is held outside the index, so a key at its pod cap with confirmed
// entries only still matches a hinted pod the key does not hold. A
// speculative index row for that pod is dropped at the cap.
func TestMatchBlockKeysHintAtFullKey(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	keys := []kvblock.BlockHash{10}
	confirmed := []kvblock.PodEntry{
		{PodIdentifier: "pod-a", DeviceTier: "gpu"},
		{PodIdentifier: "pod-b", DeviceTier: "gpu"},
	}
	newFullIndex := func() kvblock.Index {
		idx, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 16, PodCacheSize: len(confirmed)})
		require.NoError(t, err)
		require.NoError(t, idx.Add(ctx, keys, keys, confirmed))
		return idx
	}

	rows := newFullIndex()
	require.NoError(t, rows.Add(ctx, nil, keys, []kvblock.PodEntry{{PodIdentifier: "pod-c", Speculative: true}}))
	got, err := kvcache.NewIndexerForTest(&mockTokenProcessor{}, rows, kvcache.DefaultKVCacheBackendConfig()).
		MatchBlockKeys(ctx, keys, nil)
	require.NoError(t, err)
	assert.NotContains(t, got, "pod-c")

	speculative := kvcache.NewSpeculativeIndex(time.Minute)
	speculative.Add(keys, "pod-c")
	hinted := kvcache.NewIndexerForTest(&mockTokenProcessor{}, newFullIndex(), kvcache.DefaultKVCacheBackendConfig())
	hinted.SetSpeculativeIndex(speculative)
	got, err = hinted.MatchBlockKeys(ctx, keys, nil)
	require.NoError(t, err)
	assertPodMatches(t, map[string]kvcache.PodMatch{
		"pod-a": {WeightedScore: 1, MatchedBlocks: 1, ConfirmedBlocks: 1, BlocksByTier: map[string]int{"gpu": 1}},
		"pod-b": {WeightedScore: 1, MatchedBlocks: 1, ConfirmedBlocks: 1, BlocksByTier: map[string]int{"gpu": 1}},
		"pod-c": {WeightedScore: 1, MatchedBlocks: 1, BlocksByTier: map[string]int{kvcache.SpeculativeTier: 1}},
	}, got)
}

// A hint scores as the speculative index row it stands in for. On random
// fixtures, an index holding confirmed entries plus speculative rows and an
// index holding the same confirmed entries with the rows recorded as hints
// produce the same matches: under plain weights, a tail-only tier, and a
// configured speculative weight, with gaps, pod filters, and pods holding
// several tiers or ranks at a key.
//
// The materializing feeder is compared on the entries Lookup returns for the
// confirmed fixture, since Lookup stops at the first key the index lacks.
func TestSpeculativeHintsMatchIndexRows(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(t.Context())
	tiers := []string{"gpu", "cpu", "storage", "disk"}
	configs := map[string][]*kvcache.KVCacheBackendConfig{
		"plain weights":      {{Name: "gpu", Weight: 1.0}, {Name: "cpu", Weight: 0.8}, {Name: "storage", Weight: 0.6}},
		"tail-only storage":  tailOnlyBackends(),
		"speculative at 0.5": append(tailOnlyBackends(), &kvcache.KVCacheBackendConfig{Name: kvcache.SpeculativeTier, Weight: 0.5}),
	}

	for name, backends := range configs {
		t.Run(name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(7)) //#nosec G404 -- fixed-seed deterministic test fixture
			for iter := 0; iter < 400; iter++ {
				numKeys, numPods := 1+rng.Intn(16), 1+rng.Intn(5)
				keys := make([]kvblock.BlockHash, numKeys)
				for i := range keys {
					keys[i] = kvblock.BlockHash(i + 1)
				}
				pod := func(p int) string { return fmt.Sprintf("pod-%d", p) }

				// Half the fixtures repeat one key's entries along a stretch,
				// the shape the matcher folds as a run.
				runs := rng.Intn(2) == 0
				confirmed := make(map[kvblock.BlockHash][]kvblock.PodEntry, numKeys)
				var previous []kvblock.PodEntry
				for _, key := range keys {
					if runs && len(previous) > 0 && rng.Float64() < 0.7 {
						confirmed[key] = previous
						continue
					}
					var entries []kvblock.PodEntry
					for p := 0; p < numPods; p++ {
						if rng.Float64() >= 0.75 {
							continue
						}
						for n := 1 + rng.Intn(2); n > 0; n-- {
							entry := kvblock.PodEntry{PodIdentifier: pod(p), DeviceTier: tiers[rng.Intn(len(tiers))]}
							if rng.Intn(3) == 0 {
								entry.HasGroup, entry.GroupIdx = true, kvblock.GroupID(rng.Intn(3))
							}
							entries = append(entries, entry)
						}
					}
					previous = entries
					if len(entries) > 0 {
						confirmed[key] = entries
					}
				}
				hints := map[string]int{}
				for p := 0; p < numPods; p++ {
					if rng.Intn(2) == 0 {
						hints[pod(p)] = 1 + rng.Intn(numKeys)
					}
				}
				filter := sets.New[string]()
				if rng.Intn(2) == 0 {
					for p := 0; p < numPods; p++ {
						if rng.Intn(2) == 0 {
							filter.Insert(pod(p))
						}
					}
				}

				withRows := func(fixture map[kvblock.BlockHash][]kvblock.PodEntry) map[kvblock.BlockHash][]kvblock.PodEntry {
					out := make(map[kvblock.BlockHash][]kvblock.PodEntry, numKeys)
					for key, entries := range fixture {
						out[key] = append([]kvblock.PodEntry(nil), entries...)
					}
					for p, length := range hints {
						for _, key := range keys[:length] {
							out[key] = append(out[key], kvblock.PodEntry{PodIdentifier: p, Speculative: true})
						}
					}
					return out
				}
				matchRows := func(fixture map[kvblock.BlockHash][]kvblock.PodEntry) map[string]kvcache.PodMatch {
					matcher, idx := newMatcher(t, backends)
					populateIndex(t, idx, withRows(fixture))
					want, err := matcher.MatchBlockKeys(ctx, keys, filter)
					require.NoError(t, err)
					return want
				}

				speculative := kvcache.NewSpeculativeIndex(time.Minute)
				for p, length := range hints {
					speculative.Add(keys[:length], p)
				}
				walked, materialized, idx := hintMatcher(t, backends, speculative)
				populateIndex(t, idx, confirmed)

				got, err := walked.MatchBlockKeys(ctx, keys, filter)
				require.NoError(t, err)
				assertPodMatches(t, matchRows(confirmed), got)

				looked, err := idx.Lookup(ctx, keys, filter)
				require.NoError(t, err)
				got, err = materialized.MatchBlockKeys(ctx, keys, filter)
				require.NoError(t, err)
				assertPodMatches(t, matchRows(looked), got)
			}
		})
	}
}
