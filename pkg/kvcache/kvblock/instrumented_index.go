// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvblock

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/apimachinery/pkg/util/sets"
)

type instrumentedIndex struct {
	next Index
}

// NewInstrumentedIndex wraps an Index and emits metrics for Add, Evict, and
// Lookup.
func NewInstrumentedIndex(next Index) Index {
	return &instrumentedIndex{next: next}
}

func (m *instrumentedIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	err := m.next.Add(ctx, engineKeys, requestKeys, entries)
	metrics.Admissions.Add(float64(len(requestKeys)))
	return err
}

func (m *instrumentedIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error {
	err := m.next.Evict(ctx, key, keyType, entries)
	metrics.Evictions.Add(float64(len(entries)))
	return err
}

func (m *instrumentedIndex) Lookup(
	ctx context.Context,
	requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	timer := prometheus.NewTimer(metrics.LookupLatency)
	defer timer.ObserveDuration()

	metrics.LookupRequests.Inc()

	pods, err := m.next.Lookup(ctx, requestKeys, podIdentifierSet)
	if err != nil {
		return nil, err
	}

	// Synchronous: callers own requestKeys and the result map after return,
	// so a deferred fold would race their reuse.
	recordHitMetrics(requestKeys, pods)

	return pods, nil
}

func (m *instrumentedIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	return m.next.GetRequestKey(ctx, engineKey)
}

func (m *instrumentedIndex) Clear(ctx context.Context, podIdentifier string) error {
	return m.next.Clear(ctx, podIdentifier)
}

func recordHitMetrics(requestKeys []BlockHash, keyToPods map[BlockHash][]PodEntry) {
	maxHit := maxContiguousPodHits(requestKeys, keyToPods)
	metrics.MaxPodHitCount.Add(float64(maxHit))
	metrics.LookupHits.Add(float64(maxHit))
}

// maxContiguousPodHits returns the longest contiguous prefix chain any single
// pod holds, counting from the first request key. One state map serves the
// whole fold; a pod stays in the chain while its last-seen key is the
// preceding one, and duplicate device tiers at a key count once.
func maxContiguousPodHits(requestKeys []BlockHash, keyToPods map[BlockHash][]PodEntry) int {
	if len(requestKeys) == 0 {
		return 0
	}
	type podState struct {
		lastKey int
		count   int
	}
	state := make(map[string]podState, len(keyToPods[requestKeys[0]]))
	best := 0
	for i, key := range requestKeys {
		entries := keyToPods[key]
		if len(entries) == 0 {
			break // no pod holds this key: every chain from key 0 ends here
		}
		advanced := false
		for _, e := range entries {
			s, ok := state[e.PodIdentifier]
			switch {
			case i == 0 && !ok:
				state[e.PodIdentifier] = podState{lastKey: 0, count: 1}
				advanced = true
				if best == 0 {
					best = 1
				}
			case ok && s.lastKey == i-1:
				s.lastKey = i
				s.count++
				state[e.PodIdentifier] = s
				advanced = true
				if s.count > best {
					best = s.count
				}
			default:
				// Duplicate tier at this key, or a pod outside the chain.
			}
		}
		if !advanced {
			break // no chain advanced at this key; later keys cannot either
		}
	}
	return best
}
