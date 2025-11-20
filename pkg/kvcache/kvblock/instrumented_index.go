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

	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/metrics"
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

func (m *instrumentedIndex) Add(ctx context.Context, keys []Key, entries []PodEntry) error {
	err := m.next.Add(ctx, keys, entries)
	metrics.Admissions.Add(float64(len(keys)))
	return err
}

func (m *instrumentedIndex) Evict(ctx context.Context, key Key, entries []PodEntry) error {
	err := m.next.Evict(ctx, key, entries)
	metrics.Evictions.Add(float64(len(entries)))
	return err
}

func (m *instrumentedIndex) Lookup(
	ctx context.Context,
	keys []Key,
	podIdentifierSet sets.Set[string],
) (map[Key][]PodEntry, error) {
	timer := prometheus.NewTimer(metrics.LookupLatency)
	defer timer.ObserveDuration()

	metrics.LookupRequests.Inc()

	pods, err := m.next.Lookup(ctx, keys, podIdentifierSet)
	if err != nil {
		return nil, err
	}

	go recordHitMetrics(pods)

	return pods, nil
}

func recordHitMetrics(keyToPods map[Key][]PodEntry) {
	podCount := make(map[string]int)
	for _, pods := range keyToPods {
		for _, p := range pods {
			// First time seeing this pod in current lookup window.
			// set to 1 because counts are local to this call (not cumulative over time).
			// This ensures compatibility with sliding window attention (SWA) and cache eviction,
			// where only recent hits within the active window are considered.
			podCount[p.PodIdentifier]++
		}
	}

	maxHit := 0
	for _, count := range podCount {
		if count > maxHit {
			maxHit = count
		}
	}

	metrics.MaxPodHitCount.Add(float64(maxHit))
	metrics.LookupHits.Add(float64(maxHit))
}
