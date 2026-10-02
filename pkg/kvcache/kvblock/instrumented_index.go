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

// instrumentedWalker carries the KeyWalker capability of the wrapped index.
type instrumentedWalker struct {
	*instrumentedIndex
	walker KeyWalker
}

// NewInstrumentedIndex wraps an Index and emits metrics for Add, Evict,
// Lookup, and WalkKeys. The wrapper is a KeyWalker exactly when next is one.
// Read metrics count and time Lookup and WalkKeys calls; contiguous-chain
// hit metrics are recorded by the kvcache matcher.
func NewInstrumentedIndex(next Index) Index {
	m := &instrumentedIndex{next: next}
	if walker, ok := next.(KeyWalker); ok {
		return &instrumentedWalker{instrumentedIndex: m, walker: walker}
	}
	return m
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

	return m.next.Lookup(ctx, requestKeys, podIdentifierSet)
}

// WalkKeys forwards the walk as a lookup request.
func (m *instrumentedWalker) WalkKeys(ctx context.Context, requestKeys []BlockHash,
	visit func(pos int, found bool, entries []EntryRef) bool,
) error {
	timer := prometheus.NewTimer(metrics.LookupLatency)
	defer timer.ObserveDuration()

	metrics.LookupRequests.Inc()

	return m.walker.WalkKeys(ctx, requestKeys, visit)
}

func (m *instrumentedIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	return m.next.GetRequestKey(ctx, engineKey)
}

func (m *instrumentedIndex) Clear(ctx context.Context, podIdentifier string) error {
	return m.next.Clear(ctx, podIdentifier)
}
