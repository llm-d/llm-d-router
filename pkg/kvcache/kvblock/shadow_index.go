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

package kvblock

import (
	"context"
	"sort"
	"sync/atomic"

	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
	"k8s.io/apimachinery/pkg/util/sets"
)

// ShadowIndex validates one index against another in production: every
// write fans into both backends and reads are served by the primary, while
// a sampled fraction of reads compares the two backends' entry sets key by
// key and exports divergence counters. The comparison is over per-key
// PodEntry sets in name space, which every backend produces through Lookup;
// scoring is shared consumer code, so entry-set equality implies equal match
// results. Divergence is expected to be nonzero at a small rate: a write
// fanned to the primary but not yet to the shadow, and each backend's own
// budget eviction, both legally diverge the views. Acceptance is therefore a
// rate budget on divergent over compared keys evaluated on the exported
// counters, not exact equality.
//
// The shadow must never fail or slow the primary's contract: shadow write
// errors are counted and swallowed, and comparison runs after the primary's
// result is already in hand. Shadow errors leave the shadow permanently
// behind the primary, which the divergence counters surface. GetRequestKey
// resolves through the primary only; the engine-key mapping is not compared.
//
// Memory cost is roughly the sum of both backends while enabled.
type shadowIndex struct {
	primary Index
	shadow  Index
	// sampleEvery is the read sampling period; zero disables comparison.
	sampleEvery uint64
	reads       atomic.Uint64
}

// shadowWalker carries the KeyWalker and OrdinalResolver capabilities of the
// primary index.
type shadowWalker struct {
	*shadowIndex
	walker KeyWalker
	resolverForwarder
}

// NewShadowIndex wraps primary and fans writes into shadow. Every
// sampleEvery-th read compares the backends; sampleEvery <= 0 fans writes
// without comparing. The wrapper is a KeyWalker exactly when the primary is
// one, and resolves ordinals through the primary's interner.
func NewShadowIndex(primary, shadow Index, sampleEvery int) Index {
	if sampleEvery < 0 {
		sampleEvery = 0
	}
	s := &shadowIndex{primary: primary, shadow: shadow, sampleEvery: uint64(sampleEvery)} //#nosec G115 -- clamped non-negative
	if walker, ok := primary.(KeyWalker); ok {
		resolver, _ := primary.(OrdinalResolver)
		return &shadowWalker{shadowIndex: s, walker: walker, resolverForwarder: resolverForwarder{resolver: resolver}}
	}
	return s
}

func (s *shadowIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	err := s.primary.Add(ctx, engineKeys, requestKeys, entries)
	if serr := s.shadow.Add(ctx, engineKeys, requestKeys, entries); serr != nil {
		metrics.IndexShadowErrors.Inc()
	}
	return err
}

func (s *shadowIndex) Evict(ctx context.Context, keyType KeyType, keys []BlockHash, entries []PodEntry) error {
	err := s.primary.Evict(ctx, keyType, keys, entries)
	if serr := s.shadow.Evict(ctx, keyType, keys, entries); serr != nil {
		metrics.IndexShadowErrors.Inc()
	}
	return err
}

func (s *shadowIndex) Clear(ctx context.Context, podIdentifier string) error {
	err := s.primary.Clear(ctx, podIdentifier)
	if serr := s.shadow.Clear(ctx, podIdentifier); serr != nil {
		metrics.IndexShadowErrors.Inc()
	}
	return err
}

func (s *shadowIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	return s.primary.GetRequestKey(ctx, engineKey)
}

func (s *shadowIndex) Lookup(
	ctx context.Context,
	requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	result, err := s.primary.Lookup(ctx, requestKeys, podIdentifierSet)
	if err == nil && s.sampleRead(len(requestKeys)) {
		s.compareAgainst(ctx, requestKeys, podIdentifierSet)
	}
	return result, err
}

// WalkKeys serves the walk from the primary and, on a sampled read,
// compares the backends' views of the whole key list afterwards. The
// walk's early stop does not bound the comparison: the entry sets are
// the contract being compared, not one walk's chain.
func (m *shadowWalker) WalkKeys(ctx context.Context, requestKeys []BlockHash,
	visit func(pos int, found bool, entries []WalkEntry) bool,
) error {
	err := m.walker.WalkKeys(ctx, requestKeys, visit)
	if err == nil && m.sampleRead(len(requestKeys)) {
		m.compareAgainst(ctx, requestKeys, nil)
	}
	return err
}

// sampleRead reports whether this read is a comparison sample. The counter
// runs over reads, not keys, so the sampling period is per read call.
func (s *shadowIndex) sampleRead(keys int) bool {
	if s.sampleEvery == 0 || keys == 0 {
		return false
	}
	return s.reads.Add(1)%s.sampleEvery == 0
}

// compareAgainst counts keys whose entry sets differ between the two
// backends under the same filter. Lookup stops a key list at its first
// missing key, so the comparison queries each key on its own: a mid-list
// miss in one backend must not mark every later key divergent in the other.
func (s *shadowIndex) compareAgainst(
	ctx context.Context,
	requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) {
	divergent := 0
	one := make([]BlockHash, 1)
	for _, key := range requestKeys {
		one[0] = key
		primaryResult, err := s.primary.Lookup(ctx, one, podIdentifierSet)
		if err != nil {
			metrics.IndexShadowErrors.Inc()
			return
		}
		shadowResult, err := s.shadow.Lookup(ctx, one, podIdentifierSet)
		if err != nil {
			metrics.IndexShadowErrors.Inc()
			return
		}
		if !equalPodEntries(primaryResult[key], shadowResult[key]) {
			divergent++
		}
	}
	metrics.IndexShadowCompared.Add(float64(len(requestKeys)))
	metrics.IndexShadowDivergent.Add(float64(divergent))
}

// equalPodEntries reports whether two entry lists hold the same multiset of
// PodEntry values. Both lists are copied before sorting: the primary's
// Lookup result belongs to its caller.
func equalPodEntries(a, b []PodEntry) bool {
	if len(a) != len(b) {
		return false
	}
	if len(a) == 0 {
		return true
	}
	sortedA := append([]PodEntry(nil), a...)
	sortedB := append([]PodEntry(nil), b...)
	sort.Slice(sortedA, func(i, j int) bool { return podEntryLess(sortedA[i], sortedA[j]) })
	sort.Slice(sortedB, func(i, j int) bool { return podEntryLess(sortedB[i], sortedB[j]) })
	for i := range sortedA {
		if sortedA[i] != sortedB[i] {
			return false
		}
	}
	return true
}

func podEntryLess(a, b PodEntry) bool {
	if a.PodIdentifier != b.PodIdentifier {
		return a.PodIdentifier < b.PodIdentifier
	}
	if a.DeviceTier != b.DeviceTier {
		return a.DeviceTier < b.DeviceTier
	}
	if a.Speculative != b.Speculative {
		return a.Speculative
	}
	if a.HasGroup != b.HasGroup {
		return a.HasGroup
	}
	return a.GroupIdx < b.GroupIdx
}
