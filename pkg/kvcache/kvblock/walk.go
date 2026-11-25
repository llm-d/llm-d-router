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
	"math"
)

// EntryRef is one indexed entry with the ordinals the index assigned to its
// pod identifier and device tier. Two entries share a PodOrdinal exactly
// when they share a PodIdentifier, and a TierOrdinal exactly when they share
// a DeviceTier. An ordinal is assigned on the first Add of its name and
// belongs to that name for the index lifetime: Evict and Clear never reclaim
// or reassign one, and an Add whose new names would exceed the index's caps
// fails before writing anything. Live entries hold a sparse subset of the
// assigned range, so consumers key request-local tables by ordinals and
// never size state by them.
type EntryRef struct {
	PodEntry
	PodOrdinal  uint32
	TierOrdinal uint32
}

// WalkEntry is the decoded walk payload: one (pod ordinal, tier ordinal,
// speculative) triple per (pod, tier) pair a key holds. The ordinals are the
// index's EntryRef ordinals; Speculative marks an entry stored without
// engine confirmation. The reserved ordinal math.MaxUint32 never appears as
// a TierOrdinal: no interner assigns it, and consumers reserve it for the
// speculative chain.
type WalkEntry struct {
	PodOrdinal  uint32
	TierOrdinal uint32
	Speculative bool
}

// ReservedTierOrdinal is the tier ordinal consumers reserve for the
// speculative chain. Interners assign ordinals from zero upward, so it never
// names a real tier.
const ReservedTierOrdinal = math.MaxUint32

// OrdinalResolver is an optional Index capability: the index's name-to-
// ordinal assignment for pod identifiers and device tiers. A consumer uses
// it to move string-keyed request state into ordinal space once per request
// (a pod filter, configured tier weights), and to resolve ordinals back to
// names when materializing string-keyed results. Every KeyWalker also
// implements OrdinalResolver: the walk payload carries ordinals, so a
// consumer must be able to interpret them.
//
// An ordinal's meaning never changes over the index lifetime (see EntryRef),
// so a name resolved at the start of a request still names the same ordinal
// at result time. A name not yet interned has no ordinal and matches
// nothing, which is correct for filter inversion.
type OrdinalResolver interface {
	// PodOrdinal returns the ordinal assigned to the pod identifier, and
	// whether it has one.
	PodOrdinal(name string) (uint32, bool)
	// PodName returns the pod identifier assigned to ordinal, and whether
	// it has one.
	PodName(ordinal uint32) (string, bool)
	// TierOrdinal returns the ordinal assigned to the device tier, and
	// whether it has one.
	TierOrdinal(name string) (uint32, bool)
	// TierName returns the device tier assigned to ordinal, and whether it
	// has one.
	TierName(ordinal uint32) (string, bool)
}

// resolverForwarder forwards ordinal resolution to a wrapped index, so a
// decorator that carries the wrapped KeyWalker inherits the OrdinalResolver
// methods by embedding instead of restating them.
type resolverForwarder struct {
	resolver OrdinalResolver
}

func (f resolverForwarder) PodOrdinal(name string) (uint32, bool) {
	return f.resolver.PodOrdinal(name)
}

func (f resolverForwarder) PodName(ordinal uint32) (string, bool) {
	return f.resolver.PodName(ordinal)
}

func (f resolverForwarder) TierOrdinal(name string) (uint32, bool) {
	return f.resolver.TierOrdinal(name)
}

func (f resolverForwarder) TierName(ordinal uint32) (string, bool) {
	return f.resolver.TierName(ordinal)
}

// KeyWalker is an optional Index capability: visit requestKeys in input
// order without materializing per-key entry slices.
//
// The walk contract:
//
//   - visit runs once per position of requestKeys, in order, until it
//     returns false, the keys run out, or ctx is cancelled. A key listed
//     twice is visited at both positions, each visit reading the index anew.
//   - found reports whether the index held a generation of the key when the
//     position was looked up. A miss is visited with found=false and no
//     entries; the walk continues.
//   - entries is every unfiltered (pod, tier) pair in that generation, one
//     WalkEntry each, in an order the consumer must not rely on. It is
//     borrowed: read-only and valid only until visit returns, after which
//     the index may reorder or overwrite its backing array. Consumers copy
//     what they keep.
//   - visit must not call back into the index. If visit panics, any lock
//     held for the call is released and the panic propagates to the caller.
//   - A visit sees an internally consistent generation, exclusive of writers
//     and other visits using that generation. Capacity eviction may detach it
//     and a later Add may install a new generation of the same key while
//     visit runs. There is no snapshot across positions: a concurrent Add,
//     Evict, or Clear may be visible at some positions and not at others.
//   - Cancellation is polled at the first position, every 256th position
//     after it, and once more when the walk ends, so a cancelled ctx always
//     yields ctx.Err(); visits may run between the cancellation and the
//     next poll.
//   - When the walk ends, for any reason, the visited prefix is refreshed
//     per the implementation's recency rule under one acquisition, skipping
//     keys absent by then: InMemoryIndex promotes the prefix through the
//     last key found to most recently used in position order, and positions
//     after the last key found are not promoted. Lookup promotes the same
//     way, so a prefix that is only read stays resident under capacity
//     pressure.
type KeyWalker interface {
	WalkKeys(ctx context.Context, requestKeys []BlockHash,
		visit func(pos int, found bool, entries []WalkEntry) bool) error
}

// Walk batches start small and double, so a walk that stops early peeks at
// most walkBatchMin keys or about twice the keys it visits, whichever is more.
const (
	walkBatchMin = 8
	walkBatchMax = 256
)

// WalkKeys implements KeyWalker. Keys are peeked in batches under one
// acquisition of the LRU's shared lock, and each key's entries are visited
// under that key's own lock after the shared lock is released, since evicting
// a key takes its lock before the LRU's. The visited prefix is promoted in a
// deferred call, so every exit path refreshes what was read.
func (m *InMemoryIndex) WalkKeys(ctx context.Context, requestKeys []BlockHash,
	visit func(pos int, found bool, entries []WalkEntry) bool,
) error {
	visited := 0
	// Every exit, cancellation included, refreshes what was read.
	defer func() { m.data.Promote(requestKeys[:visited]) }()
	var buf [walkBatchMax]*PodCache
	// scratch materializes WalkEntries from the visited PodCache
	// generations; one buffer serves the whole walk.
	scratch := make([]WalkEntry, 0, 8)
	batch := walkBatchMin
	for start := 0; start < len(requestKeys); {
		end := min(start+batch, len(requestKeys))
		for i, pc := range m.data.PeekInto(requestKeys[start:end], buf[:]) {
			pos := start + i
			if pos&cancellationCheckMask == 0 && ctx.Err() != nil {
				return ctx.Err()
			}
			if pc == nil {
				if !visit(pos, false, nil) {
					return ctx.Err()
				}
				continue
			}
			visited = pos + 1
			if !pc.visitEntries(pos, visit, &scratch) {
				return ctx.Err()
			}
		}
		start = end
		batch = min(2*batch, walkBatchMax)
	}
	return ctx.Err()
}

// visitEntries materializes the cache's entries as WalkEntries, one per
// distinct (pod, tier, speculative) triple, and hands them to visit under
// the cache's lock. Duplicate triples (the same pod and tier under different
// groups) fold to one entry: the walk payload is per (pod, tier) pair.
// Duplicates can only coexist when entries differ in their group fields, so
// the duplicate scan runs only while the cache holds such an entry; the
// grouped flag is recomputed on the first visit after a mutation.
func (pc *PodCache) visitEntries(pos int, visit func(int, bool, []WalkEntry) bool, scratch *[]WalkEntry) bool {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if !pc.groupedKnown {
		pc.grouped = false
		for i := range pc.entries {
			if pc.entries[i].HasGroup || pc.entries[i].GroupIdx != 0 {
				pc.grouped = true
				break
			}
		}
		pc.groupedKnown = true
	}
	entries := (*scratch)[:0]
	for i := range pc.entries {
		ref := &pc.entries[i]
		e := WalkEntry{PodOrdinal: ref.PodOrdinal, TierOrdinal: ref.TierOrdinal, Speculative: ref.Speculative}
		if pc.grouped {
			dup := false
			for j := range entries {
				if entries[j] == e {
					dup = true
					break
				}
			}
			if dup {
				continue
			}
		}
		entries = append(entries, e)
	}
	*scratch = entries
	return visit(pos, true, entries)
}
