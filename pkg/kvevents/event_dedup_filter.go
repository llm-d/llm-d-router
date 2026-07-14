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

package kvevents

import "sync"

// noGroupIdx is the sentinel groupIdx used in a dedup scope when an event
// carries no KV-cache group. It mirrors the "no group" PodEntry state
// (HasGroup=false), so grouped and ungrouped copies of the same block hash are
// reference-counted independently — matching how the index treats them as
// distinct entries. Real event group indices are non-negative (the engine
// adapters reject a negative group_idx), so this sentinel cannot collide with a
// genuine group.
const noGroupIdx = -1

// noDataParallelRank is the sentinel data-parallel rank used in a dedup scope.
//
// On current main the index identity (kvblock.PodEntry) is pod-level and does
// NOT distinguish data-parallel ranks, so every scope uses this sentinel and
// reference counts aggregate across ranks — which is exactly what the pod-level
// index requires (a block is still resident on the pod until every rank has
// removed it). The value matches PR #370's NoDataParallelRank (-1) convention.
//
// TODO(#370): once DataParallelRank is propagated onto EventBatch and into
// PodEntry, source the rank from the event in pool.go so the dedup scope
// becomes DP-aware in lockstep with the (then DP-aware) index identity. No
// change to this file is required — only the scope construction at the call
// sites.
const noDataParallelRank = -1

// groupIdxOrNoGroup maps an optional event group index to the dedup-scope int,
// substituting noGroupIdx for an absent group.
func groupIdxOrNoGroup(groupIdx *int) int {
	if groupIdx == nil {
		return noGroupIdx
	}
	return *groupIdx
}

// blockScope identifies the set of block reference counts that share a single
// index eviction identity for one pod. Its fields mirror the dimensions of
// kvblock.PodEntry that determine which stored entry an eviction targets: pod,
// device tier, KV-cache group, and (in future, see noDataParallelRank)
// data-parallel rank.
type blockScope struct {
	podIdentifier    string
	deviceTier       string
	groupIdx         int
	dataParallelRank int
}

// dedupKey is the per-block reference-count key within a single pod's bucket.
type dedupKey struct {
	deviceTier       string
	groupIdx         int
	dataParallelRank int
	blockHash        uint64
}

func (s blockScope) key(blockHash uint64) dedupKey {
	return dedupKey{
		deviceTier:       s.deviceTier,
		groupIdx:         s.groupIdx,
		dataParallelRank: s.dataParallelRank,
		blockHash:        blockHash,
	}
}

// eventDedupFilter reference-counts BlockStored/BlockRemoved announcements so
// that duplicate removes do not evict a block another announcement still
// references. vLLM's OffloadingConnector legitimately emits such duplicates in
// chunk mode: when a shared prefix is not aligned to the offloaded chunk size,
// two sibling chunks list the same constituent block hash, so the same hash is
// stored and removed more than once on the wire.
//
// The filter mirrors the wire event stream rather than the index state: every
// BlockStored increments the count for its hashes unconditionally (duplicates
// included), and a BlockRemoved only forwards a hash to the index once its
// count returns to zero. Removes for never-seen hashes pass through defensively
// (the index treats an unknown evict as a no-op), and a count never goes
// negative. The index may independently evict an entry (LRU/cost) without a
// wire remove; that only makes the filter's count an over-estimate, which stays
// safe because a suppressed or forwarded evict on an already-absent entry is a
// no-op either way.
//
// Memory footprint: the per-pod map is bounded by the block hashes currently
// outstanding on the wire. Under a correct emitter every announced store is
// matched by a remove when the engine evicts the block (vLLM's offload tracker
// pops on eviction and emits BlockRemoved), so the map tracks the engine's
// offload-pool capacity and drains in steady state; an entry is also reclaimed
// the moment its count returns to zero, and clear() drops a whole pod on
// AllBlocksCleared. The only growth paths are lost ZMQ removes or an
// index-internal eviction with no matching wire remove, both of which leave a
// harmless over-estimate (see above) rather than a correctness error. This
// mirrors the bounded-by-the-wire behavior of Dynamo's EventDedupFilter.
//
// It is safe for concurrent use: the single mutex guards the whole cross-pod
// map. The Pool additionally shards work by pod identifier (see Pool.AddTask),
// so a given pod's events are serialized onto one worker and mutex contention
// is low in practice.
type eventDedupFilter struct {
	mu   sync.Mutex
	refs map[string]map[dedupKey]int // podIdentifier -> per-block reference count
}

func newEventDedupFilter() *eventDedupFilter {
	return &eventDedupFilter{refs: make(map[string]map[dedupKey]int)}
}

// trackStore records one reference for each stored block hash within scope.
// Duplicate stores intentionally increment the count.
func (f *eventDedupFilter) trackStore(scope blockScope, blockHashes []uint64) {
	if f == nil || len(blockHashes) == 0 {
		return
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	bucket := f.refs[scope.podIdentifier]
	if bucket == nil {
		bucket = make(map[dedupKey]int)
		f.refs[scope.podIdentifier] = bucket
	}
	for _, h := range blockHashes {
		bucket[scope.key(h)]++
	}
}

// filterRemove decrements the reference count for each removed block hash and
// returns only the hashes that should actually be evicted from the index: a
// hash is forwarded when its count reaches zero, or when it was never tracked
// (defensive pass-through). The returned slice preserves the input order and
// multiplicity.
func (f *eventDedupFilter) filterRemove(scope blockScope, blockHashes []uint64) []uint64 {
	if f == nil || len(blockHashes) == 0 {
		return blockHashes
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	bucket := f.refs[scope.podIdentifier]
	kept := make([]uint64, 0, len(blockHashes))
	for _, h := range blockHashes {
		if bucket == nil {
			kept = append(kept, h) // unknown pod -> defensive pass-through
			continue
		}
		k := scope.key(h)
		count, ok := bucket[k]
		if !ok || count <= 0 {
			kept = append(kept, h) // never tracked -> defensive pass-through
			continue
		}
		count--
		if count == 0 {
			delete(bucket, k)
			kept = append(kept, h) // last reference released -> evict for real
			continue
		}
		bucket[k] = count // still referenced -> suppress this remove
	}
	if bucket != nil && len(bucket) == 0 {
		delete(f.refs, scope.podIdentifier)
	}
	return kept
}

// clear drops all reference counts for a pod. It is invoked on
// AllBlocksCleared, in lockstep with the index's pod-wide eager clear, so the
// filter does not retain stale references after the engine resets its prefix
// cache.
func (f *eventDedupFilter) clear(podIdentifier string) {
	if f == nil {
		return
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.refs, podIdentifier)
}
