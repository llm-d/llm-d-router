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

package kvevents

import (
	"slices"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// anchorNode holds what a later store needs to continue the canonical hash
// chain from any engine block of one BlockStored event whose engine blocks are
// smaller than the canonical block.
type anchorNode struct {
	// tokens are the event's tokens, blockSize per engine block, and engineKeys
	// its engine keys.
	tokens     []uint32
	blockSize  int
	engineKeys []kvblock.BlockHash
	// features are the event's extra features per engine block, cut after the
	// last non-nil entry.
	features []*kvblock.BlockExtraFeatures
	// head is the residual the event chains from: the engine blocks between the
	// last canonical boundary before the event and its first engine block.
	head residual
	// keys are the request keys of the canonical blocks completed by head and tokens.
	keys []kvblock.BlockHash
}

// residual is a run of engine blocks that follows a canonical boundary and
// ends before the next one.
type residual struct {
	// key is the request key of the canonical block ending at the boundary.
	key kvblock.BlockHash
	// tokens, engineKeys and features describe the run's engine blocks.
	tokens     []uint32
	engineKeys []kvblock.BlockHash
	features   []*kvblock.BlockExtraFeatures
}

// headBlocks returns how many of the event's leading engine blocks fall in the
// canonical block that head starts, or 0 when the event does not complete it.
func (n *anchorNode) headBlocks(canonicalBlockSize int) int {
	if len(n.head.tokens) == 0 || len(n.keys) == 0 {
		return 0
	}
	return (canonicalBlockSize - len(n.head.tokens)) / n.blockSize
}

// anchorRef locates one engine block within its event.
type anchorRef struct {
	node  *anchorNode
	block int32
}

// resume returns the residual a store chains from when its parent is the
// referenced engine block. The residual holds copies, so a node keeping it
// does not hold the parent event's arrays.
func (r anchorRef) resume(canonicalBlockSize int) residual {
	n := r.node
	blocks := int(r.block) + 1
	end := blocks * n.blockSize
	span := len(n.head.tokens) + end
	if completed := span / canonicalBlockSize; completed > 0 {
		first := blocks - span%canonicalBlockSize/n.blockSize
		return residual{
			key:        n.keys[completed-1],
			tokens:     slices.Clone(n.tokens[first*n.blockSize : end]),
			engineKeys: slices.Clone(n.engineKeys[first:blocks]),
			features:   trimFeatures(featureRange(n.features, first, blocks)),
		}
	}
	if len(n.head.tokens) == 0 {
		return residual{
			key:        n.head.key,
			tokens:     slices.Clone(n.tokens[:end]),
			engineKeys: slices.Clone(n.engineKeys[:blocks]),
			features:   trimFeatures(featureRange(n.features, 0, blocks)),
		}
	}
	return residual{
		key:        n.head.key,
		tokens:     slices.Concat(n.head.tokens, n.tokens[:end]),
		engineKeys: slices.Concat(n.head.engineKeys, n.engineKeys[:blocks]),
		features: trimFeatures(joinFeatures(n.head.features, len(n.head.engineKeys),
			featureRange(n.features, 0, blocks), blocks)),
	}
}

// featureRange returns features[from:to] padded with nil to to-from entries,
// or nil when the range holds no extra features.
func featureRange(features []*kvblock.BlockExtraFeatures, from, to int) []*kvblock.BlockExtraFeatures {
	if from >= len(features) || !slices.ContainsFunc(features[from:min(to, len(features))],
		func(f *kvblock.BlockExtraFeatures) bool { return f != nil }) {
		return nil
	}
	out := make([]*kvblock.BlockExtraFeatures, to-from)
	copy(out, features[from:min(to, len(features))])
	return out
}

// joinFeatures concatenates the extra features of two runs of engine blocks,
// either of which may be nil or shorter than its block count.
func joinFeatures(a []*kvblock.BlockExtraFeatures, aBlocks int,
	b []*kvblock.BlockExtraFeatures, bBlocks int,
) []*kvblock.BlockExtraFeatures {
	if a == nil && b == nil {
		return nil
	}
	out := make([]*kvblock.BlockExtraFeatures, aBlocks+bBlocks)
	copy(out, a)
	copy(out[aBlocks:], b)
	return out
}

// trimFeatures cuts features after its last non-nil entry.
func trimFeatures(features []*kvblock.BlockExtraFeatures) []*kvblock.BlockExtraFeatures {
	for i := len(features) - 1; i >= 0; i-- {
		if features[i] != nil {
			return features[:i+1]
		}
	}
	return nil
}

// anchorEntry is one anchored engine block and the number of tracked stores of
// it that no remove has released. It repeats anchorRef's fields so that an
// entry takes 16 bytes.
type anchorEntry struct {
	node   *anchorNode
	block  int32
	stores int32
}

func (e anchorEntry) ref() anchorRef {
	return anchorRef{node: e.node, block: e.block}
}

// podAnchors holds one pod's anchors.
type podAnchors struct {
	entries map[kvblock.BlockHash]anchorEntry
	// dependents maps an engine block to the request keys that later stores
	// completed over it. Such a request key is mapped in the index from the
	// later store's engine keys only, so removing the block has to evict it by
	// request key.
	dependents map[kvblock.BlockHash][]kvblock.BlockHash
}

// partialBlockAnchors maps each engine block of a sub-canonical store to its
// position on the canonical hash chain, per pod. An anchor depends only on the
// hash chain, so one entry serves every device tier and KV-cache group holding
// the block. Its store count follows the dedup filter's reference counts summed
// over those scopes, and it lives until the count returns to zero or the pod is
// cleared. An event's tokens stay in memory while any of its blocks is anchored.
type partialBlockAnchors struct {
	mu   sync.Mutex
	pods map[string]*podAnchors
}

func newPartialBlockAnchors() *partialBlockAnchors {
	return &partialBlockAnchors{pods: make(map[string]*podAnchors)}
}

func (a *partialBlockAnchors) lookup(podIdentifier string, engineKey kvblock.BlockHash) (anchorRef, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	pod := a.pods[podIdentifier]
	if pod == nil {
		return anchorRef{}, false
	}
	entry, ok := pod.entries[engineKey]
	return entry.ref(), ok
}

// record anchors the engine blocks of node and counts one store for each. A
// block that is already anchored keeps its anchor: the same engine hash is the
// same position on the chain. The request key node completes over its head is
// registered on the head's engine blocks.
func (a *partialBlockAnchors) record(podIdentifier string, node *anchorNode, canonicalBlockSize int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	pod := a.pods[podIdentifier]
	if pod == nil {
		pod = &podAnchors{entries: make(map[kvblock.BlockHash]anchorEntry, len(node.engineKeys))}
		a.pods[podIdentifier] = pod
	}
	for i, key := range node.engineKeys {
		entry, ok := pod.entries[key]
		if !ok {
			entry.node, entry.block = node, int32(i) //#nosec G115 -- bounded by the event's block count
		}
		entry.stores++
		pod.entries[key] = entry
	}
	if node.headBlocks(canonicalBlockSize) == 0 {
		return
	}
	if pod.dependents == nil {
		pod.dependents = make(map[kvblock.BlockHash][]kvblock.BlockHash)
	}
	for _, key := range node.head.engineKeys {
		if !slices.Contains(pod.dependents[key], node.keys[0]) {
			pod.dependents[key] = append(pod.dependents[key], node.keys[0])
		}
	}
}

// retain counts one more store for each of engineKeys that is anchored. It
// mirrors a store that announces blocks without their tokens.
func (a *partialBlockAnchors) retain(podIdentifier string, engineKeys []kvblock.BlockHash) {
	a.mu.Lock()
	defer a.mu.Unlock()
	pod := a.pods[podIdentifier]
	if pod == nil {
		return
	}
	for _, key := range engineKeys {
		if entry, ok := pod.entries[key]; ok {
			entry.stores++
			pod.entries[key] = entry
		}
	}
}

// remove mirrors one BlockRemoved event after the dedup filter. Each hash in
// removed releases one store of its engine block, except the hashes in
// untracked, a subsequence of removed that matched no store in the event's
// scope. The anchor goes with its last store. It returns the request keys that
// later stores completed over the blocks in evicted, which the caller evicts
// from the index for the event's scope.
func (a *partialBlockAnchors) remove(
	podIdentifier string, modelSeed uint64, removed, untracked, evicted []uint64, canonicalBlockSize int,
) []kvblock.BlockHash {
	a.mu.Lock()
	defer a.mu.Unlock()
	pod := a.pods[podIdentifier]
	if pod == nil {
		return nil
	}

	var dependents []kvblock.BlockHash
	if len(pod.dependents) > 0 {
		for _, hash := range evicted {
			for _, requestKey := range pod.dependents[namespacedEngineKey(modelSeed, hash)] {
				if !slices.Contains(dependents, requestKey) {
					dependents = append(dependents, requestKey)
				}
			}
		}
	}

	for _, hash := range removed {
		if len(untracked) > 0 && untracked[0] == hash {
			untracked = untracked[1:]
			continue
		}
		key := namespacedEngineKey(modelSeed, hash)
		entry, ok := pod.entries[key]
		if !ok {
			continue
		}
		if entry.stores > 1 {
			entry.stores--
			pod.entries[key] = entry
			continue
		}
		delete(pod.entries, key)
		delete(pod.dependents, key)
		// The request key this event completed over its head is mapped from its
		// first engine blocks and leaves the index with them.
		if n := entry.node; int(entry.block) < n.headBlocks(canonicalBlockSize) {
			for _, headKey := range n.head.engineKeys {
				pod.forget(headKey, n.keys[0])
			}
		}
	}
	if len(pod.entries) == 0 {
		delete(a.pods, podIdentifier)
	}
	return dependents
}

// forget drops requestKey from the dependents of engineKey.
func (p *podAnchors) forget(engineKey, requestKey kvblock.BlockHash) {
	keys := p.dependents[engineKey]
	i := slices.Index(keys, requestKey)
	if i < 0 {
		return
	}
	if len(keys) == 1 {
		delete(p.dependents, engineKey)
		return
	}
	p.dependents[engineKey] = slices.Delete(keys, i, i+1)
}

func (a *partialBlockAnchors) clear(podIdentifier string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.pods, podIdentifier)
}
