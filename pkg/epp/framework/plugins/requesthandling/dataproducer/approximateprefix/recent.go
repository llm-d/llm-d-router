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

package approximateprefix

import (
	"encoding/binary"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/common/clamp"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// windowSlot is one sequenced element. A drop slot retracts dropGen.
// Empty hashes are a withdrawn prompt.
type windowSlot struct {
	hashes  []blockHash
	drop    bool
	dropGen uint64
}

// endpointWindow holds one endpoint's sequenced elements. slots[start:] are
// live. Trimmed elements stay in front of start until compaction moves the live
// tail to the front, so trimming is amortized constant per element.
type endpointWindow struct {
	slots []windowSlot
	start int
	// total is the hash count of live content slots.
	total int
	// base is the generation of slots[start]. It advances by one for each
	// element trimmed, so a generation always names the same element.
	base uint64
}

// recentWindow buffers recently recorded prompt hash sequences per endpoint.
// Each content slot is one prompt's hashes starting at block 0. Cap is on total
// hashes across prompts so cross-replica publish payloads stay small.
type recentWindow struct {
	mu      sync.Mutex
	pending map[string]*endpointWindow
	max     int
}

func newRecentWindow(max int) *recentWindow {
	if max <= 0 {
		max = defaultLRUCapacityPerServer
	}
	return &recentWindow{
		pending: make(map[string]*endpointWindow),
		max:     max,
	}
}

// record appends one prompt's hashes for endpointID and trims oldest prompts
// until the total hash count is at most max. A single oversized prompt is
// head-capped to max so matching still anchors at block 0. It returns the
// stored entry, which replace identifies by its backing array.
func (w *recentWindow) record(endpointID string, hashes []blockHash) []blockHash {
	if len(hashes) == 0 {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	cp := w.headCopy(hashes)
	buf := w.pending[endpointID]
	if buf == nil {
		buf = &endpointWindow{}
		w.pending[endpointID] = buf
	}
	buf.slots = append(buf.slots, windowSlot{hashes: cp})
	buf.total += len(cp)
	buf.trim(w.max)
	return cp
}

// replace swaps the entry that record returned as entry for hashes. A stream
// element is published once, so the old slot is withdrawn and the new hashes
// are appended as the newest slot. It does nothing when the entry has already
// been withdrawn or trimmed, or the endpoint forgotten, so entries recorded
// since, including an equal prompt of another request, are left as they are.
func (w *recentWindow) replace(endpointID string, entry, hashes []blockHash) {
	if len(entry) == 0 || len(hashes) == 0 {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	buf := w.pending[endpointID]
	if buf == nil {
		return
	}
	for j := buf.start; j < len(buf.slots); j++ {
		slot := &buf.slots[j]
		if slot.drop || len(slot.hashes) == 0 || &slot.hashes[0] != &entry[0] {
			continue
		}
		cp := w.headCopy(hashes)
		gen := buf.base + clamp.Uint64(j-buf.start)
		buf.total -= len(slot.hashes)
		slot.hashes = nil
		buf.slots = append(buf.slots, windowSlot{drop: true, dropGen: gen}, windowSlot{hashes: cp})
		buf.total += len(cp)
		buf.trim(w.max)
		return
	}
}

// headCopy copies hashes, capped to max so matching still anchors at block 0.
func (w *recentWindow) headCopy(hashes []blockHash) []blockHash {
	if len(hashes) > w.max {
		hashes = hashes[:w.max]
	}
	cp := make([]blockHash, len(hashes))
	copy(cp, hashes)
	return cp
}

// snapshot returns a deep copy of pending prompts for endpointID without clearing.
// Peers read the window as a whole on every publish, so draining would wipe
// peer-visible hashes on the next idle publish tick.
func (w *recentWindow) snapshot(endpointID string) [][]blockHash {
	prompts, _ := w.snapshotWindow(endpointID)
	live := prompts[:0:0]
	for _, p := range prompts {
		if len(p) > 0 {
			live = append(live, p)
		}
	}
	if len(live) == 0 {
		return nil
	}
	return live
}

// snapshotWindow returns content slots, including empty withdrawn prompts, and
// the generation of the first element. Retraction slots are omitted.
func (w *recentWindow) snapshotWindow(endpointID string) ([][]blockHash, uint64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	buf := w.pending[endpointID]
	if buf == nil {
		return nil, 0
	}
	var prompts [][]blockHash
	for _, slot := range buf.slots[buf.start:] {
		if slot.drop {
			continue
		}
		prompts = append(prompts, append([]blockHash(nil), slot.hashes...))
	}
	return prompts, buf.base
}

// sequenced is the value incremental replication publishes for endpointID.
func (w *recentWindow) sequenced(endpointID string) *attrprefix.PrefixHashSet {
	w.mu.Lock()
	defer w.mu.Unlock()
	set := &attrprefix.PrefixHashSet{}
	buf := w.pending[endpointID]
	if buf == nil {
		return set
	}
	set.Base = buf.base
	for _, slot := range buf.slots[buf.start:] {
		if slot.drop {
			set.AppendRetract(slot.dropGen)
			continue
		}
		set.AppendContent(hashesToUint64(slot.hashes))
	}
	return set
}

// forget drops the buffer for endpointID when the endpoint leaves the pool.
func (w *recentWindow) forget(endpointID string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.pending, endpointID)
}

// removePrompt empties a matching prompt and appends a retraction of its
// generation, then trims. Hashes use the same head cap as record. A prompt
// already trimmed out is a no-op.
func (w *recentWindow) removePrompt(endpointID string, hashes []blockHash) {
	if len(hashes) == 0 {
		return
	}
	prompt := hashes
	if len(prompt) > w.max {
		prompt = prompt[:w.max]
	}
	key := promptKey(prompt)
	w.mu.Lock()
	defer w.mu.Unlock()
	buf := w.pending[endpointID]
	if buf == nil {
		return
	}
	matched := false
	for j := buf.start; j < len(buf.slots); j++ {
		slot := &buf.slots[j]
		if slot.drop || promptKey(slot.hashes) != key {
			continue
		}
		gen := buf.base + clamp.Uint64(j-buf.start)
		buf.total -= len(slot.hashes)
		slot.hashes = nil
		buf.slots = append(buf.slots, windowSlot{drop: true, dropGen: gen})
		matched = true
	}
	if !matched {
		return
	}
	buf.trim(w.max)
}

// trim drops the oldest slots until the hash count and the slot count are each
// at most max, then drops a leading run of withdrawn slots. Those hold no
// hashes, so the hash cap alone does not bound them.
func (e *endpointWindow) trim(max int) {
	for e.total > max && len(e.slots)-e.start > 1 {
		e.dropFront()
	}
	for len(e.slots)-e.start > max {
		e.dropFront()
	}
	for len(e.slots)-e.start > 0 && withdrawnSlot(e.slots[e.start]) {
		e.dropFront()
	}
	e.compact()
}

func (e *endpointWindow) dropFront() {
	slot := e.slots[e.start]
	if !slot.drop {
		e.total -= len(slot.hashes)
	}
	e.slots[e.start] = windowSlot{}
	e.start++
	e.base++
}

// compact moves the live slots to the front once the trimmed prefix is at
// least as long as the live part.
func (e *endpointWindow) compact() {
	live := len(e.slots) - e.start
	if e.start == 0 || e.start < live {
		return
	}
	n := copy(e.slots, e.slots[e.start:])
	clear(e.slots[n:])
	e.slots = e.slots[:n]
	e.start = 0
}

func withdrawnSlot(slot windowSlot) bool {
	return slot.drop || len(slot.hashes) == 0
}

func hashesToUint64(hashes []blockHash) []uint64 {
	if len(hashes) == 0 {
		return []uint64{}
	}
	out := make([]uint64, len(hashes))
	for i, h := range hashes {
		out[i] = uint64(h)
	}
	return out
}

func promptsToUint64(prompts [][]blockHash) [][]uint64 {
	if len(prompts) == 0 {
		return nil
	}
	out := make([][]uint64, len(prompts))
	for i, p := range prompts {
		cp := make([]uint64, len(p))
		for j, h := range p {
			cp[j] = uint64(h)
		}
		out[i] = cp
	}
	return out
}

func uint64sToBlockHashes(prompt []uint64) []blockHash {
	out := make([]blockHash, len(prompt))
	for i, h := range prompt {
		out[i] = blockHash(h)
	}
	return out
}

// promptKey identifies a prompt hash sequence for dedup. Length-prefixed so
// adjacent equal hashes from different prompts cannot collide.
func promptKey(hashes []blockHash) string {
	if len(hashes) == 0 {
		return ""
	}
	b := make([]byte, 8+8*len(hashes))
	binary.LittleEndian.PutUint64(b[:8], uint64(len(hashes)))
	for i, h := range hashes {
		binary.LittleEndian.PutUint64(b[8+8*i:], uint64(h))
	}
	return string(b)
}

// promptKeyUint64 is promptKey for sync-attribute prompts that are still
// []uint64, avoiding a blockHash conversion on the steady-state merge path.
func promptKeyUint64(hashes []uint64) string {
	if len(hashes) == 0 {
		return ""
	}
	b := make([]byte, 8+8*len(hashes))
	binary.LittleEndian.PutUint64(b[:8], uint64(len(hashes)))
	for i, h := range hashes {
		binary.LittleEndian.PutUint64(b[8+8*i:], h)
	}
	return string(b)
}
