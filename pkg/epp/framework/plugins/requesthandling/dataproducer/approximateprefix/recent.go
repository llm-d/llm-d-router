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
)

// recentWindow buffers recently recorded prompt hash sequences per endpoint.
// Each inner slice is one prompt's hashes starting at block 0. Cap is on total
// hashes across prompts so cross-replica publish payloads stay small.
type recentWindow struct {
	mu      sync.Mutex
	pending map[string]*endpointWindow
	max     int
}

type endpointWindow struct {
	prompts [][]blockHash
	start   int
	total   int
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
	buf.evict(w.max - len(cp))
	// Compact only after enough evictions to amortize copying the retained records.
	if buf.start > 0 && buf.start >= len(buf.prompts)-buf.start {
		n := copy(buf.prompts, buf.prompts[buf.start:])
		clear(buf.prompts[n:])
		buf.prompts = buf.prompts[:n]
		buf.start = 0
	}
	buf.prompts = append(buf.prompts, cp)
	buf.total += len(cp)
	return cp
}

// replace swaps the entry that record returned as entry for hashes, keeping its
// position. It does nothing when the entry has already been trimmed or the
// endpoint forgotten, so entries recorded since, including an equal prompt of
// another request, are left as they are.
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
	for i := buf.start; i < len(buf.prompts); i++ {
		if p := buf.prompts[i]; &p[0] == &entry[0] {
			cp := w.headCopy(hashes)
			buf.prompts[i] = cp
			buf.total += len(cp) - len(p)
			buf.evict(w.max)
			return
		}
	}
}

// evict drops the oldest prompts until the total hash count is at most limit.
func (b *endpointWindow) evict(limit int) {
	for b.total > limit {
		b.total -= len(b.prompts[b.start])
		b.prompts[b.start] = nil
		b.start++
	}
}

func (w *recentWindow) headCopy(hashes []blockHash) []blockHash {
	if len(hashes) > w.max {
		hashes = hashes[:w.max]
	}
	cp := make([]blockHash, len(hashes))
	copy(cp, hashes)
	return cp
}

// snapshot returns a deep copy of pending prompts for endpointID without clearing.
// The syncer replaces each replica field on every Set, so draining would wipe
// peer-visible hashes on the next idle publish tick.
func (w *recentWindow) snapshot(endpointID string) [][]blockHash {
	w.mu.Lock()
	defer w.mu.Unlock()
	buf := w.pending[endpointID]
	if buf == nil {
		return nil
	}
	return copyPromptHashes(buf.prompts[buf.start:])
}

// forget drops the buffer for endpointID when the endpoint leaves the pool.
func (w *recentWindow) forget(endpointID string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.pending, endpointID)
}

// removePrompt drops one prompt's hashes from endpointID's buffer so a request
// that failed stops being published to peers. hashes are normalized with the
// same head-cap record applies, so a recording that was capped is still
// matched. A prompt already trimmed out by a later record is a no-op.
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
	out := make([][]blockHash, 0, len(buf.prompts)-buf.start)
	for _, p := range buf.prompts[buf.start:] {
		if promptKey(p) != key {
			out = append(out, p)
		} else {
			buf.total -= len(p)
		}
	}
	buf.prompts = out
	buf.start = 0
}

func copyPromptHashes(prompts [][]blockHash) [][]blockHash {
	if len(prompts) == 0 {
		return nil
	}
	out := make([][]blockHash, len(prompts))
	for i, p := range prompts {
		cp := make([]blockHash, len(p))
		copy(cp, p)
		out[i] = cp
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
