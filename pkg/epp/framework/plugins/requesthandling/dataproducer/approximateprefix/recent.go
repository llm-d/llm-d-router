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
	pending map[string][][]blockHash
	max     int
}

func newRecentWindow(max int) *recentWindow {
	if max <= 0 {
		max = defaultLRUCapacityPerServer
	}
	return &recentWindow{
		pending: make(map[string][][]blockHash),
		max:     max,
	}
}

// record appends one prompt's hashes for endpointID and trims oldest prompts
// until the total hash count is at most max. A single oversized prompt is
// head-capped to max so matching still anchors at block 0.
func (w *recentWindow) record(endpointID string, hashes []blockHash) {
	if len(hashes) == 0 {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	prompt := hashes
	if len(prompt) > w.max {
		prompt = prompt[:w.max]
	}
	cp := make([]blockHash, len(prompt))
	copy(cp, prompt)

	buf := w.pending[endpointID]
	buf = append(buf, cp)
	w.pending[endpointID] = trimPromptHashes(buf, w.max)
}

// snapshot returns a deep copy of pending prompts for endpointID without clearing.
// The syncer replaces each replica field on every Set, so draining would wipe
// peer-visible hashes on the next idle publish tick.
func (w *recentWindow) snapshot(endpointID string) [][]blockHash {
	w.mu.Lock()
	defer w.mu.Unlock()
	return copyPromptHashes(w.pending[endpointID])
}

// forget drops the buffer for endpointID when the endpoint leaves the pool.
func (w *recentWindow) forget(endpointID string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.pending, endpointID)
}

// trimPromptHashes drops oldest whole prompts until total hashes <= max.
// If one prompt still exceeds max, it is head-capped.
func trimPromptHashes(prompts [][]blockHash, max int) [][]blockHash {
	if max <= 0 || len(prompts) == 0 {
		return prompts
	}
	total := promptHashCount(prompts)
	for total > max && len(prompts) > 1 {
		total -= len(prompts[0])
		prompts = prompts[1:]
	}
	if total > max {
		head := make([]blockHash, max)
		copy(head, prompts[0][:max])
		return [][]blockHash{head}
	}
	// Own the slice so a later append cannot overwrite retained prompts.
	out := make([][]blockHash, len(prompts))
	copy(out, prompts)
	return out
}

func promptHashCount(prompts [][]blockHash) int {
	n := 0
	for _, p := range prompts {
		n += len(p)
	}
	return n
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
