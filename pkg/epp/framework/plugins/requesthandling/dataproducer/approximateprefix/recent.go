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
)

// recentWindow buffers recently recorded prompt hash sequences per endpoint.
// Each inner slice is one prompt's hashes starting at block 0. Cap is on total
// hashes across prompts so cross-replica publish payloads stay small.
type recentWindow struct {
	mu      sync.Mutex
	pending map[string][][]blockHash
	// base is the generation of pending[endpointID][0]. It advances by one for
	// each prompt trimmed, so a generation always names the same prompt.
	base map[string]uint64
	max  int
}

func newRecentWindow(max int) *recentWindow {
	if max <= 0 {
		max = defaultLRUCapacityPerServer
	}
	return &recentWindow{
		pending: make(map[string][][]blockHash),
		base:    make(map[string]uint64),
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
	trimmed, dropped := trimPromptHashes(buf, w.max)
	w.pending[endpointID] = trimmed
	w.base[endpointID] += clamp.Uint64(dropped)
}

// snapshot returns a deep copy of pending prompts for endpointID without clearing.
// Peers read the window as a whole on every publish, so draining would wipe
// peer-visible hashes on the next idle publish tick.
func (w *recentWindow) snapshot(endpointID string) [][]blockHash {
	prompts, _ := w.snapshotWindow(endpointID)
	return prompts
}

// snapshotWindow is snapshot plus the generation of the first prompt.
func (w *recentWindow) snapshotWindow(endpointID string) ([][]blockHash, uint64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return copyPromptHashes(w.pending[endpointID]), w.base[endpointID]
}

// forget drops the buffer for endpointID when the endpoint leaves the pool.
func (w *recentWindow) forget(endpointID string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.pending, endpointID)
	delete(w.base, endpointID)
}

// trimPromptHashes drops oldest whole prompts until total hashes <= max and
// reports how many it dropped. record caps each prompt at max before calling
// it, so the last prompt always fits.
func trimPromptHashes(prompts [][]blockHash, max int) ([][]blockHash, int) {
	if max <= 0 || len(prompts) == 0 {
		return prompts, 0
	}
	total := promptHashCount(prompts)
	dropped := 0
	for total > max && len(prompts) > 1 {
		total -= len(prompts[0])
		prompts = prompts[1:]
		dropped++
	}
	// Own the slice so a later append cannot overwrite retained prompts.
	out := make([][]blockHash, len(prompts))
	copy(out, prompts)
	return out, dropped
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
