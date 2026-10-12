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
	"sync"
)

// recentWindow buffers recently recorded hashes for each endpoint. Cap keeps
// cross-replica publish payloads small.
type recentWindow struct {
	mu      sync.Mutex
	pending map[string][]blockHash
	max     int
}

func newRecentWindow(max int) *recentWindow {
	if max <= 0 {
		max = defaultMaxSyncedHashes
	}
	return &recentWindow{
		pending: make(map[string][]blockHash),
		max:     max,
	}
}

// record appends head-capped hashes for endpointID and trims the buffer to max.
func (w *recentWindow) record(endpointID string, hashes []blockHash) {
	if len(hashes) == 0 {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	head := hashes
	if len(head) > w.max {
		head = head[:w.max]
	}
	buf := w.pending[endpointID]
	buf = append(buf, head...)
	if len(buf) > w.max {
		// Own the trimmed slice so a later append cannot overwrite retained hashes.
		buf = append([]blockHash(nil), buf[len(buf)-w.max:]...)
	}
	w.pending[endpointID] = buf
}

// snapshot returns a copy of pending hashes for endpointID without clearing.
// The syncer replaces each replica field on every Set, so draining would wipe
// peer-visible hashes on the next idle publish tick.
func (w *recentWindow) snapshot(endpointID string) []blockHash {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := w.pending[endpointID]
	if len(out) == 0 {
		return nil
	}
	cp := make([]blockHash, len(out))
	copy(cp, out)
	return cp
}

// forget drops the buffer for endpointID when the endpoint leaves the pool.
func (w *recentWindow) forget(endpointID string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.pending, endpointID)
}
