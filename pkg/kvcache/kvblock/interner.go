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

import "sync"

// interner assigns dense uint32 ordinals to strings. An ordinal is assigned
// on an intern call and belongs to its name for the interner's lifetime:
// it is never reclaimed or reassigned, and the append-only names table
// always resolves it back. Concurrent readers (ordinalOf, nameOf) share the
// table with writers through the RWMutex; the write path holds the lock
// across a whole batch so a batch is assigned all or nothing.
type interner struct {
	mu    sync.RWMutex
	ids   map[string]uint32
	names []string
	limit int
}

func newInterner(limit int) *interner {
	return &interner{ids: make(map[string]uint32), limit: limit}
}

// fitsLocked reports whether the names not yet interned fit under the
// limit. Called with mu held for writing.
func (in *interner) fitsLocked(names func(yield func(string))) bool {
	pending := 0
	seen := map[string]struct{}{}
	names(func(s string) {
		if _, ok := in.ids[s]; ok {
			return
		}
		if _, dup := seen[s]; dup {
			return
		}
		seen[s] = struct{}{}
		pending++
	})
	return len(in.ids)+pending <= in.limit
}

// internLocked returns the ordinal for s, assigning the next free one on
// first use. Called with mu held for writing, after fitsLocked.
func (in *interner) internLocked(s string) uint32 {
	if id, ok := in.ids[s]; ok {
		return id
	}
	id := uint32(len(in.ids)) //#nosec G115 -- fitsLocked caps len(in.ids) at the interner limit, at most 1<<20
	in.ids[s] = id
	in.names = append(in.names, s)
	return id
}

// ordinalOf returns the ordinal assigned to s, and whether it has one.
func (in *interner) ordinalOf(s string) (uint32, bool) {
	in.mu.RLock()
	defer in.mu.RUnlock()
	id, ok := in.ids[s]
	return id, ok
}

// nameOf returns the name assigned to ordinal, and whether it has one.
func (in *interner) nameOf(ordinal uint32) (string, bool) {
	in.mu.RLock()
	defer in.mu.RUnlock()
	if uint64(ordinal) >= uint64(len(in.names)) { //#nosec G115 -- compared in uint64 to avoid truncation
		return "", false
	}
	return in.names[ordinal], true
}
