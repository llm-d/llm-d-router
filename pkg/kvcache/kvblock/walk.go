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
	"sync"
)

// EntryRef is one indexed entry with the ordinals the index assigned to its
// pod and device tier. Ordinals are stable for the index lifetime and never
// reused, so consumers can key request-local tables by them; live entries
// hold a sparse subset of the assigned range, so consumers must not size
// state by it.
type EntryRef struct {
	PodEntry
	PodOrdinal  uint32
	TierOrdinal uint32
}

// KeyWalker is an optional Index capability: visit requestKeys in input
// order without materializing per-key entry slices.
type KeyWalker interface {
	// WalkKeys calls visit once per position, in order, until visit returns
	// false, the keys run out, or ctx is cancelled. Cancellation is observed
	// at checkpoints along the walk and when the walk completes, and always
	// makes the walk return ctx.Err(). A miss is reported with found=false
	// and no entries.
	// entries is borrowed from the index: read-only and valid only until
	// visit returns. A walk refreshes the recency of the keys it visited
	// once it ends, as Lookup does, so a prefix that is only read stays
	// resident under capacity pressure.
	WalkKeys(ctx context.Context, requestKeys []BlockHash,
		visit func(pos int, found bool, entries []EntryRef) bool) error
}

// WalkKeys implements KeyWalker. Each key's entries are visited under that
// key's lock, so a visit excludes writers to the same key and nothing else.
func (m *InMemoryIndex) WalkKeys(ctx context.Context, requestKeys []BlockHash,
	visit func(pos int, found bool, entries []EntryRef) bool,
) error {
	visited := 0
	// Every exit, cancellation included, refreshes what was read.
	defer func() { m.data.Promote(requestKeys[:visited]) }()
	for pos, key := range requestKeys {
		if pos&cancellationCheckMask == 0 && ctx.Err() != nil {
			return ctx.Err()
		}
		pc, found := m.data.Peek(key)
		if !found || pc == nil {
			if !visit(pos, false, nil) {
				return ctx.Err()
			}
			continue
		}
		visited = pos + 1
		pc.mu.Lock()
		more := visit(pos, true, pc.entries)
		pc.mu.Unlock()
		if !more {
			return ctx.Err()
		}
	}
	return ctx.Err()
}

// interner assigns dense uint32 ordinals to strings, stable for its lifetime
// and never reused, up to a fixed number of distinct strings. Callers hold mu
// across a whole batch so a batch is assigned all or nothing.
type interner struct {
	mu    sync.Mutex
	ids   map[string]uint32
	limit int
}

func newInterner(limit int) *interner {
	return &interner{ids: make(map[string]uint32), limit: limit}
}

// fitsLocked reports whether the names not yet interned fit under the
// limit. Called with mu held.
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
// first use. Called with mu held, after fitsLocked.
func (in *interner) internLocked(s string) uint32 {
	if id, ok := in.ids[s]; ok {
		return id
	}
	id := uint32(len(in.ids))
	in.ids[s] = id
	return id
}
