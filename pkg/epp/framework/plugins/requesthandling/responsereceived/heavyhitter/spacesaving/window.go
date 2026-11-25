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

// Package spacesaving keeps the heaviest keys over a sliding time window. Time is split into
// slices; each slice holds a weighted Space-Saving summary of at most Slots keys, and a merged
// view holds each key's sum across the retained slices.
package spacesaving

import (
	"cmp"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"
)

// Config sets the window geometry. Window must be an exact multiple of Slice; the slice count
// is Window/Slice, and the retained span is between Window-Slice and Window. Slots is the
// Space-Saving summary size per slice: a key whose weight in a slice exceeds that slice's
// total divided by Slots is guaranteed a slot there.
type Config struct {
	Window time.Duration
	Slice  time.Duration
	Slots  int
}

// Entry is one key's windowed weight estimate. Weight - Error is a lower bound on the true
// windowed weight, and Weight is within (window total / slots per slice) of it. Error counts
// only the load inherited at slot takeover: load lost to an eviction carries no Error, so an
// Error of 0 does not by itself mean the value is exact. Counts are exact while no eviction
// happened during the window.
type Entry struct {
	ID     string
	Weight int64
	Error  int64
}

// Snapshot is a point-in-time read of the window. Entries holds at most the requested number of
// heaviest keys, heaviest first with ties by ID ascending. Total is the exact windowed weight of
// all keys, Tracked the number of keys in the merged view, and Evictions the cumulative slot
// takeovers since construction.
type Snapshot struct {
	Entries   []Entry
	Total     int64
	Tracked   int
	Evictions uint64
}

// slot is one Space-Saving counter. err is the count the key inherited when it took the slot
// over, so count - err is the weight this key itself added to the slice through this slot.
type slot struct {
	id    string // interned: the same string as the merged entry's id
	count int64
	err   int64
}

// slice is one ring slot. The live slice keeps slots as a min-heap on count; a closed slice is
// unordered and kept only so it can be subtracted from the merged view when it expires.
type slice struct {
	slots []slot
	total int64
}

type mergedEntry struct {
	id     string
	weight int64
	err    int64
}

// Window is the sliding-window summary. All public methods are safe for concurrent use.
type Window struct {
	mu        sync.Mutex
	origin    time.Time
	sliceDur  time.Duration
	slots     int
	ring      []slice
	epoch     int64            // epoch of the live slice, ring[epoch % len(ring)]
	index     map[string]int32 // live slice only: id -> heap position
	merged    map[string]*mergedEntry
	total     int64
	evictions uint64
}

// New builds a window from cfg. origin is the zero of its time base; every method's now is
// interpreted as now.Sub(origin), so origin should come from time.Now() for wall-clock use.
func New(cfg Config, origin time.Time) (*Window, error) {
	switch {
	case cfg.Window <= 0:
		return nil, fmt.Errorf("window must be > 0, got %v", cfg.Window)
	case cfg.Slice <= 0:
		return nil, fmt.Errorf("slice must be > 0, got %v", cfg.Slice)
	case cfg.Slice > cfg.Window:
		return nil, fmt.Errorf("slice %v must not exceed window %v", cfg.Slice, cfg.Window)
	case cfg.Window%cfg.Slice != 0:
		return nil, fmt.Errorf("window %v must be a multiple of slice %v, so the retained span never exceeds the window", cfg.Window, cfg.Slice)
	case cfg.Slots < 1:
		return nil, fmt.Errorf("slots must be >= 1, got %d", cfg.Slots)
	}
	n := int(cfg.Window / cfg.Slice)
	if n < 2 {
		return nil, fmt.Errorf("window %v over slice %v yields %d slices, need at least 2", cfg.Window, cfg.Slice, n)
	}
	ring := make([]slice, n)
	for i := range ring {
		ring[i].slots = make([]slot, 0, cfg.Slots)
	}
	return &Window{
		origin:   origin,
		sliceDur: cfg.Slice,
		slots:    cfg.Slots,
		ring:     ring,
		index:    make(map[string]int32, cfg.Slots),
		merged:   make(map[string]*mergedEntry),
	}, nil
}

// Add charges weight to id at now. A non-positive weight is ignored.
func (w *Window) Add(now time.Time, id string, weight int64) {
	if weight <= 0 {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.advance(now)

	live := &w.ring[w.epoch%int64(len(w.ring))]
	live.total += weight
	w.total += weight

	if pos, ok := w.index[id]; ok {
		sl := &live.slots[pos]
		sl.count += weight
		w.merged[sl.id].weight += weight
		w.siftDown(live, int(pos))
		return
	}

	if len(live.slots) < w.slots {
		me := w.mergedEntryFor(id)
		live.slots = append(live.slots, slot{id: me.id, count: weight})
		me.weight += weight
		w.index[me.id] = int32(len(live.slots) - 1)
		w.siftUp(live, len(live.slots)-1)
		return
	}

	old := live.slots[0] // the minimum slot; id is not indexed, so id != old.id
	delete(w.index, old.id)
	w.unmerge(old.id, old.count, old.err)
	me := w.mergedEntryFor(id)
	live.slots[0] = slot{id: me.id, count: old.count + weight, err: old.count}
	w.index[me.id] = 0
	me.weight += old.count + weight
	me.err += old.count
	w.evictions++
	w.siftDown(live, 0)
}

// Lookup returns id's windowed entry, reporting false when id carries no weight in the window.
func (w *Window) Lookup(now time.Time, id string) (Entry, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.advance(now)
	me, ok := w.merged[id]
	if !ok {
		return Entry{}, false
	}
	return Entry{ID: me.id, Weight: me.weight, Error: me.err}, true
}

// Snapshot returns the window's state at now. k selects the heaviest entries to return: k < 0
// returns all, k == 0 returns none, and the totals are reported for every k.
func (w *Window) Snapshot(now time.Time, k int) Snapshot {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.advance(now)

	snap := Snapshot{Total: w.total, Tracked: len(w.merged), Evictions: w.evictions}
	if k <= 0 || len(w.merged) == 0 {
		if k < 0 {
			snap.Entries = w.sortEntries(nil)
		}
		return snap
	}
	if k > len(w.merged) {
		k = len(w.merged)
	}

	// Bounded min-heap whose root is the worst kept entry: the lowest weight, then the highest
	// ID. A candidate replaces the root only when it beats it, so the kept set is the k best.
	top := make([]Entry, 0, k)
	for _, me := range w.merged {
		e := Entry{ID: me.id, Weight: me.weight, Error: me.err}
		if len(top) < k {
			top = append(top, e)
			siftWorstUp(top, len(top)-1)
			continue
		}
		if worseEntry(top[0], e) {
			top[0] = e
			siftWorstDown(top, 0)
		}
	}
	snap.Entries = w.sortEntries(top)
	return snap
}

func (w *Window) sortEntries(entries []Entry) []Entry {
	if entries == nil {
		entries = make([]Entry, 0, len(w.merged))
		for _, me := range w.merged {
			entries = append(entries, Entry{ID: me.id, Weight: me.weight, Error: me.err})
		}
	}
	slices.SortFunc(entries, func(a, b Entry) int {
		if a.Weight != b.Weight {
			return cmp.Compare(b.Weight, a.Weight) // heaviest first
		}
		return strings.Compare(a.ID, b.ID)
	})
	return entries
}

// advance rotates the ring to the epoch of now. Slices older than the window are expired
// wholesale; an epoch at or before the current one is a no-op. The caller holds the lock.
func (w *Window) advance(now time.Time) {
	e := int64(now.Sub(w.origin) / w.sliceDur)
	if e <= w.epoch {
		return
	}
	if e-w.epoch >= int64(len(w.ring)) {
		for i := range w.ring {
			w.expire(i)
		}
	} else {
		for i := int64(1); i <= e-w.epoch; i++ {
			w.expire(int((w.epoch + i) % int64(len(w.ring))))
		}
	}
	// Every expired slice is reset, and the new live slice is always among them, so the index
	// no longer describes any slot.
	w.epoch = e
	clear(w.index)
}

// expire drops a ring slice from the merged view and resets it. The caller holds the lock.
func (w *Window) expire(idx int) {
	s := &w.ring[idx]
	for _, sl := range s.slots {
		w.unmerge(sl.id, sl.count, sl.err)
	}
	w.total -= s.total
	clear(s.slots)
	s.slots = s.slots[:0]
	s.total = 0
}

// unmerge subtracts one slot's contribution; the entry is deleted when nothing remains. The
// caller holds the lock.
func (w *Window) unmerge(id string, count, err int64) {
	me := w.merged[id]
	me.weight -= count
	me.err -= err
	if me.weight == 0 {
		delete(w.merged, id)
	}
}

// mergedEntryFor returns the merged entry for id, creating one when absent. The entry owns the
// canonical ID string: the caller's may share backing memory with a larger buffer. The caller
// holds the lock.
func (w *Window) mergedEntryFor(id string) *mergedEntry {
	if me, ok := w.merged[id]; ok {
		return me
	}
	me := &mergedEntry{id: strings.Clone(id)}
	w.merged[me.id] = me
	return me
}

// siftUp and siftDown maintain the live slice as a min-heap on count, keeping the index
// positions current. The caller holds the lock.

func (w *Window) siftUp(s *slice, i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if s.slots[parent].count <= s.slots[i].count {
			return
		}
		w.swap(s, parent, i)
		i = parent
	}
}

func (w *Window) siftDown(s *slice, i int) {
	n := len(s.slots)
	for {
		left, right, min := 2*i+1, 2*i+2, i
		if left < n && s.slots[left].count < s.slots[min].count {
			min = left
		}
		if right < n && s.slots[right].count < s.slots[min].count {
			min = right
		}
		if min == i {
			return
		}
		w.swap(s, i, min)
		i = min
	}
}

func (w *Window) swap(s *slice, a, b int) {
	s.slots[a], s.slots[b] = s.slots[b], s.slots[a]
	w.index[s.slots[a].id] = int32(a)
	w.index[s.slots[b].id] = int32(b)
}

// worseEntry reports whether a ranks below b in the snapshot order (heaviest first, ties by ID
// ascending); the worst kept entry is the root of the selection heap.
func worseEntry(a, b Entry) bool {
	if a.Weight != b.Weight {
		return a.Weight < b.Weight
	}
	return a.ID > b.ID
}

func siftWorstUp(top []Entry, i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !worseEntry(top[i], top[parent]) {
			return
		}
		top[parent], top[i] = top[i], top[parent]
		i = parent
	}
}

func siftWorstDown(top []Entry, i int) {
	n := len(top)
	for {
		left, right, min := 2*i+1, 2*i+2, i
		if left < n && worseEntry(top[left], top[min]) {
			min = left
		}
		if right < n && worseEntry(top[right], top[min]) {
			min = right
		}
		if min == i {
			return
		}
		top[min], top[i] = top[i], top[min]
		i = min
	}
}
