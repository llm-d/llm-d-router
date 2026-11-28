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

package spacesaving

import (
	"fmt"
	"math/rand/v2"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var testOrigin = time.Unix(1000, 0)

func newWindow(t *testing.T, cfg Config) *Window {
	t.Helper()
	w, err := New(cfg, testOrigin)
	require.NoError(t, err)
	return w
}

// checkInvariants asserts the internal-state contract: the merged view holds exactly the union of
// the ring slices' slot IDs, each entry's weight and error are the sums of its slots' counts and
// errors with the error below the weight, the totals agree, and the index describes the live
// slice. Call it only when no other goroutine mutates the window.
func checkInvariants(t *testing.T, w *Window) {
	t.Helper()
	type sums struct{ weight, err int64 }
	perID := make(map[string]sums)
	var ringTotal int64
	for i := range w.ring {
		s := &w.ring[i]
		require.LessOrEqual(t, len(s.slots), w.slots, "slice %d exceeds the slot capacity", i)
		for _, sl := range s.slots {
			sum := perID[sl.id]
			sum.weight += sl.count
			sum.err += sl.err
			perID[sl.id] = sum
		}
		ringTotal += s.total
	}
	assert.Equal(t, ringTotal, w.total, "ring slice totals equal the window total")
	assert.Equal(t, len(perID), len(w.merged), "merged keys are the union of the slot IDs")

	var mergedSum int64
	for id, me := range w.merged {
		sum, ok := perID[id]
		require.True(t, ok, "merged key %q holds no slot", id)
		assert.Equal(t, sum.weight, me.weight, "weight of %q", id)
		assert.Equal(t, sum.err, me.err, "error of %q", id)
		assert.GreaterOrEqual(t, me.err, int64(0), "error of %q", id)
		assert.Less(t, me.err, me.weight, "error of %q", id)
		mergedSum += me.weight
	}
	assert.Equal(t, mergedSum, w.total, "merged weights sum to the window total")

	live := &w.ring[w.epoch%int64(len(w.ring))]
	liveIDs := make(map[string]struct{}, len(live.slots))
	for i, sl := range live.slots {
		pos, ok := w.index[sl.id]
		require.True(t, ok, "live slot %d (%q) missing from the index", i, sl.id)
		assert.EqualValues(t, i, pos, "index position of %q", sl.id)
		liveIDs[sl.id] = struct{}{}
	}
	assert.Equal(t, len(liveIDs), len(live.slots), "live slice holds distinct IDs")
	assert.Equal(t, len(liveIDs), len(w.index), "index covers exactly the live slice")
}

func TestNew_InvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{"zero_window", Config{Window: 0, Slice: time.Second, Slots: 2}},
		{"negative_window", Config{Window: -time.Second, Slice: time.Second, Slots: 2}},
		{"zero_slice", Config{Window: time.Minute, Slice: 0, Slots: 2}},
		{"slice_over_window", Config{Window: time.Second, Slice: 2 * time.Second, Slots: 2}},
		{"window_not_slice_multiple", Config{Window: 10 * time.Second, Slice: 3 * time.Second, Slots: 2}},
		{"single_slice", Config{Window: time.Second, Slice: time.Second, Slots: 2}},
		{"zero_slots", Config{Window: time.Minute, Slice: time.Second, Slots: 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New(tt.cfg, testOrigin)
			assert.Error(t, err)
		})
	}
}

func TestAdd_ExactBelowCapacity(t *testing.T) {
	w := newWindow(t, Config{Window: 3 * time.Second, Slice: time.Second, Slots: 4})

	want := map[string]int64{"a": 0, "b": 0, "c": 0}
	for i, id := range []string{"a", "b", "c"} {
		for j := range 3 {
			weight := int64(10*(i+1) + j)
			w.Add(testOrigin.Add(time.Duration(i)*time.Second+time.Duration(j)*10*time.Millisecond), id, weight)
			want[id] += weight
		}
	}

	snap := w.Snapshot(testOrigin.Add(2*time.Second+50*time.Millisecond), -1)
	assert.Zero(t, snap.Evictions)
	assert.Equal(t, len(want), snap.Tracked)
	assert.Equal(t, int64(33+63+93), snap.Total)
	for _, e := range snap.Entries {
		assert.Equal(t, want[e.ID], e.Weight, "weight of %q", e.ID)
		assert.Zero(t, e.Error, "error of %q", e.ID)
	}
	checkInvariants(t, w)
}

func TestAdd_IgnoresNonPositiveWeight(t *testing.T) {
	w := newWindow(t, Config{Window: time.Minute, Slice: time.Second, Slots: 4})

	w.Add(testOrigin, "a", 0)
	w.Add(testOrigin, "a", -5)

	snap := w.Snapshot(testOrigin, -1)
	assert.Zero(t, snap.Total)
	assert.Zero(t, snap.Tracked)
	assert.Empty(t, snap.Entries)
	checkInvariants(t, w)
}

func TestAdd_Takeover(t *testing.T) {
	w := newWindow(t, Config{Window: 3 * time.Second, Slice: time.Second, Slots: 2})

	w.Add(testOrigin, "a", 10)
	w.Add(testOrigin, "b", 5)
	// c has no slot and the table is full: c takes over b, the minimum, inheriting its count.
	w.Add(testOrigin, "c", 7)

	entry, ok := w.Lookup(testOrigin, "c")
	require.True(t, ok)
	assert.Equal(t, Entry{ID: "c", Weight: 12, Error: 5}, entry)
	_, ok = w.Lookup(testOrigin, "b")
	assert.False(t, ok, "b held no slot in another slice")
	entry, ok = w.Lookup(testOrigin, "a")
	require.True(t, ok)
	assert.Equal(t, Entry{ID: "a", Weight: 10, Error: 0}, entry)
	snap := w.Snapshot(testOrigin, -1)
	assert.Equal(t, uint64(1), snap.Evictions)
	assert.Equal(t, int64(22), snap.Total)

	// a gains a slot in the next slice, then loses it to e: its merged weight drops back to the
	// first slice's sum instead of disappearing, because it still holds a slot there.
	next := testOrigin.Add(time.Second)
	w.Add(next, "a", 2)
	w.Add(next, "d", 4)
	w.Add(next, "e", 5)

	entry, ok = w.Lookup(next, "a")
	require.True(t, ok)
	assert.Equal(t, Entry{ID: "a", Weight: 10, Error: 0}, entry)
	entry, ok = w.Lookup(next, "e")
	require.True(t, ok)
	assert.Equal(t, Entry{ID: "e", Weight: 7, Error: 2}, entry)
	snap = w.Snapshot(next, -1)
	assert.Equal(t, uint64(2), snap.Evictions)
	assert.Equal(t, int64(33), snap.Total)
	checkInvariants(t, w)
}

func TestRotation(t *testing.T) {
	w := newWindow(t, Config{Window: 3 * time.Second, Slice: time.Second, Slots: 4})

	// One nanosecond before the boundary lands in the first slice; the boundary itself moves to
	// the next one, which the different expiry below relies on.
	w.Add(testOrigin.Add(time.Second-time.Nanosecond), "a", 1)
	w.Add(testOrigin.Add(time.Second), "b", 2)

	// At epoch 3 the first slice (epoch 0) has aged out of the 3-slice window; the second
	// remains.
	atThree := testOrigin.Add(3 * time.Second)
	w.Add(atThree, "c", 4)
	_, ok := w.Lookup(atThree, "a")
	assert.False(t, ok, "first slice expired")
	entry, ok := w.Lookup(atThree, "b")
	require.True(t, ok)
	assert.Equal(t, Entry{ID: "b", Weight: 2, Error: 0}, entry)

	// A gap longer than the window clears everything.
	atTen := testOrigin.Add(10 * time.Second)
	snap := w.Snapshot(atTen, -1)
	assert.Zero(t, snap.Total)
	assert.Zero(t, snap.Tracked)
	assert.Empty(t, snap.Entries)

	w.Add(atTen, "d", 8)
	// A read at a time before the current epoch does not rewind the ring.
	snap = w.Snapshot(testOrigin.Add(9*time.Second), -1)
	assert.Equal(t, int64(8), snap.Total)
	assert.Equal(t, 1, snap.Tracked)
	checkInvariants(t, w)
}

func TestSnapshot_OrderAndLimit(t *testing.T) {
	w := newWindow(t, Config{Window: 2 * time.Second, Slice: time.Second, Slots: 8})
	for id, weight := range map[string]int64{"x": 5, "y": 5, "z": 9, "w": 1, "v": 5} {
		w.Add(testOrigin, id, weight)
	}

	snap := w.Snapshot(testOrigin, -1)
	assert.Equal(t, []Entry{
		{ID: "z", Weight: 9},
		{ID: "v", Weight: 5},
		{ID: "x", Weight: 5},
		{ID: "y", Weight: 5},
		{ID: "w", Weight: 1},
	}, snap.Entries)

	snap = w.Snapshot(testOrigin, 0)
	assert.Empty(t, snap.Entries)
	assert.Equal(t, int64(25), snap.Total)
	assert.Equal(t, 5, snap.Tracked)

	snap = w.Snapshot(testOrigin, 2)
	assert.Equal(t, []Entry{{ID: "z", Weight: 9}, {ID: "v", Weight: 5}}, snap.Entries)

	snap = w.Snapshot(testOrigin, 100)
	assert.Len(t, snap.Entries, 5)
}

func TestLookup(t *testing.T) {
	w := newWindow(t, Config{Window: 2 * time.Second, Slice: time.Second, Slots: 4})
	w.Add(testOrigin, "a", 7)

	entry, ok := w.Lookup(testOrigin, "a")
	require.True(t, ok)
	assert.Equal(t, Entry{ID: "a", Weight: 7, Error: 0}, entry)

	_, ok = w.Lookup(testOrigin, "zz")
	assert.False(t, ok)

	_, ok = w.Lookup(testOrigin.Add(2*time.Second), "a")
	assert.False(t, ok, "a aged out of the window")
}

// oracleEvent is one charged weight as recorded by the exact reference in
// TestAccuracyAgainstOracle.
type oracleEvent struct {
	epoch  int64
	id     string
	weight int64
}

// assertAccuracy compares the window against the exact weights of the events whose epochs fall
// in the retained range [live-slices+1, live].
func assertAccuracy(t *testing.T, w *Window, history []oracleEvent, now time.Time, slots, sliceCount int) {
	t.Helper()
	snap := w.Snapshot(now, -1)

	liveEpoch := int64(now.Sub(testOrigin) / time.Second)
	lo := liveEpoch - int64(sliceCount) + 1
	trueWeight := make(map[string]int64)
	var total int64
	for _, ev := range history {
		if ev.epoch >= lo {
			trueWeight[ev.id] += ev.weight
			total += ev.weight
		}
	}
	assert.Equal(t, total, snap.Total, "window total is exact")

	var entrySum int64
	for _, e := range snap.Entries {
		entrySum += e.Weight
		trueW, ok := trueWeight[e.ID]
		require.True(t, ok, "tracked id %q is absent from the oracle", e.ID)
		assert.LessOrEqual(t, e.Weight-e.Error, trueW, "lower bound of %q", e.ID)
		assert.LessOrEqual(t, (e.Weight-trueW)*int64(slots), snap.Total, "error bound of %q", e.ID)
	}
	assert.Equal(t, entrySum, snap.Total, "entry weights sum to the window total")

	for id, trueW := range trueWeight {
		if trueW*int64(slots) > snap.Total { // above the retention line
			_, ok := w.Lookup(now, id)
			assert.True(t, ok, "id %q above the retention line is not tracked", id)
		}
	}
}

func TestAccuracyAgainstOracle(t *testing.T) {
	const (
		slices = 10
		slots  = 32
		events = 20000
		ids    = 1000
	)
	w := newWindow(t, Config{Window: slices * time.Second, Slice: time.Second, Slots: slots})
	rng := rand.New(rand.NewPCG(1, 2))
	zipf := rand.NewZipf(rng, 1.2, 1.5, ids-1)

	history := make([]oracleEvent, 0, events)
	var clock time.Duration
	for i := range events {
		// An average step of 1.5ms over 20000 events spans three 10s windows.
		clock += time.Duration(rng.Int64N(int64(3 * time.Millisecond)))
		now := testOrigin.Add(clock)
		ev := oracleEvent{
			epoch:  int64(clock / time.Second),
			id:     fmt.Sprintf("user-%03d", zipf.Uint64()),
			weight: int64(1 + rng.IntN(1000)),
		}
		w.Add(now, ev.id, ev.weight)
		history = append(history, ev)

		if (i+1)%2000 == 0 {
			assertAccuracy(t, w, history, now, slots, slices)
		}
	}
	snap := w.Snapshot(testOrigin.Add(clock), -1)
	assert.Positive(t, snap.Evictions, "the sequence must exercise takeovers")
	checkInvariants(t, w)
}

func TestConcurrentAddAndSnapshot(t *testing.T) {
	const (
		writers   = 8
		perWriter = 1000
	)
	w := newWindow(t, Config{Window: time.Minute, Slice: time.Second, Slots: 64})
	now := testOrigin

	var want int64
	var wg sync.WaitGroup
	for g := range writers {
		want += int64(perWriter * (100 + g))
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range perWriter {
				w.Add(now, fmt.Sprintf("w%d-%d", g, i), int64(100+g))
			}
		}()
	}

	done := make(chan struct{})
	var reader sync.WaitGroup
	reader.Add(1)
	go func() {
		defer reader.Done()
		for {
			select {
			case <-done:
				return
			default:
				w.Snapshot(now, 10)
				w.Lookup(now, "w0-0")
			}
		}
	}()

	wg.Wait()
	close(done)
	reader.Wait()

	snap := w.Snapshot(now, -1)
	assert.Equal(t, want, snap.Total)
}

func TestSnapshot_EmptyWindow(t *testing.T) {
	w := newWindow(t, Config{Window: time.Minute, Slice: time.Second, Slots: 4})

	snap := w.Snapshot(testOrigin, -1)
	assert.Empty(t, snap.Entries)
	assert.Zero(t, snap.Total)
	assert.Zero(t, snap.Tracked)

	snap = w.Snapshot(testOrigin, 5)
	assert.Empty(t, snap.Entries)
}
