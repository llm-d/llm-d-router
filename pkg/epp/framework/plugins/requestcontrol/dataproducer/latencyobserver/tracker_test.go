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

package latencyobserver

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// values extracts the TTFTs of a window, in the order returned.
func values(window []observation) []float64 {
	out := make([]float64, len(window))
	for i, o := range window {
		out[i] = o.value
	}
	return out
}

func TestSlidingWindowTracker(t *testing.T) {
	now := time.Now()

	t.Run("sorts by value, carrying each in-flight count with it", func(t *testing.T) {
		tracker := newSlidingWindowTracker(8)
		tracker.add(0.9, 20, now)
		tracker.add(0.1, 2, now.Add(time.Millisecond))

		window := tracker.window(now.Add(time.Second), time.Minute, 0)
		require.Len(t, window, 2)
		assert.Equal(t, []float64{0.1, 0.9}, values(window))
		assert.Equal(t, int64(2), window[0].inflight, "the fast one keeps its own count")
	})

	t.Run("overwrites the oldest once full", func(t *testing.T) {
		tracker := newSlidingWindowTracker(3)
		for i, v := range []float64{0.1, 0.2, 0.3, 0.4, 0.5} {
			tracker.add(v, 0, now.Add(time.Duration(i)*time.Millisecond))
		}
		assert.Equal(t, []float64{0.3, 0.4, 0.5}, values(tracker.window(now.Add(time.Second), time.Minute, 0)))
	})

	t.Run("bounds by age and by count, keeping the newest", func(t *testing.T) {
		tracker := newSlidingWindowTracker(8)
		tracker.add(0.1, 0, now.Add(-10*time.Minute)) // stale
		tracker.add(0.2, 0, now.Add(-time.Minute))
		tracker.add(0.3, 0, now)

		assert.Equal(t, []float64{0.2, 0.3}, values(tracker.window(now, 3*time.Minute, 0)), "by age")
		assert.Equal(t, []float64{0.3}, values(tracker.window(now, 3*time.Minute, 1)), "by count")
	})
}

// Both percentile helpers interpolate between neighbours and must tolerate an
// empty slice: they index sorted[lo], so a zero-length window would panic.
func TestPercentiles(t *testing.T) {
	sorted := []observation{{value: 0}, {value: 1}, {value: 2}, {value: 3}, {value: 4}}

	assert.InDelta(t, 2.0, percentileOf(sorted, 0.5), 1e-9)
	assert.InDelta(t, 0.5, percentileOf(sorted, 0.125), 1e-9, "interpolated between 0 and 1")
	assert.InDelta(t, 4.0, percentileOf(sorted, 1), 1e-9, "upper bound")
	assert.Zero(t, percentileOf(nil, 0.5))

	assert.InDelta(t, 2.0, percentileFloat64([]float64{0, 1, 2, 3, 4}, 0.5), 1e-9)
	assert.Zero(t, percentileFloat64(nil, 0.5))
}

func TestDropMax(t *testing.T) {
	assert.Equal(t, []float64{0.1, 0.2}, dropMax([]float64{0.1, 0.2, 5.0}), "keeps the low end the floor reads")
	assert.Empty(t, dropMax(nil))
}

// bandInflight averages the [p-0.1, p+0.1] band, clamped at both ends, and
// divides by the band width -- so an empty slice must not reach the division.
func TestBandInflight(t *testing.T) {
	sorted := make([]observation, 10)
	for i := range sorted {
		sorted[i] = observation{value: float64(i) / 10, inflight: int64(i)}
	}

	assert.InDelta(t, 4.0, bandInflight(sorted, 0.50), 1e-9, "band [0.4,0.6] of n=10 -> indices 3..5")
	assert.InDelta(t, 0.5, bandInflight(sorted, 0.10), 1e-9, "clamped at the start")
	assert.InDelta(t, 8.0, bandInflight(sorted, 0.95), 1e-9, "clamped at the end")
	assert.Zero(t, bandInflight(nil, 0.5))
}
