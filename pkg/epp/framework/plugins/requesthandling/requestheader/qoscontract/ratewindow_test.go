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

package qoscontract

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSlidingWindow(t *testing.T) {
	start := time.Date(2026, 11, 24, 12, 0, 0, 0, time.UTC)
	at := func(ms int) time.Time { return start.Add(time.Duration(ms) * time.Millisecond) }
	w := newSlidingWindow(time.Second, 3)

	for _, ms := range []int{0, 200, 400} {
		require.False(t, w.full(at(ms)), "at %dms", ms)
		w.add(at(ms))
	}
	assert.True(t, w.full(at(999)), "3 admissions within the last second")
	assert.Equal(t, time.Millisecond, w.retryAfter(at(999)))
	assert.False(t, w.full(at(1000)), "the admission at 0ms leaves the window at 1000ms")

	// Wrap the ring several times: each admission replaces the oldest.
	for ms := 1000; ms < 5000; ms += 400 {
		require.False(t, w.full(at(ms)), "at %dms", ms)
		w.add(at(ms))
	}
	// The ring holds the last three admissions: 3800, 4200 and 4600.
	assert.True(t, w.full(at(4700)))
	assert.Equal(t, 100*time.Millisecond, w.retryAfter(at(4700)), "3800 leaves at 4800")
	assert.False(t, w.full(at(4800)))
}

func TestRateLimit(t *testing.T) {
	cases := []struct {
		name   string
		rps    float64
		window time.Duration
		want   int
	}{
		{name: "whole", rps: 10, window: time.Second, want: 10},
		{name: "fraction floors", rps: 2.5, window: time.Second, want: 2},
		{name: "rate below one per second", rps: 0.5, window: 4 * time.Second, want: 2},
		{name: "float rounding does not lose a request", rps: 0.29, window: 100 * time.Second, want: 29},
		{name: "huge product is clamped above the cap", rps: 1e300, window: time.Hour, want: maxRateWindowRequests + 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, got := rateLimit(TeamConfig{MaxReqsPerSec: tc.rps}, tc.window)
			assert.Equal(t, tc.want, got)
		})
	}
}
