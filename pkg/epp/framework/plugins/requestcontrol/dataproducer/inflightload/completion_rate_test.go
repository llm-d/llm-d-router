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

package inflightload

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestCompletionTracker(t *testing.T) {
	t0 := time.Unix(1_000_000, 0)

	tests := []struct {
		name    string
		records []time.Duration // offsets from t0
		readAt  time.Duration
		want    float64
	}{
		{
			name: "no completions",
			want: 0,
		},
		{
			name:    "completions in one second",
			records: []time.Duration{0, 0, 0},
			want:    3.0 / completionWindowSeconds,
		},
		{
			name:    "completions spread over the window",
			records: []time.Duration{0, 10 * time.Second, 29 * time.Second},
			readAt:  29 * time.Second,
			want:    3.0 / completionWindowSeconds,
		},
		{
			name:    "completions older than the window are excluded",
			records: []time.Duration{0, 5 * time.Second},
			readAt:  32 * time.Second,
			want:    1.0 / completionWindowSeconds,
		},
		{
			name:    "reused bucket starts from zero",
			records: []time.Duration{0, 0, completionWindowSeconds * time.Second},
			readAt:  completionWindowSeconds * time.Second,
			want:    1.0 / completionWindowSeconds,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := newCompletionTracker()
			for _, off := range tt.records {
				tr.record("ep", t0.Add(off))
			}
			require.InDelta(t, tt.want, tr.rate("ep", t0.Add(tt.readAt)), 1e-9)
		})
	}
}

func TestCompletionTrackerEndpoints(t *testing.T) {
	t0 := time.Unix(1_000_000, 0)
	tr := newCompletionTracker()
	tr.record("a", t0)
	tr.record("a", t0)
	tr.record("b", t0)

	require.InDelta(t, 2.0/completionWindowSeconds, tr.rate("a", t0), 1e-9)
	require.InDelta(t, 1.0/completionWindowSeconds, tr.rate("b", t0), 1e-9)
	require.Zero(t, tr.rate("unknown", t0))

	tr.delete("a")
	require.Zero(t, tr.rate("a", t0))
	require.InDelta(t, 1.0/completionWindowSeconds, tr.rate("b", t0), 1e-9)
}

func TestCompletionTrackerConcurrent(t *testing.T) {
	t0 := time.Unix(1_000_000, 0)
	tr := newCompletionTracker()

	const workers, perWorker = 8, 100
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range perWorker {
				tr.record("ep", t0)
				_ = tr.rate("ep", t0)
			}
		}()
	}
	wg.Wait()

	require.InDelta(t, float64(workers*perWorker)/completionWindowSeconds, tr.rate("ep", t0), 1e-9)
}
