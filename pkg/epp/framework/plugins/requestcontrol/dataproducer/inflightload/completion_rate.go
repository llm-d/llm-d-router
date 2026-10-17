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
	"time"
)

// completionWindowSeconds is the trailing window over which completion rates are averaged.
const completionWindowSeconds = 30

// completionWindow counts an endpoint's completions in one-second buckets.
type completionWindow struct {
	mu      sync.Mutex
	seconds [completionWindowSeconds]int64 // unix second each bucket holds
	counts  [completionWindowSeconds]int64
}

func (w *completionWindow) record(now time.Time) {
	sec := now.Unix()
	i := sec % completionWindowSeconds
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.seconds[i] != sec {
		w.seconds[i] = sec
		w.counts[i] = 0
	}
	w.counts[i]++
}

func (w *completionWindow) rate(now time.Time) float64 {
	sec := now.Unix()
	var total int64
	w.mu.Lock()
	defer w.mu.Unlock()
	for i := range w.counts {
		if sec-w.seconds[i] < completionWindowSeconds {
			total += w.counts[i]
		}
	}
	return float64(total) / completionWindowSeconds
}

// completionTracker holds a completionWindow per endpoint.
type completionTracker struct {
	mu      sync.RWMutex
	windows map[string]*completionWindow
}

func newCompletionTracker() *completionTracker {
	return &completionTracker{windows: make(map[string]*completionWindow)}
}

func (t *completionTracker) record(endpointID string, now time.Time) {
	t.mu.RLock()
	w, exists := t.windows[endpointID]
	t.mu.RUnlock()

	if !exists {
		t.mu.Lock()
		if w, exists = t.windows[endpointID]; !exists {
			w = &completionWindow{}
			t.windows[endpointID] = w
		}
		t.mu.Unlock()
	}
	w.record(now)
}

// rate returns the endpoint's completions per second over the window, or 0 for
// an endpoint with no recorded completions.
func (t *completionTracker) rate(endpointID string, now time.Time) float64 {
	t.mu.RLock()
	w, exists := t.windows[endpointID]
	t.mu.RUnlock()

	if !exists {
		return 0
	}
	return w.rate(now)
}

func (t *completionTracker) delete(endpointID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.windows, endpointID)
}
