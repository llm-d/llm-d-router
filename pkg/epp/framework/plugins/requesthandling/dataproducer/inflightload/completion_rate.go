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

const (
	// completionWindowSeconds is the trailing window over which completion rates are averaged.
	completionWindowSeconds = 30
	// completionFallbackWindowSeconds is the longer trailing window used when the
	// short window holds fewer than minWindowCompletions completions.
	completionFallbackWindowSeconds = 120
	// minWindowCompletions is the fewest completions in the short window for its
	// rate to be used.
	minWindowCompletions = 5
)

// completionWindow counts an endpoint's completions in one-second buckets.
type completionWindow struct {
	mu      sync.Mutex
	seconds [completionFallbackWindowSeconds]int64 // unix second each bucket holds
	counts  [completionFallbackWindowSeconds]int64
}

func (w *completionWindow) record(now time.Time) {
	sec := now.Unix()
	i := sec % completionFallbackWindowSeconds
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.seconds[i] != sec {
		w.seconds[i] = sec
		w.counts[i] = 0
	}
	w.counts[i]++
}

// rate returns completions per second over the short window, or over the
// fallback window when the short one holds fewer than minWindowCompletions.
func (w *completionWindow) rate(now time.Time) float64 {
	sec := now.Unix()
	var short, long int64
	w.mu.Lock()
	defer w.mu.Unlock()
	for i := range w.counts {
		age := sec - w.seconds[i]
		if age < 0 || age >= completionFallbackWindowSeconds {
			continue
		}
		long += w.counts[i]
		if age < completionWindowSeconds {
			short += w.counts[i]
		}
	}
	if short >= minWindowCompletions {
		return float64(short) / completionWindowSeconds
	}
	return float64(long) / completionFallbackWindowSeconds
}

// completionTracker holds a completionWindow per registered endpoint.
type completionTracker struct {
	mu      sync.RWMutex
	windows map[string]*completionWindow
}

func newCompletionTracker() *completionTracker {
	return &completionTracker{windows: make(map[string]*completionWindow)}
}

// add registers an endpoint. An endpoint that is already registered keeps its window.
func (t *completionTracker) add(endpointID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, exists := t.windows[endpointID]; !exists {
		t.windows[endpointID] = &completionWindow{}
	}
}

// record counts a completion on a registered endpoint. Completions for an
// endpoint that is not registered, such as one already deleted, are dropped.
func (t *completionTracker) record(endpointID string, now time.Time) {
	t.mu.RLock()
	w, exists := t.windows[endpointID]
	t.mu.RUnlock()
	if exists {
		w.record(now)
	}
}

// rate returns the endpoint's completions per second, or 0 for an endpoint
// that is not registered.
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
