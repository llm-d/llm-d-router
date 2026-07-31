/*
Copyright 2026 The Kubernetes Authors.

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

package zmqmetrics

import (
	"math"
	"sync"
)

type prefillSample struct {
	computedTokens          int
	batchExecutionLatencyMs float64
}

type ringBuffer struct {
	samples []prefillSample
	idx     int
	count   int
}

type prefillTracker struct {
	mu       sync.RWMutex
	capacity int
	buffers  map[string]*ringBuffer
}

func newPrefillTracker(capacity int) *prefillTracker {
	return &prefillTracker{
		capacity: capacity,
		buffers:  make(map[string]*ringBuffer),
	}
}

func (t *prefillTracker) AddReading(epKey string, computedTokens int, latencyMs float64) float64 {
	if computedTokens <= 0 || latencyMs <= 0 || math.IsNaN(latencyMs) || math.IsInf(latencyMs, 0) {
		t.mu.RLock()
		buf, exists := t.buffers[epKey]
		var currentRate float64
		if exists {
			currentRate = buf.computeRate()
		}
		t.mu.RUnlock()
		return currentRate
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	buf, exists := t.buffers[epKey]
	if !exists {
		buf = &ringBuffer{
			samples: make([]prefillSample, t.capacity),
		}
		t.buffers[epKey] = buf
	}

	buf.samples[buf.idx] = prefillSample{
		computedTokens:          computedTokens,
		batchExecutionLatencyMs: latencyMs,
	}
	buf.idx = (buf.idx + 1) % t.capacity
	if buf.count < t.capacity {
		buf.count++
	}

	return buf.computeRate()
}

func (b *ringBuffer) computeRate() float64 {
	if b.count == 0 {
		return 0
	}
	var totalTokens int
	var totalLatencyMs float64

	for i := 0; i < b.count; i++ {
		totalTokens += b.samples[i].computedTokens
		totalLatencyMs += b.samples[i].batchExecutionLatencyMs
	}

	if totalLatencyMs <= 0 || math.IsNaN(totalLatencyMs) || math.IsInf(totalLatencyMs, 0) {
		return 0
	}

	rate := float64(totalTokens) / (totalLatencyMs / 1000.0)
	if math.IsNaN(rate) || math.IsInf(rate, 0) {
		return 0
	}
	return rate
}

func (t *prefillTracker) RemoveEndpoint(epKey string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.buffers, epKey)
}
