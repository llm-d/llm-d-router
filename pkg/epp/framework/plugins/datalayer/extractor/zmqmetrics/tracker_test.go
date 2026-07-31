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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPrefillTracker_AddReading(t *testing.T) {
	tracker := newPrefillTracker(3)
	epKey := "default/pod-1"

	// Reading 1: 1000 tokens in 50ms -> 20000 tokens/sec
	rate1 := tracker.AddReading(epKey, 1000, 50.0)
	assert.InDelta(t, 20000.0, rate1, 0.001)

	// Reading 2: 2000 tokens in 100ms -> total (1000+2000)/(50+100)ms = 3000 tokens / 0.15s = 20000 tokens/sec
	rate2 := tracker.AddReading(epKey, 2000, 100.0)
	assert.InDelta(t, 20000.0, rate2, 0.001)

	// Reading 3: 500 tokens in 10ms -> total 3500 tokens / 160ms = 21875 tokens/sec
	rate3 := tracker.AddReading(epKey, 500, 10.0)
	assert.InDelta(t, 21875.0, rate3, 0.001)

	// Reading 4: ring buffer capacity is 3. Replaces reading 1 (1000, 50ms).
	// Buffer has: (2000, 100ms), (500, 10ms), (1000, 20ms)
	// total = 3500 tokens / 130ms = 26923.0769 tokens/sec
	rate4 := tracker.AddReading(epKey, 1000, 20.0)
	assert.InDelta(t, 26923.0769, rate4, 0.01)
}

func TestPrefillTracker_InvalidInputs(t *testing.T) {
	tracker := newPrefillTracker(3)
	epKey := "default/pod-1"

	// Initial reading
	tracker.AddReading(epKey, 1000, 50.0)

	// Invalid readings (zero tokens, negative latency, NaN, Inf) should return current rate without mutating buffer
	assert.InDelta(t, 20000.0, tracker.AddReading(epKey, 0, 50.0), 0.001)
	assert.InDelta(t, 20000.0, tracker.AddReading(epKey, 1000, 0), 0.001)
	assert.InDelta(t, 20000.0, tracker.AddReading(epKey, 1000, -10.0), 0.001)
	assert.InDelta(t, 20000.0, tracker.AddReading(epKey, 1000, math.NaN()), 0.001)
	assert.InDelta(t, 20000.0, tracker.AddReading(epKey, 1000, math.Inf(1)), 0.001)
}

func TestPrefillTracker_RemoveEndpoint(t *testing.T) {
	tracker := newPrefillTracker(3)
	epKey := "default/pod-1"

	tracker.AddReading(epKey, 1000, 50.0)
	assert.Len(t, tracker.buffers, 1)

	tracker.RemoveEndpoint(epKey)
	assert.Len(t, tracker.buffers, 0)
}
