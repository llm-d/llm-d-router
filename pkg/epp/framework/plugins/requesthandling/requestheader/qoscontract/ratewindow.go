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

import "time"

// slidingWindow is an exact sliding-log rate limit: it holds the times of the last limit requests
// it admitted, so a request is over the limit when limit requests were admitted within the last
// window. The oldest time is at next. Callers serialize access.
type slidingWindow struct {
	window time.Duration
	times  []time.Time // ring buffer of the last len(times) admissions
	next   int
	n      int
}

func newSlidingWindow(window time.Duration, limit int) *slidingWindow {
	return &slidingWindow{window: window, times: make([]time.Time, limit)}
}

// full reports whether limit requests were admitted within the window ending at now.
func (w *slidingWindow) full(now time.Time) bool {
	return w.n == len(w.times) && now.Sub(w.times[w.next]) < w.window
}

// add records a request admitted at now, replacing the oldest when the ring is full.
func (w *slidingWindow) add(now time.Time) {
	w.times[w.next] = now
	w.next = (w.next + 1) % len(w.times)
	if w.n < len(w.times) {
		w.n++
	}
}

// retryAfter returns the time from now until the oldest admission leaves the window, when the
// next request fits. It is meaningful only when full(now).
func (w *slidingWindow) retryAfter(now time.Time) time.Duration {
	return w.times[w.next].Add(w.window).Sub(now)
}
