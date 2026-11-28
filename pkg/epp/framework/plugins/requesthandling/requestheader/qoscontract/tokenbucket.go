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
	"math"
	"time"
)

// tokenBucket limits a token rate: its level refills at rate tokens per second up to capacity, and
// each request within the contract takes its estimated tokens from it. Unlike a sliding window, a
// bucket keeps no per-request history, so the difference between a request's estimate and its
// actual tokens is applied whenever the request completes. An overrun can take the level below
// zero, a debt the refill pays back before the next request fits. Callers serialize access.
type tokenBucket struct {
	rate     float64 // tokens per second
	capacity float64 // the largest burst, rate * window
	level    float64
	last     time.Time
}

// newTokenBucket returns a full bucket for rate tokens per second, with a burst of rate * window.
func newTokenBucket(rate float64, window time.Duration, now time.Time) *tokenBucket {
	capacity := rate * window.Seconds()
	return &tokenBucket{rate: rate, capacity: capacity, level: capacity, last: now}
}

// refill adds the tokens accrued since the last refill, up to capacity. A now before the last
// refill, from a wall clock that stepped back, adds nothing.
func (b *tokenBucket) refill(now time.Time) {
	if now.After(b.last) {
		b.level = math.Min(b.capacity, b.level+b.rate*now.Sub(b.last).Seconds())
		b.last = now
	}
}

// fits reports whether a request of cost tokens fits at now. A request larger than the whole burst
// fits when the bucket is full, so it is not refused forever.
func (b *tokenBucket) fits(now time.Time, cost float64) bool {
	b.refill(now)
	return b.level >= cost || b.level >= b.capacity
}

// take charges cost tokens. Callers check fits first.
func (b *tokenBucket) take(cost float64) {
	b.level -= cost
}

// settle applies delta, the reservation minus the actual tokens, at now: a refund when positive, an
// overrun when negative. The level stays at most capacity and may fall below zero.
func (b *tokenBucket) settle(now time.Time, delta float64) {
	b.refill(now)
	b.level = math.Min(b.capacity, b.level+delta)
}

// retryAfter returns the time until a request of cost tokens fits, when no other request arrives.
// It is meaningful only after fits(now, cost) reported false.
func (b *tokenBucket) retryAfter(cost float64) time.Duration {
	deficit := math.Min(cost, b.capacity) - b.level
	if deficit <= 0 {
		return 0
	}
	return time.Duration(deficit / b.rate * float64(time.Second))
}
