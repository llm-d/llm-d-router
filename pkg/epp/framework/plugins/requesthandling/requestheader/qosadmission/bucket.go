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

package qosadmission

import (
	"math"
	"time"
)

// tokenBucket is a lazily refilled token bucket. It refills continuously at ratePerSecond up
// to capacity, and holds the debt of a refund-adjusted charge as a negative level. The caller
// serializes access.
type tokenBucket struct {
	ratePerSecond float64
	capacity      float64
	level         float64
	updated       time.Time
}

func newTokenBucket(tokensPerMinute, burstSeconds float64, now time.Time) *tokenBucket {
	rate := tokensPerMinute / 60
	capacity := rate * burstSeconds
	return &tokenBucket{ratePerSecond: rate, capacity: capacity, level: capacity, updated: now}
}

func (b *tokenBucket) refill(now time.Time) {
	elapsed := now.Sub(b.updated).Seconds()
	if elapsed > 0 {
		b.level = math.Min(b.capacity, b.level+elapsed*b.ratePerSecond)
		b.updated = now
	}
}

// tryTake charges cost if the bucket holds at least cost tokens.
func (b *tokenBucket) tryTake(cost float64, now time.Time) bool {
	b.refill(now)
	if b.level < cost {
		return false
	}
	b.level -= cost
	return true
}

// adjust adds delta tokens: positive for a refund, negative for a charge above the reservation.
// The level never exceeds capacity; it may go negative, which delays later admissions.
func (b *tokenBucket) adjust(delta float64, now time.Time) {
	b.refill(now)
	b.level = math.Min(b.capacity, b.level+delta)
}

// remaining returns the tokens available now, floored at zero.
func (b *tokenBucket) remaining(now time.Time) float64 {
	b.refill(now)
	return math.Max(0, b.level)
}

// waitFor returns how long until the bucket holds cost tokens. Costs above capacity never fit,
// so the wait is to a full bucket.
func (b *tokenBucket) waitFor(cost float64, now time.Time) time.Duration {
	b.refill(now)
	target := math.Min(cost, b.capacity)
	if b.level >= target || b.ratePerSecond <= 0 {
		return 0
	}
	return time.Duration((target - b.level) / b.ratePerSecond * float64(time.Second))
}

// untilFull returns how long until the bucket is full.
func (b *tokenBucket) untilFull(now time.Time) time.Duration {
	return b.waitFor(b.capacity, now)
}
