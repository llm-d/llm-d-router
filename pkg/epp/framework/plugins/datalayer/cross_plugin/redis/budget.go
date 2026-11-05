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

package redis

import (
	"math"
	"sync"
	"time"
)

// byteBudget is a token bucket of bytes that refills at rate per second and
// holds at most one second of rate. A transfer may start while any budget is
// left and overdraw it; the debt delays later transfers. A nil budget is
// unlimited.
type byteBudget struct {
	mu     sync.Mutex
	rate   float64
	tokens float64
	last   time.Time
	now    func() time.Time
}

func newByteBudget(bytesPerSecond int64) *byteBudget {
	b := &byteBudget{rate: float64(bytesPerSecond), tokens: float64(bytesPerSecond), now: time.Now}
	b.last = b.now()
	return b
}

func (b *byteBudget) refillLocked() {
	now := b.now()
	b.tokens = min(b.rate, b.tokens+b.rate*now.Sub(b.last).Seconds())
	b.last = now
}

// take charges n bytes and reports true if any budget was left.
func (b *byteBudget) take(n int) bool {
	if b == nil {
		return true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refillLocked()
	if b.tokens <= 0 {
		return false
	}
	b.tokens -= float64(n)
	return true
}

// available returns the bytes left, which is negative while in debt.
func (b *byteBudget) available() float64 {
	if b == nil {
		return math.Inf(1)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refillLocked()
	return b.tokens
}

// charge adds n bytes to what was spent. A negative n refunds.
func (b *byteBudget) charge(n int) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.tokens -= float64(n)
}

// reserve deducts want elements of avg bytes under one lock and reports how
// many were reserved. Like take, it allows one overdraft when any budget is
// left. A nil budget is unlimited.
func (b *byteBudget) reserve(avg, want int64) (count int64, reserved int, ok bool) {
	if want <= 0 {
		return 0, 0, true
	}
	if avg < 1 {
		avg = 1
	}
	if b == nil {
		return want, 0, true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refillLocked()
	if b.tokens <= 0 {
		return 0, 0, false
	}
	count = min(max(int64(b.tokens)/avg, 1), want)
	reserved = int(count * avg)
	b.tokens -= float64(reserved)
	return count, reserved, true
}
