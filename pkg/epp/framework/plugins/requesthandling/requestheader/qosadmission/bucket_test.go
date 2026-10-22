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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestTokenBucket_StartsFullAtBurstCapacity(t *testing.T) {
	now := time.Unix(0, 0)
	// 6000 tokens/min = 100/s; 10s burst = 1000 capacity.
	b := newTokenBucket(6000, 10, now)
	assert.InDelta(t, 1000, b.remaining(now), 1e-9)
}

func TestTokenBucket_TakeAndRefill(t *testing.T) {
	now := time.Unix(0, 0)
	b := newTokenBucket(6000, 10, now)

	assert.True(t, b.tryTake(1000, now))
	assert.False(t, b.tryTake(1, now), "an empty bucket rejects")

	later := now.Add(2 * time.Second)
	assert.InDelta(t, 200, b.remaining(later), 1e-9)
	assert.True(t, b.tryTake(200, later))
}

func TestTokenBucket_RefillCapsAtCapacity(t *testing.T) {
	now := time.Unix(0, 0)
	b := newTokenBucket(6000, 10, now)
	assert.InDelta(t, 1000, b.remaining(now.Add(time.Hour)), 1e-9)
}

func TestTokenBucket_AdjustRefundAndDebt(t *testing.T) {
	now := time.Unix(0, 0)
	b := newTokenBucket(6000, 10, now)
	b.tryTake(800, now)

	b.adjust(300, now)
	assert.InDelta(t, 500, b.remaining(now), 1e-9)

	b.adjust(5000, now)
	assert.InDelta(t, 1000, b.remaining(now), 1e-9, "a refund cannot exceed capacity")

	b.adjust(-1500, now)
	assert.InDelta(t, 0, b.remaining(now), 1e-9, "debt reads as zero remaining")
	assert.Equal(t, 5*time.Second, b.waitFor(0, now), "debt of 500 at 100/s clears in 5s")
}

func TestTokenBucket_WaitFor(t *testing.T) {
	now := time.Unix(0, 0)
	b := newTokenBucket(6000, 10, now)
	b.tryTake(1000, now)

	assert.Equal(t, 3*time.Second, b.waitFor(300, now))
	assert.Equal(t, 10*time.Second, b.waitFor(5000, now), "a cost above capacity waits for a full bucket")
	assert.Equal(t, 10*time.Second, b.untilFull(now))
	assert.Equal(t, time.Duration(0), b.waitFor(0, now.Add(time.Minute)))
}
