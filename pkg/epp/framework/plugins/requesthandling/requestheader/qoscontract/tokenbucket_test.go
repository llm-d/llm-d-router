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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var bucketStart = time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC)

func TestTokenBucket_RefillsAtRateUpToCapacity(t *testing.T) {
	b := newTokenBucket(100, 2*time.Second, bucketStart)
	assert.InDelta(t, 200, b.level, 1e-9, "a bucket starts full: rate * window")
	require.True(t, b.fits(bucketStart, 200))
	b.take(150)

	assert.False(t, b.fits(bucketStart, 100))
	assert.Equal(t, 500*time.Millisecond, b.retryAfter(100), "50 tokens short at 100 per second")
	assert.True(t, b.fits(bucketStart.Add(500*time.Millisecond), 100))

	b.refill(bucketStart.Add(time.Hour))
	assert.InDelta(t, 200, b.level, 1e-9, "refill stops at capacity")
}

func TestTokenBucket_RequestAboveCapacityFitsWhenFull(t *testing.T) {
	b := newTokenBucket(100, time.Second, bucketStart)
	require.True(t, b.fits(bucketStart, 250), "a full bucket admits a request larger than the burst")
	b.take(250)
	assert.InDelta(t, -150, b.level, 1e-9)

	assert.False(t, b.fits(bucketStart.Add(time.Second), 250), "it fits again only once the bucket is full")
	assert.Equal(t, 1500*time.Millisecond, b.retryAfter(250), "at -50 the bucket is 150 tokens short of full")
	assert.True(t, b.fits(bucketStart.Add(2500*time.Millisecond), 250))
}

func TestTokenBucket_SettleRefundsAndOverruns(t *testing.T) {
	b := newTokenBucket(100, time.Second, bucketStart)
	b.take(80)
	b.settle(bucketStart, 50)
	assert.InDelta(t, 70, b.level, 1e-9, "a refund comes back")
	b.settle(bucketStart, 100)
	assert.InDelta(t, 100, b.level, 1e-9, "a refund does not lift the level above capacity")
	b.settle(bucketStart, -150)
	assert.InDelta(t, -50, b.level, 1e-9, "an overrun is a debt")
	assert.False(t, b.fits(bucketStart.Add(400*time.Millisecond), 0), "the debt is paid back before any request fits")
	assert.True(t, b.fits(bucketStart.Add(500*time.Millisecond), 0))
}

func TestTokenBucket_ClockSteppingBackAddsNothing(t *testing.T) {
	b := newTokenBucket(100, time.Second, bucketStart)
	b.take(100)
	assert.False(t, b.fits(bucketStart.Add(-time.Hour), 1))
	assert.InDelta(t, 0, b.level, 1e-9)
	assert.True(t, b.fits(bucketStart.Add(20*time.Millisecond), 1), "refill counts from the latest time seen")
}

func TestTokenBucket_FractionalRate(t *testing.T) {
	b := newTokenBucket(0.5, 10*time.Second, bucketStart)
	assert.InDelta(t, 5, b.capacity, 1e-9)
	b.take(5)
	assert.Equal(t, 2*time.Second, b.retryAfter(1), "one token at half a token per second")
	assert.Equal(t, time.Duration(0), b.retryAfter(0), "no wait for a request that fits")
}
