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

package internal

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
)

const drainTestPriority = 10

// fullBandStats returns stats whose band capacity check fails for any request size, leaving the global scope untripped.
func fullBandStats() contracts.AggregateStats {
	return contracts.AggregateStats{
		PerPriorityBandStats: map[int]contracts.PriorityBandStats{
			drainTestPriority: {Priority: drainTestPriority, CapacityRequests: 5, Len: 5},
		},
	}
}

func TestDrainEstimator(t *testing.T) {
	t.Parallel()
	base := time.Unix(1000, 0)

	t.Run("omits the hint before any measurement", func(t *testing.T) {
		t.Parallel()
		e := newDrainEstimator(base)
		assert.Zero(t, e.retryAfterHint(fullBandStats(), drainTestPriority, 100, base),
			"a scope with no observed dispatches has no basis for a projection")
	})

	t.Run("omits the hint on a measured-zero rate", func(t *testing.T) {
		t.Parallel()
		e := newDrainEstimator(base)
		e.maybeFold(base.Add(2 * time.Second))
		e.maybeFold(base.Add(4 * time.Second))
		assert.Zero(t, e.retryAfterHint(fullBandStats(), drainTestPriority, 100, base.Add(4*time.Second)),
			"zero drain gives no bound on the wait, so no hint is emitted")
	})

	t.Run("projects one slot from the band rate, rounded up", func(t *testing.T) {
		t.Parallel()
		e := newDrainEstimator(base)
		e.recordDispatch(drainTestPriority)
		e.recordDispatch(drainTestPriority)
		now := base.Add(2 * time.Second)
		e.maybeFold(now) // rate = 0.3 * (2/2) = 0.3/s; projection = 3.33s.
		assert.Equal(t, 4*time.Second, e.retryAfterHint(fullBandStats(), drainTestPriority, 100, now))
	})

	t.Run("omits sub-second projections", func(t *testing.T) {
		t.Parallel()
		e := newDrainEstimator(base)
		for range 10 {
			e.recordDispatch(drainTestPriority)
		}
		now := base.Add(2 * time.Second)
		e.maybeFold(now) // rate = 0.3 * (10/2) = 1.5/s; projection = 0.67s.
		assert.Zero(t, e.retryAfterHint(fullBandStats(), drainTestPriority, 100, now),
			"a whole-second floor would over-throttle a fast-draining band")
	})

	t.Run("caps long projections", func(t *testing.T) {
		t.Parallel()
		e := newDrainEstimator(base)
		e.recordDispatch(drainTestPriority)
		now := base.Add(100 * time.Second)
		e.maybeFold(now) // rate = 0.3 * (1/100) = 0.003/s; projection = 333s.
		assert.Equal(t, retryAfterHintCap, e.retryAfterHint(fullBandStats(), drainTestPriority, 100, now))
	})

	t.Run("current-window dispatches correct a decayed rate", func(t *testing.T) {
		t.Parallel()
		e := newDrainEstimator(base)
		e.recordDispatch(drainTestPriority)
		e.recordDispatch(drainTestPriority)
		e.maybeFold(base.Add(2 * time.Second)) // rate = 0.3/s.
		e.maybeFold(base.Add(4 * time.Second)) // idle folds decay the EWMA toward zero.
		e.maybeFold(base.Add(6 * time.Second))
		now := base.Add(6*time.Second + 500*time.Millisecond)
		withoutBurst := e.retryAfterHint(fullBandStats(), drainTestPriority, 100, now)
		assert.Equal(t, 7*time.Second, withoutBurst, "decayed rate 0.147/s projects 6.8s")

		for range 5 {
			e.recordDispatch(drainTestPriority)
		}
		assert.Zero(t, e.retryAfterHint(fullBandStats(), drainTestPriority, 100, now),
			"5 dispatches in the open 500ms window prove a 10/s drain, under the one-second floor")
	})

	t.Run("uses the global rate when only the global limit tripped", func(t *testing.T) {
		t.Parallel()
		e := newDrainEstimator(base)
		e.recordDispatch(drainTestPriority)
		now := base.Add(2 * time.Second)
		e.maybeFold(now) // global rate = 0.3 * (1/2) = 0.15/s; projection = 6.67s.
		stats := contracts.AggregateStats{
			TotalCapacityRequests: 5,
			TotalLen:              5,
			PerPriorityBandStats: map[int]contracts.PriorityBandStats{
				drainTestPriority: {Priority: drainTestPriority, CapacityRequests: 100},
			},
		}
		assert.Equal(t, 7*time.Second, e.retryAfterHint(stats, drainTestPriority, 100, now))
	})

	t.Run("takes the longer projection when both scopes tripped", func(t *testing.T) {
		t.Parallel()
		e := newDrainEstimator(base)
		const fastPriority = 20
		for range 10 {
			e.recordDispatch(fastPriority)
		}
		e.recordDispatch(drainTestPriority)
		now := base.Add(2 * time.Second)
		e.maybeFold(now) // band rate = 0.15/s (6.67s); global rate = 1.65/s (0.61s).
		stats := contracts.AggregateStats{
			TotalCapacityRequests: 5,
			TotalLen:              5,
			PerPriorityBandStats: map[int]contracts.PriorityBandStats{
				drainTestPriority: {Priority: drainTestPriority, CapacityRequests: 5, Len: 5},
			},
		}
		assert.Equal(t, 7*time.Second, e.retryAfterHint(stats, drainTestPriority, 100, now),
			"the slower tripped scope bounds the wait")
	})
}
