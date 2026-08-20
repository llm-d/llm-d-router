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
	"math"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
)

const (
	// drainFoldInterval is how often accumulated dispatch counts are folded into the rate EWMAs.
	drainFoldInterval = time.Second

	// drainEWMAAlpha weights the most recent fold window against the accumulated rate.
	drainEWMAAlpha = 0.3

	// retryAfterHintCap bounds the emitted hint. The value is advisory (RFC 9110) and consumers bound it independently;
	// the cap keeps a decayed estimate from projecting arbitrarily far.
	retryAfterHintCap = 30 * time.Second
)

// drainEstimator tracks dispatch completion rates per priority band and globally, and projects the wait until a slot
// frees in a rejecting scope. It is owned by the Processor and accessed only from the Run goroutine, so it needs no
// synchronization.
type drainEstimator struct {
	perBand  map[int]*drainRate
	global   drainRate
	lastFold time.Time
}

// drainRate is one scope's dispatch-rate state: the EWMA over completed fold windows and the count accumulated in the
// current window.
type drainRate struct {
	count  uint64
	rate   float64 // dispatches per second
	folded bool
}

func newDrainEstimator(now time.Time) *drainEstimator {
	return &drainEstimator{perBand: make(map[int]*drainRate), lastFold: now}
}

// recordDispatch counts one dispatch completion against the band and the global scope.
func (e *drainEstimator) recordDispatch(priority int) {
	b := e.perBand[priority]
	if b == nil {
		b = &drainRate{}
		e.perBand[priority] = b
	}
	b.count++
	e.global.count++
}

// maybeFold folds accumulated counts into the rate EWMAs once per drainFoldInterval.
func (e *drainEstimator) maybeFold(now time.Time) {
	elapsed := now.Sub(e.lastFold)
	if elapsed < drainFoldInterval {
		return
	}
	seconds := elapsed.Seconds()
	e.global.fold(seconds)
	for _, b := range e.perBand {
		b.fold(seconds)
	}
	e.lastFold = now
}

func (r *drainRate) fold(elapsedSeconds float64) {
	r.rate = drainEWMAAlpha*(float64(r.count)/elapsedSeconds) + (1-drainEWMAAlpha)*r.rate
	r.count = 0
	r.folded = true
}

// current returns the scope's dispatch rate in requests per second, and whether the scope has any measurement at all.
// The rate is the EWMA, raised to the current window's instantaneous rate when that is higher, so a burst that resumes
// dispatching after an idle stretch corrects a decayed estimate without waiting for the next fold.
func (r *drainRate) current(sinceFold time.Duration) (float64, bool) {
	rate := r.rate
	if sec := sinceFold.Seconds(); sec > 0 && r.count > 0 {
		if inst := float64(r.count) / sec; inst > rate {
			rate = inst
		}
	}
	return rate, r.folded || r.count > 0
}

// retryAfterHint projects the wait until one slot frees in the scope whose capacity check rejected the request: the
// priority band, the global limit, or both (the longer projection wins). Byte-tripped scopes use the same request-rate
// projection, on the approximation that one dispatch frees an average request's bytes. The hint is rounded up to whole
// seconds and capped at retryAfterHintCap. It is zero, meaning no hint, when no tripped scope has a measured nonzero
// rate or when the projection is under one second: a whole-second floor would over-throttle a fast-draining scope, and
// absence tells the client to keep its own retry policy.
func (e *drainEstimator) retryAfterHint(
	stats contracts.AggregateStats,
	priority int,
	itemByteSize uint64,
	now time.Time,
) time.Duration {
	sinceFold := now.Sub(e.lastFold)

	var projection float64 // seconds
	if globalTripped(stats, itemByteSize) {
		if rate, ok := e.global.current(sinceFold); ok && rate > 0 {
			projection = 1 / rate
		}
	}
	if band, ok := stats.PerPriorityBandStats[priority]; ok && bandTripped(band, itemByteSize) {
		if b := e.perBand[priority]; b != nil {
			if rate, ok := b.current(sinceFold); ok && rate > 0 {
				projection = math.Max(projection, 1/rate)
			}
		}
	}
	if projection < 1 {
		return 0
	}
	hint := time.Duration(math.Ceil(projection)) * time.Second
	if hint > retryAfterHintCap {
		return retryAfterHintCap
	}
	return hint
}

// globalTripped mirrors hasCapacity's global comparisons.
func globalTripped(stats contracts.AggregateStats, itemByteSize uint64) bool {
	return (stats.TotalCapacityBytes > 0 && stats.TotalByteSize+itemByteSize > stats.TotalCapacityBytes) ||
		(stats.TotalCapacityRequests > 0 && stats.TotalLen+1 > stats.TotalCapacityRequests)
}

// bandTripped mirrors hasCapacity's per-band comparisons.
func bandTripped(band contracts.PriorityBandStats, itemByteSize uint64) bool {
	return (band.CapacityBytes > 0 && band.ByteSize+itemByteSize > band.CapacityBytes) ||
		(band.CapacityRequests > 0 && band.Len+1 > band.CapacityRequests)
}
