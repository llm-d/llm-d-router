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
	"slices"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
)

const (
	// drainEWMAAlpha weights each newly observed contended inter-dispatch interval in the per-scope
	// exponential moving average.
	drainEWMAAlpha = 0.3

	// drainMinHintThreshold suppresses sub-half-second projections so brief contention does not emit
	// a 1s Retry-After floor.
	drainMinHintThreshold = 500 * time.Millisecond

	// retryAfterHintCap bounds both individual interval samples (preventing a single outage from
	// dominating subsequent EWMA estimates) and the emitted Retry-After delay.
	retryAfterHintCap = 30 * time.Second

	// drainIdleExpiry clears a scope's cached EWMA after continuous idle time so recovery after a
	// quiet stretch is not anchored to stale slow history.
	drainIdleExpiry = 60 * time.Second
)

// scopeDrain tracks contended dispatch pacing for one aggregation scope (a single priority band or
// the registry-wide aggregate). All methods run exclusively on the Processor.Run goroutine.
type scopeDrain struct {
	busy             bool
	hasEstimate      bool
	lastTransitionAt time.Time
	accumulatedBusy  time.Duration
	meanInterval     float64
}

func (s *scopeDrain) reset() {
	s.hasEstimate = false
	s.meanInterval = 0
	s.accumulatedBusy = 0
}

func (s *scopeDrain) isIdleExpired(now time.Time) bool {
	return !s.busy && !s.lastTransitionAt.IsZero() && now.Sub(s.lastTransitionAt) >= drainIdleExpiry
}

func (s *scopeDrain) busyDuration(now time.Time) time.Duration {
	dur := s.accumulatedBusy
	if s.busy && !s.lastTransitionAt.IsZero() {
		elapsed := min(max(time.Duration(0), now.Sub(s.lastTransitionAt)), retryAfterHintCap)
		dur = min(dur+elapsed, retryAfterHintCap)
	}
	return dur
}

func (s *scopeDrain) sync(nowBusy bool, now time.Time) {
	if !s.busy {
		if s.isIdleExpired(now) {
			s.reset()
		}
		if nowBusy {
			s.busy = true
			s.lastTransitionAt = now
		}
		return
	}
	if !nowBusy {
		// Fold elapsed busy time before pausing so a TTL sweep or empty-pool transition preserves
		// zero-drain stall progress without charging idle time while paused.
		s.accumulatedBusy = s.busyDuration(now)
		s.busy = false
		s.lastTransitionAt = now
	}
}

func (s *scopeDrain) recordContendedDispatch(now time.Time) {
	dt := s.busyDuration(now).Seconds()
	if !s.hasEstimate {
		s.meanInterval = dt
	} else {
		s.meanInterval += drainEWMAAlpha * (dt - s.meanInterval)
	}
	s.hasEstimate = true
	s.accumulatedBusy = 0
	s.lastTransitionAt = now
}

func (s *scopeDrain) projection(dim contracts.CapacityDimension, itemByteSize uint64, now time.Time) (waitSec float64, feasible bool) {
	// An item larger than the scope's total byte capacity can never fit regardless of drain rate.
	if dim.CapacityBytes > 0 && itemByteSize > dim.CapacityBytes {
		return 0, false
	}

	var reqDispatches float64
	if dim.CapacityRequests > 0 && dim.Len >= dim.CapacityRequests {
		reqDispatches = float64(dim.Len-dim.CapacityRequests) + 1
	}

	var byteDispatches float64
	if dim.CapacityBytes > 0 {
		freeBytes := dim.CapacityBytes - itemByteSize
		if dim.ByteSize > freeBytes {
			byteDeficit := dim.ByteSize - freeBytes
			if dim.Len == 0 {
				// Guard against near-consistent snapshot tears where a concurrent cleanup sweep
				// zeroed Len before ByteSize.
				byteDispatches = 1.0
			} else {
				byteDispatches = min(float64(dim.Len), max(1.0, float64(dim.Len)*float64(byteDeficit)/float64(dim.ByteSize)))
			}
		}
	}

	dispatches := max(reqDispatches, byteDispatches)
	if dispatches == 0 || s == nil || s.isIdleExpired(now) {
		return 0, true
	}

	var mean float64
	if s.hasEstimate {
		mean = s.meanInterval
	}
	slotSec := max(mean, s.busyDuration(now).Seconds())
	return dispatches * slotSec, true
}

// drainEstimator maintains per-band and global contended drain-interval estimates to project
// RFC 9110 Retry-After hints on queue-capacity rejections. Owned exclusively by Processor.Run.
type drainEstimator struct {
	global  scopeDrain
	perBand map[int]*scopeDrain
}

func newDrainEstimator() *drainEstimator {
	return &drainEstimator{
		perBand: make(map[int]*scopeDrain),
	}
}

func (e *drainEstimator) syncBusy(priorities []int, registry contracts.FlowRegistry, poolEmpty bool, now time.Time) {
	var (
		globalLen   uint64
		anyBandBusy bool
	)
	for _, priority := range priorities {
		snapshot, err := registry.CapacitySnapshot(priority)
		if err != nil {
			// The band was removed between listing priority levels and taking the snapshot.
			continue
		}
		globalLen = snapshot.Global.Len
		bandNowBusy := !poolEmpty && snapshot.Band.Len > 0
		if bandNowBusy {
			anyBandBusy = true
		}
		band, ok := e.perBand[priority]
		if !ok {
			if !bandNowBusy {
				continue
			}
			band = &scopeDrain{}
			e.perBand[priority] = band
		}
		band.sync(bandNowBusy, now)
	}

	globalNowBusy := anyBandBusy || (!poolEmpty && globalLen > 0)
	e.global.sync(globalNowBusy, now)

	for p := range e.perBand {
		if !slices.Contains(priorities, p) {
			delete(e.perBand, p)
		}
	}
}

func (e *drainEstimator) recordDispatch(priority int, now time.Time) {
	band, ok := e.perBand[priority]
	if !ok || !band.busy {
		// Pass-through dispatch through an idle band: clear paused stall accumulation on this band
		// (and on global when no other band is queued) without distorting contended EWMAs.
		if ok {
			band.accumulatedBusy = 0
		}
		if !e.global.busy {
			e.global.accumulatedBusy = 0
		}
		return
	}
	band.recordContendedDispatch(now)
	e.global.recordContendedDispatch(now)
}

func (e *drainEstimator) retryAfterHint(
	snapshot contracts.CapacitySnapshot,
	priority int,
	itemByteSize uint64,
	now time.Time,
) time.Duration {
	globalWait, globalFeasible := e.global.projection(snapshot.Global, itemByteSize, now)
	if !globalFeasible {
		return 0
	}
	bandWait, bandFeasible := e.perBand[priority].projection(snapshot.Band, itemByteSize, now)
	if !bandFeasible {
		return 0
	}

	waitSec := max(globalWait, bandWait)
	if waitSec < drainMinHintThreshold.Seconds() {
		return 0
	}
	secs := int64(math.Ceil(min(waitSec, retryAfterHintCap.Seconds())))
	return time.Duration(secs) * time.Second
}
