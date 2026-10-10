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

package steps

import (
	"sync/atomic"

	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
)

// forceStreamBudget bounds the total bytes concurrent forced requests may buffer
// at once. One decode step instance serves every request, so a single budget is
// process-global and is the only limit on buffered memory: the coordinator pod
// carries no memory limit, so an exhausted budget sends the request down the
// non-buffering pass-through rather than growing the heap.
type forceStreamBudget struct {
	current atomic.Int64
	max     int64
	// perRequestMax caps a single request's reservation, so one large request
	// cannot claim the whole budget and starve concurrent candidates.
	perRequestMax int64
}

func newForceStreamBudget(maxBytes, perRequestMax int64) *forceStreamBudget {
	return &forceStreamBudget{max: maxBytes, perRequestMax: perRequestMax}
}

// tryReserve adds n to the live total if it stays within max, reporting whether
// it did. The compare-and-swap retries only against a competing reservation, so
// a caller that loses the race re-reads the total instead of over-committing.
// n must be in (0, max]; estimateReservation guarantees that before any call.
// current stays in [0, max], so max-current never underflows an int64.
func (b *forceStreamBudget) tryReserve(n int64) bool {
	for {
		cur := b.current.Load()
		if n > b.max-cur {
			return false
		}
		if b.current.CompareAndSwap(cur, cur+n) {
			coordmetrics.AddForceStreamBufferedBytes(n)
			return true
		}
	}
}

// release returns a prior reservation of n bytes. Each tryReserve that returns
// true is balanced by exactly one release, so the live total cannot drift.
func (b *forceStreamBudget) release(n int64) {
	b.current.Add(-n)
	coordmetrics.SubForceStreamBufferedBytes(n)
}
