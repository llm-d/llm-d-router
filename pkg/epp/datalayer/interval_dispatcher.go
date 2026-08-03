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

package datalayer

import (
	"context"
	"fmt"
	"math"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// intervalDispatcher wraps a PollingDispatcher and only forwards Dispatch
// every `period` ticks. On non-firing ticks Dispatch returns nil immediately.
type intervalDispatcher struct {
	fwkdl.PollingDispatcher
	period    int // number of base ticks between dispatches (e.g. 20 = once per second at 50ms base)
	remaining int // countdown: decremented each tick, fires and resets to period when it hits 0
}

// newIntervalDispatcher wraps d so that its Dispatch fires once every period
// base ticks. period <= 1 returns d unwrapped (every-tick is the default).
func newIntervalDispatcher(d fwkdl.PollingDispatcher, period int) fwkdl.PollingDispatcher {
	if period <= 1 {
		return d
	}
	return &intervalDispatcher{
		PollingDispatcher: d,
		period:            period,
		remaining:         1,
	}
}

func (d *intervalDispatcher) Dispatch(ctx context.Context, ep fwkdl.Endpoint) error {
	d.remaining--
	if d.remaining > 0 {
		return nil
	}
	d.remaining = d.period
	return d.PollingDispatcher.Dispatch(ctx, ep)
}

// periodTicks converts a configured scrape interval into a positive number of
// base ticks. interval <= 0 means every base tick (backward-compatible default).
// Non-exact multiples are rounded to the nearest base-tick multiple (minimum 1).
func periodTicks(interval, base time.Duration) (int, error) {
	if base <= 0 {
		return 0, fmt.Errorf("base tick must be positive, got %s", base)
	}
	if interval <= 0 {
		return 1, nil
	}
	ticks := int(math.Round(float64(interval) / float64(base)))
	if ticks < 1 {
		ticks = 1
	}
	return ticks, nil
}
