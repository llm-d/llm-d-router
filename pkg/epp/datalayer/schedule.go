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
	"fmt"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// ScheduledDispatcher pairs a PollingDispatcher with how often the Collector
// should invoke it, measured in base ticks (Runtime.pollingInterval).
type ScheduledDispatcher struct {
	Dispatcher  fwkdl.PollingDispatcher
	PeriodTicks int // must be >= 1; 1 = every base tick
}

// PeriodTicks converts a configured scrape interval into a positive number of
// base ticks. interval <= 0 means every base tick (backward-compatible default).
// A non-zero interval must be a positive multiple of base so cadences stay on
// the Collector's single ticker (no arbitrary durations such as 53ms).
func PeriodTicks(interval, base time.Duration) (int, error) {
	if base <= 0 {
		return 0, fmt.Errorf("base tick must be positive, got %s", base)
	}
	if interval <= 0 {
		return 1, nil
	}
	if interval < base || interval%base != 0 {
		return 0, fmt.Errorf("interval %s must be a positive multiple of base tick %s", interval, base)
	}
	return int(interval / base), nil
}
