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

// Package duration holds helpers for plugin parameters given in seconds.
package duration

import (
	"fmt"
	"math"
	"time"
)

// ValidateSeconds rejects a number of seconds that does not convert to a
// positive time.Duration. A ticker panics on a non-positive interval.
func ValidateSeconds(field string, seconds float64) error {
	if seconds <= 0 {
		return fmt.Errorf("%s must be > 0, got %v", field, seconds)
	}
	// float64(math.MaxInt64) rounds up to 2^63, the first value outside int64.
	nanos := seconds * float64(time.Second)
	if nanos >= float64(math.MaxInt64) {
		return fmt.Errorf("%s %v exceeds the representable duration", field, seconds)
	}
	if time.Duration(nanos) <= 0 {
		return fmt.Errorf("%s %v is too small to represent as a duration", field, seconds)
	}
	return nil
}
