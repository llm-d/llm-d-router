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

// Package clamp converts between integer widths and signedness at their boundaries instead of
// wrapping, for sites where the value is expected to stay in range but the compiler cannot prove it.
package clamp

import "math"

// Uint64 converts a signed integer to uint64, clamping to 0 instead of wrapping if v is negative.
func Uint64[T int | int32 | int64](v T) uint64 {
	if v < 0 {
		return 0
	}
	return uint64(v)
}

// Int64 converts v to int64, clamping to math.MaxInt64 instead of wrapping if v exceeds the signed range.
func Int64(v uint64) int64 {
	if v > math.MaxInt64 {
		return math.MaxInt64
	}
	return int64(v)
}

// Int converts v to int, clamping to math.MaxInt instead of wrapping if v exceeds the signed range.
func Int(v uint64) int {
	if v > math.MaxInt {
		return math.MaxInt
	}
	return int(v)
}

// Uint32 converts a signed integer to uint32, clamping to 0 if v is negative and to
// math.MaxUint32 if v exceeds the unsigned 32-bit range.
func Uint32[T int | int32 | int64](v T) uint32 {
	if v < 0 {
		return 0
	}
	if uint64(v) > math.MaxUint32 {
		return math.MaxUint32
	}
	return uint32(v)
}
