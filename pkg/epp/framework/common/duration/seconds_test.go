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

package duration

import (
	"math"
	"testing"
)

func TestValidateSeconds(t *testing.T) {
	tests := []struct {
		name    string
		seconds float64
		wantErr string
	}{
		{name: "one second", seconds: 1},
		{name: "one nanosecond", seconds: 1e-9},
		{name: "zero", seconds: 0, wantErr: "sweep must be > 0, got 0"},
		{name: "negative", seconds: -1, wantErr: "sweep must be > 0, got -1"},
		{name: "below one nanosecond", seconds: 1e-10, wantErr: "sweep 1e-10 is too small to represent as a duration"},
		{name: "past int64 nanoseconds", seconds: 1e10, wantErr: "sweep 1e+10 exceeds the representable duration"},
		{name: "infinity", seconds: math.Inf(1), wantErr: "sweep +Inf exceeds the representable duration"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateSeconds("sweep", tt.seconds)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("ValidateSeconds(%v) = %v, want nil", tt.seconds, err)
				}
				return
			}
			if err == nil || err.Error() != tt.wantErr {
				t.Fatalf("ValidateSeconds(%v) = %v, want %q", tt.seconds, err, tt.wantErr)
			}
		})
	}
}
