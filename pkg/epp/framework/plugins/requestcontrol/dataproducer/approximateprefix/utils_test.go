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

package approximateprefix

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCalculateHitRatioStats(t *testing.T) {
	tests := []struct {
		name       string
		matchLens  []float64
		wantMax    float64
		wantAvg    float64
		wantStdDev float64
	}{
		{
			name:       "empty slice yields zero",
			matchLens:  []float64{},
			wantMax:    0,
			wantAvg:    0,
			wantStdDev: 0,
		},
		{
			name:       "single value yields zero stddev",
			matchLens:  []float64{5},
			wantMax:    5,
			wantAvg:    5,
			wantStdDev: 0,
		},
		{
			name:       "multiple values yield correct avg and stddev",
			matchLens:  []float64{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0},
			wantMax:    9,
			wantAvg:    5,
			wantStdDev: 2.14,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			max, avg, stddev := calculateHitRatioStats(tt.matchLens)
			assert.InDelta(t, tt.wantMax, max, 1e-9)
			assert.InDelta(t, tt.wantAvg, avg, 1e-9)
			assert.InDelta(t, tt.wantStdDev, stddev, 1e-9)
		})
	}
}
