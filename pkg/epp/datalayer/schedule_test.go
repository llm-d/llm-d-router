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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPeriodTicks(t *testing.T) {
	base := 50 * time.Millisecond
	tests := []struct {
		name     string
		interval time.Duration
		base     time.Duration
		want     int
		wantErr  bool
	}{
		{name: "zero interval defaults to every tick", interval: 0, base: base, want: 1},
		{name: "negative interval defaults to every tick", interval: -time.Second, base: base, want: 1},
		{name: "equal to base", interval: base, base: base, want: 1},
		{name: "1s is 20 ticks", interval: time.Second, base: base, want: 20},
		{name: "5s is 100 ticks", interval: 5 * time.Second, base: base, want: 100},
		{name: "not a multiple", interval: 75 * time.Millisecond, base: base, wantErr: true},
		{name: "smaller than base", interval: 25 * time.Millisecond, base: base, wantErr: true},
		{name: "non-positive base", interval: time.Second, base: 0, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := PeriodTicks(tt.interval, tt.base)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}
