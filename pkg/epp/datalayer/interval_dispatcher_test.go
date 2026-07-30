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
	"errors"
	"sync/atomic"
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
		{name: "not a multiple rounds to nearest", interval: 75 * time.Millisecond, base: base, want: 2},
		{name: "not a multiple rounds up", interval: 130 * time.Millisecond, base: base, want: 3},
		{name: "not a multiple rounds down", interval: 120 * time.Millisecond, base: base, want: 2},
		{name: "smaller than base clamps to 1", interval: 25 * time.Millisecond, base: base, want: 1},
		{name: "non-positive base", interval: time.Second, base: 0, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := periodTicks(tt.interval, tt.base)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestIntervalDispatcher(t *testing.T) {
	tests := []struct {
		name      string
		period    int
		ticks     int
		wantCalls int64
	}{
		{name: "period 1 fires every tick", period: 1, ticks: 5, wantCalls: 5},
		{name: "period 2 fires every other tick", period: 2, ticks: 6, wantCalls: 3},
		{name: "period 3 fires every third tick", period: 3, ticks: 7, wantCalls: 3},
		{name: "period 0 returns unwrapped", period: 0, ticks: 4, wantCalls: 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inner := &errSource{kind: "test", err: nil}
			d := newIntervalDispatcher(inner, tt.period)
			for i := 0; i < tt.ticks; i++ {
				_ = d.Dispatch(context.Background(), defaultEndpoint())
			}
			assert.Equal(t, tt.wantCalls, atomic.LoadInt64(&inner.CallCount))
		})
	}
}

func TestIntervalDispatcher_PropagatesError(t *testing.T) {
	wantErr := errors.New("boom")
	inner := &errSource{kind: "test", err: wantErr}
	d := newIntervalDispatcher(inner, 1)
	err := d.Dispatch(context.Background(), defaultEndpoint())
	assert.ErrorIs(t, err, wantErr)
}

func TestIntervalDispatcher_IndependentCounters(t *testing.T) {
	inner := &errSource{kind: "test", err: nil}
	original := newIntervalDispatcher(inner, 3)

	clone := newIntervalDispatcher(
		original.(*intervalDispatcher).PollingDispatcher,
		original.(*intervalDispatcher).period,
	)

	ep := defaultEndpoint()
	ctx := context.Background()

	// Fire original once (remaining 1->0, fires, remaining=3)
	_ = original.Dispatch(ctx, ep)
	assert.Equal(t, int64(1), atomic.LoadInt64(&inner.CallCount))

	// Clone should also fire on its first call (independent counter)
	_ = clone.Dispatch(ctx, ep)
	assert.Equal(t, int64(2), atomic.LoadInt64(&inner.CallCount))
}
