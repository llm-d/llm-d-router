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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
)

type fakeDrainRegistry struct {
	*mocks.MockRegistryDataPlane
	snapshots map[int]contracts.CapacitySnapshot
}

func newFakeDrainRegistry(snaps map[int]contracts.CapacitySnapshot) *fakeDrainRegistry {
	r := &fakeDrainRegistry{
		MockRegistryDataPlane: &mocks.MockRegistryDataPlane{},
		snapshots:             snaps,
	}
	r.CapacitySnapshotFunc = func(priority int) (contracts.CapacitySnapshot, error) {
		snap, ok := r.snapshots[priority]
		if !ok {
			return contracts.CapacitySnapshot{}, contracts.ErrPriorityBandNotFound
		}
		return snap, nil
	}
	return r
}

func reqSnap(bandLen, bandCap, globalLen, globalCap uint64) contracts.CapacitySnapshot {
	return contracts.CapacitySnapshot{
		Band:   contracts.CapacityDimension{Len: bandLen, CapacityRequests: bandCap},
		Global: contracts.CapacityDimension{Len: globalLen, CapacityRequests: globalCap},
	}
}

// seedEstimator initializes band 0 and global to a contended interval through the public estimator
// sequence (syncBusy -> recordDispatch -> syncBusy).
func seedEstimator(interval time.Duration) (*drainEstimator, time.Time) {
	est := newDrainEstimator()
	t0 := time.Unix(1000, 0)
	busy := reqSnap(1, 1, 1, 10)
	reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: busy})
	est.syncBusy([]int{0}, reg, false, t0)
	now := t0.Add(interval)
	est.recordDispatch(0, now)
	est.syncBusy([]int{0}, reg, false, now)
	return est, now
}

func TestDrainEstimator_SteadyAndNoisySlowDrain(t *testing.T) {
	t.Parallel()

	t.Run("steady_5s_contended_drain", func(t *testing.T) {
		t.Parallel()
		est := newDrainEstimator()
		now := time.Unix(1000, 0)
		snap := reqSnap(2, 2, 2, 10)
		reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: snap})
		est.syncBusy([]int{0}, reg, false, now)

		for range 6 {
			now = now.Add(5 * time.Second)
			est.recordDispatch(0, now)
			est.syncBusy([]int{0}, reg, false, now)
		}

		assert.Equal(t, 5*time.Second, est.retryAfterHint(snap, 0, 100, now))
		assert.Equal(t, 10*time.Second, est.retryAfterHint(reqSnap(3, 2, 3, 10), 0, 100, now))
	})

	t.Run("noisy_multi_second_slow_drain", func(t *testing.T) {
		t.Parallel()
		est := newDrainEstimator()
		now := time.Unix(1000, 0)
		snap := reqSnap(2, 2, 2, 10)
		reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: snap})
		est.syncBusy([]int{0}, reg, false, now)

		for _, d := range []time.Duration{3 * time.Second, 7 * time.Second, 4 * time.Second, 6 * time.Second, 5 * time.Second} {
			now = now.Add(d)
			est.recordDispatch(0, now)
			est.syncBusy([]int{0}, reg, false, now)
		}

		hint := est.retryAfterHint(snap, 0, 100, now)
		assert.GreaterOrEqual(t, hint, 4*time.Second)
		assert.LessOrEqual(t, hint, 6*time.Second)
	})
}

func TestDrainEstimator_IdleExpiry(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		idleGap  time.Duration
		wantHint time.Duration
	}{
		{name: "idle_30s_preserves_estimate", idleGap: 30 * time.Second, wantHint: 4 * time.Second},
		{name: "idle_60s_expires_stale_estimate", idleGap: drainIdleExpiry, wantHint: 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			est, t1 := seedEstimator(4 * time.Second)
			busy := reqSnap(1, 1, 1, 10)
			reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: reqSnap(0, 1, 0, 10)})
			est.syncBusy([]int{0}, reg, false, t1)

			t2 := t1.Add(tc.idleGap)
			// Verify both read-side expiry (before syncBusy) and state after re-entering busy.
			assert.Equal(t, tc.wantHint, est.retryAfterHint(busy, 0, 100, t2))
			reg.snapshots[0] = busy
			est.syncBusy([]int{0}, reg, false, t2)
			assert.Equal(t, tc.wantHint, est.retryAfterHint(busy, 0, 100, t2))
		})
	}
}

func TestDrainEstimator_PassThroughIsolation(t *testing.T) {
	t.Parallel()

	t.Run("cold_pool_pass_through_burst", func(t *testing.T) {
		t.Parallel()
		// Space pass-throughs by 2s (> 0.5s threshold) so erroneous recording would emit a 2s hint.
		est := newDrainEstimator()
		now := time.Unix(1000, 0)
		reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: reqSnap(0, 4, 0, 4)})
		for range 64 {
			now = now.Add(2 * time.Second)
			est.recordDispatch(0, now)
			est.syncBusy([]int{0}, reg, false, now)
		}
		assert.Zero(t, est.retryAfterHint(reqSnap(4, 4, 4, 4), 0, 100, now))

		// Seed a 5s estimate, drain idle, and run 64 1ms pass-throughs: estimate stays 5s.
		seeded, t1 := seedEstimator(5 * time.Second)
		idleReg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: reqSnap(0, 4, 0, 4)})
		seeded.syncBusy([]int{0}, idleReg, false, t1)
		for range 64 {
			t1 = t1.Add(time.Millisecond)
			seeded.recordDispatch(0, t1)
			seeded.syncBusy([]int{0}, idleReg, false, t1)
		}
		assert.Equal(t, 5*time.Second, seeded.retryAfterHint(reqSnap(4, 4, 4, 4), 0, 100, t1))
	})

	t.Run("mixed_band_pass_throughs_preserve_gated_band_and_global", func(t *testing.T) {
		t.Parallel()
		const highPrio, lowPrio = 10, 0
		est := newDrainEstimator()
		now := time.Unix(1000, 0)
		lowSnap := reqSnap(2, 2, 2, 2)
		highSnap := reqSnap(0, 2, 2, 2)
		reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{highPrio: highSnap, lowPrio: lowSnap})
		est.syncBusy([]int{highPrio, lowPrio}, reg, false, now)

		for range 20 {
			now = now.Add(200 * time.Millisecond)
			est.recordDispatch(highPrio, now)
			est.syncBusy([]int{highPrio, lowPrio}, reg, false, now)
		}

		// Neither Low nor Global observed a contended dispatch; stall age on both reached 4s.
		assert.Equal(t, 4*time.Second, est.retryAfterHint(lowSnap, lowPrio, 100, now))
		assert.Equal(t, 4*time.Second, est.retryAfterHint(highSnap, highPrio, 100, now))
	})

	t.Run("intra_band_fairness_jump_counts_as_contended", func(t *testing.T) {
		t.Parallel()
		est := newDrainEstimator()
		t0 := time.Unix(1000, 0)
		busy := reqSnap(2, 2, 2, 10)
		reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: busy})
		est.syncBusy([]int{0}, reg, false, t0)

		t1 := t0.Add(3 * time.Second)
		est.recordDispatch(0, t1)
		est.syncBusy([]int{0}, reg, false, t1)

		// 1s after the dispatch, recorded 3s estimates hold at 3s on both band and global;
		// an unrecorded dispatch would have stalled to 4s.
		assert.Equal(t, 3*time.Second, est.retryAfterHint(busy, 0, 100, t1.Add(time.Second)))
		assert.Equal(t, 3*time.Second, est.retryAfterHint(reqSnap(1, 10, 2, 2), 0, 100, t1.Add(time.Second)))
	})

	t.Run("idle_pass_through_clears_paused_stall_on_band_and_global", func(t *testing.T) {
		t.Parallel()
		est := newDrainEstimator()
		t0 := time.Unix(1000, 0)
		busy := reqSnap(1, 1, 1, 1)
		idle := reqSnap(0, 1, 0, 1)
		reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: busy})

		// Stall for 5s, then pause via TTL sweep so both band and global retain 5s accumulatedBusy.
		est.syncBusy([]int{0}, reg, false, t0)
		t1 := t0.Add(5 * time.Second)
		reg.snapshots[0] = idle
		est.syncBusy([]int{0}, reg, false, t1)

		// Recovery pass-through while both band and global are idle clears paused stall on both.
		t2 := t1.Add(time.Second)
		est.recordDispatch(0, t2)
		est.syncBusy([]int{0}, reg, false, t2)

		assert.Zero(t, est.retryAfterHint(reqSnap(1, 1, 1, 10), 0, 100, t2))
		assert.Zero(t, est.retryAfterHint(reqSnap(0, 10, 1, 1), 0, 100, t2))
	})
}

func TestDrainEstimator_StallRampToCap(t *testing.T) {
	t.Parallel()

	est := newDrainEstimator()
	t0 := time.Unix(1000, 0)
	busy := reqSnap(1, 1, 1, 10)
	est.syncBusy([]int{0}, newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: busy}), false, t0)

	for _, tc := range []struct {
		elapsed time.Duration
		want    time.Duration
	}{
		{499 * time.Millisecond, 0},
		{500 * time.Millisecond, 1 * time.Second},
		{1 * time.Second, 1 * time.Second},
		{1001 * time.Millisecond, 2 * time.Second},
		{15 * time.Second, 15 * time.Second},
		{30 * time.Second, 30 * time.Second},
		{45 * time.Second, 30 * time.Second},
	} {
		assert.Equal(t, tc.want, est.retryAfterHint(busy, 0, 100, t0.Add(tc.elapsed)), "elapsed=%s", tc.elapsed)
	}
}

func TestDrainEstimator_MultiDispatchStallFloor(t *testing.T) {
	t.Parallel()

	t.Run("stall_floor_applies_to_first_dispatch_only_when_mean_is_known", func(t *testing.T) {
		t.Parallel()
		// 0.05s mean + 2.0s in-progress stall with 10 required dispatches:
		// 2.0 + 9*0.05 = 2.45s -> 3s (rather than 10*2.0 = 20s).
		est, now := seedEstimator(50 * time.Millisecond)
		assert.Equal(t, 3*time.Second, est.retryAfterHint(reqSnap(10, 1, 10, 100), 0, 100, now.Add(2*time.Second)))

		// 0.25s mean + 1.2s in-progress stall with 25 required dispatches:
		// 1.2 + 24*0.25 = 7.2s -> 8s (rather than 25*1.2 = 30s).
		est2, now2 := seedEstimator(250 * time.Millisecond)
		assert.Equal(t, 8*time.Second, est2.retryAfterHint(reqSnap(25, 1, 25, 100), 0, 100, now2.Add(1200*time.Millisecond)))
	})

	t.Run("no_estimate_scales_stall_across_all_required_dispatches", func(t *testing.T) {
		t.Parallel()
		est := newDrainEstimator()
		t0 := time.Unix(1000, 0)
		busy := reqSnap(3, 1, 3, 10)
		est.syncBusy([]int{0}, newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: busy}), false, t0)

		// Without a dispatch estimate, 3 required dispatches * 2s stall = 6s.
		assert.Equal(t, 6*time.Second, est.retryAfterHint(busy, 0, 100, t0.Add(2*time.Second)))
	})
}

func TestDrainEstimator_StallPauseAndResume(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		pauseSnap contracts.CapacitySnapshot
		poolEmpty bool
	}{
		{name: "ttl_cleanup_sweep_pauses_busy_clock", pauseSnap: reqSnap(0, 1, 0, 10), poolEmpty: false},
		{name: "scale_from_zero_pool_empty_pauses_busy_clock", pauseSnap: reqSnap(1, 1, 1, 10), poolEmpty: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			est := newDrainEstimator()
			t0 := time.Unix(1000, 0)
			busy := reqSnap(1, 1, 1, 10)
			reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: busy})

			// Stall 4s, pause for 20s (< 60s expiry), resume for 1s -> total busy stall = 5s.
			est.syncBusy([]int{0}, reg, false, t0)
			t1 := t0.Add(4 * time.Second)
			reg.snapshots[0] = tc.pauseSnap
			est.syncBusy([]int{0}, reg, tc.poolEmpty, t1)

			t2 := t1.Add(20 * time.Second)
			reg.snapshots[0] = busy
			est.syncBusy([]int{0}, reg, false, t2)
			assert.Equal(t, 5*time.Second, est.retryAfterHint(busy, 0, 100, t2.Add(time.Second)))
		})
	}
}

func TestDrainEstimator_SampleClampAndFastRecovery(t *testing.T) {
	t.Parallel()

	est := newDrainEstimator()
	t0 := time.Unix(1000, 0)
	busy := reqSnap(1, 1, 1, 10)
	reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: busy})

	// 120s stall dispatch must clamp sample to 30s.
	est.syncBusy([]int{0}, reg, false, t0)
	now := t0.Add(120 * time.Second)
	est.recordDispatch(0, now)
	est.syncBusy([]int{0}, reg, false, now)
	assert.Equal(t, 30*time.Second, est.retryAfterHint(busy, 0, 100, now))

	// At alpha=0.3 and 50ms spacing (20 req/s), 1st fast dispatch drops 30s -> ceil(21.015s) = 22s
	// (an unclamped 120s sample would still emit 30s), and dispatch 12 drops below 0.5s.
	const fastInterval = 50 * time.Millisecond
	now = now.Add(fastInterval)
	est.recordDispatch(0, now)
	est.syncBusy([]int{0}, reg, false, now)
	assert.Equal(t, 22*time.Second, est.retryAfterHint(busy, 0, 100, now))

	for i := 2; i <= 11; i++ {
		now = now.Add(fastInterval)
		est.recordDispatch(0, now)
		est.syncBusy([]int{0}, reg, false, now)
		assert.Positive(t, est.retryAfterHint(busy, 0, 100, now), "dispatch %d should still emit hint", i)
	}

	now = now.Add(fastInterval)
	est.recordDispatch(0, now)
	est.syncBusy([]int{0}, reg, false, now)
	assert.Zero(t, est.retryAfterHint(busy, 0, 100, now))
}

func TestDrainEstimator_BurstyUnreservedScrapeDrain(t *testing.T) {
	t.Parallel()

	est, now := seedEstimator(5 * time.Second)
	busy := reqSnap(2, 2, 2, 10)
	reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: busy})

	for range 10 {
		now = now.Add(time.Millisecond)
		est.recordDispatch(0, now)
		est.syncBusy([]int{0}, reg, false, now)
	}

	// Known trade-off: rapid unreserved burst depresses EWMA below threshold until stallAge takes over.
	assert.Zero(t, est.retryAfterHint(busy, 0, 100, now))
	assert.Equal(t, 1*time.Second, est.retryAfterHint(busy, 0, 100, now.Add(500*time.Millisecond)))
	assert.Equal(t, 5*time.Second, est.retryAfterHint(busy, 0, 100, now.Add(5*time.Second)))
}

func TestDrainEstimator_Projection(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		snap      contracts.CapacitySnapshot
		priority  int
		itemBytes uint64
		want      time.Duration
	}{
		{
			name:      "byte_scaling_fractional_dispatches",
			snap:      contracts.CapacitySnapshot{Band: contracts.CapacityDimension{Len: 10, ByteSize: 1000, CapacityBytes: 1000}},
			itemBytes: 350,
			want:      7 * time.Second,
		},
		{
			name:      "byte_scaling_floors_at_one_dispatch",
			snap:      contracts.CapacitySnapshot{Band: contracts.CapacityDimension{Len: 10, ByteSize: 1000, CapacityBytes: 1000}},
			itemBytes: 10,
			want:      2 * time.Second,
		},
		{
			name:      "byte_scaling_caps_at_len",
			snap:      contracts.CapacitySnapshot{Band: contracts.CapacityDimension{Len: 2, ByteSize: 1500, CapacityBytes: 1000}},
			itemBytes: 1000,
			want:      4 * time.Second,
		},
		{
			name:      "zero_len_tear_falls_back_to_one_dispatch",
			snap:      contracts.CapacitySnapshot{Band: contracts.CapacityDimension{Len: 0, ByteSize: 1000, CapacityBytes: 1000}},
			itemBytes: 200,
			want:      2 * time.Second,
		},
		{
			name: "max_uint64_bounds_avoid_overflow",
			snap: contracts.CapacitySnapshot{Band: contracts.CapacityDimension{
				Len:              math.MaxUint64,
				ByteSize:         math.MaxUint64,
				CapacityBytes:    math.MaxUint64,
				CapacityRequests: 1,
			}},
			itemBytes: 1024,
			want:      retryAfterHintCap,
		},
		{
			name:      "zero_byte_size_does_not_trip",
			snap:      contracts.CapacitySnapshot{Band: contracts.CapacityDimension{Len: 5, ByteSize: 0, CapacityBytes: 1000}},
			itemBytes: 500,
			want:      0,
		},
		{
			name: "oversized_for_unseen_band_omits_hint",
			snap: contracts.CapacitySnapshot{
				Band:   contracts.CapacityDimension{Len: 0, ByteSize: 0, CapacityBytes: 500},
				Global: contracts.CapacityDimension{Len: 5, ByteSize: 900, CapacityBytes: 1000},
			},
			priority:  99,
			itemBytes: 600,
			want:      0,
		},
		{
			name: "oversized_for_global_omits_hint",
			snap: contracts.CapacitySnapshot{
				Band:   contracts.CapacityDimension{Len: 2, CapacityRequests: 2, CapacityBytes: 2000},
				Global: contracts.CapacityDimension{Len: 2, ByteSize: 500, CapacityBytes: 500},
			},
			itemBytes: 600,
			want:      0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			est, now := seedEstimator(2 * time.Second)
			assert.Equal(t, tc.want, est.retryAfterHint(tc.snap, tc.priority, tc.itemBytes, now))
		})
	}
}

func TestDrainEstimator_MaxOfGlobalAndBand(t *testing.T) {
	t.Parallel()

	t.Run("band_slower_than_global", func(t *testing.T) {
		t.Parallel()
		est := newDrainEstimator()
		now := time.Unix(1000, 0)
		snap0 := reqSnap(2, 2, 4, 4)
		snap1 := reqSnap(2, 2, 4, 4)
		reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{0: snap0, 1: snap1})
		est.syncBusy([]int{1, 0}, reg, false, now)

		// Band 1 dispatches every 2s (global = 2s) while band 0 stalls to 8s.
		for range 4 {
			now = now.Add(2 * time.Second)
			est.recordDispatch(1, now)
			est.syncBusy([]int{1, 0}, reg, false, now)
		}
		assert.Equal(t, 8*time.Second, est.retryAfterHint(snap0, 0, 100, now))
	})

	t.Run("global_slower_than_band", func(t *testing.T) {
		t.Parallel()
		est, t1 := seedEstimator(2 * time.Second)
		// Band 0 goes idle remembering 2s while band 1 stalls global for 9s.
		reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{
			0: reqSnap(0, 2, 2, 4),
			1: reqSnap(2, 2, 2, 4),
		})
		est.syncBusy([]int{1, 0}, reg, false, t1)
		assert.Equal(t, 9*time.Second, est.retryAfterHint(reqSnap(2, 2, 4, 4), 0, 100, t1.Add(9*time.Second)))
	})
}

func TestDrainEstimator_PrunesRemovedBands(t *testing.T) {
	t.Parallel()

	est := newDrainEstimator()
	t0 := time.Unix(1000, 0)
	band10 := reqSnap(1, 1, 2, 10)
	band0 := reqSnap(1, 1, 2, 10)
	reg := newFakeDrainRegistry(map[int]contracts.CapacitySnapshot{10: band10, 0: band0})

	// Seed a 5s estimate on band 10, then drop band 10 from priorities.
	est.syncBusy([]int{10, 0}, reg, false, t0)
	t1 := t0.Add(5 * time.Second)
	est.recordDispatch(10, t1)
	est.syncBusy([]int{10, 0}, reg, false, t1)
	assert.Equal(t, 5*time.Second, est.retryAfterHint(band10, 10, 100, t1))

	delete(reg.snapshots, 10)
	est.syncBusy([]int{0}, reg, false, t1.Add(time.Second))
	assert.Zero(t, est.retryAfterHint(band10, 10, 100, t1.Add(time.Second)))
}
