/*
Copyright 2026 The Kubernetes Authors.

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

package stateapi

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestMemoryInflightStore_ReserveReleaseSnapshot(t *testing.T) {
	t.Parallel()
	s := newMemoryInflightStore()

	s.Reserve("req-1", "ep-1", 100)
	s.Reserve("req-2", "ep-1", 50)

	snap := s.SnapshotBatch([]string{"ep-1", "ep-unknown"})
	require.Equal(t, InflightSnapshot{Requests: 2, Tokens: 150}, snap["ep-1"])
	_, hasUnknown := snap["ep-unknown"]
	require.False(t, hasUnknown)

	s.Release("req-1", "ep-1", 100)
	snap = s.SnapshotBatch([]string{"ep-1"})
	require.Equal(t, InflightSnapshot{Requests: 1, Tokens: 50}, snap["ep-1"])
}

func TestMemoryInflightStore_ReserveIsIdempotent(t *testing.T) {
	t.Parallel()
	s := newMemoryInflightStore()

	s.Reserve("req-1", "ep-1", 100)
	s.Reserve("req-1", "ep-1", 100) // retried call, same requestID

	snap := s.SnapshotBatch([]string{"ep-1"})
	require.Equal(t, InflightSnapshot{Requests: 1, Tokens: 100}, snap["ep-1"])
}

func TestMemoryInflightStore_ReleaseIsIdempotent(t *testing.T) {
	t.Parallel()
	s := newMemoryInflightStore()

	s.Reserve("req-1", "ep-1", 100)
	s.Release("req-1", "ep-1", 100)
	s.Release("req-1", "ep-1", 100) // retried release must not double-decrement

	snap := s.SnapshotBatch([]string{"ep-1"})
	require.Equal(t, InflightSnapshot{}, snap["ep-1"])
}

func TestMemoryInflightStore_ReleaseWithoutReserveIsNoOp(t *testing.T) {
	t.Parallel()
	s := newMemoryInflightStore()

	s.Release("req-unknown", "ep-1", 100)

	snap := s.SnapshotBatch([]string{"ep-1"})
	_, ok := snap["ep-1"]
	require.False(t, ok)
}

func TestMemoryInflightStore_DeleteEndpoint(t *testing.T) {
	t.Parallel()
	s := newMemoryInflightStore()

	s.Reserve("req-1", "ep-1", 100)
	s.DeleteEndpoint("ep-1")

	snap := s.SnapshotBatch([]string{"ep-1"})
	_, ok := snap["ep-1"]
	require.False(t, ok)
}

func TestMemoryPrefixStore_CommitMatchRemove(t *testing.T) {
	t.Parallel()
	s := newMemoryPrefixStore()

	s.Commit("req-1", "ep-1", []uint64{1, 2})
	s.Commit("req-2", "ep-2", []uint64{2})

	require.ElementsMatch(t, []string{"ep-1"}, s.Match(1))
	require.ElementsMatch(t, []string{"ep-1", "ep-2"}, s.Match(2))

	s.RemoveEndpoint("ep-1")
	require.Empty(t, s.Match(1))
	require.ElementsMatch(t, []string{"ep-2"}, s.Match(2))
}

func TestMemoryConcurrencyStore_AdmitReleaseRespectsGlobalMax(t *testing.T) {
	t.Parallel()
	s := newMemoryConcurrencyStore(context.Background(), Config{GlobalMaxConcurrency: 2, LeaseTTL: time.Minute})
	key := FlowKey{ID: "tenant-a", Priority: 0}

	require.Equal(t, ConcurrencyOutcomeAdmitted, s.Admit("req-1", key))
	require.Equal(t, ConcurrencyOutcomeAdmitted, s.Admit("req-2", key))
	require.Equal(t, ConcurrencyOutcomeRejected, s.Admit("req-3", key))

	s.ReleaseConcurrency("req-1", key)
	require.Equal(t, ConcurrencyOutcomeAdmitted, s.Admit("req-3", key))
}

// TestMemoryConcurrencyStore_AdmitUnderConcurrencyRespectsGlobalMax guards the
// property store.go documents as "globalMaxConcurrency (remote authority,
// precise)": concurrent Admit calls for the same FlowKey but different
// requestIDs land on different leaseShards (shard is keyed by requestID), so
// the shard mutex alone cannot make the shared counter's check-then-increment
// atomic across requestIDs. tryAdmit's CAS loop is what actually closes this;
// -race cannot catch a regression here because every access to counter is
// already a properly synchronized atomic op — the bug is a lost admission
// decision (TOCTOU), not a data race.
func TestMemoryConcurrencyStore_AdmitUnderConcurrencyRespectsGlobalMax(t *testing.T) {
	t.Parallel()
	const globalMax = 8
	const racers = 64 // concurrent Admits contending for the single remaining slot
	const rounds = 20

	for round := 0; round < rounds; round++ {
		s := newMemoryConcurrencyStore(context.Background(), Config{GlobalMaxConcurrency: globalMax, LeaseTTL: time.Minute})
		key := FlowKey{ID: "tenant-a", Priority: 0}

		// Fill globalMax-1 slots sequentially (uncontended) so every racer
		// below observes the identical stale counter value (globalMax-1) when
		// it Loads, instead of needing to collide mid-ramp against a counter
		// that's also moving.
		for i := 0; i < globalMax-1; i++ {
			require.Equal(t, ConcurrencyOutcomeAdmitted, s.Admit(fmt.Sprintf("round-%d-seed-%d", round, i), key))
		}

		var admitted atomic.Int64
		var ready, wg sync.WaitGroup
		start := make(chan struct{})
		ready.Add(racers)
		wg.Add(racers)
		for i := 0; i < racers; i++ {
			go func(i int) {
				defer wg.Done()
				ready.Done()
				<-start // all racers fire Admit at once, all seeing the same stale counter value
				if s.Admit(fmt.Sprintf("round-%d-racer-%d", round, i), key) == ConcurrencyOutcomeAdmitted {
					admitted.Add(1)
				}
			}(i)
		}
		ready.Wait()
		close(start)
		wg.Wait()

		require.LessOrEqualf(t, admitted.Load(), int64(1),
			"round %d: only 1 of the %d racers contending for the single remaining slot should be admitted, got %d",
			round, racers, admitted.Load())
	}
}

func TestMemoryConcurrencyStore_AdmitIsIdempotent(t *testing.T) {
	t.Parallel()
	s := newMemoryConcurrencyStore(context.Background(), Config{GlobalMaxConcurrency: 1, LeaseTTL: time.Minute})
	key := FlowKey{ID: "tenant-a", Priority: 0}

	require.Equal(t, ConcurrencyOutcomeAdmitted, s.Admit("req-1", key))
	// A retried Admit for the same requestID must not consume a second slot.
	require.Equal(t, ConcurrencyOutcomeAdmitted, s.Admit("req-1", key))
	require.Equal(t, ConcurrencyOutcomeRejected, s.Admit("req-2", key))
}

func TestMemoryConcurrencyStore_ReleaseWithoutAdmitIsNoOp(t *testing.T) {
	t.Parallel()
	s := newMemoryConcurrencyStore(context.Background(), Config{GlobalMaxConcurrency: 1, LeaseTTL: time.Minute})
	key := FlowKey{ID: "tenant-a", Priority: 0}

	s.ReleaseConcurrency("req-unknown", key)

	require.Equal(t, ConcurrencyOutcomeAdmitted, s.Admit("req-1", key))
}

// TestMemoryConcurrencyStore_SweepReclaimsLeakedLeases is the load-bearing
// test for the spike's throughput numbers: an unreclaimed lease (e.g. from a
// crashed stateless replica that never calls Release) permanently shrinks
// effective fleet concurrency, which would show up as "stateless mode
// degrades over a long run" — a false negative unrelated to the architecture.
func TestMemoryConcurrencyStore_SweepReclaimsLeakedLeases(t *testing.T) {
	t.Parallel()
	var mu sync.Mutex
	fakeNow := time.Now()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s := newMemoryConcurrencyStore(ctx, Config{
		GlobalMaxConcurrency: 1,
		LeaseTTL:             10 * time.Millisecond,
		SweepInterval:        5 * time.Millisecond,
	})
	s.counterMu.Lock()
	s.now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return fakeNow
	}
	s.counterMu.Unlock()

	key := FlowKey{ID: "tenant-a", Priority: 0}
	require.Equal(t, ConcurrencyOutcomeAdmitted, s.Admit("leaked-req", key))
	require.Equal(t, ConcurrencyOutcomeRejected, s.Admit("req-2", key))

	// Advance the fake clock past the lease TTL and let the sweep goroutine run.
	mu.Lock()
	fakeNow = fakeNow.Add(time.Hour)
	mu.Unlock()

	require.Eventually(t, func() bool {
		return s.Admit("req-3", key) == ConcurrencyOutcomeAdmitted
	}, time.Second, 5*time.Millisecond, "leaked lease was never reclaimed by the sweep")
}
