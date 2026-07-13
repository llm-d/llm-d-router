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
	"sync"
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
	s.mu.Lock()
	s.now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return fakeNow
	}
	s.mu.Unlock()

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
