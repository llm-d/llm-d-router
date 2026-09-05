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

package queue

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
)

// genScorer builds a deterministic ScoreFunc keyed on the item's immutable ID. score reports each ID's
// base value at even generations and gen(base) at odd ones, so a test flips the whole order by a single
// Bump without mutating any shared state: the func reads only the ID (fixed at Add) and the policy's
// atomic generation, so it is lock-free even while a rebuild reads it from another goroutine.
//
// The base map is written once, before the queue is populated, and only read afterwards. Tests that need
// no reordering pass a nil gen and leave the map fixed; the two flip tests pass gen to invert it.
type genScorer struct {
	policy *mocks.MockScoringOrderingPolicy
	base   map[string]float64
	gen    func(base float64) float64
}

func newGenScorer() *genScorer {
	return &genScorer{base: make(map[string]float64)}
}

func (g *genScorer) score(item flowcontrol.QueueItemAccessor) float64 {
	b := g.base[item.OriginalRequest().ID()]
	if g.gen != nil && g.policy.Generation()%2 == 1 {
		return g.gen(b)
	}
	return b
}

// scoredItem records an item's base score and adds it, mirroring the real ordering: a request's score is
// known to the policy before the queue books it at Add time.
func scoredItem(
	t *testing.T,
	q *priorityQueue,
	scorer *genScorer,
	id string,
	score float64,
	enqueue time.Time,
) *mocks.MockQueueItemAccessor {
	t.Helper()
	scorer.base[id] = score
	item := itemAt(10, id, enqueue)
	q.Add(item)
	return item
}

// drainIDs empties the queue through Peek followed by Remove, returning the dispatch order. This exercises
// the same pair of calls the dispatch path makes, so a cached-score regression that only manifests during
// Remove's re-sift is caught here rather than by Peek alone.
func drainIDs(t *testing.T, q *priorityQueue) []string {
	t.Helper()
	var ids []string
	for q.Len() > 0 {
		head := q.Peek()
		require.NotNil(t, head, "Peek must not return nil while Len is positive")
		ids = append(ids, head.OriginalRequest().ID())
		_, err := q.Remove(head.Handle())
		require.NoError(t, err)
		assertHeapProperty(t, q, "after removing head")
	}
	return ids
}

// TestScoringQueue_ModeDetection verifies the constructor's optional-interface upgrade: a scoring policy puts
// the queue in scoring mode and a static one does not, with exactly one policy field populated either way.
func TestScoringQueue_ModeDetection(t *testing.T) {
	t.Parallel()

	scoring := newPriorityQueue(mocks.NewMockScoringOrderingPolicy("detect", newGenScorer().score))
	assert.NotNil(t, scoring.heap.scoringPolicy, "a scoring policy must put the queue in scoring mode")
	assert.Nil(t, scoring.heap.policy, "scoring mode must not also populate the static policy field")

	static := newPriorityQueue(enqueueTimePolicy)
	assert.Nil(t, static.heap.scoringPolicy, "a static policy must not put the queue in scoring mode")
	assert.NotNil(t, static.heap.policy, "static mode must populate the static policy field")
}

// TestScoringQueue_ReordersOnGenerationBump verifies the core contract: the queue orders by cached score
// (highest first), ignores score changes until the generation moves, and re-sorts when it does.
func TestScoringQueue_ReordersOnGenerationBump(t *testing.T) {
	t.Parallel()
	scorer := newGenScorer()
	scorer.gen = func(base float64) float64 { return 4 - base } // odd generations invert 1,2,3 -> 3,2,1
	policy := mocks.NewMockScoringOrderingPolicy("reorder", scorer.score)
	scorer.policy = policy
	q := newPriorityQueue(policy)
	now := time.Now()

	scoredItem(t, q, scorer, "a", 1, now)
	scoredItem(t, q, scorer, "b", 2, now.Add(time.Millisecond))
	scoredItem(t, q, scorer, "c", 3, now.Add(2*time.Millisecond))

	require.Equal(t, "c", q.Peek().OriginalRequest().ID(), "highest score must be at the head")

	// The scorer would report the inverted order at an odd generation, but until a bump the queue must
	// keep serving the cached even-generation order.
	assert.Equal(t, "c", q.Peek().OriginalRequest().ID(), "must not re-sort without a generation bump")

	policy.Bump()
	assert.Equal(t, "a", q.Peek().OriginalRequest().ID(), "must re-sort on a generation bump")

	assert.Equal(t, []string{"a", "b", "c"}, drainIDs(t, q), "drain order must follow the refreshed scores")
}

// TestScoringQueue_NoRefreshWithoutGenerationBump asserts the throttle contract by counting Score calls: the
// queue must call Score exactly once per item at Add, never on a Peek whose generation is unchanged, and
// exactly once per item on the first Peek after a bump no matter how many Peeks follow.
//
// The counts are exact rather than bounded on purpose. An implementation that recomputed without storing the
// new generation, or stored it before recomputing, would still pass a looser assertion while doubling the
// rebuild cost of every dispatch cycle under a fairness policy that peeks each queue twice.
func TestScoringQueue_NoRefreshWithoutGenerationBump(t *testing.T) {
	t.Parallel()
	scorer := newGenScorer()
	policy := mocks.NewMockScoringOrderingPolicy("throttle", scorer.score)
	scorer.policy = policy
	q := newPriorityQueue(policy)
	now := time.Now()

	const itemCount = 3
	for i := range itemCount {
		scoredItem(t, q, scorer, fmt.Sprintf("item-%d", i), 1, now.Add(time.Duration(i)*time.Millisecond))
	}
	require.EqualValues(t, itemCount, policy.ScoreCalls(), "Add must score each item exactly once")

	for range 10 {
		q.Peek()
	}
	assert.EqualValues(t, itemCount, policy.ScoreCalls(),
		"Peek must not call Score while the generation is unchanged")

	policy.Bump()
	for range 10 {
		q.Peek()
	}
	assert.EqualValues(t, 2*itemCount, policy.ScoreCalls(),
		"one bump must cost exactly one Score per item, however many Peeks follow")
}

// TestScoringQueue_RefreshIsDeduplicated verifies the double-checked locking in refreshIfStale: many
// concurrent Peeks observing one stale generation must together trigger exactly one O(n) rebuild.
func TestScoringQueue_RefreshIsDeduplicated(t *testing.T) {
	t.Parallel()
	scorer := newGenScorer()
	policy := mocks.NewMockScoringOrderingPolicy("dedupe", scorer.score)
	scorer.policy = policy
	q := newPriorityQueue(policy)
	now := time.Now()

	const itemCount = 200
	for i := range itemCount {
		scoredItem(t, q, scorer, fmt.Sprintf("item-%d", i), float64(i),
			now.Add(time.Duration(i)*time.Microsecond))
	}
	require.EqualValues(t, itemCount, policy.ScoreCalls())

	policy.Bump()

	const peekers = 32
	var wg sync.WaitGroup
	start := make(chan struct{})
	for range peekers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			q.Peek()
		}()
	}
	close(start)
	wg.Wait()

	assert.EqualValues(t, 2*itemCount, policy.ScoreCalls(),
		"concurrent Peeks after one bump must together trigger exactly one rebuild")
}

// TestScoringQueue_RefreshPreservesAccounting verifies that a rebuild changes ordering but not statistics.
// managedQueue derives the registry's aggregates from Len/ByteSize deltas measured around each mutation, and
// it does not wrap Peek, so a refresh that altered either count would desynchronize those aggregates with no
// local symptom.
func TestScoringQueue_RefreshPreservesAccounting(t *testing.T) {
	t.Parallel()
	scorer := newGenScorer()
	scorer.gen = func(base float64) float64 { return -base } // odd generations invert every score
	policy := mocks.NewMockScoringOrderingPolicy("accounting", scorer.score)
	scorer.policy = policy
	q := newPriorityQueue(policy)
	now := time.Now()

	const itemCount = 50
	var wantBytes uint64
	for i := range itemCount {
		id := fmt.Sprintf("item-%d", i)
		scorer.base[id] = float64(i)
		byteSize := uint64(10 + i)
		item := itemAt(byteSize, id, now.Add(time.Duration(i)*time.Microsecond))
		q.Add(item)
		wantBytes += byteSize
	}
	require.Equal(t, itemCount, q.Len())
	require.Equal(t, wantBytes, q.ByteSize())
	require.Equal(t, fmt.Sprintf("item-%d", itemCount-1), q.Peek().OriginalRequest().ID())

	// A bump inverts every score, so the rebuild must reorder the entire heap.
	policy.Bump()
	require.NotNil(t, q.Peek())

	assert.Equal(t, itemCount, q.Len(), "a rebuild must not change Len")
	assert.Equal(t, wantBytes, q.ByteSize(), "a rebuild must not change ByteSize")
	assert.Equal(t, "item-0", q.Peek().OriginalRequest().ID(), "but it must change the order")
}

// TestScoringQueue_TieBreakMatchesCompareByScore verifies that the queue's cached comparator and the
// framework's live CompareByScore break ties identically. The fairness tier compares heads across queues
// through CompareByScore and has no access to any cache, so a divergence here would let a band-level pick
// disagree with the queue's own head for equal-scored items.
func TestScoringQueue_TieBreakMatchesCompareByScore(t *testing.T) {
	t.Parallel()
	scorer := newGenScorer()
	policy := mocks.NewMockScoringOrderingPolicy("tiebreak", scorer.score)
	scorer.policy = policy
	q := newPriorityQueue(policy)
	now := time.Now()

	// Added late-first so the tie-break, not insertion order, decides the head.
	late := scoredItem(t, q, scorer, "late", 5, now.Add(time.Second))
	early := scoredItem(t, q, scorer, "early", 5, now)

	require.Equal(t, "early", q.Peek().OriginalRequest().ID(),
		"equal scores must be broken by earlier EnqueueTime")
	assert.True(t, flowcontrol.CompareByScore(policy, early, late), "CompareByScore must agree")
	assert.False(t, flowcontrol.CompareByScore(policy, late, early), "CompareByScore must be antisymmetric")
}

// TestScoringQueue_NaNScoreIsLowestPriority verifies the NaN backstop. ScoringOrderingPolicy forbids
// returning NaN, but a NaN compares false against every value and would break the strict weak ordering the
// heap requires, so both the queue's cache and CompareByScore normalize it to negative infinity.
func TestScoringQueue_NaNScoreIsLowestPriority(t *testing.T) {
	t.Parallel()
	scorer := newGenScorer()
	policy := mocks.NewMockScoringOrderingPolicy("nan", scorer.score)
	scorer.policy = policy
	q := newPriorityQueue(policy)
	now := time.Now()

	nan := scoredItem(t, q, scorer, "nan", math.NaN(), now)
	negInf := scoredItem(t, q, scorer, "neg-inf", math.Inf(-1), now.Add(time.Millisecond))
	finite := scoredItem(t, q, scorer, "finite", -1e300, now.Add(2*time.Millisecond))

	assert.Equal(t, "finite", q.Peek().OriginalRequest().ID(),
		"a finite score must outrank both NaN and negative infinity")

	// NaN normalizes to negative infinity, so it ties with a genuine -Inf and the tie-break decides.
	assert.True(t, flowcontrol.CompareByScore(policy, nan, negInf),
		"a NaN score must tie with negative infinity and lose only on EnqueueTime")
	assert.True(t, flowcontrol.CompareByScore(policy, finite, nan), "a finite score must beat NaN")
	assert.Equal(t, []string{"finite", "nan", "neg-inf"}, drainIDs(t, q))
}

// TestScoringQueue_RandomizedInterleavings is the regression test for the bug this mode exists to fix. A
// comparator that read live policy state would violate the heap property as soon as a score changed under it;
// a comparator reading cached scores cannot, no matter how adds, removes, mutations and bumps interleave.
//
// The mutate-without-bump case is the important one: the cache stays internally consistent and therefore so
// does the heap, even though the policy now disagrees with both.
func TestScoringQueue_RandomizedInterleavings(t *testing.T) {
	t.Parallel()
	// This test runs on a single goroutine, so it mutates the scorer's base map directly (case 2) to model
	// a score changing without a generation bump -- the situation the cache must ignore until it rebuilds.
	scorer := newGenScorer()
	policy := mocks.NewMockScoringOrderingPolicy("fuzz", scorer.score)
	scorer.policy = policy
	q := newPriorityQueue(policy)
	// Fixed seed: a failure must be reproducible from the test output alone.
	rng := rand.New(rand.NewSource(0xC0FFEE)) //nolint:gosec // deterministic test input, not security-relevant
	now := time.Now()

	var live []*mocks.MockQueueItemAccessor
	nextID := 0

	for step := range 5000 {
		switch rng.Intn(4) {
		case 0: // Add an item with a random score.
			id := fmt.Sprintf("item-%d", nextID)
			nextID++
			item := scoredItem(t, q, scorer, id, rng.NormFloat64(),
				now.Add(time.Duration(rng.Intn(1000))*time.Microsecond))
			live = append(live, item)
		case 1: // Remove a random live item.
			if len(live) == 0 {
				break
			}
			victim := rng.Intn(len(live))
			_, err := q.Remove(live[victim].Handle())
			require.NoError(t, err, "step %d: removing a live handle must succeed", step)
			live = append(live[:victim], live[victim+1:]...)
		case 2: // Mutate a score without bumping: the queue must not observe it.
			if len(live) == 0 {
				break
			}
			scorer.base[live[rng.Intn(len(live))].OriginalRequest().ID()] = rng.NormFloat64()
		case 3: // Bump, then Peek to force the refresh.
			policy.Bump()
			q.Peek()
		}
		assertHeapProperty(t, q, "step %d", step)
	}

	// Draining must yield non-increasing cached scores.
	prev := math.Inf(1)
	for q.Len() > 0 {
		head := q.Peek()
		require.NotNil(t, head)
		hi, ok := head.Handle().(*heapItem)
		require.True(t, ok)
		require.LessOrEqual(t, hi.score, prev, "drain order must be non-increasing in cached score")
		prev = hi.score
		_, err := q.Remove(head.Handle())
		require.NoError(t, err)
	}
}

// TestScoringQueue_Concurrency stress-tests the write-lock discipline under the race detector. The rebuild
// must hold the write lock across the whole pass -- both the Score loop and heap.Init -- or a concurrent Add
// or Cleanup will observe a partially rebuilt heap.
func TestScoringQueue_Concurrency(t *testing.T) {
	t.Parallel()
	// Score reads the item's immutable byte size, so a concurrently added item carries its score from
	// construction and no goroutine writes a shared score store: the only concurrency under test is the
	// queue's own write-lock discipline, not the mock's.
	byteScore := func(item flowcontrol.QueueItemAccessor) float64 {
		return float64(item.OriginalRequest().ByteSize())
	}
	policy := mocks.NewMockScoringOrderingPolicy("concurrency", byteScore)
	q := newPriorityQueue(policy)
	now := time.Now()

	const seeded = 100
	for i := range seeded {
		q.Add(itemAt(uint64(i+1), fmt.Sprintf("seed-%d", i), now.Add(time.Duration(i)*time.Microsecond)))
	}

	const (
		goroutinesPerRole = 4
		opsPerGoroutine   = 100
	)
	var wg sync.WaitGroup
	start := make(chan struct{})

	spawn := func(work func(worker, op int)) {
		for worker := range goroutinesPerRole {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				for op := range opsPerGoroutine {
					work(worker, op)
				}
			}()
		}
	}

	spawn(func(_, _ int) { q.Peek() })
	spawn(func(_, _ int) { policy.Bump() })
	spawn(func(worker, op int) {
		id := fmt.Sprintf("added-%d-%d", worker, op)
		q.Add(itemAt(uint64(op+1), id, now.Add(time.Duration(op)*time.Microsecond)))
	})
	spawn(func(_, _ int) {
		// Remove roughly half the queue's contents, contending with the rebuild for the write lock.
		q.Cleanup(func(item flowcontrol.QueueItemAccessor) bool {
			return item.OriginalRequest().ByteSize()%20 == 0
		})
	})

	close(start)
	wg.Wait()

	assertHeapProperty(t, q, "after concurrent stress")
	assert.Equal(t, len(q.heap.items), q.Len(), "Len must match the heap's actual contents")
}
