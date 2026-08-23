/*
Copyright 2025 The Kubernetes Authors.

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
	"sync"
	"sync/atomic"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
)

var benchmarkFlowKey = flowcontrol.FlowKey{ID: "benchmark-flow"}

// BenchmarkQueues runs a series of benchmarks against the priority queue implementation.
func BenchmarkQueues(b *testing.B) {
	b.Run(PriorityQueueName, func(b *testing.B) {
		q := New(enqueueTimePolicy)

		b.Run("AddRemove", func(b *testing.B) {
			benchmarkAddRemove(b, q)
		})

		b.Run("AddPeekRemove", func(b *testing.B) {
			benchmarkAddPeekRemove(b, q)
		})

		b.Run("BulkAddThenBulkRemove", func(b *testing.B) {
			benchmarkBulkAddThenBulkRemove(b, q)
		})

		b.Run("HighContention", func(b *testing.B) {
			benchmarkHighContention(b, q)
		})
	})
}

// BenchmarkQueuesScoring mirrors BenchmarkQueues under a ScoringOrderingPolicy, so the two are
// benchstat-comparable at identical sizes and structure. Every sub-benchmark bumps the policy's generation
// before each Peek, forcing refreshIfStale's full rebuild every time rather than the throttled
// once-per-interval rebuild a real policy produces -- the worst case, not the common one.
func BenchmarkQueuesScoring(b *testing.B) {
	b.Run(PriorityQueueName, func(b *testing.B) {
		policy := mocks.NewMockScoringOrderingPolicy("bench", func(item flowcontrol.QueueItemAccessor) float64 {
			return float64(item.OriginalRequest().ByteSize())
		})
		q := New(policy)

		b.Run("AddRemove", func(b *testing.B) {
			benchmarkAddRemove(b, q)
		})

		b.Run("AddPeekRemove", func(b *testing.B) {
			benchmarkAddPeekRemoveScoring(b, q, policy)
		})

		b.Run("BulkAddThenBulkRemove", func(b *testing.B) {
			for _, n := range bumpEveryNSweep {
				b.Run(fmt.Sprintf("bumpEveryN=%d", n), func(b *testing.B) {
					benchmarkBulkAddThenBulkRemoveScoring(b, q, policy, n)
				})
			}
		})

		b.Run("HighContention", func(b *testing.B) {
			for _, n := range bumpEveryNSweep {
				b.Run(fmt.Sprintf("bumpEveryN=%d", n), func(b *testing.B) {
					benchmarkHighContentionScoring(b, q, policy, n)
				})
			}
		})
	})
}

// bumpEveryNSweep is the set of throttle ratios (bump once every N operations) used to bracket rebuild cost
// between the worst case (N=1, every operation rebuilds) and a throttled approximation of real usage.
// b.Loop()'s iteration count is calibrated by the testing framework at runtime, not chosen up front, so a
// time-based throttle (bump once every duration D) is not reproducible across runs: the framework would
// recalibrate iteration count against the now-slower loop, changing how many bumps land within D each run.
// An iteration-count ratio has no such feedback loop and stays deterministic and benchstat-comparable.
var bumpEveryNSweep = []int{1, 10, 100, 1000}

// benchmarkAddRemove measures the throughput of tightly coupled Add and Remove operations in parallel. This is a good
// measure of the base overhead of the queue's data structure and locking mechanism.
func benchmarkAddRemove(b *testing.B, q contracts.SafeQueue) {
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			item := mocks.NewMockQueueItemAccessor(1, "item", benchmarkFlowKey)
			q.Add(item)
			_, err := q.Remove(item.Handle())
			if err != nil {
				b.Fatalf("Remove failed: %v", err)
			}
		}
	})
}

// benchmarkAddPeekRemove measures the throughput of a serial Add, Peek, and Remove sequence. This simulates a
// common consumer pattern where a single worker peeks at an item before deciding to process and remove it.
func benchmarkAddPeekRemove(b *testing.B, q contracts.SafeQueue) {
	// Pre-add one item so Peek doesn't fail on the first iteration.
	initialItem := mocks.NewMockQueueItemAccessor(1, "initial", benchmarkFlowKey)
	q.Add(initialItem)

	b.ReportAllocs()

	for b.Loop() {
		item := mocks.NewMockQueueItemAccessor(1, "item", benchmarkFlowKey)
		q.Add(item)
		peeked := q.Peek()
		if peeked == nil {
			// In a concurrent benchmark, this could happen if the queue becomes empty.
			// In a serial one, it's a fatal error.
			b.Fatal("Peek failed")
		}

		_, err := q.Remove(peeked.Handle())
		if err != nil {
			b.Fatalf("Remove failed: %v", err)
		}
	}
}

// benchmarkBulkAddThenBulkRemove measures performance of filling the queue up with a batch of items and then draining
// it. This can reveal performance characteristics related to how the data structure grows and shrinks.
func benchmarkBulkAddThenBulkRemove(b *testing.B, q contracts.SafeQueue) {
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		// Add a batch of items
		items := make([]flowcontrol.QueueItemAccessor, 100)
		for j := range items {
			item := mocks.NewMockQueueItemAccessor(1, fmt.Sprintf("bulk-%d-%d", i, j), benchmarkFlowKey)
			items[j] = item
			q.Add(item)
		}

		// Remove the same number of items
		for range items {
			peeked := q.Peek()
			if peeked == nil {
				b.Fatal("Peek failed")
			}
			if _, err := q.Remove(peeked.Handle()); err != nil {
				b.Fatalf("Remove failed: %v", err)
			}
		}
	}
}

// benchmarkHighContention simulates a more realistic workload with multiple producers and consumers operating on the
// queue concurrently.
func benchmarkHighContention(b *testing.B, q contracts.SafeQueue) {
	// Pre-fill the queue to ensure consumers have work to do immediately.
	for i := range 1000 {
		item := mocks.NewMockQueueItemAccessor(1, fmt.Sprintf("prefill-%d", i), benchmarkFlowKey)
		q.Add(item)
	}

	stopCh := make(chan struct{})
	var wgProducers sync.WaitGroup

	// Start producer goroutines to run in the background.
	for range 4 {
		wgProducers.Go(func() {
			for {
				select {
				case <-stopCh:
					return
				default:
					item := mocks.NewMockQueueItemAccessor(1, "item", benchmarkFlowKey)
					q.Add(item)
				}
			}
		})
	}

	b.ReportAllocs()
	b.ResetTimer()

	// Consumers drive the benchmark.
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			peeked := q.Peek()
			if peeked != nil {
				_, _ = q.Remove(peeked.Handle())
			}
		}
	})

	b.StopTimer()
	close(stopCh) // Signal producers to stop.
	wgProducers.Wait()
}

// benchmarkAddPeekRemoveScoring mirrors benchmarkAddPeekRemove, bumping the policy before every Peek so each
// iteration forces refreshIfStale's rebuild rather than hitting the cached fast path after the first Peek.
func benchmarkAddPeekRemoveScoring(b *testing.B, q contracts.SafeQueue, policy *mocks.MockScoringOrderingPolicy) {
	initialItem := mocks.NewMockQueueItemAccessor(1, "initial", benchmarkFlowKey)
	q.Add(initialItem)

	b.ReportAllocs()

	for b.Loop() {
		item := mocks.NewMockQueueItemAccessor(1, "item", benchmarkFlowKey)
		q.Add(item)

		policy.Bump()
		peeked := q.Peek()
		if peeked == nil {
			b.Fatal("Peek failed")
		}

		_, err := q.Remove(peeked.Handle())
		if err != nil {
			b.Fatalf("Remove failed: %v", err)
		}
	}
}

// benchmarkBulkAddThenBulkRemoveScoring mirrors benchmarkBulkAddThenBulkRemove, bumping the policy every N
// Peeks in the drain loop so refreshIfStale's rebuild fires at a throttle ratio between the worst case (N=1,
// every Peek against the shrinking heap) and an amortized approximation of a real throttled policy.
func benchmarkBulkAddThenBulkRemoveScoring(b *testing.B, q contracts.SafeQueue, policy *mocks.MockScoringOrderingPolicy, bumpEveryN int) {
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		items := make([]flowcontrol.QueueItemAccessor, 100)
		for j := range items {
			item := mocks.NewMockQueueItemAccessor(1, fmt.Sprintf("bulk-%d-%d", i, j), benchmarkFlowKey)
			items[j] = item
			q.Add(item)
		}

		for j := range items {
			if j%bumpEveryN == 0 {
				policy.Bump()
			}
			peeked := q.Peek()
			if peeked == nil {
				b.Fatal("Peek failed")
			}
			if _, err := q.Remove(peeked.Handle()); err != nil {
				b.Fatalf("Remove failed: %v", err)
			}
		}
	}
}

// benchmarkHighContentionScoring mirrors benchmarkHighContention exactly, bumping the policy every N consumer
// Peeks so refreshIfStale's rebuild fires at a throttle ratio between the worst case (N=1, every Peek contends
// for the write lock instead of only the read lock the cached fast path would take) and an amortized
// approximation of a real throttled policy. The consumer count is tracked with an atomic counter rather than
// a per-goroutine index: RunParallel shares this callback across goroutines, so a plain counter would race.
func benchmarkHighContentionScoring(b *testing.B, q contracts.SafeQueue, policy *mocks.MockScoringOrderingPolicy, bumpEveryN int) {
	for i := range 1000 {
		item := mocks.NewMockQueueItemAccessor(1, fmt.Sprintf("prefill-%d", i), benchmarkFlowKey)
		q.Add(item)
	}

	stopCh := make(chan struct{})
	var wgProducers sync.WaitGroup

	// Start producer goroutines to run in the background.
	for range 4 {
		wgProducers.Go(func() {
			for {
				select {
				case <-stopCh:
					return
				default:
					item := mocks.NewMockQueueItemAccessor(1, "item", benchmarkFlowKey)
					q.Add(item)
				}
			}
		})
	}

	var peekCount atomic.Int64

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if peekCount.Add(1)%int64(bumpEveryN) == 0 {
				policy.Bump()
			}
			peeked := q.Peek()
			if peeked != nil {
				_, _ = q.Remove(peeked.Handle())
			}
		}
	})

	b.StopTimer()
	close(stopCh)
	wgProducers.Wait()
}
