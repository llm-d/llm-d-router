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

package contracts

import (
	"sync"
	"time"

	"k8s.io/utils/clock"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// standingWindow measures how long one scope has stayed backlogged.
type standingWindow struct {
	interval, ttl time.Duration

	mu          sync.Mutex
	since       time.Time
	lastBacklog time.Time
}

// observe records whether the scope showed backlog at now. Relief restarts the window only once
// the scope has shown none for a short budget: with a few dispatches per short budget, a queue held
// at that budget still dispatches an occasional short wait.
func (w *standingWindow) observe(backlogged bool, now time.Time) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if backlogged {
		w.lastBacklog = now
	} else if now.Sub(w.lastBacklog) >= w.ttl {
		w.since = now
	}
}

// cutoff returns the latest enqueue time that has exhausted the standing budget, or zero until a
// full observation window and short budget have elapsed with endpoints present.
func (w *standingWindow) cutoff(now, regimeSince time.Time) time.Time {
	w.mu.Lock()
	since := w.since
	w.mu.Unlock()
	if regimeSince.After(since) {
		since = regimeSince
	}
	cutoff := now.Add(-w.ttl)
	if since.IsZero() || since.Add(w.interval).After(cutoff) {
		return time.Time{}
	}
	return cutoff
}

// StandingBand tracks continuous backlog across the flow queues of one priority band.
type StandingBand struct {
	window  standingWindow
	clock   clock.PassiveClock
	backlog func() int64

	mu         sync.Mutex
	lastBucket int64
	dispatched [8]int64
}

// NewStandingBand creates backlog tracking for a band. backlog reports the requests queued in it.
func NewStandingBand(interval, ttl time.Duration, clk clock.PassiveClock, backlog func() int64) *StandingBand {
	return &StandingBand{window: standingWindow{interval: interval, ttl: ttl}, clock: clk, backlog: backlog}
}

// NewQueue creates the tracker for one of the band's flow queues.
func (b *StandingBand) NewQueue() *StandingQueue {
	q := NewStandingQueue(b.window.interval, b.window.ttl, b.clock)
	q.band = b
	return q
}

// recordDispatch counts a dispatch. The band shows backlog while it holds at least as many requests
// as it dispatched during the last half short budget. The order of service within a band depends on
// its fairness policy, so one request's wait says little about the others.
func (b *StandingBand) recordDispatch(now time.Time) {
	width := max(int64(b.window.ttl/2)/int64(len(b.dispatched)), 1)
	bucket := now.UnixNano() / width
	b.mu.Lock()
	if bucket-b.lastBucket >= int64(len(b.dispatched)) {
		b.dispatched = [8]int64{}
	} else {
		for i := b.lastBucket + 1; i <= bucket; i++ {
			b.dispatched[i%int64(len(b.dispatched))] = 0
		}
	}
	b.lastBucket = bucket
	b.dispatched[bucket%int64(len(b.dispatched))]++
	var recent int64
	for _, n := range b.dispatched {
		recent += n
	}
	b.mu.Unlock()
	b.window.observe(b.backlog() >= recent, now)
}

// StandingQueue tracks continuous backlog for one flow with a configured short wait budget.
type StandingQueue struct {
	window standingWindow
	clock  clock.PassiveClock
	band   *StandingBand
}

// NewStandingQueue creates delay tracking for a single flow.
func NewStandingQueue(interval, ttl time.Duration, clk clock.PassiveClock) *StandingQueue {
	return &StandingQueue{window: standingWindow{interval: interval, ttl: ttl}, clock: clk}
}

// Start begins observation when the first item enters an empty queue. The caller has not yet added
// the item to the band's count, so a zero band backlog means the band was empty too.
func (s *StandingQueue) Start() {
	now := s.clock.Now()
	s.window.observe(false, now)
	if s.band != nil && s.band.backlog() == 0 {
		s.band.window.observe(false, now)
	}
}

// RecordDispatch records a dispatch. The flow shows backlog when the request waited at least half
// the short budget.
func (s *StandingQueue) RecordDispatch(enqueued, now time.Time) {
	s.window.observe(now.Sub(enqueued) >= s.window.ttl-s.window.ttl/2, now)
	if s.band != nil {
		s.band.recordDispatch(now)
	}
}

// ExpiryCutoff returns the latest enqueue time that has exhausted the standing budget of the flow
// or of its band, or zero while neither is standing.
func (s *StandingQueue) ExpiryCutoff(now, regimeSince time.Time) time.Time {
	cutoff := s.window.cutoff(now, regimeSince)
	if cutoff.IsZero() && s.band != nil {
		cutoff = s.band.window.cutoff(now, regimeSince)
	}
	return cutoff
}

// PredicateFunc defines a function that returns true if a given item matches a certain condition.
// It is used by SafeQueue.Cleanup to filter items.
type PredicateFunc func(item flowcontrol.QueueItemAccessor) bool

// SafeQueue defines the contract for a single, concurrent-safe queue implementation.
// This interface is designed for in-memory, synchronous flow control. Implementations are expected to be unbounded;
// capacity management occurs outside the queue implementation.
// All implementations MUST be goroutine-safe.
type SafeQueue interface {
	flowcontrol.QueueInspectionMethods

	// Add enqueues an item. It must associate a new, unique QueueItemHandle with the item by calling
	// item.SetHandle().
	// Contract: The caller MUST NOT provide a nil item.
	Add(item flowcontrol.QueueItemAccessor)

	// Remove atomically finds and removes the item identified by the given handle.
	// On success, implementations MUST invalidate the provided handle by calling handle.Invalidate().
	// Returns the removed item.
	// Returns ErrInvalidQueueItemHandle if the handle is invalid.
	// Returns ErrQueueItemNotFound if the handle is valid but the item is not in the queue.
	Remove(handle flowcontrol.QueueItemHandle) (removedItem flowcontrol.QueueItemAccessor, err error)

	// Cleanup iterates through the queue and atomically removes all items for which the predicate returns true,
	// returning them in a slice.
	// The handle for each removed item MUST be invalidated.
	Cleanup(predicate PredicateFunc) (cleanedItems []flowcontrol.QueueItemAccessor)

	// Drain atomically removes all items from the queue and returns them in a slice.
	// The handle for all removed items MUST be invalidated. The queue MUST be empty after this operation.
	Drain() (drainedItems []flowcontrol.QueueItemAccessor)
}
