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

// Package queue provides the priority-ordered SafeQueue used by flow control.
package queue

import (
	"container/heap"
	"sync"
	"sync/atomic"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// PriorityQueueName identifies the priority queue implementation (used for benchmark labels).
//
// This queue provides a concurrent-safe priority queue whose ordering is maintained by an internal
// container/heap. Items are ordered by the configured OrderingPolicy, with the highest-priority
// item (per the policy) at the head.
//
// Each item's position in the heap is tracked on its handle, enabling O(log n) targeted removal.
const PriorityQueueName = "PriorityQueue"

// New creates a SafeQueue ordered by the given OrderingPolicy. Peek returns the highest-priority
// item per the policy.
func New(policy flowcontrol.OrderingPolicy) contracts.SafeQueue {
	return newPriorityQueue(policy)
}

// heapItem holds a queued item together with its current position in the heap. It doubles as the
// item's flowcontrol.QueueItemHandle, allowing O(log n) removal by index without a side lookup
// table.
type heapItem struct {
	item flowcontrol.QueueItemAccessor
	// byteSize is the item's byte size as reported at Add time. Removal paths debit this booked
	// value rather than re-reading the request, so the queue's byte accounting stays exact even if
	// a caller-implemented FlowControlRequest.ByteSize() is not stable across calls.
	byteSize uint64
	// score is the item's cached ordering key, used only in scoring mode. The comparator reads this rather
	// than calling the policy, so a heap operation sees values fixed before it began and stays transitive
	// while the policy's state moves. Booked at Add, rewritten for every item on refresh.
	score         float64
	index         int // position in itemHeap.items; set to -1 once removed.
	isInvalidated bool
}

// Handle returns the heap item itself, which is used as the handle.
func (h *heapItem) Handle() any { return h }

// Invalidate marks the handle as invalid.
func (h *heapItem) Invalidate() { h.isInvalidated = true }

// IsInvalidated returns true if the handle has been invalidated.
func (h *heapItem) IsInvalidated() bool { return h.isInvalidated }

var _ flowcontrol.QueueItemHandle = &heapItem{}

// itemHeap implements container/heap.Interface. Its methods and items slice are guarded by the owning
// priorityQueue's mutex; the policy fields are immutable after construction and cachedGeneration is atomic,
// so the owner reads both without the mutex.
type itemHeap struct {
	items []*heapItem
	// Exactly one of policy and scoringPolicy is populated: the nil one's counterpart is the queue's mode.
	// scoringPolicy suffices for scoring mode because ScoringOrderingPolicy embeds OrderingPolicy.
	policy        flowcontrol.OrderingPolicy        // static mode
	scoringPolicy flowcontrol.ScoringOrderingPolicy // scoring mode; owner calls Score/Generation through it
	// cachedGeneration is the generation every score in items was computed at. refreshIfStale reads it
	// unlocked on the fast path and writes it under the lock after a rebuild. Meaningless in static mode.
	cachedGeneration atomic.Uint64
}

func (h *itemHeap) Len() int { return len(h.items) }

// Less orders the heap so the highest-priority item sits at the root: policy.Less in static mode.
//
// In scoring mode it reads cached scores, never live policy state -- what keeps it transitive for a drifting
// key: higher score first, ties by earlier EnqueueTime. The tie-break MUST match flowcontrol.CompareByScore,
// which the fairness tier uses to compare heads across queues; divergence lets a band-level pick disagree
// with the queue's own head for equal-scored items.
func (h *itemHeap) Less(i, j int) bool {
	if h.scoringPolicy == nil {
		return h.policy.Less(h.items[i].item, h.items[j].item)
	}
	a, b := h.items[i], h.items[j]
	if a.score != b.score {
		return a.score > b.score
	}
	return a.item.EnqueueTime().Before(b.item.EnqueueTime())
}

func (h *itemHeap) Swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
	h.items[i].index = i
	h.items[j].index = j
}

func (h *itemHeap) Push(x any) {
	hi := x.(*heapItem)
	hi.index = len(h.items)
	h.items = append(h.items, hi)
}

func (h *itemHeap) Pop() any {
	old := h.items
	n := len(old)
	hi := old[n-1]
	old[n-1] = nil // Avoid retaining the removed item.
	hi.index = -1  // Mark as no longer in the heap.
	h.items = old[:n-1]
	return hi
}

// priorityQueue is a concurrent-safe SafeQueue over a container/heap, ordered by the configured policy with
// the highest-priority item at the head. len and byteSize are atomic snapshots maintained inside the same
// critical section as every mutation, so they cannot drift from the heap's contents.
type priorityQueue struct {
	heap     *itemHeap
	len      atomic.Int64
	byteSize atomic.Uint64
	mu       sync.RWMutex
}

// newPriorityQueue creates a new priority queue with the given policy. The optional ScoringOrderingPolicy
// upgrade is checked once here; a policy implementing it goes to scoringPolicy (scoring mode), any other to
// policy (the unchanged static path).
//
// cachedGeneration is left at zero rather than seeded: the cache is empty so no generation is honest, the
// first Peek's rebuild iterates zero items, and seeding would call plugin code while the registry holds its
// lock (FlowRegistry.buildFlowComponents).
func newPriorityQueue(policy flowcontrol.OrderingPolicy) *priorityQueue {
	h := &itemHeap{items: make([]*heapItem, 0)}
	if scoringPolicy, ok := policy.(flowcontrol.ScoringOrderingPolicy); ok {
		h.scoringPolicy = scoringPolicy
	} else {
		h.policy = policy
	}
	return &priorityQueue{heap: h}
}

// --- SafeQueue Interface Implementation ---

// Len returns the number of items in the queue.
func (pq *priorityQueue) Len() int {
	return int(pq.len.Load())
}

// ByteSize returns the total byte size of all items in the queue.
func (pq *priorityQueue) ByteSize() uint64 {
	return pq.byteSize.Load()
}

// Peek returns the highest-priority item without removing it.
// Time complexity: O(1) in static mode, amortized O(1) in scoring mode.
//
// In scoring mode Peek is the refresh point: it is the last read before the value is used, so refreshing
// here (rather than on a ticker) bounds how stale a head can be at comparison time.
func (pq *priorityQueue) Peek() flowcontrol.QueueItemAccessor {
	if pq.heap.scoringPolicy != nil {
		pq.refreshIfStale()
	}

	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if len(pq.heap.items) == 0 {
		return nil
	}
	return pq.heap.items[0].item
}

// refreshIfStale rebuilds every cached score and re-heapifies if the policy's generation has moved. The
// unlocked pre-check makes the common no-op case one atomic load; the re-check under the lock is
// double-checked locking, since sync.RWMutex cannot upgrade.
//
// The caller must have established that the queue is in scoring mode.
func (pq *priorityQueue) refreshIfStale() {
	// Capture gen once. Stamping cachedGeneration with a value re-read after the rebuild could mark scores
	// current at an epoch they were not read at and skip the next real change. Capturing before the Score
	// pass instead keeps the stamp <= the state the scores came from: if the policy advanced meanwhile, the
	// stamp is stale-low and the next Peek simply rebuilds again -- never a skipped rebuild.
	gen := pq.heap.scoringPolicy.Generation()
	if gen == pq.heap.cachedGeneration.Load() {
		return
	}

	pq.mu.Lock()
	defer pq.mu.Unlock()

	if gen == pq.heap.cachedGeneration.Load() {
		return
	}

	for _, hi := range pq.heap.items {
		hi.score = pq.computeScore(hi.item)
	}
	heap.Init(pq.heap)
	pq.heap.cachedGeneration.Store(gen)
}

// Add adds an item to the queue.
// Time complexity: O(log n).
//
// In scoring mode the score is computed before the lock, so Score never runs in the critical section. It
// reflects state at Add time, which may be newer than the incumbents' cached scores; see
// ScoringOrderingPolicy for the resulting bounded skew.
func (pq *priorityQueue) Add(item flowcontrol.QueueItemAccessor) {
	hi := &heapItem{item: item, byteSize: item.OriginalRequest().ByteSize()}
	if pq.heap.scoringPolicy != nil {
		hi.score = pq.computeScore(item)
	}
	item.SetHandle(hi)

	pq.mu.Lock()
	heap.Push(pq.heap, hi)
	pq.len.Store(int64(len(pq.heap.items)))
	pq.byteSize.Add(hi.byteSize)
	pq.mu.Unlock()
}

// Remove removes an item from the queue.
// Time complexity: O(log n).
func (pq *priorityQueue) Remove(handle flowcontrol.QueueItemHandle) (flowcontrol.QueueItemAccessor, error) {
	if handle == nil {
		return nil, contracts.ErrInvalidQueueItemHandle
	}
	hi, ok := handle.(*heapItem)
	if !ok {
		return nil, contracts.ErrInvalidQueueItemHandle
	}

	pq.mu.Lock()
	defer pq.mu.Unlock()

	if hi.IsInvalidated() {
		return nil, contracts.ErrInvalidQueueItemHandle
	}

	// Validate membership by identity: a *heapItem is created in Add and only ever lives in a single
	// queue's slice, so a matching pointer at its tracked index proves it belongs to this queue and
	// is still present. This also guards against a stale index (e.g., the item was concurrently
	// removed) reading out of bounds or removing the wrong item.
	i := hi.index
	if i < 0 || i >= len(pq.heap.items) || pq.heap.items[i] != hi {
		return nil, contracts.ErrQueueItemNotFound
	}

	heap.Remove(pq.heap, i)
	pq.len.Store(int64(len(pq.heap.items)))
	pq.byteSize.Add(^hi.byteSize + 1) // Atomic subtraction of the booked size.
	hi.Invalidate()
	return hi.item, nil
}

// Cleanup removes items from the queue that satisfy the predicate.
func (pq *priorityQueue) Cleanup(predicate contracts.PredicateFunc) []flowcontrol.QueueItemAccessor {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	var removedItems []flowcontrol.QueueItemAccessor

	// Compact survivors in place: the kept count never exceeds the read index, so survivors can be
	// written back into the existing backing array instead of allocating a second slice.
	items := pq.heap.items
	kept := 0
	for _, hi := range items {
		if predicate(hi.item) {
			removedItems = append(removedItems, hi.item)
			hi.Invalidate()
			hi.index = -1
			pq.byteSize.Add(^hi.byteSize + 1) // Atomic subtraction of the booked size.
			continue
		}
		items[kept] = hi
		hi.index = kept
		kept++
	}

	if len(removedItems) > 0 {
		// Clear the vacated tail so removed items aren't retained by the backing array.
		for i := kept; i < len(items); i++ {
			items[i] = nil
		}
		pq.heap.items = items[:kept]
		pq.len.Store(int64(len(pq.heap.items)))
		// Re-establish the heap property on the remaining items.
		heap.Init(pq.heap)
	}

	return removedItems
}

// Drain removes all items from the queue.
func (pq *priorityQueue) Drain() []flowcontrol.QueueItemAccessor {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	drainedItems := make([]flowcontrol.QueueItemAccessor, len(pq.heap.items))
	for i, hi := range pq.heap.items {
		drainedItems[i] = hi.item
		hi.Invalidate()
		hi.index = -1
	}

	pq.heap.items = make([]*heapItem, 0)
	pq.len.Store(0)
	pq.byteSize.Store(0)

	return drainedItems
}

// --- Scoring mode ---

// computeScore reads the policy's score and normalizes NaN, the queue's backstop against a non-conforming
// policy. The caller must have established that the queue is in scoring mode.
func (pq *priorityQueue) computeScore(item flowcontrol.QueueItemAccessor) float64 {
	return flowcontrol.NormalizeScore(pq.heap.scoringPolicy.Score(item))
}
