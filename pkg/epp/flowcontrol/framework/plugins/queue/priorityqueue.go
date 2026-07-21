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
	item          flowcontrol.QueueItemAccessor
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

// itemHeap implements container/heap.Interface. It is NOT goroutine-safe; the owning
// priorityQueue guards all access with its mutex.
type itemHeap struct {
	items  []*heapItem
	policy flowcontrol.OrderingPolicy
}

func (h *itemHeap) Len() int { return len(h.items) }

// Less orders the heap so that the highest-priority item (per the policy) sits at the root.
// policy.Less(a, b) reports that 'a' has higher priority than 'b', which is exactly the order
// container/heap uses to select the root.
func (h *itemHeap) Less(i, j int) bool {
	return h.policy.Less(h.items[i].item, h.items[j].item)
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

// priorityQueue implements the SafeQueue interface using a container/heap.
// The heap is ordered by the provided policy, with higher priority considered closer to the head.
// This implementation is concurrent-safe.
type priorityQueue struct {
	heap     *itemHeap
	byteSize atomic.Uint64
	mu       sync.RWMutex
}

// newPriorityQueue creates a new priority queue with the given policy.
func newPriorityQueue(policy flowcontrol.OrderingPolicy) *priorityQueue {
	return &priorityQueue{
		heap: &itemHeap{
			items:  make([]*heapItem, 0),
			policy: policy,
		},
	}
}

// --- SafeQueue Interface Implementation ---

// Len returns the number of items in the queue.
func (pq *priorityQueue) Len() int {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	return len(pq.heap.items)
}

// ByteSize returns the total byte size of all items in the queue.
func (pq *priorityQueue) ByteSize() uint64 {
	return pq.byteSize.Load()
}

// Peek returns the highest-priority item without removing it.
// Time complexity: O(1).
func (pq *priorityQueue) Peek() flowcontrol.QueueItemAccessor {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if len(pq.heap.items) == 0 {
		return nil
	}
	return pq.heap.items[0].item
}

// Add adds an item to the queue.
// Time complexity: O(log n).
func (pq *priorityQueue) Add(item flowcontrol.QueueItemAccessor) {
	hi := &heapItem{item: item}
	item.SetHandle(hi)

	pq.mu.Lock()
	heap.Push(pq.heap, hi)
	pq.mu.Unlock()

	pq.byteSize.Add(item.OriginalRequest().ByteSize())
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
	pq.byteSize.Add(^hi.item.OriginalRequest().ByteSize() + 1) // Atomic subtraction.
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
			pq.byteSize.Add(^hi.item.OriginalRequest().ByteSize() + 1) // Atomic subtraction.
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
	pq.byteSize.Store(0)

	return drainedItems
}
