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

package registry

import (
	"slices"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// fairQueue decorates the SafeQueue of each flow in a band that enables fair admission. It keeps the queue in the
// band's length index and its items in Add order, so the band's longest queue and that queue's last-added item are
// found in O(1) amortized. Its owning managedQueue serializes every call under mq.mu.
type fairQueue struct {
	contracts.SafeQueue
	mq    *managedQueue
	index *queueLenIndex

	// added holds the queue's items in Add order. Every removal drops removed items from both ends, so the last entry
	// is always queued, and Cleanup, which scans the whole queue anyway, drops them all. Removed items between queued
	// ones, such as those dispatched out of Add order, stay referenced until they outnumber the queued items and added
	// is compacted: at most as many as the queue holds. A removed item is recognized by its invalidated handle.
	added []flowcontrol.QueueItemAccessor

	// indexPrev, indexNext and indexLen place the queue in index. index.mu protects them.
	indexPrev, indexNext *fairQueue
	indexLen             int
}

func (q *fairQueue) Add(item flowcontrol.QueueItemAccessor) {
	q.SafeQueue.Add(item)
	q.added = append(q.added, item)
	q.index.update(q, q.Len())
}

func (q *fairQueue) Remove(handle flowcontrol.QueueItemHandle) (flowcontrol.QueueItemAccessor, error) {
	item, err := q.SafeQueue.Remove(handle)
	if err == nil {
		q.removed()
	}
	return item, err
}

func (q *fairQueue) Cleanup(predicate contracts.PredicateFunc) []flowcontrol.QueueItemAccessor {
	items := q.SafeQueue.Cleanup(predicate)
	if len(items) > 0 {
		q.added = slices.DeleteFunc(q.added, isRemoved)
		q.removed()
	}
	return items
}

func (q *fairQueue) Drain() []flowcontrol.QueueItemAccessor {
	items := q.SafeQueue.Drain()
	if len(items) > 0 {
		q.removed()
	}
	return items
}

// removed records that items left the queue: it moves the queue in the index and drops removed items from the ends of
// added. Compacting added once removed items outnumber queued ones keeps it within twice the queue length at O(1)
// amortized cost per removal. An emptied queue keeps added's cleared array, so a flow that empties and refills does
// not allocate on every Add.
func (q *fairQueue) removed() {
	n := q.Len()
	q.index.update(q, n)
	if n == 0 {
		clear(q.added)
		q.added = q.added[:0]
		return
	}
	for len(q.added) > 0 && isRemoved(q.added[0]) {
		q.added[0] = nil
		q.added = q.added[1:]
	}
	for last := len(q.added) - 1; last >= 0 && isRemoved(q.added[last]); last-- {
		q.added[last] = nil
		q.added = q.added[:last]
	}
	if len(q.added) > 2*n {
		q.added = slices.DeleteFunc(q.added, isRemoved)
	}
}

// lastAdded returns the last-added item still in the queue, or nil if the queue is empty.
func (q *fairQueue) lastAdded() flowcontrol.QueueItemAccessor {
	if len(q.added) == 0 {
		return nil
	}
	return q.added[len(q.added)-1]
}

func isRemoved(item flowcontrol.QueueItemAccessor) bool {
	return item.Handle().IsInvalidated()
}

// queueLenIndex orders a band's non-empty fair queues by length so the longest is found in O(1). buckets[n] heads an
// intrusive list of the queues holding n items. Add and dispatch move a queue by one bucket; a Cleanup or Drain that
// removes k items lowers max by at most k. buckets never shrinks: 8 bytes per slot up to the longest queue seen.
type queueLenIndex struct {
	// mu serializes updates: the Run goroutine adds and dispatches while cleanup-sweep workers remove items from other
	// queues of the same band. It is a leaf lock taken inside a queue's critical section.
	mu      sync.Mutex
	buckets []*fairQueue
	max     int
}

func newQueueLenIndex() *queueLenIndex {
	return &queueLenIndex{buckets: make([]*fairQueue, 1)}
}

// update moves q to the bucket for n items, or out of the index when n is zero.
func (x *queueLenIndex) update(q *fairQueue, n int) {
	x.mu.Lock()
	defer x.mu.Unlock()
	if q.indexPrev != nil {
		q.indexPrev.indexNext = q.indexNext
	} else if x.buckets[q.indexLen] == q {
		x.buckets[q.indexLen] = q.indexNext
	}
	if q.indexNext != nil {
		q.indexNext.indexPrev = q.indexPrev
	}
	q.indexPrev, q.indexNext, q.indexLen = nil, nil, n
	if n > 0 {
		for len(x.buckets) <= n {
			x.buckets = append(x.buckets, nil)
		}
		if head := x.buckets[n]; head != nil {
			head.indexPrev = q
			q.indexNext = head
		}
		x.buckets[n] = q
		x.max = max(x.max, n)
	}
	for x.max > 0 && x.buckets[x.max] == nil {
		x.max--
	}
}

// longest returns the managed queue of a fair queue holding the most items, or nil if the index is nil or every queue
// is empty.
func (x *queueLenIndex) longest() contracts.ManagedQueue {
	if x == nil {
		return nil
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	if q := x.buckets[x.max]; q != nil {
		return q.mq
	}
	return nil
}
