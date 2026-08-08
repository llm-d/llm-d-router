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
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var testFlowKey = flowcontrol.FlowKey{ID: "test-flow-1", Priority: 0}

// enqueueTimePolicy orders items by their enqueue time (earliest first).
var enqueueTimePolicy = &mocks.MockOrderingPolicy{
	TypedNameV: plugin.TypedName{Name: "enqueue_time_asc"},
	LessFunc: func(a, b flowcontrol.QueueItemAccessor) bool {
		return a.EnqueueTime().Before(b.EnqueueTime())
	},
}

// byteSizePolicy orders items by their byte size (smaller first).
var byteSizePolicy = &mocks.MockOrderingPolicy{
	TypedNameV: plugin.TypedName{Name: "byte_size_asc"},
	LessFunc: func(a, b flowcontrol.QueueItemAccessor) bool {
		return a.OriginalRequest().ByteSize() < b.OriginalRequest().ByteSize()
	},
}

// reverseEnqueueTimePolicy orders items by their enqueue time (latest first).
var reverseEnqueueTimePolicy = &mocks.MockOrderingPolicy{
	TypedNameV: plugin.TypedName{Name: "enqueue_time_desc"},
	LessFunc: func(a, b flowcontrol.QueueItemAccessor) bool {
		return a.EnqueueTime().After(b.EnqueueTime())
	},
}

// itemAt builds a mock item with the given byte size and enqueue time.
func itemAt(byteSize uint64, id string, enqueue time.Time) *mocks.MockQueueItemAccessor {
	item := mocks.NewMockQueueItemAccessor(byteSize, id, testFlowKey)
	item.EnqueueTimeV = enqueue
	return item
}

// TestPriorityQueue_Ordering verifies that the queue dispatches items in the order dictated by its
// configured policy, while keeping Len/ByteSize consistent across the Add/Peek/Remove lifecycle.
// itemsInOrder is the expected dispatch order (head first).
func TestPriorityQueue_Ordering(t *testing.T) {
	t.Parallel()
	now := time.Now()

	testCases := []struct {
		name         string
		policy       flowcontrol.OrderingPolicy
		itemsInOrder []*mocks.MockQueueItemAccessor
	}{
		{
			name:   "FIFO_ByEnqueueTime",
			policy: enqueueTimePolicy,
			itemsInOrder: []*mocks.MockQueueItemAccessor{
				itemAt(100, "fifo-1", now.Add(-2*time.Second)),
				itemAt(50, "fifo-2", now.Add(-1*time.Second)),
				itemAt(20, "fifo-3", now),
			},
		},
		{
			name:   "LIFO_ByEnqueueTime",
			policy: reverseEnqueueTimePolicy,
			itemsInOrder: []*mocks.MockQueueItemAccessor{
				itemAt(20, "lifo-1", now),
				itemAt(50, "lifo-2", now.Add(-1*time.Second)),
				itemAt(100, "lifo-3", now.Add(-2*time.Second)),
			},
		},
		{
			name:   "BySize",
			policy: byteSizePolicy,
			itemsInOrder: []*mocks.MockQueueItemAccessor{
				itemAt(20, "size-small", now),
				itemAt(50, "size-medium", now),
				itemAt(100, "size-large", now),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assertLifecycleAndOrdering(t, New(tc.policy), tc.itemsInOrder)
		})
	}
}

// assertLifecycleAndOrdering adds every item, then drains via Peek/Remove, asserting the queue
// returns them in itemsInOrder and keeps Len/ByteSize consistent throughout.
func assertLifecycleAndOrdering(t *testing.T, q contracts.SafeQueue, itemsInOrder []*mocks.MockQueueItemAccessor) {
	t.Helper()

	assert.Nil(t, q.Peek(), "Peek on an empty queue must return nil")

	var wantLen int
	var wantBytes uint64
	for _, item := range itemsInOrder {
		q.Add(item)
		require.NotNil(t, item.Handle(), "Add must assign a handle")
		require.False(t, item.Handle().IsInvalidated(), "a new handle must not be invalidated")

		wantLen++
		wantBytes += item.OriginalRequest().ByteSize()
		assert.Equal(t, wantLen, q.Len(), "Len after Add")
		assert.Equal(t, wantBytes, q.ByteSize(), "ByteSize after Add")
	}

	for i, want := range itemsInOrder {
		peeked := q.Peek()
		require.NotNil(t, peeked, "Peek (iteration %d)", i)
		assert.Equal(t, want.OriginalRequest().ID(), peeked.OriginalRequest().ID(), "Peek must return the head (iteration %d)", i)
		assert.Equal(t, wantLen, q.Len(), "Peek must not change Len (iteration %d)", i)

		removed, err := q.Remove(peeked.Handle())
		require.NoError(t, err, "Remove of the head (iteration %d)", i)
		assert.Equal(t, want.OriginalRequest().ID(), removed.OriginalRequest().ID(), "Remove must return the head (iteration %d)", i)
		assert.True(t, peeked.Handle().IsInvalidated(), "Remove must invalidate the handle (iteration %d)", i)

		wantLen--
		wantBytes -= removed.OriginalRequest().ByteSize()
		assert.Equal(t, wantLen, q.Len(), "Len after Remove (iteration %d)", i)
		assert.Equal(t, wantBytes, q.ByteSize(), "ByteSize after Remove (iteration %d)", i)
	}

	assert.Zero(t, q.Len(), "Len must be 0 once drained")
	assert.Zero(t, q.ByteSize(), "ByteSize must be 0 once drained")
	assert.Nil(t, q.Peek(), "Peek on a drained queue must return nil")
}

func TestPriorityQueue_Remove(t *testing.T) {
	t.Parallel()

	t.Run("RejectsInvalidHandles", func(t *testing.T) {
		t.Parallel()
		q := New(enqueueTimePolicy)
		q.Add(mocks.NewMockQueueItemAccessor(100, "item", testFlowKey))

		otherQ := New(enqueueTimePolicy)
		otherItem := mocks.NewMockQueueItemAccessor(10, "other", flowcontrol.FlowKey{ID: "other-flow"})
		otherQ.Add(otherItem)

		invalidated := &mocks.MockQueueItemHandle{}
		invalidated.Invalidate()

		testCases := []struct {
			name      string
			handle    flowcontrol.QueueItemHandle
			expectErr error
		}{
			{"nil", nil, contracts.ErrInvalidQueueItemHandle},
			{"invalidated", invalidated, contracts.ErrInvalidQueueItemHandle},
			{"foreign type", &mocks.MockQueueItemHandle{}, contracts.ErrInvalidQueueItemHandle},
			{"from another queue", otherItem.Handle(), contracts.ErrQueueItemNotFound},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				wantLen, wantBytes := q.Len(), q.ByteSize()
				_, err := q.Remove(tc.handle)
				assert.ErrorIs(t, err, tc.expectErr)
				assert.Equal(t, wantLen, q.Len(), "Len must be unchanged after a failed Remove")
				assert.Equal(t, wantBytes, q.ByteSize(), "ByteSize must be unchanged after a failed Remove")
			})
		}
	})

	t.Run("RemovesNonHeadItem", func(t *testing.T) {
		t.Parallel()
		q := New(enqueueTimePolicy)
		now := time.Now()
		head := itemAt(10, "head", now.Add(-3*time.Second))
		mid := itemAt(20, "mid", now.Add(-2*time.Second))
		tail := itemAt(30, "tail", now.Add(-1*time.Second))
		q.Add(head)
		q.Add(mid)
		q.Add(tail)

		removed, err := q.Remove(mid.Handle())
		require.NoError(t, err)
		assert.Equal(t, "mid", removed.OriginalRequest().ID())
		assert.True(t, mid.Handle().IsInvalidated(), "Remove must invalidate the handle")
		assert.Equal(t, 2, q.Len())
		assert.Equal(t, head.OriginalRequest().ByteSize()+tail.OriginalRequest().ByteSize(), q.ByteSize())

		_, err = q.Remove(mid.Handle())
		assert.ErrorIs(t, err, contracts.ErrInvalidQueueItemHandle, "removing with a stale handle must fail")
	})
}

func TestPriorityQueue_Cleanup(t *testing.T) {
	t.Parallel()

	// oddByteSize matches items whose byte size is odd.
	oddByteSize := func(item flowcontrol.QueueItemAccessor) bool {
		return item.OriginalRequest().ByteSize()%2 != 0
	}

	t.Run("EmptyQueue", func(t *testing.T) {
		t.Parallel()
		q := New(enqueueTimePolicy)
		assert.Empty(t, q.Cleanup(oddByteSize))
		assert.Zero(t, q.Len())
		assert.Zero(t, q.ByteSize())
	})

	t.Run("MatchesNone_KeepsAll", func(t *testing.T) {
		t.Parallel()
		q := New(enqueueTimePolicy)
		k1 := mocks.NewMockQueueItemAccessor(10, "k1", testFlowKey)
		k2 := mocks.NewMockQueueItemAccessor(12, "k2", testFlowKey)
		q.Add(k1)
		q.Add(k2)

		assert.Empty(t, q.Cleanup(func(flowcontrol.QueueItemAccessor) bool { return false }))
		assert.Equal(t, 2, q.Len())
		assert.Equal(t, uint64(22), q.ByteSize())
		assert.False(t, k1.Handle().IsInvalidated(), "kept handle must not be invalidated")
		assert.False(t, k2.Handle().IsInvalidated(), "kept handle must not be invalidated")
	})

	t.Run("MatchesAll_RemovesAll", func(t *testing.T) {
		t.Parallel()
		q := New(enqueueTimePolicy)
		r1 := mocks.NewMockQueueItemAccessor(11, "r1", testFlowKey)
		r2 := mocks.NewMockQueueItemAccessor(13, "r2", testFlowKey)
		q.Add(r1)
		q.Add(r2)

		assert.Len(t, q.Cleanup(func(flowcontrol.QueueItemAccessor) bool { return true }), 2)
		assert.Zero(t, q.Len())
		assert.Zero(t, q.ByteSize())
		assert.True(t, r1.Handle().IsInvalidated(), "removed handle must be invalidated")
		assert.True(t, r2.Handle().IsInvalidated(), "removed handle must be invalidated")
	})

	t.Run("MatchesSubset_RemovesAndReheapifies", func(t *testing.T) {
		t.Parallel()
		q := New(enqueueTimePolicy)
		keep1 := mocks.NewMockQueueItemAccessor(20, "keep1", testFlowKey)
		remove1 := mocks.NewMockQueueItemAccessor(11, "remove1", testFlowKey)
		keep2 := mocks.NewMockQueueItemAccessor(22, "keep2", testFlowKey)
		remove2 := mocks.NewMockQueueItemAccessor(33, "remove2", testFlowKey)
		q.Add(keep1)
		q.Add(remove1)
		q.Add(keep2)
		q.Add(remove2)

		removed := q.Cleanup(oddByteSize)
		assert.Len(t, removed, 2)
		assert.Equal(t, 2, q.Len())
		assert.Equal(t, uint64(42), q.ByteSize(), "ByteSize must reflect only the kept items")
		assert.True(t, remove1.Handle().IsInvalidated())
		assert.True(t, remove2.Handle().IsInvalidated())
		assert.False(t, keep1.Handle().IsInvalidated())
		assert.False(t, keep2.Handle().IsInvalidated())

		// The kept items must still drain in a valid order (heap property preserved).
		var remainingIDs []string
		for q.Len() > 0 {
			item, err := q.Remove(q.Peek().Handle())
			require.NoError(t, err)
			remainingIDs = append(remainingIDs, item.OriginalRequest().ID())
		}
		sort.Strings(remainingIDs)
		assert.Equal(t, []string{"keep1", "keep2"}, remainingIDs)
	})
}

func TestPriorityQueue_Drain(t *testing.T) {
	t.Parallel()

	t.Run("NonEmpty_ReturnsAndInvalidatesAll", func(t *testing.T) {
		t.Parallel()
		q := New(enqueueTimePolicy)
		d1 := mocks.NewMockQueueItemAccessor(10, "d1", testFlowKey)
		d2 := mocks.NewMockQueueItemAccessor(20, "d2", testFlowKey)
		q.Add(d1)
		q.Add(d2)

		drained := q.Drain()
		assert.Len(t, drained, 2)
		assert.Zero(t, q.Len())
		assert.Zero(t, q.ByteSize())
		assert.True(t, d1.Handle().IsInvalidated())
		assert.True(t, d2.Handle().IsInvalidated())
	})

	t.Run("Empty_IsIdempotent", func(t *testing.T) {
		t.Parallel()
		q := New(enqueueTimePolicy)
		assert.Empty(t, q.Drain())
		assert.Empty(t, q.Drain())
		assert.Zero(t, q.Len())
		assert.Zero(t, q.ByteSize())
	})
}

// TestPriorityQueue_Concurrency drives a mix of concurrent operations and verifies that the
// accounting stays consistent: items drained at the end must equal initial + adds - removes.
func TestPriorityQueue_Concurrency(t *testing.T) {
	t.Parallel()
	q := New(enqueueTimePolicy)

	const (
		numGoroutines   = 10
		initialItems    = 200
		opsPerGoroutine = 50
	)

	// handleChan is a concurrent-safe pool of handles goroutines pull from to test Remove.
	handleChan := make(chan flowcontrol.QueueItemHandle, initialItems+(numGoroutines*opsPerGoroutine))
	for i := range initialItems {
		item := mocks.NewMockQueueItemAccessor(1, fmt.Sprintf("init-%d", i), testFlowKey)
		q.Add(item)
		handleChan <- item.Handle()
	}

	var wg sync.WaitGroup
	var adds, removes atomic.Uint64
	for i := range numGoroutines {
		wg.Add(1)
		go func(routineID int) {
			defer wg.Done()
			for j := range opsPerGoroutine {
				switch (j + routineID) % 4 {
				case 0: // Add
					item := mocks.NewMockQueueItemAccessor(1, fmt.Sprintf("add-%d-%d", routineID, j), testFlowKey)
					q.Add(item)
					adds.Add(1)
					handleChan <- item.Handle()
				case 1: // Remove
					select {
					case handle := <-handleChan:
						if handle != nil && !handle.IsInvalidated() {
							if _, err := q.Remove(handle); err == nil {
								removes.Add(1)
							} else {
								assert.ErrorIs(t, err, contracts.ErrInvalidQueueItemHandle, "racing Remove must fail cleanly")
							}
						}
					default:
					}
				case 2: // Inspect
					_ = q.Len()
					_ = q.ByteSize()
					if peeked := q.Peek(); q.Len() == 0 {
						assert.Nil(t, peeked, "Peek on empty queue must be nil")
					}
				case 3: // Cleanup (no-op predicate)
					q.Cleanup(func(flowcontrol.QueueItemAccessor) bool { return false })
				}
			}
		}(i)
	}

	wg.Wait()
	close(handleChan)

	drained := q.Drain()
	for _, item := range drained {
		require.True(t, item.Handle().IsInvalidated(), "every drained handle must be invalidated")
	}
	assert.Equal(t, int(initialItems)+int(adds.Load())-int(removes.Load()), len(drained),
		"drained count must equal initial + adds - removes")
	assert.Zero(t, q.Len())
	assert.Zero(t, q.ByteSize())
}

// TestPriorityQueue_InternalProperty is a white-box test that the heap invariant holds after a
// series of Add and Remove operations.
func TestPriorityQueue_InternalProperty(t *testing.T) {
	t.Parallel()
	q := newPriorityQueue(enqueueTimePolicy)

	items := make([]*mocks.MockQueueItemAccessor, 20)
	now := time.Now()
	for i := range items {
		// Add items in a varied order of enqueue times.
		items[i] = itemAt(10, "item", now.Add(time.Duration((i%5-2)*10)*time.Second))
		q.Add(items[i])
		assertHeapProperty(t, q, "after adding item %d", i)
	}

	for _, i := range []int{15, 7, 11} {
		_, err := q.Remove(items[i].Handle())
		require.NoError(t, err, "Remove should not fail for item %d", i)
		assertHeapProperty(t, q, "after removing item %d", i)
	}

	for q.Len() > 0 {
		head := q.Peek()
		require.NotNil(t, head)
		_, err := q.Remove(head.Handle())
		require.NoError(t, err)
		assertHeapProperty(t, q, "after removing head item")
	}
}

// assertHeapProperty checks that the slice satisfies the (max-by-policy) heap property: no child may
// outrank its parent, and every item's tracked index must match its slice position.
func assertHeapProperty(t *testing.T, q *priorityQueue, msgAndArgs ...any) {
	t.Helper()
	items := q.heap.items
	for i, hi := range items {
		require.Equal(t, i, hi.index, "item's tracked index must match its slice position. %v", msgAndArgs)

		for _, child := range []int{2*i + 1, 2*i + 2} {
			if child >= len(items) {
				continue
			}
			// policy.Less(a, b) == true means 'a' has higher priority than 'b'. A child must never
			// outrank its parent.
			require.False(t, q.heap.policy.Less(items[child].item, items[i].item),
				"child %d must not outrank parent %d. %v", child, i, msgAndArgs)
		}
	}
}

// TestPriorityQueue_ByteSize_BookedAtAdd verifies that byte accounting settles against the size
// booked at Add time. A caller-implemented FlowControlRequest whose ByteSize() changes between Add
// and removal must not be able to drift the queue's byte counter.
func TestPriorityQueue_ByteSize_BookedAtAdd(t *testing.T) {
	t.Parallel()

	t.Run("Remove", func(t *testing.T) {
		t.Parallel()
		q := newPriorityQueue(enqueueTimePolicy)
		item := mocks.NewMockQueueItemAccessor(100, "req-unstable", testFlowKey)
		q.Add(item)
		require.Equal(t, uint64(100), q.ByteSize(), "Byte size must reflect the size booked at Add")

		// The request misreports its size after admission; removal must debit the booked 100.
		item.OriginalRequestV.(*mocks.MockFlowControlRequest).ByteSizeV = 5000
		_, err := q.Remove(item.Handle())
		require.NoError(t, err)
		assert.Zero(t, q.ByteSize(), "Removal must debit the booked size, not the re-read size")
		assert.Zero(t, q.Len())
	})

	t.Run("Cleanup", func(t *testing.T) {
		t.Parallel()
		q := newPriorityQueue(enqueueTimePolicy)
		itemA := mocks.NewMockQueueItemAccessor(50, "req-a", testFlowKey)
		itemB := mocks.NewMockQueueItemAccessor(75, "req-b", testFlowKey)
		q.Add(itemA)
		q.Add(itemB)
		require.Equal(t, uint64(125), q.ByteSize())

		itemA.OriginalRequestV.(*mocks.MockFlowControlRequest).ByteSizeV = 1
		removed := q.Cleanup(func(i flowcontrol.QueueItemAccessor) bool {
			return i.OriginalRequest().ID() == "req-a"
		})
		require.Len(t, removed, 1)
		assert.Equal(t, uint64(75), q.ByteSize(), "Cleanup must debit the booked size, not the re-read size")
		assert.Equal(t, 1, q.Len())
	})
}
