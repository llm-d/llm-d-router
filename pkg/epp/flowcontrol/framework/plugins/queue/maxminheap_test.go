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
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
)

// TestMaxMinHeap_InternalProperty validates that the max-min heap property is maintained after a series of `Add` and
// `Remove` operations. This is a white-box test to ensure the internal data structure is always in a valid state.
func TestMaxMinHeap_InternalProperty(t *testing.T) {
	t.Parallel()
	q := newMaxMinHeap(enqueueTimePolicy)

	items := make([]*mocks.MockQueueItemAccessor, 20)
	now := time.Now()
	for i := range items {
		// Add items in a somewhat random order of enqueue times
		items[i] = mocks.NewMockQueueItemAccessor(10, "item", flowcontrol.FlowKey{ID: "flow"})
		items[i].EnqueueTimeV = now.Add(time.Duration((i%5-2)*10) * time.Second)
		q.Add(items[i])
		assertHeapProperty(t, q, "after adding item %d", i)
	}

	// Remove a few items from the middle and validate the heap property
	for _, i := range []int{15, 7, 11} {
		handle := items[i].Handle()
		_, err := q.Remove(handle)
		require.NoError(t, err, "Remove should not fail for item %d", i)
		assertHeapProperty(t, q, "after removing item %d", i)
	}

	// Remove remaining items from the head and validate each time
	for q.Len() > 0 {
		head := q.PeekHead()
		require.NotNil(t, head)
		_, err := q.Remove(head.Handle())
		require.NoError(t, err)
		assertHeapProperty(t, q, "after removing head item")
	}
}

// TestMaxMinHeap_Remove_CrossSubtreeSwap demonstrates that Remove corrupts the heap when the replacement element
// (swapped from the end of the array) belongs to a different subtree than the removed element.
// The root cause is that Remove calls down() but not up(), so ancestor-level violations go unfixed.
func TestMaxMinHeap_Remove_CrossSubtreeSwap(t *testing.T) {
	t.Parallel()

	// 6-element max-min heap (FCFS: earlier time = higher priority):
	//
	//        0          L0 (max) -- highest priority
	//       / \
	//      5   50       L1 (min) -- lowest priority in subtrees
	//     / \   \
	//    1   4   40     L2 (max) -- leaves
	//
	// Left subtree of root:  {5, 1, 4}   (index 1, bounded by min-ancestor time=5)
	// Right subtree of root: {50, 40}    (index 2, bounded by min-ancestor time=50)
	times := []int{0, 5, 50, 1, 4, 40}

	q := newMaxMinHeap(enqueueTimePolicy)
	flowKey := flowcontrol.FlowKey{ID: "f", Priority: 0}
	base := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	for i, sec := range times {
		item := mocks.NewMockQueueItemAccessor(10, fmt.Sprintf("t%d", sec), flowKey)
		item.EnqueueTimeV = base.Add(time.Duration(sec) * time.Second)
		hi := &heapItem{item: item, index: i}
		item.SetHandle(hi)
		q.items = append(q.items, item)
		q.handles[item.Handle()] = hi
	}
	q.byteSize.Store(60)

	// Remove index 3 (time=1, left subtree leaf). Last element (index 5, time=40, right subtree) swaps in.
	// down(3) has no children -> no-op. Heap becomes [0, 5, 50, 40, 4].
	// Node 1 (min, time=5) must be lowest priority in its subtree {5, 40, 4}.
	// But time=40 has lower priority than time=5 -> VIOLATION.
	_, err := q.Remove(q.items[3].Handle())
	require.NoError(t, err)

	// Verify every node is the max (or min) of its entire subtree -- not just its direct children.
	for i := range q.items {
		isMin := isMinLevel(i)
		for d := range q.items {
			if !isAncestor(i, d) {
				continue
			}
			if isMin {
				require.Falsef(t, q.policy.Less(q.items[i], q.items[d]),
					"min-level node %d (t=%v) has higher priority than descendant %d (t=%v)",
					i, q.items[i].EnqueueTime().Sub(base)/time.Second,
					d, q.items[d].EnqueueTime().Sub(base)/time.Second)
			} else {
				require.Falsef(t, q.policy.Less(q.items[d], q.items[i]),
					"max-level node %d (t=%v) has lower priority than descendant %d (t=%v)",
					i, q.items[i].EnqueueTime().Sub(base)/time.Second,
					d, q.items[d].EnqueueTime().Sub(base)/time.Second)
			}
		}
	}
}

// isAncestor reports whether a is a strict ancestor of d in a binary heap.
func isAncestor(a, d int) bool {
	if d <= a {
		return false
	}
	for d > a {
		d = (d - 1) / 2
	}
	return d == a
}

// assertHeapProperty checks if the slice of items satisfies the max-min heap property.
func assertHeapProperty(t *testing.T, h *maxMinHeap, msgAndArgs ...any) {
	t.Helper()
	if len(h.items) > 0 {
		verifyNode(t, h, 0, msgAndArgs...)
	}
}

// verifyNode recursively checks that the subtree at index `i` satisfies the max-min heap property.
func verifyNode(t *testing.T, h *maxMinHeap, i int, msgAndArgs ...any) {
	t.Helper()
	n := len(h.items)
	if i >= n {
		return
	}

	level := int(math.Floor(math.Log2(float64(i + 1))))
	isMinLevel := level%2 != 0

	leftChild := 2*i + 1
	rightChild := 2*i + 2

	// Check children
	if leftChild < n {
		if isMinLevel {
			require.False(t, h.policy.Less(h.items[i], h.items[leftChild]),
				"min-level node %d has child %d with smaller value. %v", i, leftChild, msgAndArgs)
		} else { // isMaxLevel
			require.False(t, h.policy.Less(h.items[leftChild], h.items[i]),
				"max-level node %d has child %d with larger value. %v", i, leftChild, msgAndArgs)
		}
		verifyNode(t, h, leftChild, msgAndArgs...)
	}

	if rightChild < n {
		if isMinLevel {
			require.False(t, h.policy.Less(h.items[i], h.items[rightChild]),
				"min-level node %d has child %d with smaller value. %v", i, rightChild, msgAndArgs)
		} else { // isMaxLevel
			require.False(t, h.policy.Less(h.items[rightChild], h.items[i]),
				"max-level node %d has child %d with larger value. %v", i, rightChild, msgAndArgs)
		}
		verifyNode(t, h, rightChild, msgAndArgs...)
	}
}
