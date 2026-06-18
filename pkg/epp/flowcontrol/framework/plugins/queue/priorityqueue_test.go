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
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
)

// TestPriorityQueue_InternalProperty validates that the heap property is maintained after a series
// of Add and Remove operations. This is a white-box test to ensure the internal data structure
// is always in a valid state.
func TestPriorityQueue_InternalProperty(t *testing.T) {
	t.Parallel()
	q := newPriorityQueue(enqueueTimePolicy)

	items := make([]*mocks.MockQueueItemAccessor, 20)
	now := time.Now()
	for i := range items {
		// Add items in a somewhat random order of enqueue times.
		items[i] = mocks.NewMockQueueItemAccessor(10, "item", flowcontrol.FlowKey{ID: "flow"})
		items[i].EnqueueTimeV = now.Add(time.Duration((i%5-2)*10) * time.Second)
		q.Add(items[i])
		assertHeapProperty(t, q, "after adding item %d", i)
	}

	// Remove a few items from the middle and validate the heap property.
	for _, i := range []int{15, 7, 11} {
		handle := items[i].Handle()
		_, err := q.Remove(handle)
		require.NoError(t, err, "Remove should not fail for item %d", i)
		assertHeapProperty(t, q, "after removing item %d", i)
	}

	// Remove remaining items from the head and validate each time.
	for q.Len() > 0 {
		head := q.Peek()
		require.NotNil(t, head)
		_, err := q.Remove(head.Handle())
		require.NoError(t, err)
		assertHeapProperty(t, q, "after removing head item")
	}
}

// assertHeapProperty checks that the slice of items satisfies the (max-by-policy) heap property:
// no child may outrank its parent, and every item's tracked index must match its slice position.
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
			// have higher priority than its parent.
			require.False(t, q.heap.policy.Less(items[child].item, items[i].item),
				"child %d must not outrank parent %d. %v", child, i, msgAndArgs)
		}
	}
}
