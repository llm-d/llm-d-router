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

package flowcontrol

// QueueInspectionMethods defines SafeQueue's read-only methods.
type QueueInspectionMethods interface {
	// Name returns a string identifier for the concrete queue implementation type (e.g., "PriorityQueue").
	Name() string

	// Len returns the current number of items in the queue.
	Len() int

	// ByteSize returns the current total byte size of all items in the queue.
	ByteSize() uint64

	// Peek returns the item at the "head" of the queue (the item with the highest priority according to the queue's
	// ordering) without removing it.
	// Returns nil if the queue is empty.
	Peek() QueueItemAccessor
}
