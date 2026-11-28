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

package crossreplica

import "sync"

// Dispatching counts, per priority band, the requests that left the queue and are not on an endpoint yet. Peers
// read them as band demand: until a request's endpoint load is published, the queue alone would show the band
// emptier than it is.
type Dispatching struct {
	mu     sync.Mutex
	byID   map[string]int
	counts map[int]uint64
}

// NewDispatching returns an empty counter.
func NewDispatching() *Dispatching {
	return &Dispatching{byID: map[string]int{}, counts: map[int]uint64{}}
}

// Add records a request dispatched from the priority band. A repeated request ID is ignored.
func (d *Dispatching) Add(requestID string, priority int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if _, ok := d.byID[requestID]; ok {
		return
	}
	d.byID[requestID] = priority
	d.counts[priority]++
}

// Remove forgets a request once its endpoint load is recorded. Unknown request IDs are ignored.
func (d *Dispatching) Remove(requestID string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	priority, ok := d.byID[requestID]
	if !ok {
		return
	}
	delete(d.byID, requestID)
	if d.counts[priority]--; d.counts[priority] == 0 {
		delete(d.counts, priority)
	}
}

// Count returns the requests of the priority band that are on their way to an endpoint.
func (d *Dispatching) Count(priority int) uint64 {
	if d == nil {
		return 0
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.counts[priority]
}
