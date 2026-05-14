/*
Copyright 2026 The Kubernetes Authors.

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

package eviction

import (
	"sync"
	"sync/atomic"

	errcommon "github.com/llm-d/llm-d-inference-scheduler/pkg/common/error"
)

// evictionEntry holds the eviction channel and an optional reason for the eviction.
type evictionEntry struct {
	ch     chan struct{}
	reason atomic.Value // stores errcommon.RequestDroppedReason
}

// EvictionRegistry is a shared registry that maps request IDs to eviction channels.
// It bridges the RequestEvictor (which decides what to evict) and the ext_proc Process()
// goroutine (which owns the stream needed to send ImmediateResponse).
//
// Lifecycle:
//   - PreRequest: RequestEvictor creates an eviction channel and registers it via Register().
//   - Process(): after HandleRequest returns, looks up the channel via Get() and selects on it.
//   - EvictN: evictor closes the channel via the EvictionItem.EvictCh reference.
//   - Process() defer: removes the channel via Deregister().
//
// All methods are goroutine-safe.
type EvictionRegistry struct {
	entries sync.Map // requestID (string) → *evictionEntry
}

// NewEvictionRegistry creates a new EvictionRegistry.
func NewEvictionRegistry() *EvictionRegistry {
	return &EvictionRegistry{}
}

// Register stores an eviction channel for the given request ID.
func (r *EvictionRegistry) Register(requestID string, ch chan struct{}) {
	r.entries.Store(requestID, &evictionEntry{ch: ch})
}

// Get returns the eviction channel for the given request ID, or nil if not found.
func (r *EvictionRegistry) Get(requestID string) chan struct{} {
	v, ok := r.entries.Load(requestID)
	if !ok {
		return nil
	}
	return v.(*evictionEntry).ch
}

// SetReason records the eviction reason for a request before the channel is closed.
func (r *EvictionRegistry) SetReason(requestID string, reason errcommon.RequestDroppedReason) {
	v, ok := r.entries.Load(requestID)
	if !ok {
		return
	}
	v.(*evictionEntry).reason.Store(reason)
}

// GetReason returns the eviction reason for a request, or empty string if not found.
func (r *EvictionRegistry) GetReason(requestID string) errcommon.RequestDroppedReason {
	v, ok := r.entries.Load(requestID)
	if !ok {
		return ""
	}
	reason, _ := v.(*evictionEntry).reason.Load().(errcommon.RequestDroppedReason)
	return reason
}

// Deregister removes the eviction entry for the given request ID.
func (r *EvictionRegistry) Deregister(requestID string) {
	r.entries.Delete(requestID)
}
