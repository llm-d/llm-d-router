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

package registry

import (
	"sync"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// managedQueue implements `contracts.ManagedQueue`. It decorates a SafeQueue with registry
// integration: aggregate statistics propagation and a read-only policy-facing accessor.
//
// # Statistics Ownership
//
// Per-queue statistics (`Len`, `ByteSize`) have a single owner: the underlying SafeQueue. This
// wrapper does not keep its own copy; reads delegate to the queue. Registry aggregates are updated
// with deltas measured from the queue's reported stats before and after each mutation, all within
// the same critical section, so the aggregates are sums of true per-queue histories and cannot go
// negative. There is no duplicated state and therefore no cross-copy invariant to enforce.
//
// # Invariant Protection
//
// To keep aggregates consistent with queue state, the following must be upheld:
//  1. Exclusive Access: All mutations on the underlying `SafeQueue` MUST be performed exclusively through this wrapper.
//  2. Non-Autonomous State: The underlying queue must not change state autonomously (e.g., no internal TTL eviction).
type managedQueue struct {
	// --- Immutable Identity & Dependencies (set at construction) ---
	key    flowcontrol.FlowKey
	policy flowcontrol.OrderingPolicy
	logger logr.Logger

	// onStatsDelta is the callback used to propagate statistics changes up to the registry.
	onStatsDelta propagateStatsDeltaFunc

	// --- State Protected by `mu` ---

	// mu protects all mutating operations. It ensures that the mutation of the underlying `queue`
	// and the propagation of the measured statistics delta occur as a single, atomic transaction.
	mu sync.Mutex
	// queue is the underlying, concurrency-safe queue implementation that this `managedQueue` decorates.
	// Its state must only be modified while holding `mu`.
	queue contracts.SafeQueue

	flowQueueAccessor *flowQueueAccessor
}

var _ contracts.ManagedQueue = &managedQueue{}

// newManagedQueue creates a new instance of a `managedQueue`.
func newManagedQueue(
	queue contracts.SafeQueue,
	policy flowcontrol.OrderingPolicy,
	key flowcontrol.FlowKey,
	logger logr.Logger,
	onStatsDelta propagateStatsDeltaFunc,
) *managedQueue {
	mqLogger := logger.WithName("managed-queue").WithValues("flowKey", key)
	mq := &managedQueue{
		queue:        queue,
		policy:       policy,
		key:          key,
		onStatsDelta: onStatsDelta,
		logger:       mqLogger,
	}
	mq.flowQueueAccessor = &flowQueueAccessor{mq: mq}
	return mq
}

// FlowQueueAccessor returns a read-only, flow-aware view of this queue.
func (mq *managedQueue) FlowQueueAccessor() flowcontrol.FlowQueueAccessor {
	return mq.flowQueueAccessor
}

// Add enqueues an item into the underlying queue and propagates the measured statistics delta.
func (mq *managedQueue) Add(item flowcontrol.QueueItemAccessor) error {
	mq.mu.Lock()
	defer mq.mu.Unlock()

	mq.applyAndPropagateLocked(func() { mq.queue.Add(item) })
	mq.logger.V(logging.TRACE).Info("Request added to queue", "requestID", item.OriginalRequest().ID())
	return nil
}

// Remove wraps the underlying SafeQueue.Remove and propagates the measured statistics delta.
func (mq *managedQueue) Remove(handle flowcontrol.QueueItemHandle) (flowcontrol.QueueItemAccessor, error) {
	mq.mu.Lock()
	defer mq.mu.Unlock()

	var removedItem flowcontrol.QueueItemAccessor
	var err error
	mq.applyAndPropagateLocked(func() { removedItem, err = mq.queue.Remove(handle) })
	if err != nil {
		return nil, err
	}
	mq.logger.V(logging.TRACE).Info("Request removed from queue", "requestID", removedItem.OriginalRequest().ID())
	return removedItem, nil
}

// Cleanup wraps the underlying SafeQueue.Cleanup and propagates the measured statistics delta.
func (mq *managedQueue) Cleanup(predicate contracts.PredicateFunc) []flowcontrol.QueueItemAccessor {
	mq.mu.Lock()
	defer mq.mu.Unlock()

	var cleanedItems []flowcontrol.QueueItemAccessor
	mq.applyAndPropagateLocked(func() { cleanedItems = mq.queue.Cleanup(predicate) })
	if len(cleanedItems) == 0 {
		return nil
	}
	if v := mq.logger.V(logging.DEBUG); v.Enabled() {
		reqIDs := make([]string, 0, len(cleanedItems))
		for _, item := range cleanedItems {
			if req := item.OriginalRequest(); req != nil {
				reqIDs = append(reqIDs, req.ID())
			}
		}
		v.Info("Cleaned up queue", "removedItemCount", len(cleanedItems), "requestIDs", reqIDs)
	} else {
		mq.logger.V(logging.DEBUG).Info("Cleaned up queue", "removedItemCount", len(cleanedItems))
	}
	return cleanedItems
}

// Drain wraps the underlying SafeQueue.Drain and propagates the measured statistics delta.
func (mq *managedQueue) Drain() []flowcontrol.QueueItemAccessor {
	mq.mu.Lock()
	defer mq.mu.Unlock()

	var drainedItems []flowcontrol.QueueItemAccessor
	mq.applyAndPropagateLocked(func() { drainedItems = mq.queue.Drain() })
	if len(drainedItems) == 0 {
		return nil
	}
	if v := mq.logger.V(logging.DEBUG); v.Enabled() {
		reqIDs := make([]string, 0, len(drainedItems))
		for _, item := range drainedItems {
			if req := item.OriginalRequest(); req != nil {
				reqIDs = append(reqIDs, req.ID())
			}
		}
		v.Info("Drained queue", "itemCount", len(drainedItems), "requestIDs", reqIDs)
	} else {
		mq.logger.V(logging.DEBUG).Info("Drained queue", "itemCount", len(drainedItems))
	}
	return drainedItems
}

// Len returns the current number of items in the queue.
func (mq *managedQueue) Len() int {
	return mq.queue.Len()
}

// ByteSize returns the current total byte size of all items in the queue.
func (mq *managedQueue) ByteSize() uint64 {
	return mq.queue.ByteSize()
}

// applyAndPropagateLocked runs mutate on the underlying queue and propagates the statistics delta
// measured across it, keeping the registry aggregates consistent with the queue's actual state.
// It must be called while holding the `managedQueue.mu` lock.
func (mq *managedQueue) applyAndPropagateLocked(mutate func()) {
	beforeLen := mq.queue.Len()
	beforeBytes := mq.queue.ByteSize()

	mutate()

	lenDelta := int64(mq.queue.Len() - beforeLen)
	byteSizeDelta := int64(mq.queue.ByteSize()) - int64(beforeBytes)
	if lenDelta == 0 && byteSizeDelta == 0 {
		return
	}
	// Propagate the delta up to the registry. This propagation is lock-free and eventually consistent.
	mq.onStatsDelta(mq.key.Priority, lenDelta, byteSizeDelta)
}

// --- `flowQueueAccessor` ---

// flowQueueAccessor implements FlowQueueAccessor. It provides a read-only, policy-facing view.
//
// # Role: The Read-Only Proxy
//
// This wrapper protects system invariants. It acts as a proxy that exposes only the read-only methods.
// This prevents policy plugins from using type assertions to access the concrete `*managedQueue` and calling mutation
// methods, which would bypass statistics tracking.
type flowQueueAccessor struct {
	mq *managedQueue
}

var _ flowcontrol.FlowQueueAccessor = &flowQueueAccessor{}

// --- Read-only pass-through methods to the underlying SafeQueue ---
func (a *flowQueueAccessor) Peek() flowcontrol.QueueItemAccessor { return a.mq.queue.Peek() }

// --- Read-only methods from the managedQueue wrapper ---
func (a *flowQueueAccessor) Len() int                                   { return a.mq.Len() }
func (a *flowQueueAccessor) ByteSize() uint64                           { return a.mq.ByteSize() }
func (a *flowQueueAccessor) OrderingPolicy() flowcontrol.OrderingPolicy { return a.mq.policy }
func (a *flowQueueAccessor) FlowKey() flowcontrol.FlowKey               { return a.mq.key }
