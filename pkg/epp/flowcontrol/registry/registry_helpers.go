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
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// priorityBand holds all managedQueues and configuration for a single priority level.
type priorityBand struct {
	// --- Immutable (set at construction) ---

	// fairnessPolicy is the singleton plugin instance governing this band.
	// It is duplicated here from the config to allow lock-free access on the hot path.
	fairnessPolicy flowcontrol.FairnessPolicy

	// policyState holds the opaque, mutable state for the fairness policy.
	// It is initialized once at creation via fairnessPolicy.NewState() and exposed via GetPolicyState().
	policyState any

	// config is the local copy of the band's definition. It is immutable after the band is published to the
	// registry; hot-path readers (CapacitySnapshot) rely on this to read capacities lock-free.
	config PriorityBandConfig

	// stats aggregates the occupancy of this band's queues (see occupancyStats).
	stats occupancyStats

	// --- State Protected by the registry's mu ---

	// queues holds all managedQueue instances within this band, keyed by their logical ID string.
	// The priority is implicit from the parent priorityBand.
	queues map[string]*managedQueue

	// priorityBandAccessor is a preallocated flowcontrol.PriorityBandAccessor for this priorityBand
	priorityBandAccessor *priorityBandAccessor

	// activeQueues indexes the subset of `queues` that currently hold items, keyed by logical ID
	// (values are *managedQueue). It is maintained by each queue's empty<->non-empty transitions
	// (serialized per queue under the queue's own mutex) and read lock-free by IterateQueues, which
	// keeps the dispatch hot path O(active flows) with zero allocation instead of O(registered
	// flows) with a snapshot. The view is eventually consistent: a queue is always present here by
	// the time an Add returns, but may linger briefly after draining; readers must tolerate
	// observing an empty queue.
	activeQueues sync.Map
}

// capacityDimension returns this band's current occupancy against its configured limits.
func (b *priorityBand) capacityDimension() contracts.CapacityDimension {
	return contracts.CapacityDimension{
		Len:              uint64(b.stats.len.Load()),
		ByteSize:         uint64(b.stats.byteSize.Load()),
		CapacityRequests: b.config.MaxRequests,
		CapacityBytes:    b.config.MaxBytes,
	}
}

// setQueueActivity is the onActiveTransition callback for this band's queues. It runs inside the
// queue's critical section, so it must remain lock-free (sync.Map only, never the registry mutex).
// applyAndPropagateLocked invokes it, and applies the stats delta, while the queue mutex is held.
func (b *priorityBand) setQueueActivity(mq *managedQueue, active bool) {
	if active {
		b.activeQueues.Store(mq.key.ID, mq)
	} else {
		// Deactivation must be conditional on the entry still belonging to this queue. A cleanup-sweep
		// worker can drain a queue through a handle resolved before deleteFlow removed it, and a
		// successor queue may have been registered under the same ID in the interim; an unconditional
		// delete would hide that live, non-empty successor from IterateQueues.
		b.activeQueues.CompareAndDelete(mq.key.ID, mq)
	}
}

// initPriorityBand constructs the runtime state for a single priority level and registers it within the registry.
// This is used during registry initialization and by addPriorityBand (dynamic provisioning).
// The caller MUST hold fr.mu (Write Lock) as this method republishes the orderedPriorityLevels slice.
func (fr *FlowRegistry) initPriorityBand(bandConfig *PriorityBandConfig) {
	policyState := bandConfig.FairnessPolicy.NewState(context.Background())
	band := &priorityBand{
		config:         *bandConfig,
		queues:         make(map[string]*managedQueue),
		fairnessPolicy: bandConfig.FairnessPolicy,
		policyState:    policyState,
	}
	band.priorityBandAccessor = &priorityBandAccessor{registry: fr, band: band}
	fr.priorityBands.Store(bandConfig.Priority, band)

	// Copy-on-write: the published slice is shared with lock-free readers and must not be mutated.
	current := *fr.orderedPriorityLevels.Load()
	updated := make([]int, 0, len(current)+1)
	updated = append(append(updated, current...), bandConfig.Priority)
	sort.Slice(updated, func(i, j int) bool { return updated[i] > updated[j] })
	fr.orderedPriorityLevels.Store(&updated)
}

// addPriorityBand dynamically provisions a new priority band.
// It looks up the definition in fr.config, which the caller must already have populated
// (see provisionPriorityBandLocked).
// addPriorityBand must be called with the registry mutex acquired for writing
func (fr *FlowRegistry) addPriorityBand(priority int) {
	// Idempotency check.
	if _, ok := fr.priorityBands.Load(priority); ok {
		return
	}

	bandConfig := fr.config.PriorityBands[priority]
	fr.initPriorityBand(bandConfig)
	fr.logger.V(logging.DEFAULT).Info("Dynamically added priority band", "priority", priority)
}

// ManagedQueue retrieves a specific `contracts.ManagedQueue` instance from the registry.
func (fr *FlowRegistry) ManagedQueue(key flowcontrol.FlowKey) (contracts.ManagedQueue, error) {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	val, ok := fr.priorityBands.Load(key.Priority)
	if !ok {
		return nil, fmt.Errorf("failed to get managed queue for flow %q: %w", key, contracts.ErrPriorityBandNotFound)
	}
	band := val.(*priorityBand)

	mq, ok := band.queues[key.ID]
	if !ok {
		return nil, fmt.Errorf("failed to get managed queue for flow %q: %w", key, contracts.ErrFlowInstanceNotFound)
	}
	return mq, nil
}

// FairnessPolicy retrieves a priority band's configured FairnessPolicy.
// This read is lock-free as the policy instance is immutable after the registry is initialized.
func (fr *FlowRegistry) FairnessPolicy(priority int) (flowcontrol.FairnessPolicy, error) {
	val, ok := fr.priorityBands.Load(priority)
	if !ok {
		return nil, fmt.Errorf("failed to get fairness policy for priority %d: %w",
			priority, contracts.ErrPriorityBandNotFound)
	}
	return val.(*priorityBand).fairnessPolicy, nil
}

// PriorityBandAccessor retrieves a read-only view for a given priority level.
func (fr *FlowRegistry) PriorityBandAccessor(priority int) (flowcontrol.PriorityBandAccessor, error) {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	val, ok := fr.priorityBands.Load(priority)
	if !ok {
		return nil, fmt.Errorf("failed to get priority band accessor for priority %d: %w",
			priority, contracts.ErrPriorityBandNotFound)
	}
	band := val.(*priorityBand)
	return band.priorityBandAccessor, nil
}

// AllOrderedPriorityLevels returns all configured priority levels, sorted in descending order.
// The returned slice is the shared, immutable published snapshot: callers MUST treat it as
// read-only. The read is a single atomic pointer load, with no lock and no allocation.
func (fr *FlowRegistry) AllOrderedPriorityLevels() []int {
	return *fr.orderedPriorityLevels.Load()
}

//  --- Internal Administrative/Lifecycle Methods ---

// synchronizeFlow is the internal administrative method for creating a flow instance.
// It is an idempotent "create if not exists" operation.
// The priorityBand of the request is guaranteed to exist during the call to synchronizeFlow
// by ensureFlowInfrastructure.
func (fr *FlowRegistry) synchronizeFlow(
	key flowcontrol.FlowKey,
	policy flowcontrol.OrderingPolicy,
	q contracts.SafeQueue,
) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	val, _ := fr.priorityBands.Load(key.Priority)
	band := val.(*priorityBand)
	if _, ok := band.queues[key.ID]; ok {
		return
	}

	fr.logger.V(logging.TRACE).Info("Creating new queue for flow instance.", "flowKey", key)

	mq := newManagedQueue(q, policy, key, fr.logger, &band.stats, &fr.totals, band.setQueueActivity)
	band.queues[key.ID] = mq
}

// deleteFlow removes a queue instance.
// Must be called with the registry write lock held
func (fr *FlowRegistry) deleteFlow(key flowcontrol.FlowKey) {
	fr.logger.V(logging.DEBUG).Info("Deleting queue instance.", "flowKey", key)
	if val, ok := fr.priorityBands.Load(key.Priority); ok {
		band := val.(*priorityBand)
		// Requests that are asynchronously finalized (e.g., due to client stream
		// cancellation or context timeout) are left in the queue for the cleanup sweep.
		// A queue deleted here may still hold such items, and the sweep may still hold a
		// ManagedQueue handle to it (handles are resolved before processing, without
		// registry locks). Draining through the wrapper both empties the queue and
		// deducts the stats in one critical section, so a later mutation through a stale
		// handle observes an empty queue and propagates nothing.
		if mq, ok := band.queues[key.ID]; ok && mq != nil {
			if mq.Len() > 0 {
				fr.logger.V(logging.DEBUG).Info("Deregistering non-empty queue during GC, draining unswept items",
					"flowKey", key, "unsweptCount", mq.Len())
				mq.Drain()
			}
		}
		delete(band.queues, key.ID)
		band.activeQueues.Delete(key.ID)
	}
}

// --- `priorityBandAccessor` ---

// priorityBandAccessor implements PriorityBandAccessor.
// It provides a read-only, concurrent-safe view of a single priority band.
type priorityBandAccessor struct {
	registry *FlowRegistry
	band     *priorityBand
}

var _ flowcontrol.PriorityBandAccessor = &priorityBandAccessor{}

// Priority returns the numerical priority level of this band.
func (a *priorityBandAccessor) Priority() int {
	a.registry.mu.RLock()
	defer a.registry.mu.RUnlock()
	return a.band.config.Priority
}

// PolicyState returns the opaque, mutable state for the fairness policy scoped to this band.
// We don't need a lock because the pointer to the state object itself is immutable.
func (a *priorityBandAccessor) PolicyState() any {
	return a.band.policyState
}

// FlowKeys returns a slice of all flow keys within this priority band.
//
// To minimize lock contention, this implementation first snapshots the flow IDs under a read lock and then constructs
// the final slice of `flowcontrol.FlowKey` structs outside of the lock.
func (a *priorityBandAccessor) FlowKeys() []flowcontrol.FlowKey {
	a.registry.mu.RLock()
	ids := make([]string, 0, len(a.band.queues))
	for id := range a.band.queues {
		ids = append(ids, id)
	}
	a.registry.mu.RUnlock()

	priority := a.band.config.Priority
	flowKeys := make([]flowcontrol.FlowKey, len(ids))
	for i, id := range ids {
		flowKeys[i] = flowcontrol.FlowKey{ID: id, Priority: priority}
	}
	return flowKeys
}

// Queue returns a FlowQueueAccessor for the specified logical `ID` within this priority band.
func (a *priorityBandAccessor) Queue(id string) flowcontrol.FlowQueueAccessor {
	a.registry.mu.RLock()
	defer a.registry.mu.RUnlock()

	mq, ok := a.band.queues[id]
	if !ok {
		return nil
	}
	return mq.FlowQueueAccessor()
}

// IterateQueues executes the given `callback` for each active (non-empty) FlowQueueAccessor in
// this priority band.
//
// It ranges over the band's lock-free active-queue index, so it takes no registry lock and
// performs no allocation, and its cost scales with the number of flows that currently hold items
// rather than the number of registered flows. The view is eventually consistent: a queue drained
// concurrently with iteration may still be visited, so callbacks must tolerate Len() == 0; a
// queue is guaranteed to be visible once the Add that made it non-empty has returned.
func (a *priorityBandAccessor) IterateQueues(callback func(queue flowcontrol.FlowQueueAccessor) bool) {
	a.band.activeQueues.Range(func(_, v any) bool {
		return callback(v.(*managedQueue).FlowQueueAccessor())
	})
}
