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
	"sync/atomic"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/flowcontrol"
)

// priorityBand holds all managedQueues and configuration for a single priority level within a shard.
type priorityBand struct {
	// --- Immutable (set at construction) ---

	// fairnessPolicy is the singleton plugin instance governing this band.
	// It is duplicated here from the config to allow lock-free access on the hot path.
	fairnessPolicy flowcontrol.FairnessPolicy

	// policyState holds the opaque, mutable state for the fairness policy.
	// It is initialized once at creation via fairnessPolicy.NewState() and exposed via GetPolicyState().
	policyState any

	// --- State Protected by the parent shard's mu ---

	// config is the local copy of the band's definition.
	// It is updated during dynamic scaling events (updateConfig), protected by the parent shard's mutex.
	config PriorityBandConfig

	// queues holds all managedQueue instances within this band, keyed by their logical ID string.
	// The priority is implicit from the parent priorityBand.
	queues map[string]*managedQueue

	// priorityBandAccessor is a preallocated flowcontrol.PriorityBandAccessor for this priorityBand
	priorityBandAccessor *priorityBandAccessor

	// --- Concurrent-Safe State (Atomics) ---

	// Band-level statistics, updated via lock-free propagation from child queues.
	byteSize atomic.Int64
	len      atomic.Int64
}

// initPriorityBand constructs the runtime state for a single priority level and registers it within the shard.
// This is used by both newShard (initialization) and addPriorityBand (dynamic provisioning).
// The caller MUST hold fr.mu (Write Lock) as this method modifies the orderedPriorityLevels slice.
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
	fr.orderedPriorityLevels = append(fr.orderedPriorityLevels, bandConfig.Priority)
	sort.Slice(fr.orderedPriorityLevels, func(i, j int) bool {
		return fr.orderedPriorityLevels[i] > fr.orderedPriorityLevels[j]
	})
}

// addPriorityBand dynamically provisions a new priority band.
// It looks up the definition in fr.config, which must have been updated by the Registry via updateConfig/repartition.
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

// AllOrderedPriorityLevels returns a snapshot of all configured priority levels,
// sorted in descending order. The returned slice is a copy, safe for the caller to iterate
// without holding any lock.
func (fr *FlowRegistry) AllOrderedPriorityLevels() []int {
	fr.mu.RLock()
	defer fr.mu.RUnlock()
	result := make([]int, len(fr.orderedPriorityLevels))
	copy(result, fr.orderedPriorityLevels)
	return result
}

//  --- Internal Administrative/Lifecycle Methods ---

// synchronizeFlow is the internal administrative method for creating a flow instance.
// It is an idempotent "create if not exists" operation.
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

	fr.logger.V(logging.TRACE).Info("Creating new queue for flow instance.",
		"flowKey", key, "queueType", q.Name())

	mq := newManagedQueue(q, policy, key, fr.logger, fr.propagateStatsDelta)
	band.queues[key.ID] = mq
}

// deleteFlow removes a queue instance.
// Must be called with the registry write lock held
func (fr *FlowRegistry) deleteFlow(key flowcontrol.FlowKey) {
	fr.logger.V(logging.DEFAULT).Info("Deleting queue instance.", "flowKey", key)
	if val, ok := fr.priorityBands.Load(key.Priority); ok {
		band := val.(*priorityBand)
		delete(band.queues, key.ID)
	}
}

// --- `priorityBandAccessor` ---

// priorityBandAccessor implements PriorityBandAccessor.
// It provides a read-only, concurrent-safe view of a single priority band within a shard.
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

	flowKeys := make([]flowcontrol.FlowKey, len(ids))
	for i, id := range ids {
		flowKeys[i] = flowcontrol.FlowKey{ID: id, Priority: a.Priority()}
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

// IterateQueues executes the given `callback` for each FlowQueueAccessor in this priority band.
//
// To minimize lock contention, this implementation snapshots the queue accessors under a read lock and then executes
// the callback on the snapshot, outside of the lock. This ensures that a potentially slow policy (the callback) does
// not block other operations on the shard.
func (a *priorityBandAccessor) IterateQueues(callback func(queue flowcontrol.FlowQueueAccessor) bool) {
	a.registry.mu.RLock()
	accessors := make([]flowcontrol.FlowQueueAccessor, 0, len(a.band.queues))
	for _, mq := range a.band.queues {
		accessors = append(accessors, mq.FlowQueueAccessor())
	}
	a.registry.mu.RUnlock()

	for _, accessor := range accessors {
		if !callback(accessor) {
			return
		}
	}
}
