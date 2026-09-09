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

package contracts

import (
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// FlowRegistry is the complete interface for the global flow control plane.
// It composes all role-based interfaces. A concrete implementation of this interface is the single source of truth for
// all flow control state.
//
// # Conformance: Implementations MUST be goroutine-safe.
//
// # Flow Lifecycle
//
// A flow instance, identified by its immutable FlowKey, has a lease-based lifecycle managed by this interface.
// Any implementation MUST adhere to this lifecycle:
//
//  1. Lease Acquisition: A client calls WithConnection to acquire a lease. This signals that the flow is in use and
//     protects it from garbage collection. If the flow instance does not exist, it is created on first use.
//     Priority bands must be provisioned by the control plane before use; they are not created on the request path.
//  2. Active State: A flow is "Active" as long as its lease count is greater than zero.
//  3. Lease Release: When the WithConnection callback returns, the lease is released.
//     When the lease count drops to zero, the flow becomes "Idle".
//  4. Garbage Collection: The implementation MUST automatically garbage collect a flow after it has remained
//     continuously Idle for a configurable duration.
type FlowRegistry interface {
	FlowRegistryDataPlane
	PriorityBandControlPlane
	FlowRegistryBackground
}

// PriorityBandControlPlane submits priority band topology updates from the control plane.
type PriorityBandControlPlane interface {
	// SubmitDesiredPriorities queues the set of priority levels that should remain provisioned.
	// Dynamic bands not in this set become eligible for removal once idle.
	SubmitDesiredPriorities(desired map[int]struct{})
}

// FlowRegistryBackground exposes hooks consumed by the Processor maintenance loop.
type FlowRegistryBackground interface {
	PriorityBandUpdateChannel() <-chan map[int]struct{}
	FlowGCTimeout() time.Duration
	ApplyDesiredPriorities(desired map[int]struct{})
	ExecuteGCCycle()
}

// FlowRegistryDataPlane defines the high-throughput, request-path interface for the registry.
type FlowRegistryDataPlane interface {
	// WithConnection manages a scoped, leased session for a given flow.
	// It is the primary and sole entry point for interacting with the data path.
	//
	// This method handles the entire lifecycle of a flow connection:
	// 1. Flow Registration: If the flow for the given FlowKey does not exist, it is created and registered
	//    automatically. Priority bands must be provisioned by the control plane before use.
	// 2. Lease Acquisition: It acquires a lifecycle lease, protecting the flow from garbage collection.
	// 3. Callback Execution: It invokes the provided function `fn`, passing in a temporary `ActiveFlowConnection` handle.
	// 4. Guaranteed Lease Release: It ensures the lease is safely released when the callback function returns.
	//
	// This functional, callback-based approach makes resource leaks impossible, as the caller is not responsible for
	// manually closing the connection.
	//
	// Errors returned by the callback `fn` are propagated up.
	// Returns `ErrFlowIDEmpty` if the provided key has an empty ID.
	WithConnection(key flowcontrol.FlowKey, fn func(conn ActiveFlowConnection) error) error

	// ManagedQueue retrieves the managed queue for the given, unique FlowKey. This is the primary method for accessing
	// a specific flow's queue for either enqueueing or dispatching requests.
	//
	// Returns an error wrapping ErrPriorityBandNotFound if the priority specified in the key is not configured, or
	// ErrFlowInstanceNotFound if no instance exists for the given key.
	ManagedQueue(key flowcontrol.FlowKey) (ManagedQueue, error)

	// FairnessPolicy retrieves the FairnessPolicy singleton configured for the specified priority band.
	// This method provides access to the immutable logic component that governs inter-flow contention.
	// The registry guarantees that a non-nil policy is returned for any active priority band.
	//
	// Returns:
	//   - FairnessPolicy: The active policy instance.
	//   - error: A wrapped ErrPriorityBandNotFound if the priority level is not configured.
	FairnessPolicy(priority int) (flowcontrol.FairnessPolicy, error)

	// PriorityBandAccessor retrieves the read-only view of the "Flow Group" for a specific priority level.
	// This accessor provides the state of all contending flows within the band and serves as the
	// primary input for FairnessPolicy execution.
	//
	// Returns an error wrapping ErrPriorityBandNotFound if the priority level is not configured.
	PriorityBandAccessor(priority int) (flowcontrol.PriorityBandAccessor, error)

	// AllOrderedPriorityLevels returns all configured priority levels, sorted in descending
	// numerical order. This order corresponds to highest priority (highest numeric value) to lowest priority (lowest
	// numeric value).
	// The returned slice provides a definitive, ordered list of priority levels for iteration, for example, by a
	// `controller.FlowController` worker's dispatch loop.
	//
	// Contract: the returned slice is a shared, immutable snapshot. Callers MUST NOT mutate it.
	// Implementations may publish it copy-on-write, so reads are lock-free and allocation-free.
	AllOrderedPriorityLevels() []int

	// CapacitySnapshot returns the current occupancy and configured capacity for the given priority band, together
	// with the registry-wide totals. It backs the admission-path capacity check, so implementations MUST keep it
	// cheap: no locks and no per-call aggregate snapshots. The snapshot is near-consistent: counters are read
	// individually and may not reflect a single instant.
	//
	// Returns an error wrapping ErrPriorityBandNotFound if the priority level is not configured.
	CapacitySnapshot(priority int) (CapacitySnapshot, error)
}

// CapacityDimension pairs current occupancy with its configured limits for one aggregation scope (a priority band or
// the registry-wide totals). A zero capacity means no limit is enforced on that dimension.
type CapacityDimension struct {
	// Len is the number of items currently queued in this scope.
	Len uint64
	// ByteSize is the total byte size of items currently queued in this scope.
	ByteSize uint64
	// CapacityRequests is the configured maximum total request count for this scope.
	CapacityRequests uint64
	// CapacityBytes is the configured maximum total byte size for this scope.
	CapacityBytes uint64
}

// CapacitySnapshot is a near-consistent view of queue occupancy against configured limits, for one priority band and
// the registry-wide aggregate.
type CapacitySnapshot struct {
	// Band holds the occupancy and limits of the priority band the snapshot was taken for.
	Band CapacityDimension
	// Global holds the registry-wide occupancy and limits.
	Global CapacityDimension
}

// ActiveFlowConnection represents a handle to a scoped, leased session on a flow.
// It provides a safe entry point to the registry's data plane.
//
// An `ActiveFlowConnection` instance is only valid for the duration of the `WithConnection` callback from which it was
// received. Callers MUST NOT store a reference to this object or use it after the callback returns.
//
// Lifecycle & Pinning:
// This interface represents an active "Lease" on the flow. As long as this object is valid (within the callback), the
// Flow Registry guarantees that the underlying Flow State is "Pinned" and protected from Garbage Collection.
type ActiveFlowConnection interface {
	// GetDataPlane returns the FlowRegistryDataPlane this connection is pinned to.
	GetDataPlane() FlowRegistryDataPlane
	// FlowKey returns the immutable identity of the flow this connection is pinned to.
	FlowKey() flowcontrol.FlowKey
}

// ManagedQueue defines the interface for a flow's queue.
// It acts as a stateful decorator that *use an underlying SafeQueue, augmenting it with statistics tracking, and
// lifecycle awareness.
//
// Conformance: Implementations MUST be goroutine-safe.
type ManagedQueue interface {
	// Add attempts to enqueue an item.
	Add(item flowcontrol.QueueItemAccessor) error

	// Remove atomically finds and removes an item from the underlying queue using its handle.
	Remove(handle flowcontrol.QueueItemHandle) (flowcontrol.QueueItemAccessor, error)

	// Cleanup removes all items from the underlying queue that satisfy the predicate.
	Cleanup(predicate PredicateFunc) []flowcontrol.QueueItemAccessor

	// Drain removes all items from the underlying queue.
	Drain() []flowcontrol.QueueItemAccessor

	// FlowQueueAccessor returns a read-only, flow-aware accessor for this queue, used by policy plugins.
	// Conformance: This method MUST NOT return nil.
	FlowQueueAccessor() flowcontrol.FlowQueueAccessor
}
