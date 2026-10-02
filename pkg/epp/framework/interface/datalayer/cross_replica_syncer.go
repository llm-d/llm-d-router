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

package datalayer

import (
	"context"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// StateKey namespaces cross-EPP shared state.
type StateKey string

// BoundCrossReplicaState synchronizes one contributor's state for one endpoint.
type BoundCrossReplicaState interface {
	// Set publishes the live local value and refreshes the peer aggregate used by
	// Get.
	Set(ctx context.Context) error

	// Get reads the live local value and combines it with the prepared peer
	// aggregate.
	Get(ctx context.Context) (any, error)

	// Delete removes this replica's value.
	Delete(ctx context.Context) error
}

// CrossReplicaSyncer synchronizes shared state across EPP replicas.
// Implementations own the storage mechanism and must provide the atomic
// consistency required by GetOrSet.
type CrossReplicaSyncer interface {
	fwkplugin.Plugin

	// Bind creates a state handle that retains the contributor's key, endpoint,
	// live local value supplier, and aggregation function.
	Bind(key StateKey, endpointID string, local func() Cloneable, aggregate func([]any) any) BoundCrossReplicaState

	// GetOrSet atomically returns the value already stored for key and id, or
	// stores candidate and returns it. This is global request-level state shared
	// across EPP replicas, not per-endpoint state. Use it only when exact
	// coordination is required. Implementations own a fixed expiration period.
	// The bool reports whether the returned value already existed. Implementations
	// must provide linearizable behavior across every EPP replica sharing the syncer.
	GetOrSet(ctx context.Context, key StateKey, id string, candidate any) (actual any, existed bool, err error)
}

// CrossReplicaContributor is an opt-in interface for endpoint extractors that
// want their installed attributes to reflect cross-replica aggregate state.
// The plugin's Extract method is unchanged; the runtime detects this interface
// and wires the store transparently. Prefer it for per-endpoint state that can
// tolerate periodic synchronization.
type CrossReplicaContributor interface {
	CrossReplicaState() CrossReplicaSpec
}

// CrossReplicaSpec declares what a CrossReplicaContributor publishes and where.
type CrossReplicaSpec struct {
	// StateKey namespaces this contributor's data in the store.
	StateKey StateKey

	// AttributeKey is the attribute map key the plugin installs in Extract.
	// The runtime overwrites this key with a store-reading closure.
	AttributeKey fwkplugin.DataKey

	// Supply returns a closure that reads the live local value for the given
	// endpoint. The runtime uses it to publish snapshots and passes it to Get for
	// request-time reads.
	Supply func(endpointID string) func() Cloneable

	// Aggregate combines per-replica values into a single aggregate. Its output
	// must have the same type and may be passed back as an input when Get combines
	// the cached peer aggregate with the live local value.
	Aggregate func(values []any) any

	// SyncDisabled opts this contributor out of cross-replica synchronization
	// even though it implements CrossReplicaContributor. When true, the runtime
	// neither publishes the plugin's state nor installs the aggregate-reading
	// attribute, so the plugin's local value is used as-is. The zero value
	// (false) keeps synchronization enabled.
	SyncDisabled bool
}
