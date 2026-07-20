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

// CrossReplicaStore is a cross-EPP state store for sharing state across replicas.
// Implementations own the sync mechanism (e.g., Redis pub/sub, gossip) and the
// aggregation strategy (e.g., sum for in-flight load, union for cache state).
type CrossReplicaStore interface {
	fwkplugin.Plugin

	// Set writes a value for the given key and endpoint. The runtime calls
	// this after each Produce cycle with a fresh snapshot.
	Set(ctx context.Context, key StateKey, endpointID string, value any) error

	// Get returns the aggregated value for the given key and endpoint across
	// all replicas. Returns (value, true, nil) on hit, (nil, false, nil) on
	// miss, or (nil, false, err) on failure.
	Get(ctx context.Context, key StateKey, endpointID string) (any, bool, error)

	// Delete removes the value for the given key and endpoint.
	Delete(ctx context.Context, key StateKey, endpointID string) error
}

// CrossReplicaContributor is an opt-in interface for endpoint extractors that
// want their installed attributes to reflect cross-replica aggregate state.
// The plugin's Extract method is unchanged; the runtime detects this interface
// and wires the store transparently.
type CrossReplicaContributor interface {
	CrossReplicaState() CrossReplicaSpec
}

// CrossReplicaSpec declares what a CrossReplicaContributor publishes and where.
type CrossReplicaSpec struct {
	// StateKey namespaces this contributor's data in the store.
	StateKey StateKey

	// AttributeKey is the attribute map key the plugin installs in Extract.
	// The runtime overwrites this key with a store-reading closure.
	AttributeKey string

	// Supply returns a closure that reads the live local value for the given
	// endpoint. The runtime calls this closure after Produce to snapshot
	// the current local state and Set it into the store.
	Supply func(endpointID string) func() Cloneable
}
