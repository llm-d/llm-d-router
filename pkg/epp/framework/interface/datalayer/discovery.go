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

package datalayer

import (
	"context"

	"k8s.io/apimachinery/pkg/types"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// DiscoveryEndpointStore is the narrow interface required by NewDiscoveryNotifier.
// Any store that implements EndpointUpsert and EndpointDelete satisfies it.
type DiscoveryEndpointStore interface {
	EndpointUpsert(ctx context.Context, meta *EndpointMetadata)
	EndpointDelete(id types.NamespacedName)
}

// NewDiscoveryNotifier wraps a DiscoveryEndpointStore as a DiscoveryNotifier.
func NewDiscoveryNotifier(store DiscoveryEndpointStore) DiscoveryNotifier {
	return &discoveryNotifier{store: store}
}

type discoveryNotifier struct {
	store DiscoveryEndpointStore
}

func (n *discoveryNotifier) Upsert(endpoint *EndpointMetadata) {
	n.store.EndpointUpsert(context.Background(), endpoint)
}

func (n *discoveryNotifier) Delete(id types.NamespacedName) {
	n.store.EndpointDelete(id)
}

// EndpointDiscovery discovers inference endpoints and drives their lifecycle in the datastore.
// Implementations are registered in the plugin registry and selected via
// EndpointPickerConfig.discovery.pluginRef.
type EndpointDiscovery interface {
	fwkplugin.Plugin

	// Start begins discovery and blocks in the caller's goroutine until ctx is
	// cancelled or a fatal error occurs. It is the caller's responsibility to
	// invoke Start in a dedicated goroutine.
	//
	// Implementations SHOULD enumerate all currently known endpoints via
	// notifier.Upsert before entering the watch loop, to avoid serving an empty
	// datastore at startup. Implementations that guarantee no missed events
	// through their watch mechanism (e.g. a Kubernetes list+watch) may fold the
	// initial enumeration into the watch sequence instead.
	Start(ctx context.Context, notifier DiscoveryNotifier) error
}

// DiscoveryNotifier is the callback through which EndpointDiscovery communicates
// endpoint state to the datastore.
//
// DiscoveryNotifier is NOT goroutine-safe. All calls must be made sequentially
// from a single goroutine. This is the source of the ordering contract below.
//
// Ordering contract: the datastore processes Upsert and Delete calls in the order
// they are received. Plugin implementations MUST preserve event order -- do not
// buffer or dispatch calls concurrently in a way that could reorder them. For
// example, an Upsert followed by a Delete for the same endpoint must arrive in
// that order, or the endpoint will be incorrectly left in the datastore.
type DiscoveryNotifier interface {
	// Upsert adds or updates an endpoint in the datastore.
	Upsert(endpoint *EndpointMetadata)
	// Delete removes an endpoint from the datastore by its namespaced name.
	Delete(id types.NamespacedName)
}
