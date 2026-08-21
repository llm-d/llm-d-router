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

	"k8s.io/apimachinery/pkg/types"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// PeerMetadata identifies a peer EPP replica participating in cross-replica
// state synchronization.
type PeerMetadata struct {
	// ID is the stable identity of the peer, unique across the peer set.
	ID types.NamespacedName
	// Address is the IP address of the peer replica.
	Address string
	// Port is the port the peer listens on for state sync, as a string. Empty
	// when the discovery source does not expose a port.
	Port string
}

// PeerStore is the narrow interface required by NewPeerNotifier. Any store that
// implements PeerUpsert and PeerDelete satisfies it.
//
// PeerStore is intentionally separate from Datastore even if they have a similar
// shape: endpoints and peers have different consumers (scheduler vs. cross-replica syncer)
// and different lifecycles (bound to the InferencePool vs. the EPP deployment).
type PeerStore interface {
	PeerUpsert(ctx context.Context, meta *PeerMetadata)
	PeerDelete(id types.NamespacedName)
}

// NewPeerNotifier wraps a PeerStore as a PeerNotifier.
func NewPeerNotifier(store PeerStore) PeerNotifier {
	return &peerNotifier{store: store}
}

type peerNotifier struct {
	store PeerStore
}

func (n *peerNotifier) Upsert(peer *PeerMetadata) {
	n.store.PeerUpsert(context.Background(), peer)
}

func (n *peerNotifier) Delete(id types.NamespacedName) {
	n.store.PeerDelete(id)
}

// PeerDiscovery discovers peer EPP replicas and drives their lifecycle through
// a PeerNotifier. Implementations are registered in the plugin registry and
// selected via EndpointPickerConfig.dataLayer.peerDiscovery.pluginRef.
type PeerDiscovery interface {
	fwkplugin.Plugin

	// Start begins discovery and blocks until ctx is cancelled or a fatal
	// error occurs. The caller invokes Start in a dedicated goroutine.
	Start(ctx context.Context, notifier PeerNotifier) error

	// Ready returns a channel that is closed once after the plugin has
	// completed its initial reconciliation with the underlying source.
	// Callers use it to gate components that depend on a populated peer set.
	Ready() <-chan struct{}
}

// PeerNotifier is the callback through which peer discovery communicates the set
// of live EPP replicas to consumers (e.g. a CrossReplicaSyncer). It mirrors
// DiscoveryNotifier: the k8s reconciler and any file-based discovery feed the
// same sink.
//
// PeerNotifier is NOT goroutine-safe. All calls must be made sequentially from a
// single goroutine so that an Upsert followed by a Delete for the same peer is
// applied in that order.
type PeerNotifier interface {
	// Upsert adds or updates a peer.
	Upsert(peer *PeerMetadata)
	// Delete removes a peer by its identity.
	Delete(id types.NamespacedName)
}
