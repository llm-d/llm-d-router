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

// Package statesync holds the cross-replica state synchronization membership.
package statesync

import (
	"context"
	"sort"
	"sync"

	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// MemoryPeerStore is an in-memory PeerStore. Peer discovery writes the live peer
// set through PeerUpsert/PeerDelete; a CrossReplicaSyncer reads it via Peers.
type MemoryPeerStore struct {
	mu    sync.RWMutex
	peers map[types.NamespacedName]fwkdl.PeerMetadata
}

var _ fwkdl.PeerStore = (*MemoryPeerStore)(nil)

// NewMemoryPeerStore returns an empty MemoryPeerStore.
func NewMemoryPeerStore() *MemoryPeerStore {
	return &MemoryPeerStore{peers: map[types.NamespacedName]fwkdl.PeerMetadata{}}
}

// PeerUpsert adds or replaces the peer identified by meta.ID.
func (s *MemoryPeerStore) PeerUpsert(_ context.Context, meta *fwkdl.PeerMetadata) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.peers[meta.ID] = *meta
}

// PeerDelete removes the peer with the given ID. Deleting an absent peer is a
// no-op.
func (s *MemoryPeerStore) PeerDelete(id types.NamespacedName) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.peers, id)
}

// Peers returns a snapshot of the current peer set, ordered by ID for
// deterministic consumption.
func (s *MemoryPeerStore) Peers() []fwkdl.PeerMetadata {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]fwkdl.PeerMetadata, 0, len(s.peers))
	for _, p := range s.peers {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ID.Namespace != out[j].ID.Namespace {
			return out[i].ID.Namespace < out[j].ID.Namespace
		}
		return out[i].ID.Name < out[j].ID.Name
	})
	return out
}
