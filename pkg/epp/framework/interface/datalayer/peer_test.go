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
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// fakePeerDiscovery is a minimal PeerDiscovery implementation for testing.
// It pushes the provided peers through the notifier during Start, signals
// Ready, then blocks until the context is cancelled.
type fakePeerDiscovery struct {
	typedName fwkplugin.TypedName
	peers     []PeerMetadata
	ready     chan struct{}
	readyOnce sync.Once
}

var _ PeerDiscovery = (*fakePeerDiscovery)(nil)

func (d *fakePeerDiscovery) TypedName() fwkplugin.TypedName { return d.typedName }
func (d *fakePeerDiscovery) Ready() <-chan struct{}         { return d.ready }

func (d *fakePeerDiscovery) Start(ctx context.Context, notifier PeerNotifier) error {
	for i := range d.peers {
		notifier.Upsert(&d.peers[i])
	}
	d.readyOnce.Do(func() { close(d.ready) })
	<-ctx.Done()
	return nil
}

// TestPeerDiscovery_EndToEnd verifies the PeerDiscovery -> PeerNotifier ->
// PeerStore flow using a fake plugin and a recording store.
func TestPeerDiscovery_EndToEnd(t *testing.T) {
	store := &fakePeerStore{}
	disc := &fakePeerDiscovery{
		typedName: fwkplugin.TypedName{Type: "fake-peer-discovery", Name: "test"},
		peers: []PeerMetadata{
			{ID: types.NamespacedName{Name: "epp-1", Namespace: "ns"}, Address: "10.0.0.1", Port: "9002"},
			{ID: types.NamespacedName{Name: "epp-2", Namespace: "ns"}, Address: "10.0.0.2", Port: "9002"},
		},
		ready: make(chan struct{}),
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	notifier := NewPeerNotifier(store)
	errCh := make(chan error, 1)
	go func() { errCh <- disc.Start(ctx, notifier) }()

	<-disc.Ready()

	require.Len(t, store.upserted, 2)
	assert.Equal(t, "10.0.0.1", store.upserted[0].Address)
	assert.Equal(t, "10.0.0.2", store.upserted[1].Address)

	// Delete a peer through the notifier and verify the store receives it.
	notifier.Delete(types.NamespacedName{Name: "epp-1", Namespace: "ns"})
	require.Len(t, store.deleted, 1)
	assert.Equal(t, "epp-1", store.deleted[0].Name)

	cancel()
	assert.NoError(t, <-errCh)
}

func TestPeerNotifier_Upsert(t *testing.T) {
	store := &fakePeerStore{}
	notifier := NewPeerNotifier(store)
	meta := &PeerMetadata{
		ID:      types.NamespacedName{Name: "epp-1", Namespace: "ns"},
		Address: "10.0.0.1",
	}

	notifier.Upsert(meta)

	assert.Len(t, store.upserted, 1)
	assert.Equal(t, meta, store.upserted[0])
	assert.Empty(t, store.deleted)
}

func TestPeerNotifier_Delete(t *testing.T) {
	store := &fakePeerStore{}
	notifier := NewPeerNotifier(store)
	id := types.NamespacedName{Name: "epp-1", Namespace: "ns"}

	notifier.Delete(id)

	assert.Empty(t, store.upserted)
	assert.Len(t, store.deleted, 1)
	assert.Equal(t, id, store.deleted[0])
}

// fakePeerStore records calls for assertion.
type fakePeerStore struct {
	upserted []*PeerMetadata
	deleted  []types.NamespacedName
}

func (f *fakePeerStore) PeerUpsert(_ context.Context, meta *PeerMetadata) {
	f.upserted = append(f.upserted, meta)
}

func (f *fakePeerStore) PeerDelete(id types.NamespacedName) {
	f.deleted = append(f.deleted, id)
}
