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

package statesync

import (
	"context"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func peer(ns, name, addr string) fwkdl.PeerMetadata {
	return fwkdl.PeerMetadata{ID: types.NamespacedName{Namespace: ns, Name: name}, Address: addr, Port: "9010"}
}

func TestMemoryPeerStore(t *testing.T) {
	ctx := context.Background()
	s := NewMemoryPeerStore()

	if got := s.Peers(); len(got) != 0 {
		t.Fatalf("empty store Peers() = %v, want empty", got)
	}

	b := peer("ns", "b", "10.0.0.2")
	a := peer("ns", "a", "10.0.0.1")
	s.PeerUpsert(ctx, &b)
	s.PeerUpsert(ctx, &a)

	// Peers is ordered by ID for deterministic consumption.
	want := []fwkdl.PeerMetadata{a, b}
	if got := s.Peers(); !cmp.Equal(got, want) {
		t.Errorf("Peers() = %v, want %v", got, want)
	}

	// Upsert replaces by ID.
	aUpdated := peer("ns", "a", "10.0.0.9")
	s.PeerUpsert(ctx, &aUpdated)
	if got := s.Peers(); !cmp.Equal(got, []fwkdl.PeerMetadata{aUpdated, b}) {
		t.Errorf("after update Peers() = %v, want a replaced", got)
	}

	s.PeerDelete(a.ID)
	if got := s.Peers(); !cmp.Equal(got, []fwkdl.PeerMetadata{b}) {
		t.Errorf("after delete Peers() = %v, want only b", got)
	}

	// Deleting an absent peer is a no-op.
	s.PeerDelete(types.NamespacedName{Namespace: "ns", Name: "missing"})
	if got := s.Peers(); len(got) != 1 {
		t.Errorf("after no-op delete Peers() = %v, want 1", got)
	}
}

// fakePeerDiscovery is a minimal PeerDiscovery plugin that pushes a fixed set
// of peers through the notifier during Start. See the "Writing a peer
// discovery plugin" section in docs/peer-discovery.md for the pattern this
// follows.
type fakePeerDiscovery struct {
	typedName fwkplugin.TypedName
	peers     []fwkdl.PeerMetadata
	ready     chan struct{}
	readyOnce sync.Once
}

var _ fwkdl.PeerDiscovery = (*fakePeerDiscovery)(nil)

func (d *fakePeerDiscovery) TypedName() fwkplugin.TypedName { return d.typedName }
func (d *fakePeerDiscovery) Ready() <-chan struct{}         { return d.ready }

func (d *fakePeerDiscovery) Start(ctx context.Context, notifier fwkdl.PeerNotifier) error {
	for i := range d.peers {
		notifier.Upsert(&d.peers[i])
	}
	d.readyOnce.Do(func() { close(d.ready) })
	<-ctx.Done()
	return nil
}

// TestPeerDiscoveryFullStack exercises the full PeerDiscovery -> PeerNotifier
// -> MemoryPeerStore pipeline with a fake plugin, verifying that peers flow
// through to the real store.
func TestPeerDiscoveryFullStack(t *testing.T) {
	store := NewMemoryPeerStore()
	disc := &fakePeerDiscovery{
		typedName: fwkplugin.TypedName{Type: "fake-peer-discovery", Name: "test"},
		peers: []fwkdl.PeerMetadata{
			{ID: types.NamespacedName{Name: "epp-1", Namespace: "ns"}, Address: "10.0.0.1", Port: "9002"},
			{ID: types.NamespacedName{Name: "epp-2", Namespace: "ns"}, Address: "10.0.0.2", Port: "9002"},
		},
		ready: make(chan struct{}),
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	notifier := fwkdl.NewPeerNotifier(store)
	errCh := make(chan error, 1)
	go func() { errCh <- disc.Start(ctx, notifier) }()

	<-disc.Ready()

	peers := store.Peers()
	require.Len(t, peers, 2)
	assert.Equal(t, "10.0.0.1", peers[0].Address)
	assert.Equal(t, "10.0.0.2", peers[1].Address)

	// Deleting a peer through the notifier removes it from the store.
	notifier.Delete(types.NamespacedName{Name: "epp-1", Namespace: "ns"})
	peers = store.Peers()
	require.Len(t, peers, 1)
	assert.Equal(t, "10.0.0.2", peers[0].Address)

	cancel()
	assert.NoError(t, <-errCh)
}
