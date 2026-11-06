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

package runner

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// addOnlyManager serves setupPeerDiscovery, which calls Add and nothing else.
type addOnlyManager struct {
	manager.Manager
	added []manager.Runnable
}

func (m *addOnlyManager) Add(r manager.Runnable) error {
	m.added = append(m.added, r)
	return nil
}

func peersConfigRef(ref string) *configapiv1.EndpointPickerConfig {
	return &configapiv1.EndpointPickerConfig{
		DataLayer: &configapiv1.DataLayerConfig{
			Discovery: &configapiv1.DiscoveryConfig{
				Peers: &configapiv1.PeerDiscoveryConfig{PluginRef: ref},
			},
		},
	}
}

// stubPeerDiscovery records the notifier it is started with.
type stubPeerDiscovery struct {
	mu       sync.Mutex
	notifier fwkdl.PeerNotifier
	ready    chan struct{}
	started  chan struct{}
}

func newStubPeerDiscovery() *stubPeerDiscovery {
	return &stubPeerDiscovery{ready: make(chan struct{}), started: make(chan struct{})}
}

func (s *stubPeerDiscovery) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "stub-peer-discovery", Name: "stub"}
}

func (s *stubPeerDiscovery) Start(ctx context.Context, notifier fwkdl.PeerNotifier) error {
	s.mu.Lock()
	s.notifier = notifier
	s.mu.Unlock()
	close(s.started)
	close(s.ready)
	<-ctx.Done()
	return nil
}

func (s *stubPeerDiscovery) Ready() <-chan struct{} { return s.ready }

var _ fwkdl.PeerDiscovery = (*stubPeerDiscovery)(nil)

// peerConsumingSyncer consumes the peer set, standing in for a peer-to-peer
// syncer with no external store.
type peerConsumingSyncer struct {
	source fwkdl.PeerSource
}

func (s *peerConsumingSyncer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "p2p-syncer", Name: "p2p"}
}

func (s *peerConsumingSyncer) SetPeerSource(source fwkdl.PeerSource) { s.source = source }

var (
	_ fwkplugin.Plugin   = (*peerConsumingSyncer)(nil)
	_ fwkdl.PeerConsumer = (*peerConsumingSyncer)(nil)
)

// externalStoreSyncer keeps its own store, so it does not consume peers.
type externalStoreSyncer struct{}

func (s *externalStoreSyncer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "redis-syncer", Name: "redis"}
}

var _ fwkplugin.Plugin = (*externalStoreSyncer)(nil)

func runnerWithPeerPlugins(t *testing.T, disc fwkplugin.Plugin, syncer fwkplugin.Plugin) *Runner {
	t.Helper()
	h := fwkplugin.NewEppHandle(context.Background(), nil)
	h.AddPlugin("my-peers", disc)
	if syncer != nil {
		h.SetCrossReplicaSyncer(syncer)
	}
	return &Runner{PluginHandle: h}
}

// The peer set a syncer receives is the one discovery writes to, so peers
// discovered after startup are visible through it.
func TestSetupPeerDiscoveryConnectsStoreToSyncer(t *testing.T) {
	disc := newStubPeerDiscovery()
	syncer := &peerConsumingSyncer{}
	r := runnerWithPeerPlugins(t, disc, syncer)
	mgr := &addOnlyManager{}

	require.NoError(t, r.setupPeerDiscovery(mgr, peersConfigRef("my-peers")))
	require.Len(t, mgr.added, 1, "discovery runs as a manager runnable")
	require.NotNil(t, syncer.source, "the syncer was given the peer set")
	assert.Empty(t, syncer.source.Peers())

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- mgr.added[0].Start(ctx) }()
	<-disc.started

	disc.mu.Lock()
	notifier := disc.notifier
	disc.mu.Unlock()
	require.NotNil(t, notifier)
	notifier.Upsert(&fwkdl.PeerMetadata{
		ID:      types.NamespacedName{Namespace: "ns", Name: "epp-1"},
		Address: "10.0.0.1",
		Port:    "9002",
	})

	peers := syncer.source.Peers()
	require.Len(t, peers, 1, "discovery and the syncer share one store")
	assert.Equal(t, "10.0.0.1", peers[0].Address)

	cancel()
	assert.NoError(t, <-errCh)
}

// A syncer backed by its own store is left alone, and discovery still runs.
func TestSetupPeerDiscoveryLeavesNonConsumingSyncer(t *testing.T) {
	r := runnerWithPeerPlugins(t, newStubPeerDiscovery(), &externalStoreSyncer{})
	mgr := &addOnlyManager{}

	require.NoError(t, r.setupPeerDiscovery(mgr, peersConfigRef("my-peers")))
	assert.Len(t, mgr.added, 1)
	assert.False(t, r.connectPeerStore(nil), "a syncer that keeps its own store takes no peer source")
}

// Peer discovery without any configured syncer starts and stays unconsumed.
func TestSetupPeerDiscoveryWithoutSyncer(t *testing.T) {
	r := runnerWithPeerPlugins(t, newStubPeerDiscovery(), nil)
	mgr := &addOnlyManager{}

	require.NoError(t, r.setupPeerDiscovery(mgr, peersConfigRef("my-peers")))
	assert.Len(t, mgr.added, 1)
	assert.False(t, r.connectPeerStore(nil))
}

// Without dataLayer.discovery.peers nothing is registered.
func TestSetupPeerDiscoveryDisabled(t *testing.T) {
	r := runnerWithPeerPlugins(t, newStubPeerDiscovery(), &peerConsumingSyncer{})
	mgr := &addOnlyManager{}

	require.NoError(t, r.setupPeerDiscovery(mgr, &configapiv1.EndpointPickerConfig{}))
	assert.Empty(t, mgr.added)
}

func TestSetupPeerDiscoveryPluginRefNotFound(t *testing.T) {
	r := runnerWithPeerPlugins(t, newStubPeerDiscovery(), nil)
	err := r.setupPeerDiscovery(&addOnlyManager{}, peersConfigRef("missing"))
	assert.ErrorContains(t, err, "missing")
}

func TestSetupPeerDiscoveryNotPeerDiscovery(t *testing.T) {
	r := runnerWithPeerPlugins(t, &externalStoreSyncer{}, nil)
	err := r.setupPeerDiscovery(&addOnlyManager{}, peersConfigRef("my-peers"))
	assert.ErrorContains(t, err, "PeerDiscovery")
}
