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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

type fakeCloneable struct{ id string }

func (f fakeCloneable) Clone() fwkdl.Cloneable { return f }

type setCall struct {
	key        fwkdl.StateKey
	endpointID string
	value      any
}

type fakeSyncer struct {
	mu   sync.Mutex
	sets []setCall
}

func (s *fakeSyncer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-syncer", Name: "fake-syncer"}
}

func (s *fakeSyncer) Set(_ context.Context, key fwkdl.StateKey, endpointID string, value any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sets = append(s.sets, setCall{key: key, endpointID: endpointID, value: value})
	return nil
}

func (s *fakeSyncer) Get(context.Context, fwkdl.StateKey, string, func([]any) any) (any, bool, error) {
	return nil, false, nil
}

func (s *fakeSyncer) Delete(context.Context, fwkdl.StateKey, string) error { return nil }

// fakeContributor is a Plugin + CrossReplicaContributor whose supplied value
// echoes the endpoint ID, so tests can assert routing to the right key.
type fakeContributor struct {
	key          fwkdl.StateKey
	syncDisabled bool
}

type fakeEndpointContributor struct {
	fakeContributor
}

func (fakeEndpointContributor) Extract(context.Context, fwkdl.EndpointEvent) error {
	return nil
}

type blockingSyncer struct {
	mu         sync.Mutex
	setStarted chan struct{}
	allowSet   chan struct{}
	startOnce  sync.Once
	state      map[string]any
	events     []string
}

type deadlineSyncer struct {
	fakeSyncer
	deadlineObserved chan bool
}

func (s *deadlineSyncer) Set(ctx context.Context, key fwkdl.StateKey, endpointID string, value any) error {
	_, ok := ctx.Deadline()
	select {
	case s.deadlineObserved <- ok:
	default:
	}
	return s.fakeSyncer.Set(ctx, key, endpointID, value)
}

func newBlockingSyncer() *blockingSyncer {
	return &blockingSyncer{
		setStarted: make(chan struct{}),
		allowSet:   make(chan struct{}),
		state:      make(map[string]any),
	}
}

func (s *blockingSyncer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "blocking-syncer", Name: "blocking-syncer"}
}

func (s *blockingSyncer) Set(_ context.Context, _ fwkdl.StateKey, endpointID string, value any) error {
	s.startOnce.Do(func() { close(s.setStarted) })
	<-s.allowSet
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state[endpointID] = value
	s.events = append(s.events, "set")
	return nil
}

func (s *blockingSyncer) Get(_ context.Context, _ fwkdl.StateKey, endpointID string, _ func([]any) any) (any, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	value, ok := s.state[endpointID]
	return value, ok, nil
}

func (s *blockingSyncer) Delete(_ context.Context, _ fwkdl.StateKey, endpointID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.state, endpointID)
	s.events = append(s.events, "delete")
	return nil
}

func (c fakeContributor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-contributor", Name: string(c.key)}
}

func (c fakeContributor) CrossReplicaState() fwkdl.CrossReplicaSpec {
	return fwkdl.CrossReplicaSpec{
		StateKey:     c.key,
		SyncDisabled: c.syncDisabled,
		Supply: func(id string) func() fwkdl.Cloneable {
			return func() fwkdl.Cloneable { return fakeCloneable{id: id} }
		},
	}
}

func testEndpoint(name string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Namespace: "ns", Name: name},
	}, nil)
}

func extractorMapWith(contributors ...fwkdl.CrossReplicaContributor) *extractorMap {
	em := newExtractorMap()
	for _, c := range contributors {
		em.Append("src", c.(fwkplugin.Plugin))
	}
	return em
}

func TestCrossReplicaPublisher_PublishesForEndpoint(t *testing.T) {
	syncer := &fakeSyncer{}
	pub := &crossReplicaPublisher{
		syncer:       syncer,
		contributors: []fwkdl.CrossReplicaContributor{fakeContributor{key: "inflight:test"}},
	}

	pub.publish(context.Background(), "ns/ep-a")

	require.Len(t, syncer.sets, 1)
	assert.Equal(t, fwkdl.StateKey("inflight:test"), syncer.sets[0].key)
	assert.Equal(t, "ns/ep-a", syncer.sets[0].endpointID)
	assert.Equal(t, fakeCloneable{id: "ns/ep-a"}, syncer.sets[0].value)
}

func TestCrossReplicaPublisher_SkipsSyncDisabled(t *testing.T) {
	em := extractorMapWith(
		fakeContributor{key: "enabled"},
		fakeContributor{key: "disabled", syncDisabled: true},
	)

	pub := newCrossReplicaPublisher(&fakeSyncer{}, em, 0)
	require.NotNil(t, pub)
	require.Len(t, pub.contributors, 1)
	assert.Equal(t, fwkdl.StateKey("enabled"), pub.contributors[0].CrossReplicaState().StateKey)
	assert.Equal(t, defaultCrossReplicaSyncInterval, pub.Interval(), "zero interval falls back to default")
}

func TestCrossReplicaPublisher_ConfiguredInterval(t *testing.T) {
	em := extractorMapWith(fakeContributor{key: "enabled"})
	pub := newCrossReplicaPublisher(&fakeSyncer{}, em, 500*time.Millisecond)
	require.NotNil(t, pub)
	assert.Equal(t, 500*time.Millisecond, pub.Interval())
}

// endpointIDs returns the distinct endpoint IDs the syncer has seen.
func (s *fakeSyncer) endpointIDs() map[string]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]int{}
	for _, c := range s.sets {
		out[c.endpointID]++
	}
	return out
}

// One loop serves every registered endpoint, rather than one goroutine each.
func TestCrossReplicaPublisher_PublishesAllEndpoints(t *testing.T) {
	syncer := &fakeSyncer{}
	r := NewRuntime(time.Second)
	r.crossReplicaPub = &crossReplicaPublisher{
		syncer:       syncer,
		contributors: []fwkdl.CrossReplicaContributor{fakeContributor{key: "inflight:test"}},
		interval:     time.Millisecond,
	}
	for _, name := range []string{"ep-a", "ep-b", "ep-c"} {
		ep := testEndpoint(name)
		require.NotNil(t, r.NewEndpoint(context.Background(), ep.GetMetadata()))
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r.crossReplicaPub.Start(ctx)

	require.Eventually(t, func() bool {
		ids := syncer.endpointIDs()
		return ids["ns/ep-a"] > 0 && ids["ns/ep-b"] > 0 && ids["ns/ep-c"] > 0
	}, 2*time.Second, 5*time.Millisecond, "every registered endpoint should be published")
}

func TestCrossReplicaPublisher_SetsPerPublishDeadline(t *testing.T) {
	syncer := &deadlineSyncer{deadlineObserved: make(chan bool, 1)}
	r := NewRuntime(time.Second)
	r.crossReplicaPub = &crossReplicaPublisher{
		syncer:       syncer,
		contributors: []fwkdl.CrossReplicaContributor{fakeContributor{key: "inflight:test"}},
		interval:     time.Millisecond,
	}
	ep := testEndpoint("ep-a")
	require.True(t, r.crossReplicaPub.RegisterEndpoint(ep.GetMetadata().GetID()))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r.crossReplicaPub.Start(ctx)

	select {
	case observed := <-syncer.deadlineObserved:
		assert.True(t, observed, "cross-replica publish context must have a deadline")
	case <-time.After(time.Second):
		t.Fatal("cross-replica publish did not run")
	}
}

// Releasing an endpoint stops its publishing without tearing down a goroutine,
// so removed endpoints cannot keep writing stale state.
func TestCrossReplicaPublisher_StopsAfterUnregister(t *testing.T) {
	syncer := &fakeSyncer{}
	r := NewRuntime(time.Second)
	r.crossReplicaPub = &crossReplicaPublisher{
		syncer:       syncer,
		contributors: []fwkdl.CrossReplicaContributor{fakeContributor{key: "inflight:test"}},
		interval:     time.Millisecond,
	}
	ep := testEndpoint("ep-gone")
	require.True(t, r.crossReplicaPub.RegisterEndpoint(ep.GetMetadata().GetID()))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r.crossReplicaPub.Start(ctx)

	require.Eventually(t, func() bool {
		return syncer.endpointIDs()["ns/ep-gone"] > 0
	}, 2*time.Second, 5*time.Millisecond, "endpoint should publish while registered")

	r.crossReplicaPub.UnregisterEndpoint(ep.GetMetadata().GetID(), nil)
	// Let any tick already in flight drain before sampling the baseline.
	time.Sleep(50 * time.Millisecond)
	baseline := syncer.endpointIDs()["ns/ep-gone"]

	time.Sleep(100 * time.Millisecond)
	assert.Equal(t, baseline, syncer.endpointIDs()["ns/ep-gone"],
		"released endpoint must stop publishing")
}

func TestReleaseEndpointWaitsForInFlightPublish(t *testing.T) {
	syncer := newBlockingSyncer()
	contributor := fakeEndpointContributor{fakeContributor: fakeContributor{key: "inflight:test"}}
	source := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "endpoint-source")

	r := NewRuntime(time.Second)
	r.crossReplicaPub = &crossReplicaPublisher{
		syncer:       syncer,
		contributors: []fwkdl.CrossReplicaContributor{contributor},
		interval:     time.Millisecond,
	}
	r.endpoint.Set(source)
	r.extractors.Append(source.TypedName().Name, contributor)

	ep := r.NewEndpoint(context.Background(), testEndpoint("ep-gone").GetMetadata())
	require.NotNil(t, ep)

	publishDone := make(chan struct{})
	go func() {
		defer close(publishDone)
		r.crossReplicaPub.publishAll(context.Background())
	}()
	<-syncer.setStarted

	releaseDone := make(chan struct{})
	go func() {
		defer close(releaseDone)
		r.ReleaseEndpoint(ep)
	}()

	select {
	case <-releaseDone:
		t.Error("ReleaseEndpoint completed while a publish was still in flight")
	case <-time.After(50 * time.Millisecond):
	}

	close(syncer.allowSet)
	select {
	case <-publishDone:
	case <-time.After(time.Second):
		t.Fatal("in-flight publish did not complete")
	}
	select {
	case <-releaseDone:
	case <-time.After(time.Second):
		t.Fatal("ReleaseEndpoint did not complete")
	}

	_, found, err := syncer.Get(context.Background(), "inflight:test", "ns/ep-gone", nil)
	require.NoError(t, err)
	assert.False(t, found, "released endpoint state must remain deleted")
	assert.Equal(t, []string{"set", "delete"}, syncer.events)
}
