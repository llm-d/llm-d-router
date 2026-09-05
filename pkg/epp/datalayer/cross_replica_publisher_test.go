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
	aggregate  func([]any) any
}

type fakeSyncer struct {
	mu      sync.Mutex
	sets    []setCall
	deletes []setCall
}

func (s *fakeSyncer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-syncer", Name: "fake-syncer"}
}

func (s *fakeSyncer) Set(_ context.Context, key fwkdl.StateKey, endpointID string, value any, aggregate func([]any) any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sets = append(s.sets, setCall{key: key, endpointID: endpointID, value: value, aggregate: aggregate})
	return nil
}

func (s *fakeSyncer) Get(context.Context, fwkdl.StateKey, string) (any, bool, error) {
	return nil, false, nil
}

func (s *fakeSyncer) Delete(_ context.Context, key fwkdl.StateKey, endpointID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.deletes = append(s.deletes, setCall{key: key, endpointID: endpointID})
	return nil
}

func (s *fakeSyncer) GetOrSet(_ context.Context, _ fwkdl.StateKey, _ string, candidate any) (any, bool, error) {
	return candidate, false, nil
}

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

type callbackEndpointContributor struct {
	fakeContributor
	onDelete func()
}

func (c callbackEndpointContributor) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type == fwkdl.EventDelete && c.onDelete != nil {
		c.onDelete()
	}
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
	deadlineObserved chan time.Time
}

type parallelSyncer struct {
	fakeSyncer
	started chan setCall
	release chan struct{}
}

func (s *deadlineSyncer) Set(ctx context.Context, key fwkdl.StateKey, endpointID string, value any, aggregate func([]any) any) error {
	deadline, _ := ctx.Deadline()
	select {
	case s.deadlineObserved <- deadline:
	default:
	}
	return s.fakeSyncer.Set(ctx, key, endpointID, value, aggregate)
}

func (s *parallelSyncer) Set(ctx context.Context, key fwkdl.StateKey, endpointID string, value any, aggregate func([]any) any) error {
	select {
	case s.started <- setCall{key: key, endpointID: endpointID, value: value}:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case <-s.release:
	case <-ctx.Done():
		return ctx.Err()
	}
	return s.fakeSyncer.Set(ctx, key, endpointID, value, aggregate)
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

func (s *blockingSyncer) Set(_ context.Context, _ fwkdl.StateKey, endpointID string, value any, _ func([]any) any) error {
	s.startOnce.Do(func() { close(s.setStarted) })
	<-s.allowSet
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state[endpointID] = value
	s.events = append(s.events, "set")
	return nil
}

func (s *blockingSyncer) Get(_ context.Context, _ fwkdl.StateKey, endpointID string) (any, bool, error) {
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

func (s *blockingSyncer) GetOrSet(_ context.Context, _ fwkdl.StateKey, _ string, candidate any) (any, bool, error) {
	return candidate, false, nil
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
		Aggregate: func(values []any) any { return len(values) },
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
	endpointID := types.NamespacedName{Namespace: "ns", Name: "ep-a"}
	require.True(t, pub.registerEndpoint(endpointID))

	pub.publish(context.Background(), endpointID)

	require.Len(t, syncer.sets, 1)
	assert.Equal(t, fwkdl.StateKey("inflight:test"), syncer.sets[0].key)
	assert.Equal(t, "ns/ep-a", syncer.sets[0].endpointID)
	assert.Equal(t, fakeCloneable{id: "ns/ep-a"}, syncer.sets[0].value)
	assert.Equal(t, 2, syncer.sets[0].aggregate([]any{"a", "b"}))
}

func TestCrossReplicaPublisher_SkipsSyncDisabled(t *testing.T) {
	em := extractorMapWith(
		fakeContributor{key: "enabled"},
		fakeContributor{key: "disabled", syncDisabled: true},
	)

	pub := newCrossReplicaPublisher(&fakeSyncer{}, em, 0, 0)
	require.NotNil(t, pub)
	require.Len(t, pub.contributors, 1)
	assert.Equal(t, fwkdl.StateKey("enabled"), pub.contributors[0].CrossReplicaState().StateKey)
	assert.Equal(t, defaultCrossReplicaSyncInterval, pub.interval, "zero interval falls back to default")
	assert.Equal(t, defaultCrossReplicaPublishTimeout, pub.publishTimeout, "zero timeout falls back to default")
}

func TestCrossReplicaPublisher_ConfiguredDurations(t *testing.T) {
	em := extractorMapWith(fakeContributor{key: "enabled"})
	pub := newCrossReplicaPublisher(&fakeSyncer{}, em, 500*time.Millisecond, 3*time.Second)
	require.NotNil(t, pub)
	assert.Equal(t, 500*time.Millisecond, pub.interval)
	assert.Equal(t, 3*time.Second, pub.publishTimeout)
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
		syncer:         syncer,
		contributors:   []fwkdl.CrossReplicaContributor{fakeContributor{key: "inflight:test"}},
		interval:       time.Millisecond,
		publishTimeout: defaultCrossReplicaPublishTimeout,
	}
	for _, name := range []string{"ep-a", "ep-b", "ep-c"} {
		ep := testEndpoint(name)
		require.NotNil(t, r.NewEndpoint(context.Background(), ep.GetMetadata()))
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r.crossReplicaPub.start(ctx)

	require.Eventually(t, func() bool {
		ids := syncer.endpointIDs()
		return ids["ns/ep-a"] > 0 && ids["ns/ep-b"] > 0 && ids["ns/ep-c"] > 0
	}, 2*time.Second, 5*time.Millisecond, "every registered endpoint should be published")
}

func TestCrossReplicaPublisher_PublishesContributorsConcurrently(t *testing.T) {
	syncer := &parallelSyncer{
		started: make(chan setCall, 2),
		release: make(chan struct{}),
	}
	pub := &crossReplicaPublisher{
		syncer: syncer,
		contributors: []fwkdl.CrossReplicaContributor{
			fakeContributor{key: "inflight:test"},
			fakeContributor{key: "queue:test"},
		},
	}
	endpointID := testEndpoint("ep-a").GetMetadata().GetID()
	require.True(t, pub.registerEndpoint(endpointID))

	done := make(chan struct{})
	go func() {
		defer close(done)
		pub.publish(context.Background(), endpointID)
	}()

	started := map[fwkdl.StateKey]bool{}
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	for len(started) < 2 {
		select {
		case call := <-syncer.started:
			started[call.key] = true
		case <-timer.C:
			close(syncer.release)
			<-done
			t.Fatalf("contributor publishing was sequential; started keys: %v", started)
		}
	}

	close(syncer.release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("parallel contributor publishing did not complete")
	}
}

func TestCrossReplicaPublisher_SkipsEndpointDeletedAfterSnapshot(t *testing.T) {
	syncer := &fakeSyncer{}
	pub := &crossReplicaPublisher{
		syncer:       syncer,
		contributors: []fwkdl.CrossReplicaContributor{fakeContributor{key: "inflight:test"}},
	}
	endpointID := testEndpoint("ep-gone").GetMetadata().GetID()
	require.True(t, pub.registerEndpoint(endpointID))

	snapshot := pub.endpointSnapshot()
	require.Len(t, snapshot, 1)
	removed, err := pub.delete(context.Background(), endpointID)
	require.NoError(t, err)
	require.True(t, removed)

	pub.publish(context.Background(), snapshot[0])
	assert.Empty(t, syncer.sets)
}

func TestCrossReplicaPublisher_DeletesAllContributorState(t *testing.T) {
	syncer := &fakeSyncer{}
	pub := &crossReplicaPublisher{
		syncer: syncer,
		contributors: []fwkdl.CrossReplicaContributor{
			fakeContributor{key: "inflight:test"},
			fakeContributor{key: "queue:test"},
		},
	}
	endpointID := testEndpoint("ep-gone").GetMetadata().GetID()
	require.True(t, pub.registerEndpoint(endpointID))

	removed, err := pub.delete(context.Background(), endpointID)
	require.NoError(t, err)
	require.True(t, removed)
	require.ElementsMatch(t, []setCall{
		{key: "inflight:test", endpointID: "ns/ep-gone"},
		{key: "queue:test", endpointID: "ns/ep-gone"},
	}, syncer.deletes)
}

func TestCrossReplicaPublisher_SetsPerPublishDeadline(t *testing.T) {
	syncer := &deadlineSyncer{deadlineObserved: make(chan time.Time, 1)}
	publishTimeout := 3 * time.Second
	r := NewRuntime(time.Second)
	r.crossReplicaPub = &crossReplicaPublisher{
		syncer:         syncer,
		contributors:   []fwkdl.CrossReplicaContributor{fakeContributor{key: "inflight:test"}},
		interval:       time.Millisecond,
		publishTimeout: publishTimeout,
	}
	ep := testEndpoint("ep-a")
	require.True(t, r.crossReplicaPub.registerEndpoint(ep.GetMetadata().GetID()))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r.crossReplicaPub.start(ctx)

	select {
	case deadline := <-syncer.deadlineObserved:
		assert.WithinDuration(t, time.Now().Add(publishTimeout), deadline, 100*time.Millisecond)
	case <-time.After(time.Second):
		t.Fatal("cross-replica publish did not run")
	}
}

// Releasing an endpoint stops its publishing without tearing down a goroutine,
// so removed endpoints cannot keep writing stale state.
func TestCrossReplicaPublisher_StopsAfterDelete(t *testing.T) {
	syncer := &fakeSyncer{}
	r := NewRuntime(time.Second)
	r.crossReplicaPub = &crossReplicaPublisher{
		syncer:         syncer,
		contributors:   []fwkdl.CrossReplicaContributor{fakeContributor{key: "inflight:test"}},
		interval:       time.Millisecond,
		publishTimeout: defaultCrossReplicaPublishTimeout,
	}
	ep := testEndpoint("ep-gone")
	require.True(t, r.crossReplicaPub.registerEndpoint(ep.GetMetadata().GetID()))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r.crossReplicaPub.start(ctx)

	require.Eventually(t, func() bool {
		return syncer.endpointIDs()["ns/ep-gone"] > 0
	}, 2*time.Second, 5*time.Millisecond, "endpoint should publish while registered")

	removed, err := r.crossReplicaPub.delete(context.Background(), ep.GetMetadata().GetID())
	require.NoError(t, err)
	require.True(t, removed)
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
		syncer:         syncer,
		contributors:   []fwkdl.CrossReplicaContributor{contributor},
		interval:       time.Millisecond,
		publishTimeout: defaultCrossReplicaPublishTimeout,
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

	_, found, err := syncer.Get(context.Background(), "inflight:test", "ns/ep-gone")
	require.NoError(t, err)
	assert.False(t, found, "released endpoint state must remain deleted")
	assert.Equal(t, []string{"set", "delete"}, syncer.events)
}

func TestReleaseEndpointDispatchesDeleteOutsidePublisherLock(t *testing.T) {
	syncer := &fakeSyncer{}
	r := NewRuntime(time.Second)
	contributor := callbackEndpointContributor{
		fakeContributor: fakeContributor{key: "inflight:test"},
		onDelete: func() {
			_, _, _ = r.crossReplicaPub.get(context.Background(), fakeContributor{key: "inflight:test"}.CrossReplicaState(), "ns/ep-gone")
		},
	}
	r.crossReplicaPub = &crossReplicaPublisher{
		syncer:       syncer,
		contributors: []fwkdl.CrossReplicaContributor{contributor},
	}
	source := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "endpoint-source")
	r.endpoint.Set(source)
	r.extractors.Append(source.TypedName().Name, contributor)

	ep := r.NewEndpoint(context.Background(), testEndpoint("ep-gone").GetMetadata())
	require.NotNil(t, ep)

	done := make(chan struct{})
	go func() {
		defer close(done)
		r.ReleaseEndpoint(ep)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("endpoint delete event was dispatched while holding the publisher lock")
	}
}
