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
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

type fakeCloneable struct{ id string }

func (f fakeCloneable) Clone() fwkdl.Cloneable { return f }

type fakeLoad int64

func (l fakeLoad) Clone() fwkdl.Cloneable { return l }

type setCall struct {
	key        fwkdl.StateKey
	endpointID string
	value      any
	aggregate  func([]any) any
}

type fakeSyncer struct {
	mu       sync.Mutex
	sets     []setCall
	deletes  []setCall
	getValue any
	getOK    bool
	getErr   error
	setErr   error
}

func (s *fakeSyncer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-syncer", Name: "fake-syncer"}
}

func (s *fakeSyncer) Set(_ context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.setErr != nil {
		return s.setErr
	}
	s.sets = append(s.sets, setCall{
		key:        spec.StateKey,
		endpointID: endpointID,
		value:      spec.Read(endpointID),
		aggregate:  spec.Aggregate,
	})
	return nil
}

func (s *fakeSyncer) Get(_ context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) (any, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.getErr != nil {
		return nil, false, s.getErr
	}
	values := []any{spec.Read(endpointID)}
	if s.getOK {
		values = append(values, s.getValue)
	}
	return spec.Aggregate(values), true, nil
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

type liveLoadContributor struct {
	local *atomic.Int64
}

func (c liveLoadContributor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "live-load", Name: "live-load"}
}

func (c liveLoadContributor) CrossReplicaState() fwkdl.CrossReplicaSpec {
	return fwkdl.CrossReplicaSpec{
		StateKey:     "live-load",
		AttributeKey: fwkplugin.NewDataKey("live-load", "live-load"),
		Read:         func(string) fwkdl.Cloneable { return fakeLoad(c.local.Load()) },
		Aggregate: func(values []any) any {
			var total fakeLoad
			for _, value := range values {
				total += value.(fakeLoad)
			}
			return total
		},
	}
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

type blockingDeleteSyncer struct {
	fakeSyncer
	started chan struct{}
	release chan struct{}
}

type parallelSyncer struct {
	fakeSyncer
	started chan setCall
	release chan struct{}
}

func (s *deadlineSyncer) Set(ctx context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) error {
	deadline, _ := ctx.Deadline()
	select {
	case s.deadlineObserved <- deadline:
	default:
	}
	return s.fakeSyncer.Set(ctx, spec, endpointID)
}

func (s *deadlineSyncer) Delete(ctx context.Context, key fwkdl.StateKey, endpointID string) error {
	deadline, _ := ctx.Deadline()
	s.deadlineObserved <- deadline
	return s.fakeSyncer.Delete(ctx, key, endpointID)
}

func (s *blockingDeleteSyncer) Delete(ctx context.Context, key fwkdl.StateKey, endpointID string) error {
	close(s.started)
	select {
	case <-s.release:
		return s.fakeSyncer.Delete(ctx, key, endpointID)
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *parallelSyncer) Set(ctx context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) error {
	select {
	case s.started <- setCall{key: spec.StateKey, endpointID: endpointID}:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case <-s.release:
	case <-ctx.Done():
		return ctx.Err()
	}
	return s.fakeSyncer.Set(ctx, spec, endpointID)
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

func (s *blockingSyncer) Set(_ context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) error {
	s.startOnce.Do(func() { close(s.setStarted) })
	<-s.allowSet
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state[endpointID] = spec.Read(endpointID)
	s.events = append(s.events, "set")
	return nil
}

func (s *blockingSyncer) Get(_ context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) (any, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	values := []any{spec.Read(endpointID)}
	if value, ok := s.state[endpointID]; ok {
		values = append(values, value)
	}
	return spec.Aggregate(values), true, nil
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
		Read:         func(id string) fwkdl.Cloneable { return fakeCloneable{id: id} },
		Aggregate:    func(values []any) any { return len(values) },
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

func testCrossReplicaPublisher(syncer fwkdl.CrossReplicaSyncer, contributors ...fwkdl.CrossReplicaContributor) *crossReplicaPublisher {
	return &crossReplicaPublisher{syncer: syncer, contributors: contributors, publishTimeout: defaultCrossReplicaPublishTimeout}
}

func TestCrossReplicaPublisher_PublishesForEndpoint(t *testing.T) {
	syncer := &fakeSyncer{}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"})
	endpointID := types.NamespacedName{Namespace: "ns", Name: "ep-a"}
	require.True(t, pub.registerEndpoint(endpointID))

	pub.publish(context.Background(), endpointID)

	require.Len(t, syncer.sets, 1)
	assert.Equal(t, fwkdl.StateKey("inflight:test"), syncer.sets[0].key)
	assert.Equal(t, "ns/ep-a", syncer.sets[0].endpointID)
	assert.Equal(t, fakeCloneable{id: "ns/ep-a"}, syncer.sets[0].value)
	assert.Equal(t, 2, syncer.sets[0].aggregate([]any{"a", "b"}))
}

func TestCrossReplicaPublisher_RateLimitsVisiblePublishFailures(t *testing.T) {
	syncer := &fakeSyncer{setErr: assert.AnError}
	pub := newCrossReplicaPublisher(syncer, extractorMapWith(fakeContributor{key: "inflight:test"}), 0, 0)
	endpointID := types.NamespacedName{Namespace: "ns", Name: "ep-a"}
	require.True(t, pub.registerEndpoint(endpointID))

	var failures atomic.Int64
	logger := funcr.New(func(_, args string) {
		if strings.Contains(args, "cross-replica publish failed") {
			failures.Add(1)
		}
	}, funcr.Options{})
	ctx := log.IntoContext(context.Background(), logger)

	pub.publish(ctx, endpointID)
	pub.publish(ctx, endpointID)
	require.Equal(t, int64(1), failures.Load())

	pub.publishFailureLog.Interval = time.Nanosecond
	pub.publish(ctx, endpointID)
	require.Equal(t, int64(2), failures.Load())
}

func TestCrossReplicaPublisher_CombinesLiveLocalWithCachedPeers(t *testing.T) {
	var local atomic.Int64
	local.Store(4)
	contributor := liveLoadContributor{local: &local}
	spec := contributor.CrossReplicaState()
	syncer := &fakeSyncer{
		getValue: fakeLoad(7),
		getOK:    true,
	}
	pub := testCrossReplicaPublisher(syncer, contributor)
	endpoint := testEndpoint("ep-a")
	require.True(t, pub.registerEndpoint(endpoint.GetMetadata().GetID()))

	pub.handleEndpointEvent(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: endpoint,
	}, contributor)

	value, ok := endpoint.GetAttributes().Get(spec.AttributeKey)
	require.True(t, ok)
	assert.Equal(t, fakeLoad(11), value)

	local.Store(5)
	value, ok = endpoint.GetAttributes().Get(spec.AttributeKey)
	require.True(t, ok)
	assert.Equal(t, fakeLoad(12), value)

	local.Store(0)
	value, ok = endpoint.GetAttributes().Get(spec.AttributeKey)
	require.True(t, ok)
	assert.Equal(t, fakeLoad(7), value)
}

func TestCrossReplicaPublisher_UsesLiveLocalOnPeerCacheMiss(t *testing.T) {
	var local atomic.Int64
	local.Store(4)
	contributor := liveLoadContributor{local: &local}
	spec := contributor.CrossReplicaState()
	pub := testCrossReplicaPublisher(&fakeSyncer{}, contributor)
	endpoint := testEndpoint("ep-a")
	require.True(t, pub.registerEndpoint(endpoint.GetMetadata().GetID()))

	pub.handleEndpointEvent(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: endpoint,
	}, contributor)

	value, ok := endpoint.GetAttributes().Get(spec.AttributeKey)
	require.True(t, ok)
	assert.Equal(t, fakeLoad(4), value)
}

func TestCrossReplicaPublisher_UsesLiveLocalOnPeerCacheError(t *testing.T) {
	var local atomic.Int64
	local.Store(4)
	contributor := liveLoadContributor{local: &local}
	spec := contributor.CrossReplicaState()
	pub := testCrossReplicaPublisher(&fakeSyncer{getErr: assert.AnError}, contributor)
	endpoint := testEndpoint("ep-a")
	require.True(t, pub.registerEndpoint(endpoint.GetMetadata().GetID()))

	pub.handleEndpointEvent(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: endpoint,
	}, contributor)

	value, ok := endpoint.GetAttributes().Get(spec.AttributeKey)
	require.True(t, ok)
	assert.Equal(t, fakeLoad(4), value)
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
	r.crossReplicaPub = testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"})
	r.crossReplicaPub.interval = time.Millisecond
	r.crossReplicaPub.publishTimeout = defaultCrossReplicaPublishTimeout
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
	pub := testCrossReplicaPublisher(
		syncer,
		fakeContributor{key: "inflight:test"},
		fakeContributor{key: "queue:test"},
	)
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
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"})
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
	pub := testCrossReplicaPublisher(
		syncer,
		fakeContributor{key: "inflight:test"},
		fakeContributor{key: "queue:test"},
	)
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
	r.crossReplicaPub = testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"})
	r.crossReplicaPub.interval = time.Millisecond
	r.crossReplicaPub.publishTimeout = publishTimeout
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

func TestCrossReplicaPublisher_DeleteDoesNotBlockReads(t *testing.T) {
	syncer := &blockingDeleteSyncer{
		fakeSyncer: fakeSyncer{getValue: fakeLoad(7), getOK: true},
		started:    make(chan struct{}),
		release:    make(chan struct{}),
	}
	var local atomic.Int64
	local.Store(4)
	contributor := liveLoadContributor{local: &local}
	pub := testCrossReplicaPublisher(syncer, contributor)
	pub.publishTimeout = 10 * time.Second
	endpoint := testEndpoint("ep-a")
	require.True(t, pub.registerEndpoint(endpoint.GetMetadata().GetID()))
	pub.handleEndpointEvent(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint,
	}, contributor)
	removedID := testEndpoint("ep-gone").GetMetadata().GetID()
	require.True(t, pub.registerEndpoint(removedID))

	deleteDone := make(chan error, 1)
	go func() {
		_, err := pub.delete(context.Background(), removedID)
		deleteDone <- err
	}()
	t.Cleanup(func() {
		close(syncer.release)
		require.NoError(t, <-deleteDone)
	})
	select {
	case <-syncer.started:
	case <-time.After(time.Second):
		t.Fatal("delete did not start")
	}

	readDone := make(chan any, 1)
	go func() {
		value, _ := endpoint.GetAttributes().Get(contributor.CrossReplicaState().AttributeKey)
		readDone <- value
	}()
	select {
	case value := <-readDone:
		require.Equal(t, fakeLoad(11), value)
	case <-time.After(time.Second):
		t.Fatal("slow syncer delete blocked an endpoint attribute read")
	}
}

func TestCrossReplicaPublisher_SharesDeleteDeadlineAcrossContributors(t *testing.T) {
	for _, parentTimeout := range []time.Duration{time.Second, time.Minute} {
		t.Run(parentTimeout.String(), func(t *testing.T) {
			syncer := &deadlineSyncer{deadlineObserved: make(chan time.Time, 2)}
			pub := testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"}, fakeContributor{key: "queue:test"})
			pub.publishTimeout = 3 * time.Second
			endpointID := testEndpoint("ep-gone").GetMetadata().GetID()
			require.True(t, pub.registerEndpoint(endpointID))
			ctx, cancel := context.WithTimeout(context.Background(), parentTimeout)
			defer cancel()
			before := time.Now()
			removed, err := pub.delete(ctx, endpointID)
			require.NoError(t, err)
			require.True(t, removed)

			first := <-syncer.deadlineObserved
			second := <-syncer.deadlineObserved
			require.Equal(t, first, second)
			require.WithinDuration(t, before.Add(min(parentTimeout, pub.publishTimeout)), first, 100*time.Millisecond)
		})
	}
}

// Releasing an endpoint stops its publishing without tearing down a goroutine,
// so removed endpoints cannot keep writing stale state.
func TestCrossReplicaPublisher_StopsAfterDelete(t *testing.T) {
	syncer := &fakeSyncer{}
	r := NewRuntime(time.Second)
	r.crossReplicaPub = testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"})
	r.crossReplicaPub.interval = time.Millisecond
	r.crossReplicaPub.publishTimeout = defaultCrossReplicaPublishTimeout
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
	r.crossReplicaPub = testCrossReplicaPublisher(syncer, contributor)
	r.crossReplicaPub.interval = time.Millisecond
	r.crossReplicaPub.publishTimeout = defaultCrossReplicaPublishTimeout
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

	require.Empty(t, r.crossReplicaPub.endpointSnapshot())
	r.crossReplicaPub.publishAll(context.Background())
	syncer.mu.Lock()
	defer syncer.mu.Unlock()
	_, found := syncer.state["ns/ep-gone"]
	assert.False(t, found, "released endpoint state must remain deleted")
	assert.Equal(t, []string{"set", "delete"}, syncer.events)
}

func TestReleaseEndpointDispatchesDeleteOutsidePublisherLock(t *testing.T) {
	syncer := &fakeSyncer{}
	r := NewRuntime(time.Second)
	contributor := callbackEndpointContributor{
		fakeContributor: fakeContributor{key: "inflight:test"},
		onDelete: func() {
			_ = r.crossReplicaPub.endpointSnapshot()
		},
	}
	r.crossReplicaPub = testCrossReplicaPublisher(syncer, contributor)
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
