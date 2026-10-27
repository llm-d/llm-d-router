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
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

type batchSyncer struct {
	fakeSyncer
	onBatch func(context.Context, fwkdl.CrossReplicaSpec, []string, time.Duration) map[string]error
}

func (s *batchSyncer) SetBatch(ctx context.Context, spec fwkdl.CrossReplicaSpec, ids []string, timeout time.Duration) map[string]error {
	return s.onBatch(ctx, spec, ids, timeout)
}

type blockingBatchSyncer struct {
	*blockingSyncer
	batches atomic.Int32
}

func (s *blockingBatchSyncer) SetBatch(ctx context.Context, spec fwkdl.CrossReplicaSpec, ids []string, _ time.Duration) map[string]error {
	s.batches.Add(1)
	failures := make(map[string]error)
	for _, id := range ids {
		if err := s.Set(ctx, spec, id); err != nil {
			failures[id] = err
		}
	}
	return failures
}

func TestCrossReplicaPublisher_BatchContributorsShareCycleContext(t *testing.T) {
	type call struct {
		ctx     context.Context
		key     fwkdl.StateKey
		ids     []string
		timeout time.Duration
	}
	started := make(chan call, 2)
	release := make(chan struct{})
	syncer := &batchSyncer{onBatch: func(ctx context.Context, spec fwkdl.CrossReplicaSpec, ids []string, timeout time.Duration) map[string]error {
		started <- call{ctx: ctx, key: spec.StateKey, ids: ids, timeout: timeout}
		<-release
		return nil
	}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"}, fakeContributor{key: "queue:test"})
	pub.publishTimeout = 200 * time.Millisecond
	ids := []types.NamespacedName{testEndpoint("ep-a").GetMetadata().GetID(), testEndpoint("ep-b").GetMetadata().GetID()}
	for _, id := range ids {
		require.True(t, pub.registerEndpoint(id))
	}
	done := make(chan struct{})
	go func() { defer close(done); pub.publishAll(context.Background()) }()
	calls := make([]call, 0, 2)
	for range 2 {
		select {
		case c := <-started:
			calls = append(calls, c)
		case <-time.After(time.Second):
			t.Fatal("contributors did not both start")
		}
	}
	require.NotEqual(t, calls[0].key, calls[1].key)
	require.Equal(t, calls[0].ctx, calls[1].ctx)
	for _, c := range calls {
		require.Equal(t, pub.publishTimeout, c.timeout)
		require.ElementsMatch(t, []string{"ns/ep-a", "ns/ep-b"}, c.ids)
		require.NoError(t, c.ctx.Err())
		_, hasDeadline := c.ctx.Deadline()
		require.False(t, hasDeadline)
	}
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("batch did not finish")
	}
	require.Empty(t, syncer.sets)
}

func TestCrossReplicaPublisher_BatchFiltersDeletedSnapshot(t *testing.T) {
	var got []string
	syncer := &batchSyncer{onBatch: func(_ context.Context, _ fwkdl.CrossReplicaSpec, ids []string, _ time.Duration) map[string]error {
		got = append(got, ids...)
		return nil
	}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"})
	gone := testEndpoint("ep-gone").GetMetadata().GetID()
	kept := testEndpoint("ep-kept").GetMetadata().GetID()
	require.True(t, pub.registerEndpoint(gone))
	require.True(t, pub.registerEndpoint(kept))
	snapshot := pub.endpointSnapshot()
	removed, err := pub.delete(context.Background(), gone)
	require.NoError(t, err)
	require.True(t, removed)
	pub.publishBatch(context.Background(), syncer, snapshot)
	require.Equal(t, []string{"ns/ep-kept"}, got)
	require.Empty(t, syncer.sets)
}

func TestCrossReplicaPublisher_BatchAggregatesFailureLogs(t *testing.T) {
	syncer := &batchSyncer{onBatch: func(_ context.Context, _ fwkdl.CrossReplicaSpec, ids []string, _ time.Duration) map[string]error {
		failures := make(map[string]error)
		for _, id := range ids {
			failures[id] = errors.New("Redis unavailable")
		}
		return failures
	}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"})
	for _, name := range []string{"ep-a", "ep-b"} {
		require.True(t, pub.registerEndpoint(testEndpoint(name).GetMetadata().GetID()))
	}
	var lines []string
	ctx := log.IntoContext(context.Background(), funcr.New(func(_, line string) { lines = append(lines, line) }, funcr.Options{}))
	pub.publishAll(ctx)
	pub.publishAll(ctx)
	require.Len(t, lines, 1)
	require.True(t, strings.Contains(lines[0], `"failedEndpoints"=2`), lines[0])
}

func TestCrossReplicaPublisher_NonBatchPublishesEachEndpoint(t *testing.T) {
	syncer := &fakeSyncer{}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"}, fakeContributor{key: "queue:test"})
	for _, name := range []string{"ep-a", "ep-b"} {
		require.True(t, pub.registerEndpoint(testEndpoint(name).GetMetadata().GetID()))
	}
	pub.publishAll(context.Background())
	require.Len(t, syncer.sets, 4)
	for _, key := range []fwkdl.StateKey{"inflight:test", "queue:test"} {
		var ids []string
		for _, call := range syncer.sets {
			if call.key == key {
				ids = append(ids, call.endpointID)
			}
		}
		require.ElementsMatch(t, []string{"ns/ep-a", "ns/ep-b"}, ids)
	}
}

func TestReleaseEndpointDuringBatchIsCleanedUp(t *testing.T) {
	syncer := &blockingBatchSyncer{blockingSyncer: newBlockingSyncer()}
	contributor := fakeEndpointContributor{fakeContributor: fakeContributor{key: "inflight:test"}}
	source := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "endpoint-source")
	r := NewRuntime(time.Second)
	r.crossReplicaPub = testCrossReplicaPublisher(syncer, contributor)
	r.endpoint.Set(source)
	r.extractors.Append(source.TypedName().Name, contributor)
	ep := r.NewEndpoint(context.Background(), testEndpoint("ep-gone").GetMetadata())
	require.NotNil(t, ep)
	publishDone := make(chan struct{})
	go func() { defer close(publishDone); r.crossReplicaPub.publishAll(context.Background()) }()
	select {
	case <-syncer.setStarted:
	case <-time.After(time.Second):
		t.Fatal("batch did not start")
	}
	releaseDone := make(chan struct{})
	go func() { defer close(releaseDone); r.ReleaseEndpoint(ep) }()
	select {
	case <-releaseDone:
	case <-time.After(time.Second):
		t.Fatal("ReleaseEndpoint waited for the in-flight batch")
	}
	close(syncer.allowSet)
	select {
	case <-publishDone:
	case <-time.After(time.Second):
		t.Fatal("batch did not finish")
	}
	require.Empty(t, r.crossReplicaPub.endpointSnapshot())
	syncer.mu.Lock()
	defer syncer.mu.Unlock()
	require.Equal(t, []string{"delete", "set", "delete"}, syncer.events)
	require.NotContains(t, syncer.state, "ns/ep-gone")
	require.Equal(t, int32(1), syncer.batches.Load())
}

func TestCrossReplicaPublisher_BatchDoesNotBlockWriters(t *testing.T) {
	specCalled := make(chan struct{})
	allowSpec := make(chan struct{})
	batchStarted := make(chan struct{})
	allowBatch := make(chan struct{})
	syncer := &batchSyncer{onBatch: func(_ context.Context, spec fwkdl.CrossReplicaSpec, _ []string, _ time.Duration) map[string]error {
		if spec.StateKey == "inflight:test" {
			close(batchStarted)
			<-allowBatch
		}
		return nil
	}}
	second := batchSpecContributor{spec: func() fwkdl.CrossReplicaSpec {
		close(specCalled)
		<-allowSpec
		return fakeContributor{key: "queue:test"}.CrossReplicaState()
	}}
	pub := testCrossReplicaPublisher(syncer, fakeContributor{key: "inflight:test"}, second)
	require.True(t, pub.registerEndpoint(testEndpoint("ep-a").GetMetadata().GetID()))
	done := make(chan struct{})
	go func() { defer close(done); pub.publishAll(context.Background()) }()

	// A stalled CrossReplicaState must not hold the endpoint lock.
	select {
	case <-specCalled:
	case <-time.After(time.Second):
		t.Fatal("spec fetch did not start")
	}
	writerDone := make(chan struct{})
	go func() { defer close(writerDone); pub.registerEndpoint(testEndpoint("ep-b").GetMetadata().GetID()) }()
	select {
	case <-writerDone:
	case <-time.After(time.Second):
		t.Fatal("writer blocked behind CrossReplicaState fetch")
	}
	close(allowSpec)

	// An in-flight SetBatch must not hold the endpoint lock either.
	select {
	case <-batchStarted:
	case <-time.After(time.Second):
		t.Fatal("batch did not start")
	}
	writerDone2 := make(chan struct{})
	go func() { defer close(writerDone2); pub.registerEndpoint(testEndpoint("ep-c").GetMetadata().GetID()) }()
	select {
	case <-writerDone2:
	case <-time.After(time.Second):
		t.Fatal("writer blocked behind in-flight batch")
	}
	close(allowBatch)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("batch did not finish")
	}
}

func TestReleaseAndReregisterDuringBatchKeepsEndpoint(t *testing.T) {
	syncer := &blockingBatchSyncer{blockingSyncer: newBlockingSyncer()}
	contributor := fakeEndpointContributor{fakeContributor: fakeContributor{key: "inflight:test"}}
	source := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "endpoint-source")
	r := NewRuntime(time.Second)
	r.crossReplicaPub = testCrossReplicaPublisher(syncer, contributor)
	r.endpoint.Set(source)
	r.extractors.Append(source.TypedName().Name, contributor)
	ep := r.NewEndpoint(context.Background(), testEndpoint("ep-gone").GetMetadata())
	require.NotNil(t, ep)
	publishDone := make(chan struct{})
	go func() { defer close(publishDone); r.crossReplicaPub.publishAll(context.Background()) }()
	select {
	case <-syncer.setStarted:
	case <-time.After(time.Second):
		t.Fatal("batch did not start")
	}
	releaseDone := make(chan struct{})
	go func() { defer close(releaseDone); r.ReleaseEndpoint(ep) }()
	select {
	case <-releaseDone:
	case <-time.After(time.Second):
		t.Fatal("ReleaseEndpoint waited for the in-flight batch")
	}
	ep = r.NewEndpoint(context.Background(), testEndpoint("ep-gone").GetMetadata())
	require.NotNil(t, ep)
	close(syncer.allowSet)
	select {
	case <-publishDone:
	case <-time.After(time.Second):
		t.Fatal("batch did not finish")
	}
	syncer.mu.Lock()
	defer syncer.mu.Unlock()
	require.Equal(t, []string{"delete", "set"}, syncer.events)
	require.Contains(t, syncer.state, "ns/ep-gone")
}

type gatedCleanupSyncer struct {
	fakeSyncer
	setStarted  chan struct{}
	setGate     chan struct{}
	delBlocked  chan struct{}
	delGate     chan struct{}
	deleteCalls atomic.Int32
}

func (s *gatedCleanupSyncer) SetBatch(ctx context.Context, spec fwkdl.CrossReplicaSpec, ids []string, _ time.Duration) map[string]error {
	select {
	case s.setStarted <- struct{}{}:
	default:
	}
	<-s.setGate
	failures := make(map[string]error)
	for _, id := range ids {
		if err := s.Set(ctx, spec, id); err != nil {
			failures[id] = err
		}
	}
	return failures
}

func (s *gatedCleanupSyncer) Delete(ctx context.Context, key fwkdl.StateKey, endpointID string) error {
	if s.deleteCalls.Add(1) > 1 {
		select {
		case s.delBlocked <- struct{}{}:
		default:
		}
		<-s.delGate
	}
	return s.fakeSyncer.Delete(ctx, key, endpointID)
}

func TestReregisterDuringCleanupRepublishes(t *testing.T) {
	syncer := &gatedCleanupSyncer{
		setStarted: make(chan struct{}, 2),
		setGate:    make(chan struct{}),
		delBlocked: make(chan struct{}, 1),
		delGate:    make(chan struct{}),
	}
	contributor := fakeEndpointContributor{fakeContributor: fakeContributor{key: "inflight:test"}}
	source := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "endpoint-source")
	r := NewRuntime(time.Second)
	r.crossReplicaPub = testCrossReplicaPublisher(syncer, contributor)
	r.endpoint.Set(source)
	r.extractors.Append(source.TypedName().Name, contributor)
	ep := r.NewEndpoint(context.Background(), testEndpoint("ep-gone").GetMetadata())
	require.NotNil(t, ep)
	publishDone := make(chan struct{})
	go func() { defer close(publishDone); r.crossReplicaPub.publishAll(context.Background()) }()
	select {
	case <-syncer.setStarted:
	case <-time.After(time.Second):
		t.Fatal("batch did not start")
	}
	releaseDone := make(chan struct{})
	go func() { defer close(releaseDone); r.ReleaseEndpoint(ep) }()
	select {
	case <-releaseDone:
	case <-time.After(time.Second):
		t.Fatal("ReleaseEndpoint waited for the in-flight batch")
	}
	close(syncer.setGate)
	select {
	case <-syncer.delBlocked:
	case <-time.After(time.Second):
		t.Fatal("cleanup delete did not start")
	}
	ep = r.NewEndpoint(context.Background(), testEndpoint("ep-gone").GetMetadata())
	require.NotNil(t, ep)
	close(syncer.delGate)
	select {
	case <-publishDone:
	case <-time.After(time.Second):
		t.Fatal("batch did not finish")
	}
	syncer.mu.Lock()
	defer syncer.mu.Unlock()
	require.Len(t, syncer.sets, 2)
	require.Len(t, syncer.deletes, 2)
	for _, call := range syncer.sets {
		require.Equal(t, "ns/ep-gone", call.endpointID)
	}
	for _, call := range syncer.deletes {
		require.Equal(t, "ns/ep-gone", call.endpointID)
	}
}

type batchSpecContributor struct{ spec func() fwkdl.CrossReplicaSpec }

func (c batchSpecContributor) CrossReplicaState() fwkdl.CrossReplicaSpec { return c.spec() }
