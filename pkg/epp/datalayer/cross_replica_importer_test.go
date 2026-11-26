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
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

type peerSnap struct {
	peer    any
	version uint64
}

// peerStore is the CrossReplicaPeerReader half of the fake syncers below.
type peerStore struct {
	mu        sync.Mutex
	snapshots map[string]peerSnap
}

func (s *peerStore) setPeer(endpointID string, peer any, version uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.snapshots == nil {
		s.snapshots = make(map[string]peerSnap)
	}
	s.snapshots[endpointID] = peerSnap{peer: peer, version: version}
}

func (s *peerStore) PeerSnapshot(_ fwkdl.CrossReplicaSpec, endpointID string) (any, uint64, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap, ok := s.snapshots[endpointID]
	return snap.peer, snap.version, ok
}

// peerBatchSyncer implements CrossReplicaBatchSyncer and CrossReplicaPeerReader.
type peerBatchSyncer struct {
	fakeSyncer
	peerStore
	failures  map[string]error
	duringSet func()
}

func (s *peerBatchSyncer) SetBatch(_ context.Context, _ fwkdl.CrossReplicaSpec, _ []string, _ time.Duration) map[string]error {
	if s.duringSet != nil {
		s.duringSet()
	}
	return s.failures
}

// peerSerialSyncer implements CrossReplicaPeerReader without batching.
type peerSerialSyncer struct {
	fakeSyncer
	peerStore
}

type importCall struct {
	endpointID string
	peer       any
}

// recordingImporter is an importing contributor. fn, when set, decides the
// result of each call; it runs before the call is recorded.
type recordingImporter struct {
	fakeEndpointContributor
	fn    func(ctx context.Context, endpointID string, peer any) error
	calls chan importCall
}

func newRecordingImporter(fn func(ctx context.Context, endpointID string, peer any) error) *recordingImporter {
	return &recordingImporter{
		fakeEndpointContributor: fakeEndpointContributor{fakeContributor: fakeContributor{key: "prefix:test"}},
		fn:                      fn,
		calls:                   make(chan importCall, 256),
	}
}

func (r *recordingImporter) ImportCrossReplica(ctx context.Context, endpointID string, peer any) error {
	var err error
	if r.fn != nil {
		err = r.fn(ctx, endpointID, peer)
	}
	r.calls <- importCall{endpointID: endpointID, peer: peer}
	return err
}

func nextImport(t *testing.T, r *recordingImporter) importCall {
	t.Helper()
	select {
	case call := <-r.calls:
		return call
	case <-time.After(2 * time.Second):
		t.Fatal("expected an import")
		return importCall{}
	}
}

func noImport(t *testing.T, r *recordingImporter) {
	t.Helper()
	select {
	case call := <-r.calls:
		t.Fatalf("unexpected import %+v", call)
	case <-time.After(50 * time.Millisecond):
	}
}

func newImportingPublisher(t *testing.T, syncer fwkdl.CrossReplicaSyncer, importer *recordingImporter) *crossReplicaPublisher {
	t.Helper()
	pub := newCrossReplicaPublisher(syncer, extractorMapWith(importer), 5*time.Millisecond, time.Second)
	require.NotNil(t, pub)
	require.NotNil(t, pub.importers[0])
	return pub
}

func startImportWorkers(t *testing.T, pub *crossReplicaPublisher) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	for _, w := range pub.importers {
		if w != nil {
			go w.run(ctx, logr.Discard())
		}
	}
}

var importKey = types.NamespacedName{Namespace: "ns", Name: "ep-a"}

func TestImportWorker_ImportsOnlyWhenPeerVersionChanges(t *testing.T) {
	syncer := &peerBatchSyncer{}
	importer := newRecordingImporter(nil)
	pub := newImportingPublisher(t, syncer, importer)
	startImportWorkers(t, pub)
	require.True(t, pub.registerEndpoint(importKey))

	syncer.setPeer(importKey.String(), "v1", 1)
	pub.publishAll(context.Background())
	require.Equal(t, importCall{endpointID: "ns/ep-a", peer: "v1"}, nextImport(t, importer))

	pub.publishAll(context.Background())
	noImport(t, importer)

	syncer.setPeer(importKey.String(), "v2", 2)
	pub.publishAll(context.Background())
	require.Equal(t, "v2", nextImport(t, importer).peer)
}

func TestImportWorker_SerialSetEnqueues(t *testing.T) {
	syncer := &peerSerialSyncer{}
	importer := newRecordingImporter(nil)
	pub := newImportingPublisher(t, syncer, importer)
	startImportWorkers(t, pub)
	require.True(t, pub.registerEndpoint(importKey))

	syncer.setPeer(importKey.String(), "v1", 1)
	pub.publishAll(context.Background())
	require.Equal(t, "v1", nextImport(t, importer).peer)
}

func TestImportWorker_FailedSetDoesNotEnqueue(t *testing.T) {
	syncer := &peerBatchSyncer{failures: map[string]error{importKey.String(): errors.New("redis down")}}
	importer := newRecordingImporter(nil)
	pub := newImportingPublisher(t, syncer, importer)
	startImportWorkers(t, pub)
	require.True(t, pub.registerEndpoint(importKey))

	syncer.setPeer(importKey.String(), "v1", 1)
	pub.publishAll(context.Background())
	noImport(t, importer)
}

func TestImportWorker_KeepsUntakenSnapshot(t *testing.T) {
	syncer := &peerBatchSyncer{}
	importer := newRecordingImporter(nil)
	pub := newImportingPublisher(t, syncer, importer)
	require.True(t, pub.registerEndpoint(importKey))

	syncer.setPeer(importKey.String(), "with-old-fields", 1)
	pub.publishAll(context.Background())
	syncer.setPeer(importKey.String(), "old-fields-expired", 2)
	pub.publishAll(context.Background())

	startImportWorkers(t, pub)
	require.Equal(t, "with-old-fields", nextImport(t, importer).peer)
	noImport(t, importer)
	pub.publishAll(context.Background())
	require.Equal(t, "old-fields-expired", nextImport(t, importer).peer)
}

func TestImportWorker_RetriesFailedSnapshotBeforeNewer(t *testing.T) {
	syncer := &peerBatchSyncer{}
	release := make(chan struct{})
	var mu sync.Mutex
	failed := false
	importer := newRecordingImporter(func(_ context.Context, _ string, _ any) error {
		mu.Lock()
		defer mu.Unlock()
		if !failed {
			failed = true
			<-release
			return errors.New("endpoint not ready")
		}
		return nil
	})
	pub := newImportingPublisher(t, syncer, importer)
	startImportWorkers(t, pub)
	require.True(t, pub.registerEndpoint(importKey))

	syncer.setPeer(importKey.String(), "v1", 1)
	pub.publishAll(context.Background())
	require.Eventually(t, func() bool {
		w := pub.importers[0]
		w.mu.Lock()
		defer w.mu.Unlock()
		return len(w.pending) == 0
	}, time.Second, time.Millisecond, "worker should have taken v1")

	syncer.setPeer(importKey.String(), "v2", 2)
	pub.publishAll(context.Background())
	close(release)

	require.Equal(t, "v1", nextImport(t, importer).peer, "failed attempt")
	retry := nextImport(t, importer)
	require.Equal(t, "v1", retry.peer, "the failed snapshot is retried ahead of the newer one")
	noImport(t, importer)
	pub.publishAll(context.Background())
	require.Equal(t, "v2", nextImport(t, importer).peer)
}

func TestImportWorker_RetryBackoffDoublesAndCaps(t *testing.T) {
	w := newImportWorker(fwkdl.CrossReplicaSpec{}, newRecordingImporter(func(context.Context, string, any) error {
		return errors.New("not ready")
	}), &peerStore{}, 200*time.Millisecond)
	gen := newEndpointGen()
	job := &importJob{gen: gen}
	backoffs := make([]time.Duration, 0, 6)
	for range 6 {
		w.runJob(logr.Discard(), "ns/ep-a", job)
		_, queued, _ := w.take(time.Now().Add(time.Hour))
		require.Same(t, job, queued)
		backoffs = append(backoffs, job.backoff)
	}
	require.Equal(t, []time.Duration{
		200 * time.Millisecond, 400 * time.Millisecond, 800 * time.Millisecond,
		1600 * time.Millisecond, 2 * time.Second, 2 * time.Second,
	}, backoffs)
}

func TestImportWorker_DrainsPendingBeforeSleeping(t *testing.T) {
	syncer := &peerBatchSyncer{}
	importer := newRecordingImporter(nil)
	pub := newImportingPublisher(t, syncer, importer)
	const endpoints = 50
	for i := range endpoints {
		key := types.NamespacedName{Namespace: "ns", Name: fmt.Sprintf("ep-%d", i)}
		require.True(t, pub.registerEndpoint(key))
		syncer.setPeer(key.String(), "v1", 1)
	}
	pub.publishAll(context.Background())

	startImportWorkers(t, pub)
	seen := map[string]bool{}
	for range endpoints {
		seen[nextImport(t, importer).endpointID] = true
	}
	require.Len(t, seen, endpoints)
}

func TestImportWorker_StopsWithPendingImports(t *testing.T) {
	syncer := &peerBatchSyncer{}
	started := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	importer := newRecordingImporter(func(context.Context, string, any) error {
		once.Do(func() {
			close(started)
			<-release
		})
		return nil
	})
	pub := newImportingPublisher(t, syncer, importer)
	for i := range 3 {
		key := types.NamespacedName{Namespace: "ns", Name: fmt.Sprintf("ep-%d", i)}
		require.True(t, pub.registerEndpoint(key))
		syncer.setPeer(key.String(), "v1", 1)
	}
	pub.publishAll(context.Background())

	ctx, cancel := context.WithCancel(context.Background())
	stopped := make(chan struct{})
	go func() {
		pub.importers[0].run(ctx, logr.Discard())
		close(stopped)
	}()
	<-started
	cancel()
	close(release)

	select {
	case <-stopped:
	case <-time.After(2 * time.Second):
		t.Fatal("worker did not stop")
	}
	nextImport(t, importer)
	noImport(t, importer)
}

func TestImportWorker_DeleteWaitsForRunningImport(t *testing.T) {
	syncer := &peerBatchSyncer{}
	started := make(chan struct{})
	returned := make(chan struct{})
	importer := newRecordingImporter(func(ctx context.Context, _ string, _ any) error {
		close(started)
		<-ctx.Done()
		time.Sleep(20 * time.Millisecond)
		close(returned)
		return ctx.Err()
	})
	pub := newImportingPublisher(t, syncer, importer)
	startImportWorkers(t, pub)
	require.True(t, pub.registerEndpoint(importKey))
	syncer.setPeer(importKey.String(), "v1", 1)
	pub.publishAll(context.Background())
	<-started

	gen := pub.beginDelete(logr.Discard(), importKey)
	select {
	case <-returned:
	default:
		t.Fatal("beginDelete returned before the canceled import returned")
	}
	w := pub.importers[0]
	w.mu.Lock()
	require.Empty(t, w.pending)
	require.Empty(t, w.imported)
	w.mu.Unlock()
	pub.finishDelete(importKey, gen)
}

func TestImportWorker_DropsJobTakenBeforeDelete(t *testing.T) {
	syncer := &peerBatchSyncer{}
	importer := newRecordingImporter(nil)
	pub := newImportingPublisher(t, syncer, importer)
	require.True(t, pub.registerEndpoint(importKey))
	syncer.setPeer(importKey.String(), "v1", 1)
	pub.publishAll(context.Background())

	w := pub.importers[0]
	endpointID, job, _ := w.take(time.Now())
	require.NotNil(t, job)
	gen := pub.beginDelete(logr.Discard(), importKey)
	w.runJob(logr.Discard(), endpointID, job)
	noImport(t, importer)
	pub.finishDelete(importKey, gen)
}

func TestImportWorker_ReregisterWaitsForDeleteAndReimports(t *testing.T) {
	syncer := &peerBatchSyncer{}
	importer := newRecordingImporter(nil)
	pub := newImportingPublisher(t, syncer, importer)
	startImportWorkers(t, pub)
	require.True(t, pub.registerEndpoint(importKey))
	syncer.setPeer(importKey.String(), "v1", 1)
	pub.publishAll(context.Background())
	nextImport(t, importer)

	gen := pub.beginDelete(logr.Discard(), importKey)
	_, err := pub.delete(context.Background(), importKey)
	require.NoError(t, err)
	registered := make(chan bool, 1)
	go func() { registered <- pub.registerEndpoint(importKey) }()
	select {
	case <-registered:
		t.Fatal("registration proceeded before the delete reached the plugins")
	case <-time.After(50 * time.Millisecond):
	}
	pub.finishDelete(importKey, gen)
	require.True(t, <-registered)

	pub.publishAll(context.Background())
	require.Equal(t, "v1", nextImport(t, importer).peer, "a new generation imports the same version again")
}

func TestImportWorker_SkipsEndpointReplacedDuringSetBatch(t *testing.T) {
	syncer := &peerBatchSyncer{}
	importer := newRecordingImporter(nil)
	pub := newImportingPublisher(t, syncer, importer)
	require.True(t, pub.registerEndpoint(importKey))
	syncer.setPeer(importKey.String(), "v1", 1)
	syncer.duringSet = func() {
		syncer.duringSet = nil
		gen := pub.beginDelete(logr.Discard(), importKey)
		_, _ = pub.delete(context.Background(), importKey)
		pub.finishDelete(importKey, gen)
		require.True(t, pub.registerEndpoint(importKey))
	}
	pub.publishAll(context.Background())

	w := pub.importers[0]
	w.mu.Lock()
	defer w.mu.Unlock()
	require.Empty(t, w.pending, "a Set captured for the old generation must not enqueue for the new one")
}

func TestCrossReplicaPublisher_NoReaderKeepsRequestPathMerge(t *testing.T) {
	importer := newRecordingImporter(nil)
	pub := newCrossReplicaPublisher(&fakeSyncer{}, extractorMapWith(importer), time.Millisecond, time.Second)
	require.Nil(t, pub.importers[0])
	require.Equal(t, []string{importer.TypedName().String()}, pub.importersWithoutReader)

	var logs strings.Builder
	logger := funcr.New(func(prefix, args string) { logs.WriteString(args + "\n") }, funcr.Options{})
	ctx, cancel := context.WithCancel(log.IntoContext(context.Background(), logger))
	pub.start(ctx)
	cancel()
	require.Contains(t, logs.String(), "does not implement CrossReplicaPeerReader")
	require.Contains(t, logs.String(), importer.TypedName().String())
}

func TestReleaseEndpointBlocksSameIDRegistrationUntilDeleteDispatched(t *testing.T) {
	syncer := &peerBatchSyncer{}
	deleteSeen := make(chan struct{})
	allowDelete := make(chan struct{})
	importer := &blockingDeleteImporter{
		recordingImporter: newRecordingImporter(nil),
		deleteSeen:        deleteSeen,
		allowDelete:       allowDelete,
	}
	source := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "endpoint-source")
	r := NewRuntime(time.Second)
	r.crossReplicaPub = newCrossReplicaPublisher(syncer, extractorMapWith(importer), time.Millisecond, time.Second)
	r.endpoint.Set(source)
	r.extractors.Append(source.TypedName().Name, importer)

	ep := r.NewEndpoint(context.Background(), testEndpoint("ep-a").GetMetadata())
	require.NotNil(t, ep)
	go r.ReleaseEndpoint(ep)
	<-deleteSeen

	recreated := make(chan fwkdl.Endpoint, 1)
	go func() { recreated <- r.NewEndpoint(context.Background(), testEndpoint("ep-a").GetMetadata()) }()
	select {
	case <-recreated:
		t.Fatal("same-ID NewEndpoint completed before the delete event finished")
	case <-time.After(50 * time.Millisecond):
	}
	close(allowDelete)
	select {
	case got := <-recreated:
		require.NotNil(t, got)
	case <-time.After(2 * time.Second):
		t.Fatal("same-ID NewEndpoint did not complete after the delete event")
	}
}

// panickingDeleteSource panics while it handles a delete event.
type panickingDeleteSource struct {
	fwkdl.EndpointSource
}

func (s panickingDeleteSource) NotifyEndpoint(ctx context.Context, event fwkdl.EndpointEvent) (*fwkdl.EndpointEvent, error) {
	if event.Type == fwkdl.EventDelete {
		panic("delete failed")
	}
	return s.EndpointSource.NotifyEndpoint(ctx, event)
}

func TestReleaseEndpointPanicDoesNotBlockSameIDRegistration(t *testing.T) {
	importer := newRecordingImporter(nil)
	source := panickingDeleteSource{notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "endpoint-source")}
	r := NewRuntime(time.Second)
	r.crossReplicaPub = newCrossReplicaPublisher(&peerBatchSyncer{}, extractorMapWith(importer), time.Millisecond, time.Second)
	r.endpoint.Set(source)
	r.extractors.Append(source.TypedName().Name, importer)

	ep := r.NewEndpoint(context.Background(), testEndpoint("ep-a").GetMetadata())
	require.NotNil(t, ep)
	require.Panics(t, func() { r.ReleaseEndpoint(ep) })

	recreated := make(chan fwkdl.Endpoint, 1)
	go func() { recreated <- r.NewEndpoint(context.Background(), testEndpoint("ep-a").GetMetadata()) }()
	select {
	case got := <-recreated:
		require.NotNil(t, got)
	case <-time.After(2 * time.Second):
		t.Fatal("same-ID NewEndpoint is blocked after a panic in ReleaseEndpoint")
	}
}

type blockingDeleteImporter struct {
	*recordingImporter
	deleteSeen  chan struct{}
	allowDelete chan struct{}
	once        sync.Once
}

func (b *blockingDeleteImporter) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type == fwkdl.EventDelete {
		b.once.Do(func() { close(b.deleteSeen) })
		<-b.allowDelete
	}
	return nil
}

func TestImportWorker_SameVersionWhileRunningNotReimported(t *testing.T) {
	syncer := &peerBatchSyncer{}
	started := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	importer := newRecordingImporter(func(context.Context, string, any) error {
		once.Do(func() {
			close(started)
			<-release
		})
		return nil
	})
	pub := newImportingPublisher(t, syncer, importer)
	startImportWorkers(t, pub)
	require.True(t, pub.registerEndpoint(importKey))
	syncer.setPeer(importKey.String(), "v1", 1)
	pub.publishAll(context.Background())
	<-started

	pub.publishAll(context.Background())
	close(release)
	require.Equal(t, "v1", nextImport(t, importer).peer)
	noImport(t, importer)
}

func TestRegisterEndpointDuringDeleteWaitsInsteadOfRejecting(t *testing.T) {
	syncer := &peerBatchSyncer{}
	importer := newRecordingImporter(nil)
	pub := newImportingPublisher(t, syncer, importer)
	require.True(t, pub.registerEndpoint(importKey))

	// The key is still in endpoints between beginDelete and delete.
	gen := pub.beginDelete(logr.Discard(), importKey)
	registered := make(chan bool, 1)
	go func() { registered <- pub.registerEndpoint(importKey) }()
	select {
	case <-registered:
		t.Fatal("registration returned while the old generation was draining")
	case <-time.After(50 * time.Millisecond):
	}
	_, err := pub.delete(context.Background(), importKey)
	require.NoError(t, err)
	pub.finishDelete(importKey, gen)
	require.True(t, <-registered)
}
