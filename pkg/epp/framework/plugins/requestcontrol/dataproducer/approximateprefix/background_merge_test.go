/*
Copyright 2026 The Kubernetes Authors.
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

package approximateprefix

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

type backgroundTestIndexer struct {
	indexerInterface
	adds      atomic.Int64
	beforeAdd func()
}

func (i *backgroundTestIndexer) Add(hashes []blockHash, pod server) {
	if i.beforeAdd != nil {
		i.beforeAdd()
	}
	i.adds.Add(1)
	i.indexerInterface.Add(hashes, pod)
}

func newBackgroundTestProducer(t *testing.T, trigger string, syncer bool) (*dataProducer, *backgroundTestIndexer) {
	t.Helper()
	cfg := defaultConfig
	cfg.AutoTune = false
	cfg.BackgroundMerge = backgroundMergeConfig{Trigger: trigger}
	h := testHandle()
	p, err := newDataProducer(t.Context(), "background-test", cfg, h)
	require.NoError(t, err)
	index := &backgroundTestIndexer{indexerInterface: p.indexerInst}
	p.indexerInst = index
	t.Cleanup(func() {
		if p.backgroundMergeDone != nil {
			select {
			case <-p.backgroundMergeDone:
			case <-time.After(time.Second):
				t.Error("background merge outlived the test context")
			}
		}
	})
	if syncer {
		// Match the loader: syncer wiring follows plugin construction.
		h.SetCrossReplicaSyncer(stubSyncerPlugin{})
	}
	return p, index
}

func registerBackgroundPeer(t *testing.T, p *dataProducer, hashes []uint64) fwkdl.Endpoint {
	t.Helper()
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "ns", Name: "peer"}}, nil)
	require.NoError(t, p.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
	ep.GetAttributes().Put(p.syncHashDK, &attrprefix.PrefixHashSet{Prompts: [][]uint64{hashes}})
	return ep
}

func TestBackgroundMergeTriggersWithoutRequests(t *testing.T) {
	for _, trigger := range []string{backgroundMergeDisabled, backgroundMergeCallback} {
		t.Run(trigger, func(t *testing.T) {
			p, index := newBackgroundTestProducer(t, trigger, true)
			hashes := []uint64{101, 102, 103}
			ep := registerBackgroundPeer(t, p, hashes)
			p.AfterCrossReplicaPublish(t.Context())
			if trigger == backgroundMergeDisabled {
				require.Nil(t, p.backgroundMergeDone)
				require.Empty(t, p.indexerInst.Pods())
				// The unchanged request fallback still consumes the peer window.
				request := &fwksched.InferenceRequest{RequestID: "fallback", Body: tokenizedBody([]uint32{1, 2, 3})}
				require.NoError(t, p.Produce(t.Context(), request, []fwksched.Endpoint{fwksched.NewEndpoint(ep.GetMetadata(), ep.GetMetrics(), ep.GetAttributes())}))
			} else {
				require.Eventually(t, func() bool {
					return p.indexerInst.MatchLongestPrefix(uint64sToBlockHashes(hashes), []ServerID{ServerID(ep.GetMetadata().ID)})[0] == 3
				}, time.Second, time.Millisecond)
			}
			matches := p.indexerInst.MatchLongestPrefix(uint64sToBlockHashes(hashes), []ServerID{ServerID(ep.GetMetadata().ID)})
			require.Equal(t, []int{3}, matches)
			require.Equal(t, int64(1), index.adds.Load())
			// Imported prompts are never echoed into the published recent window.
			require.Empty(t, p.recent.snapshot(ep.GetMetadata().ID.String()))
		})
	}
}

func TestBackgroundMergeNeedsSyncer(t *testing.T) {
	p, index := newBackgroundTestProducer(t, backgroundMergeCallback, false)
	registerBackgroundPeer(t, p, []uint64{1, 2})
	p.AfterCrossReplicaPublish(t.Context())
	require.Nil(t, p.backgroundMergeDone, "no background worker without a syncer")
	require.Zero(t, index.adds.Load())
}

func TestProduceDisabledDoesNotAcquireEndpointMergeLock(t *testing.T) {
	p, index := newBackgroundTestProducer(t, backgroundMergeDisabled, true)
	ep := registerBackgroundPeer(t, p, []uint64{101, 102, 103})
	request := &fwksched.InferenceRequest{RequestID: "disabled-with-locked-background", Body: tokenizedBody([]uint32{1, 2, 3})}
	endpoints := []fwksched.Endpoint{fwksched.NewEndpoint(ep.GetMetadata(), ep.GetMetrics(), ep.GetAttributes())}
	done := make(chan error, 1)
	registration, ok := p.registeredEndpoints.Load(ep.GetMetadata().ID.String())
	require.True(t, ok)
	endpointLock := &registration.(*registeredEndpoint).mu
	endpointLock.Lock()
	go func() { done <- p.Produce(t.Context(), request, endpoints) }()
	select {
	case err := <-done:
		endpointLock.Unlock()
		require.NoError(t, err)
		require.Equal(t, int64(1), index.adds.Load())
	case <-time.After(time.Second):
		endpointLock.Unlock()
		<-done
		t.Fatal("Produce blocked on the endpoint merge lock with trigger=disabled")
	}
}

func TestBackgroundMergeDigestAndLog(t *testing.T) {
	p, index := newBackgroundTestProducer(t, backgroundMergeCallback, true)
	registerBackgroundPeer(t, p, []uint64{1, 2, 3})
	var lines []string
	ctx := log.IntoContext(t.Context(), funcr.New(func(_, line string) { lines = append(lines, line) }, funcr.Options{Verbosity: logutil.DEBUG}))
	p.mergeInBackground(ctx)
	p.mergeInBackground(ctx)
	require.Equal(t, int64(1), index.adds.Load())
	require.Len(t, lines, 2)
	for _, fragment := range []string{`"endpoints"=1`, `"promptsAdded"=1`, `"digestSkipped"=0`, `"duration"=`} {
		require.Contains(t, lines[0], fragment)
	}
	require.Contains(t, lines[1], `"promptsAdded"=0`)
	require.Contains(t, lines[1], `"digestSkipped"=1`)
}

func TestBackgroundMergeRejectsDeletedAndRebuiltSnapshots(t *testing.T) {
	for _, rebuild := range []bool{false, true} {
		name := "deleted"
		if rebuild {
			name = "same-name-rebuilt"
		}
		t.Run(name, func(t *testing.T) {
			p, index := newBackgroundTestProducer(t, backgroundMergeCallback, true)
			old := registerBackgroundPeer(t, p, []uint64{1, 2, 3})
			snapshot := p.registeredMergeEndpoints()
			readStarted := make(chan struct{})
			releaseRead := make(chan struct{})
			old.GetAttributes().Put(p.syncHashDK, &fwkdl.DynamicAttribute{Get: func() fwkdl.Cloneable {
				close(readStarted)
				<-releaseRead
				return &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2, 3}}}
			}})
			done := make(chan struct{})
			go func() { defer close(done); p.mergeRemoteHashesInBackground(snapshot) }()
			<-readStarted
			require.NoError(t, p.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: old}))
			if rebuild {
				registerBackgroundPeer(t, p, []uint64{7, 8})
			}
			close(releaseRead)
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("merge did not finish")
			}
			require.Zero(t, index.adds.Load())
			require.Empty(t, p.indexerInst.Pods())
			if rebuild {
				p.mergeInBackground(t.Context())
				require.Equal(t, []int{0}, p.indexerInst.MatchLongestPrefix([]blockHash{1, 2, 3}, []ServerID{ServerID(old.GetMetadata().ID)}))
				require.Equal(t, []int{2}, p.indexerInst.MatchLongestPrefix([]blockHash{7, 8}, []ServerID{ServerID(old.GetMetadata().ID)}))
			} else {
				p.mergeInBackground(t.Context())
				require.Zero(t, index.adds.Load())
			}
		})
	}
}

func TestBackgroundMergeSerializesAddWithExtract(t *testing.T) {
	p, index := newBackgroundTestProducer(t, backgroundMergeCallback, true)
	ep := registerBackgroundPeer(t, p, []uint64{1, 2})
	adding := make(chan struct{})
	release := make(chan struct{})
	index.beforeAdd = func() { close(adding); <-release }
	merged := make(chan struct{})
	go func() { defer close(merged); p.mergeRemoteHashesInBackground(p.registeredMergeEndpoints()) }()
	<-adding
	deleted := make(chan struct{})
	go func() {
		defer close(deleted)
		_ = p.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: ep})
	}()
	select {
	case <-deleted:
		t.Fatal("Extract(delete) passed the identity/Add critical section")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	<-merged
	<-deleted
	require.Zero(t, p.mergeRemoteHashesInBackground(p.registeredMergeEndpoints()).promptsAdded)
}

func TestBackgroundMergeCallbackStopsOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	h := testHandleWithSyncer()
	cfg := defaultConfig
	cfg.BackgroundMerge = backgroundMergeConfig{Trigger: backgroundMergeCallback}
	p, err := newDataProducer(ctx, "cancel-test", cfg, h)
	require.NoError(t, err)
	require.NotNil(t, p.backgroundMergeDone)
	cancel()
	select {
	case <-p.backgroundMergeDone:
	case <-time.After(time.Second):
		t.Fatal("background worker did not exit when constructor context ended")
	}
	registerBackgroundPeer(t, p, []uint64{1, 2})
	p.AfterCrossReplicaPublish(ctx)
	require.Empty(t, p.indexerInst.Pods())
}

func TestBackgroundMergeConfiguration(t *testing.T) {
	for _, tc := range []struct {
		parameters string
		errorText  string
		trigger    string
	}{
		{parameters: `{}`, trigger: backgroundMergeDisabled},
		{parameters: `{"backgroundMerge":{"trigger":"disabled"}}`, trigger: backgroundMergeDisabled},
		{parameters: `{"backgroundMerge":{"trigger":"off"}}`, errorText: "must be disabled or callback"},
		{parameters: `{"backgroundMerge":{"trigger":"callback"}}`, trigger: backgroundMergeCallback},
		{parameters: `{"backgroundMerge":{"trigger":"other"}}`, errorText: "backgroundMerge.trigger"},
	} {
		t.Run(tc.parameters, func(t *testing.T) {
			dec := plugin.StrictDecoder(json.RawMessage(tc.parameters))
			got, err := ApproxPrefixCacheFactory("config-test", dec, testHandle())
			if tc.errorText != "" {
				require.ErrorContains(t, err, tc.errorText)
				return
			}
			require.NoError(t, err)
			p := got.(*dataProducer)
			require.Equal(t, tc.trigger, p.config.BackgroundMerge.Trigger)
		})
	}
}

func TestBackgroundMergeDoesNotBlockOtherEndpointExtract(t *testing.T) {
	p, index := newBackgroundTestProducer(t, backgroundMergeCallback, true)
	registerBackgroundPeer(t, p, []uint64{1, 2})
	adding, release := make(chan struct{}), make(chan struct{})
	index.beforeAdd = func() { close(adding); <-release }
	merged := make(chan struct{})
	go func() { defer close(merged); p.mergeRemoteHashesInBackground(p.registeredMergeEndpoints()) }()
	<-adding
	other := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "ns", Name: "other"}}, nil)
	extracted := make(chan error, 1)
	go func() {
		extracted <- p.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: other})
	}()
	select {
	case err := <-extracted:
		close(release)
		<-merged
		require.NoError(t, err)
	case <-time.After(time.Second):
		close(release)
		<-merged
		<-extracted
		t.Fatal("another endpoint's Extract waited for background Add")
	}
}

func TestBackgroundMergeYieldsToExtractBetweenBatches(t *testing.T) {
	for _, removeFirst := range []bool{false, true} {
		t.Run(fmt.Sprintf("delete-before-rebuild=%t", removeFirst), func(t *testing.T) {
			p, index := newBackgroundTestProducer(t, backgroundMergeCallback, true)
			old := registerBackgroundPeer(t, p, []uint64{1})
			prompts := make([][]uint64, 2*backgroundMergeBatchPrompts+1)
			for i := range prompts {
				prompts[i] = []uint64{uint64(i + 1)}
			}
			old.GetAttributes().Put(p.syncHashDK, &attrprefix.PrefixHashSet{Prompts: prompts})
			snapshot := p.registeredMergeEndpoints()
			registration := snapshot[0].registration
			adding, release := make(chan struct{}), make(chan struct{})
			index.beforeAdd = func() {
				if index.adds.Load() == 0 {
					close(adding)
					<-release
				}
			}
			merged := make(chan struct{})
			go func() { defer close(merged); p.mergeRemoteHashesInBackground(snapshot) }()
			<-adding
			rebuilt := fwkdl.NewEndpoint(old.GetMetadata(), nil)
			extracted := make(chan error, 1)
			go func() {
				if removeFirst {
					if err := p.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: old}); err != nil {
						extracted <- err
						return
					}
				}
				extracted <- p.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: rebuilt})
			}()
			// A queued writer prevents another reader batch from acquiring the lock.
			waiting := false
			deadline := time.Now().Add(time.Second)
			for time.Now().Before(deadline) {
				if !registration.mu.TryRLock() {
					waiting = true
					break
				}
				registration.mu.RUnlock()
				runtime.Gosched()
			}
			close(release)
			<-merged
			require.NoError(t, <-extracted)
			require.True(t, waiting, "Extract writer must be queued before Add is released")
			require.Equal(t, int64(backgroundMergeBatchPrompts), index.adds.Load(), "Extract runs before the next batch")
			id := ServerID(old.GetMetadata().ID)
			require.Equal(t, []int{0}, p.indexerInst.MatchLongestPrefix([]blockHash{backgroundMergeBatchPrompts + 1}, []ServerID{id}))
			_, remembered := p.peerMerged.Load(old.GetMetadata().ID.String())
			require.False(t, remembered, "an interrupted window must not be remembered as complete")
			rebuilt.GetAttributes().Put(p.syncHashDK, &attrprefix.PrefixHashSet{Prompts: prompts})
			p.mergeInBackground(t.Context())
			require.Equal(t, []int{1}, p.indexerInst.MatchLongestPrefix([]blockHash{backgroundMergeBatchPrompts + 1}, []ServerID{id}))
			require.Equal(t, int64(backgroundMergeBatchPrompts+len(prompts)), index.adds.Load())
			rebuilt.GetAttributes().Put(p.syncHashDK, &attrprefix.PrefixHashSet{Prompts: [][]uint64{{999}}})
			p.mergeInBackground(t.Context())
			require.Equal(t, []int{1}, p.indexerInst.MatchLongestPrefix([]blockHash{999}, []ServerID{id}))
			require.Equal(t, int64(backgroundMergeBatchPrompts+len(prompts)+1), index.adds.Load())
		})
	}
}

func TestBackgroundMergeCallbackCoalescesSignals(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	h := testHandle()
	cfg := defaultConfig
	cfg.BackgroundMerge.Trigger = backgroundMergeCallback
	p, err := newDataProducer(ctx, "coalescing-test", cfg, h)
	require.NoError(t, err)
	index := &backgroundTestIndexer{indexerInterface: p.indexerInst}
	p.indexerInst = index
	h.SetCrossReplicaSyncer(stubSyncerPlugin{})
	ep := registerBackgroundPeer(t, p, []uint64{1, 2})
	var reads atomic.Int64
	read := make(chan struct{}, 3)
	ep.GetAttributes().Put(p.syncHashDK, &fwkdl.DynamicAttribute{Get: func() fwkdl.Cloneable {
		reads.Add(1)
		read <- struct{}{}
		return &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2}}}
	}})
	adding, release := make(chan struct{}), make(chan struct{})
	index.beforeAdd = func() { close(adding); <-release }
	p.AfterCrossReplicaPublish(t.Context())
	<-adding
	for range 100 {
		p.AfterCrossReplicaPublish(t.Context())
	}
	require.Len(t, p.backgroundMergeSignals, 1, "only one pending pass is retained")
	close(release)
	for range 2 {
		select {
		case <-read:
		case <-time.After(time.Second):
			t.Fatal("the pending signal did not run")
		}
	}
	cancel()
	select {
	case <-p.backgroundMergeDone:
	case <-time.After(time.Second):
		t.Fatal("worker outlived the constructor context")
	}
	require.Equal(t, int64(2), reads.Load())
	require.Equal(t, int64(1), index.adds.Load())
}

func TestBackgroundMergeSkipsDeletedRegistration(t *testing.T) {
	p, index := newBackgroundTestProducer(t, backgroundMergeCallback, true)
	ep := registerBackgroundPeer(t, p, []uint64{1, 2, 3})
	endpointID := ep.GetMetadata().ID.String()
	value, ok := p.registeredEndpoints.Load(endpointID)
	require.True(t, ok)
	registration := value.(*registeredEndpoint)
	require.NoError(t, p.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: ep}))
	require.Nil(t, registration.endpoint)
	require.Empty(t, p.registeredMergeEndpoints())

	// Range can retain this registration before delete removes its map entry.
	// Keep that retired entry visible to exercise the delayed reader deterministically.
	p.registeredEndpoints.Store(endpointID, registration)
	snapshot := p.registeredMergeEndpoints()
	require.Empty(t, snapshot)
	require.Zero(t, p.mergeRemoteHashesInBackground(snapshot).promptsAdded)
	require.Zero(t, index.adds.Load())
	p.registeredEndpoints.CompareAndDelete(endpointID, registration)
}
