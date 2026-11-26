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

package approximateprefix

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// stubPeerReaderSyncer marks a syncer that supports background import.
type stubPeerReaderSyncer struct{ stubSyncerPlugin }

func (stubPeerReaderSyncer) PeerSnapshot(fwkdl.CrossReplicaSpec, string) (any, uint64, bool) {
	return nil, 0, false
}

func newImportingProducer(t *testing.T) *dataProducer {
	t.Helper()
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.LRUCapacityPerServer = 1000
	h := plugin.NewEppHandle(context.Background(), nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
	h.SetCrossReplicaSyncer(stubPeerReaderSyncer{})
	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, h)
	require.NoError(t, err)
	require.True(t, producer.peerImportOffPath())
	return producer
}

var importPodID = k8stypes.NamespacedName{Namespace: "ns", Name: "decode-0"}

func registerImportEndpoint(t *testing.T, producer *dataProducer) {
	t.Helper()
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: importPodID}, fwkdl.NewMetrics())
	require.NoError(t, producer.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
}

func matchLen(producer *dataProducer, hashes ...blockHash) int {
	return producer.indexerInst.MatchLongestPrefix(hashes, []ServerID{ServerID(importPodID)})[0]
}

func TestImportCrossReplicaBeforeRegistration(t *testing.T) {
	producer := newImportingProducer(t)
	err := producer.ImportCrossReplica(context.Background(), importPodID.String(), &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1}}})
	require.ErrorIs(t, err, errEndpointNotReady)
}

func TestImportCrossReplicaRejectsUnexpectedPeerType(t *testing.T) {
	producer := newImportingProducer(t)
	registerImportEndpoint(t, producer)
	err := producer.ImportCrossReplica(context.Background(), importPodID.String(), attrprefix.PrefixHashSet{Prompts: [][]uint64{{1}}})
	require.ErrorContains(t, err, "unexpected type")
}

func TestImportCrossReplicaAddsPeerPromptsWithoutEcho(t *testing.T) {
	producer := newImportingProducer(t)
	registerImportEndpoint(t, producer)
	endpointID := importPodID.String()
	producer.recent.record(endpointID, []blockHash{9, 9})

	peer := &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2, 3}, {4, 5}, {9, 9}}}
	require.NoError(t, producer.ImportCrossReplica(context.Background(), endpointID, peer))

	require.Equal(t, 3, matchLen(producer, 1, 2, 3))
	require.Equal(t, 2, matchLen(producer, 4, 5))
	require.Equal(t, 2, peerMergedSize(t, producer, endpointID), "local prompts are not recorded as imported")
	require.Equal(t, [][]blockHash{{9, 9}}, producer.recent.snapshot(endpointID), "imported prompts must not enter the recent window")
}

func TestImportCrossReplicaEmptyPeerClearsKeysOnly(t *testing.T) {
	producer := newImportingProducer(t)
	registerImportEndpoint(t, producer)
	endpointID := importPodID.String()
	require.NoError(t, producer.ImportCrossReplica(context.Background(), endpointID, &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2, 3}}}))

	require.NoError(t, producer.ImportCrossReplica(context.Background(), endpointID, &attrprefix.PrefixHashSet{}))
	require.Equal(t, 0, peerMergedSize(t, producer, endpointID))
	require.Equal(t, 3, matchLen(producer, 1, 2, 3), "indexer entries stay when the peer goes away")

	require.NoError(t, producer.ImportCrossReplica(context.Background(), endpointID, &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2, 3}}}))
	require.Equal(t, 1, peerMergedSize(t, producer, endpointID), "a returning peer prompt is imported again")
}

// cancelingIndexer cancels a context after a number of Adds.
type cancelingIndexer struct {
	indexerInterface
	cancelAfter int
	cancel      context.CancelFunc
	adds        atomic.Int64
}

func (c *cancelingIndexer) Add(hashes []blockHash, s server) []blockHash {
	inserted := c.indexerInterface.Add(hashes, s)
	if int(c.adds.Add(1)) == c.cancelAfter {
		c.cancel()
	}
	return inserted
}

func TestImportCrossReplicaCanceledMidwayRetriesTheRest(t *testing.T) {
	producer := newImportingProducer(t)
	registerImportEndpoint(t, producer)
	endpointID := importPodID.String()
	prompts := make([][]uint64, 100)
	for i := range prompts {
		prompts[i] = []uint64{uint64(1000 + i), uint64(2000 + i)}
	}
	peer := &attrprefix.PrefixHashSet{Prompts: prompts}

	ctx, cancel := context.WithCancel(context.Background())
	wrapped := &cancelingIndexer{indexerInterface: producer.indexerInst, cancelAfter: 40, cancel: cancel}
	producer.indexerInst = wrapped
	err := producer.ImportCrossReplica(ctx, endpointID, peer)
	require.ErrorIs(t, err, context.Canceled)
	recorded := peerMergedSize(t, producer, endpointID)
	require.Less(t, recorded, 40, "the Add that saw cancellation is not recorded")
	require.LessOrEqual(t, int64(recorded), wrapped.adds.Load(), "keys are recorded only after their Add")

	require.NoError(t, producer.ImportCrossReplica(context.Background(), endpointID, peer))
	require.Equal(t, 100, peerMergedSize(t, producer, endpointID))
	for _, prompt := range prompts {
		require.Equal(t, 2, matchLen(producer, blockHash(prompt[0]), blockHash(prompt[1])))
	}
}

func TestImportCrossReplicaCanceledBeforeStart(t *testing.T) {
	producer := newImportingProducer(t)
	registerImportEndpoint(t, producer)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := producer.ImportCrossReplica(ctx, importPodID.String(), &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2}}})
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, 0, peerMergedSize(t, producer, importPodID.String()))
	require.Equal(t, 0, matchLen(producer, 1, 2))
}

func TestProduceDoesNotReadSyncAttributeWithPeerReader(t *testing.T) {
	disableMinBlockSizeClamp(t)
	producer := newImportingProducer(t)
	_, declared := producer.Consumes().Optional[producer.syncHashDK]
	require.False(t, declared, "background import does not consume the sync attribute")

	var reads atomic.Int64
	ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: importPodID}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	ep.Put(producer.syncHashDK, &fwkdl.DynamicAttribute{Get: func() fwkdl.Cloneable {
		reads.Add(1)
		return &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2}}}
	}})
	req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "m", Body: tokenizedBody([]uint32{1, 2})}
	require.NoError(t, producer.Produce(context.Background(), req, []fwksched.Endpoint{ep}))
	require.Zero(t, reads.Load())
}

func TestMergeRemoteHashesSkipsEndpointBeingImported(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.LRUCapacityPerServer = 100
	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)
	require.False(t, producer.peerImportOffPath())

	ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: importPodID}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2, 3}}})
	keys := producer.mergedKeys(importPodID.String())
	keys.importMu.Lock()
	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	keys.importMu.Unlock()
	require.Equal(t, 0, matchLen(producer, 1, 2, 3))

	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, 3, matchLen(producer, 1, 2, 3), "a skipped endpoint is merged by a later request")
}

func TestRemoveInactivePodsKeepsRegisteredEndpoints(t *testing.T) {
	producer := newImportingProducer(t)
	registerImportEndpoint(t, producer)
	gone := k8stypes.NamespacedName{Namespace: "ns", Name: "gone"}
	producer.indexerInst.Add([]blockHash{1}, server{ServerID: ServerID(importPodID), NumOfGPUBlocks: 10})
	producer.indexerInst.Add([]blockHash{1}, server{ServerID: ServerID(gone), NumOfGPUBlocks: 10})

	producer.removeInactivePods(context.Background(), nil)

	require.ElementsMatch(t, []ServerID{ServerID(importPodID)}, producer.indexerInst.Pods(),
		"a registered endpoint missing from a stale pod list is kept")
}

func TestExtractStaleDeleteReleasesLifecycleLock(t *testing.T) {
	producer := newImportingProducer(t)
	registerImportEndpoint(t, producer)
	stale := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: importPodID}, fwkdl.NewMetrics())
	require.NoError(t, producer.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: stale}))

	done := make(chan struct{})
	go func() {
		defer close(done)
		registerImportEndpoint(t, producer)
		producer.removeInactivePods(context.Background(), nil)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("lifecycleMu still held after a stale delete")
	}
}
