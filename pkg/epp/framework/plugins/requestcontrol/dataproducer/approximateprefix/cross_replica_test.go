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
	"testing"

	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestRecentWindowCapAndSlide(t *testing.T) {
	w := newRecentWindow(3)
	w.record("ep", []blockHash{1, 2, 3, 4, 5})
	require.Equal(t, []blockHash{1, 2, 3}, w.snapshot("ep"))

	w.record("ep", []blockHash{6, 7})
	require.Equal(t, []blockHash{3, 6, 7}, w.snapshot("ep"))
	require.Equal(t, []blockHash{3, 6, 7}, w.snapshot("ep"), "snapshot must not clear")

	w.forget("ep")
	require.Nil(t, w.snapshot("ep"))
}

func TestCrossReplicaMergeLetsPeerMatch(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.MaxSyncedHashes = 64
	cfg.LRUCapacityPerServer = 100

	ctx := context.Background()
	producerA, err := newDataProducer(ctx, ApproxPrefixCachePluginType, cfg, testHandle())
	require.NoError(t, err)
	producerB, err := newDataProducer(ctx, ApproxPrefixCachePluginType, cfg, testHandle())
	require.NoError(t, err)

	podID := k8stypes.NamespacedName{Namespace: "ns", Name: "decode-0"}
	endpointID := podID.String()
	pod := server{ServerID: ServerID(podID), NumOfGPUBlocks: 100}
	hashes := []blockHash{101, 102, 103}

	producerA.indexerInst.Add(hashes, pod)
	producerA.recent.record(endpointID, hashes)

	specA := producerA.CrossReplicaState()
	require.False(t, specA.SyncDisabled)
	localA := specA.Read(endpointID)
	agg := specA.Aggregate([]any{localA})
	set, ok := agg.(*attrprefix.PrefixHashSet)
	require.True(t, ok)
	require.ElementsMatch(t, []uint64{101, 102, 103}, set.Hashes)

	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: podID},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
	// Mimic redis-syncer Get: Aggregate(live local, peer aggregate).
	ep.Put(producerB.syncHashDK, &fwkdl.DynamicAttribute{
		Get: func() fwkdl.Cloneable {
			return producerB.CrossReplicaState().Aggregate([]any{
				producerB.CrossReplicaState().Read(endpointID),
				set,
			}).(fwkdl.Cloneable)
		},
	})

	candidates := []ServerID{ServerID(podID)}
	require.Equal(t, []int{0}, producerB.indexerInst.MatchLongestPrefix(hashes, candidates))
	producerB.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, []int{3}, producerB.indexerInst.MatchLongestPrefix(hashes, candidates))

	// Merged hashes must not enter B's recent window (no echo).
	require.Nil(t, producerB.recent.snapshot(endpointID))
}

func TestCrossReplicaAggregateUnionsReplicas(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.LRUCapacityPerServer = 100
	cfg.MaxSyncedHashes = 3
	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	require.NoError(t, err)

	spec := producer.CrossReplicaState()
	a := &attrprefix.PrefixHashSet{Hashes: []uint64{1, 2, 3}}
	b := &attrprefix.PrefixHashSet{Hashes: []uint64{3, 4}}
	got := spec.Aggregate([]any{a, b}).(*attrprefix.PrefixHashSet)
	require.Equal(t, []uint64{2, 3, 4}, got.Hashes, "union capped to maxSyncedHashes, newest-N of append order")

	// Get with only local (no peer cache) still returns the local window.
	producer.recent.record("ns/decode-0", []blockHash{9, 8})
	onlyLocal := spec.Aggregate([]any{spec.Read("ns/decode-0")}).(*attrprefix.PrefixHashSet)
	require.ElementsMatch(t, []uint64{9, 8}, onlyLocal.Hashes)
}

func TestCrossReplicaDeclaresSyncHashKey(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	require.NoError(t, err)

	_, ok := producer.Produces()[producer.syncHashDK]
	require.True(t, ok, "Produces must declare PrefixHashSet so Put is allowed")
	_, ok = producer.Consumes().Optional[producer.syncHashDK]
	require.True(t, ok, "Consumes must declare PrefixHashSet so mergeRemoteHashes Get is allowed")

	syncOff := false
	cfg.SyncCrossReplicaState = &syncOff
	localOnly, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	require.NoError(t, err)
	_, ok = localOnly.Produces()[localOnly.syncHashDK]
	require.False(t, ok)
	require.Nil(t, localOnly.Consumes().Optional)
}

func TestMergeRemoteHashesSkipsLocalOnlyWindow(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.LRUCapacityPerServer = 100
	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	require.NoError(t, err)

	podID := k8stypes.NamespacedName{Namespace: "ns", Name: "decode-0"}
	endpointID := podID.String()
	hashes := []blockHash{201, 202, 203}
	producer.recent.record(endpointID, hashes)

	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: podID},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
	// Syncer-less Extract shape: attribute is local Read only.
	ep.Put(producer.syncHashDK, &fwkdl.DynamicAttribute{
		Get: func() fwkdl.Cloneable {
			return producer.CrossReplicaState().Read(endpointID)
		},
	})

	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, []int{0}, producer.indexerInst.MatchLongestPrefix(hashes, []ServerID{ServerID(podID)}),
		"local-only attribute must not populate the indexer")
}
