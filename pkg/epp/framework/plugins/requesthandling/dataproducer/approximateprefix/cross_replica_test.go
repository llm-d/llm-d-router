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

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// stubSyncerPlugin satisfies plugin.Plugin so tests can mark a CrossReplicaSyncer
// as configured without exercising Redis.
type stubSyncerPlugin struct{}

func (stubSyncerPlugin) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "test-syncer", Name: "test"}
}

func testHandleWithSyncer() plugin.Handle {
	h := plugin.NewEppHandle(context.Background(), nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
	h.SetCrossReplicaSyncer(stubSyncerPlugin{})
	return h
}

func TestRecentWindowCapAndSlide(t *testing.T) {
	w := newRecentWindow(3)
	w.record("ep", []blockHash{1, 2, 3, 4, 5})
	require.Equal(t, [][]blockHash{{1, 2, 3}}, w.snapshot("ep"), "oversized prompt head-capped")

	w.record("ep", []blockHash{6, 7})
	require.Equal(t, [][]blockHash{{6, 7}}, w.snapshot("ep"), "oldest whole prompt dropped to fit cap")
	require.Equal(t, [][]blockHash{{6, 7}}, w.snapshot("ep"), "snapshot must not clear")

	w.forget("ep")
	require.Nil(t, w.snapshot("ep"))
}

func TestRecentWindowKeepsPromptBoundaries(t *testing.T) {
	w := newRecentWindow(5)
	w.record("ep", []blockHash{1, 2})
	w.record("ep", []blockHash{3, 4})
	require.Equal(t, [][]blockHash{{1, 2}, {3, 4}}, w.snapshot("ep"))

	w.record("ep", []blockHash{5, 6})
	require.Equal(t, [][]blockHash{{3, 4}, {5, 6}}, w.snapshot("ep"),
		"drop oldest prompt when total exceeds max")
}

func TestCrossReplicaMergeLetsPeerMatch(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.MaxSyncedTokens = 64 * 64 // 64 hashes at BlockSizeTokens=64
	cfg.LRUCapacityPerServer = 100

	ctx := context.Background()
	producerA, err := newDataProducer(ctx, ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)
	producerB, err := newDataProducer(ctx, ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
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
	require.Equal(t, [][]uint64{{101, 102, 103}}, set.Prompts)

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

func TestMergeRemoteHashesPreservesPromptBoundaries(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.LRUCapacityPerServer = 2

	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)

	podID := k8stypes.NamespacedName{Namespace: "ns", Name: "decode-0"}
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: podID},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
	// Two prompts of two hashes each. A flat Add would insert oldest-first at
	// the end and keep only the old pair under capacity 2; per-prompt Add keeps
	// the newest prompt.
	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{
		Prompts: [][]uint64{{1, 2}, {3, 4}},
	})

	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	candidates := []ServerID{ServerID(podID)}
	require.Equal(t, []int{0}, producer.indexerInst.MatchLongestPrefix([]blockHash{1, 2}, candidates),
		"older prompt should be evicted under tight LRU")
	require.Equal(t, []int{2}, producer.indexerInst.MatchLongestPrefix([]blockHash{3, 4}, candidates),
		"newest prompt must survive when merged with prompt boundaries")
}

func TestMergeRemoteHashesAddsPeerPromptsOnce(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.LRUCapacityPerServer = 2

	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)

	podID := k8stypes.NamespacedName{Namespace: "ns", Name: "decode-0"}
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: podID},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{
		Prompts: [][]uint64{{1, 2}, {3, 4}},
	})
	candidates := []ServerID{ServerID(podID)}

	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, []int{2}, producer.indexerInst.MatchLongestPrefix([]blockHash{3, 4}, candidates))

	// Seed a local prompt that a re-Add of the peer window would evict under
	// capacity 2. A second merge must not LRU-touch the peer prompts.
	producer.indexerInst.Add([]blockHash{9, 10}, server{ServerID: ServerID(podID), NumOfGPUBlocks: 2})
	require.Equal(t, []int{2}, producer.indexerInst.MatchLongestPrefix([]blockHash{9, 10}, candidates))

	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, []int{2}, producer.indexerInst.MatchLongestPrefix([]blockHash{9, 10}, candidates),
		"re-merge must not re-Add peer prompts and evict newer local entries")
}

func TestMergeRemoteHashesPrunesRememberedPeerKeys(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.LRUCapacityPerServer = 2

	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)

	podID := k8stypes.NamespacedName{Namespace: "ns", Name: "decode-0"}
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: podID},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
	candidates := []ServerID{ServerID(podID)}
	endpointID := podID.String()

	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{
		Prompts: [][]uint64{{1, 2}, {3, 4}},
	})
	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, 2, peerMergedSize(t, producer, endpointID))
	require.Equal(t, []int{0}, producer.indexerInst.MatchLongestPrefix([]blockHash{1, 2}, candidates),
		"setup: older peer prompt already LRU-evicted under capacity 2")

	// Peer window slides to only {3,4}. Remembered keys must shrink so {1,2}
	// can be treated as novel if it returns.
	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{
		Prompts: [][]uint64{{3, 4}},
	})
	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, 1, peerMergedSize(t, producer, endpointID),
		"remembered peer keys must prune to the active sync window")

	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{
		Prompts: [][]uint64{{1, 2}},
	})
	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, 1, peerMergedSize(t, producer, endpointID))
	require.Equal(t, []int{2}, producer.indexerInst.MatchLongestPrefix([]blockHash{1, 2}, candidates),
		"prompt that left and re-entered the peer window must be re-indexed")
}

func peerMergedSize(t *testing.T, producer *dataProducer, endpointID string) int {
	t.Helper()
	v, ok := producer.peerMerged.Load(endpointID)
	require.True(t, ok, "expected peerMerged entry for %s", endpointID)
	s := v.(*mergedKeySet)
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.keys)
}

func TestMergeRemoteHashesSkipsUnchangedDigest(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.LRUCapacityPerServer = 100

	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)

	podID := k8stypes.NamespacedName{Namespace: "ns", Name: "decode-0"}
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: podID},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
	endpointID := podID.String()
	prompts := [][]uint64{{1, 2, 3}, {4, 5}}
	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{Prompts: prompts})

	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.True(t, producer.peerMergedDigestMatches(endpointID, (&attrprefix.PrefixHashSet{Prompts: prompts}).ContentDigest()))

	// Same content, new slice: digest match must skip without clearing keys.
	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{
		Prompts: [][]uint64{{1, 2, 3}, {4, 5}},
	})
	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, 2, peerMergedSize(t, producer, endpointID))

	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{
		Prompts: [][]uint64{{1, 2, 3}, {4, 5}, {6}},
	})
	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, 3, peerMergedSize(t, producer, endpointID),
		"content change must invalidate digest and merge the novel prompt")
}

func TestMergeRemoteHashesNoopWithoutSyncer(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.LRUCapacityPerServer = 100

	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	require.NoError(t, err)
	require.False(t, producer.syncerActive())

	podID := k8stypes.NamespacedName{Namespace: "ns", Name: "decode-0"}
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: podID},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
	ep.Put(producer.syncHashDK, &attrprefix.PrefixHashSet{
		Prompts: [][]uint64{{1, 2, 3}},
	})
	producer.mergeRemoteHashes([]fwksched.Endpoint{ep})
	require.Equal(t, []int{0}, producer.indexerInst.MatchLongestPrefix([]blockHash{1, 2, 3}, []ServerID{ServerID(podID)}),
		"merge must not run when no CrossReplicaSyncer is configured")
}

func TestCrossReplicaAggregateUnionsReplicas(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.LRUCapacityPerServer = 100
	cfg.BlockSizeTokens = 64
	cfg.MaxSyncedTokens = 3 * 64 // 3 hashes
	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)

	spec := producer.CrossReplicaState()
	a := &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2, 3}}}
	b := &attrprefix.PrefixHashSet{Prompts: [][]uint64{{3, 4}}}
	got := spec.Aggregate([]any{a, b}).(*attrprefix.PrefixHashSet)
	require.Equal(t, [][]uint64{{1, 2, 3}, {3, 4}}, got.Prompts,
		"union keeps each replica's already-capped window without re-trimming")

	// Get with only local (no peer cache) still returns the local window.
	producer.recent.record("ns/decode-0", []blockHash{9, 8})
	onlyLocal := spec.Aggregate([]any{spec.Read("ns/decode-0")}).(*attrprefix.PrefixHashSet)
	require.Equal(t, [][]uint64{{9, 8}}, onlyLocal.Prompts)
}

func TestCrossReplicaAggregateKeepsAllReplicas(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	cfg.BlockSizeTokens = 64
	cfg.MaxSyncedTokens = 64 // 1 hash
	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)

	spec := producer.CrossReplicaState()
	// Three replicas each publish a full one-hash window. A global cap of 1
	// would keep only the last replica ID's prompt under redis-syncer's sort.
	a := &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1}}}
	b := &attrprefix.PrefixHashSet{Prompts: [][]uint64{{2}}}
	c := &attrprefix.PrefixHashSet{Prompts: [][]uint64{{3}}}
	got := spec.Aggregate([]any{a, b, c}).(*attrprefix.PrefixHashSet)
	require.Equal(t, [][]uint64{{1}, {2}, {3}}, got.Prompts)
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
	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
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

func TestExtractIgnoresStaleDelete(t *testing.T) {
	syncOn := true
	cfg := defaultConfig
	cfg.SyncCrossReplicaState = &syncOn
	cfg.AutoTune = false
	producer, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)

	podID := k8stypes.NamespacedName{Namespace: "ns", Name: "decode-0"}
	endpointID := podID.String()
	live := fwkdl.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: podID},
		fwkdl.NewMetrics(),
	)
	require.NoError(t, producer.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: live,
	}))
	producer.recent.record(endpointID, []blockHash{1, 2, 3})

	stale := fwkdl.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: podID},
		fwkdl.NewMetrics(),
	)
	require.NoError(t, producer.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: stale,
	}))
	require.Equal(t, [][]blockHash{{1, 2, 3}}, producer.recent.snapshot(endpointID),
		"stale delete for a replaced endpoint must not clear the live recent window")

	require.NoError(t, producer.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: live,
	}))
	require.Nil(t, producer.recent.snapshot(endpointID))
}
