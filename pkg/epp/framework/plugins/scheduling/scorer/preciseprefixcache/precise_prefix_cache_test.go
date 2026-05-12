package preciseprefixcache

import (
	"context"
	"testing"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

// newSpeculativeFakeScorer builds a Scorer with an in-test fake kvCacheIndexer
// (real in-memory kvblock.Index + fixed ComputeBlockKeys), avoiding the
// embedded tokenizer path removed in llm-d/llm-d-kv-cache#473.
func newSpeculativeFakeScorer(t *testing.T, speculativeTTL time.Duration) (*Scorer, []kvblock.BlockHash) {
	t.Helper()
	ctx := utils.NewTestContext(t)

	blockKeys := []kvblock.BlockHash{
		kvblock.BlockHash(111),
		kvblock.BlockHash(222),
		kvblock.BlockHash(333),
	}

	index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	mockIndexer := &mockKVCacheIndexer{
		kvBlockIndex: index,
		computeBlockKeysFunc: func(_ context.Context, _ *types.RenderChatRequest, _, _ string) ([]kvblock.BlockHash, error) {
			return blockKeys, nil
		},
	}

	kvBlockScorer, err := kvcache.NewKVBlockScorer(kvcache.DefaultKVBlockScorerConfig())
	require.NoError(t, err)

	// Mirror production OnEviction so TTL expiry actually evicts from the index.
	speculativeCache := ttlcache.New[string, *speculativeEntries](
		ttlcache.WithTTL[string, *speculativeEntries](speculativeTTL),
	)
	speculativeCache.OnEviction(func(_ context.Context, reason ttlcache.EvictionReason,
		item *ttlcache.Item[string, *speculativeEntries],
	) {
		if reason != ttlcache.EvictionReasonExpired {
			return
		}
		entries := item.Value()
		for _, reqKey := range entries.blockKeys {
			//nolint:errcheck // best-effort cleanup, mirrors production
			index.Evict(context.Background(), reqKey, kvblock.RequestKey, entries.podEntries)
		}
	})

	// DiscoverPods=false skips subscriber wiring so we don't need ZMQ here.
	kvEventsConfig := kvevents.DefaultConfig()
	kvEventsConfig.DiscoverPods = false

	return &Scorer{
		typedName:          plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvCacheIndexer:     mockIndexer,
		kvBlockScorer:      kvBlockScorer,
		kvEventsConfig:     kvEventsConfig,
		pluginState:        plugin.NewPluginState(ctx),
		speculativeCache:   speculativeCache,
		speculativeTTL:     speculativeTTL,
		blockSizeTokens:    16,
		speculativeEnabled: true,
		subscriberCtx:      ctx,
	}, blockKeys
}

func speculativeRequest(id string) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		RequestID:   id,
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			// Body must be non-nil; prompt content is ignored by the fake indexer.
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "ignored-by-fake-indexer"},
			},
		},
	}
}

func testEndpoint(name, address string) scheduling.Endpoint {
	return scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Name: name},
		Address:        address,
		Port:           "8080",
	}, nil, nil)
}

func defaultProfileResult(endpoints []scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: endpoints},
		},
	}
}

// TestProduce_PopulatesPluginState verifies Produce stores block keys in PluginState.
func TestProduce_PopulatesPluginState(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer, blockKeys := newSpeculativeFakeScorer(t, 5*time.Second)

	request := speculativeRequest("test-produce")
	endpoints := []scheduling.Endpoint{testEndpoint("pod-a", "10.0.0.1")}

	err := scorer.Produce(ctx, request, endpoints)
	require.NoError(t, err)

	state, err := plugin.ReadPluginStateKey[*precisePluginState](
		scorer.pluginState, request.RequestID, stateKey)
	require.NoError(t, err)
	require.NotNil(t, state)
	assert.Equal(t, blockKeys, state.blockKeys)
}

// TestScoreReusesPluginState verifies Score reuses block keys persisted by Produce.
func TestScoreReusesPluginState(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer, blockKeys := newSpeculativeFakeScorer(t, 5*time.Second)

	// Populate index: pod-a has all blocks, pod-b only the first.
	index := scorer.kvCacheIndexer.KVBlockIndex()
	err := index.Add(ctx, []kvblock.BlockHash{kvblock.EmptyBlockHash},
		[]kvblock.BlockHash{blockKeys[0]}, []kvblock.PodEntry{
			{PodIdentifier: "10.0.0.1:8080"},
			{PodIdentifier: "10.0.0.2:8080"},
		})
	require.NoError(t, err)
	for _, k := range blockKeys[1:] {
		err := index.Add(ctx, []kvblock.BlockHash{kvblock.EmptyBlockHash},
			[]kvblock.BlockHash{k}, []kvblock.PodEntry{{PodIdentifier: "10.0.0.1:8080"}})
		require.NoError(t, err)
	}

	endpoints := []scheduling.Endpoint{
		testEndpoint("pod-a", "10.0.0.1"),
		testEndpoint("pod-b", "10.0.0.2"),
	}

	request := speculativeRequest("test-reuse")

	err = scorer.Produce(ctx, request, endpoints)
	require.NoError(t, err)

	// pod-a covers all 3 blocks (score 1.0); pod-b covers only the first of
	// three (score 1/3).
	scores := scorer.Score(ctx, scheduling.NewCycleState(), request, endpoints)
	require.Len(t, scores, 2)

	gotByAddress := map[string]float64{}
	for ep, score := range scores {
		m := ep.GetMetadata()
		gotByAddress[m.Address+":"+m.Port] = score
	}
	assert.InDelta(t, 1.0, gotByAddress["10.0.0.1:8080"], 1e-9)
	assert.InDelta(t, 1.0/3.0, gotByAddress["10.0.0.2:8080"], 1e-9)
}

// TestPreRequest_AddsSpeculativeEntries verifies PreRequest writes speculative
// entries to both the index and the TTL cache.
func TestPreRequest_AddsSpeculativeEntries(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer, blockKeys := newSpeculativeFakeScorer(t, 5*time.Second)

	endpoints := []scheduling.Endpoint{testEndpoint("pod-a", "10.0.0.1")}

	request := speculativeRequest("test-speculative")

	// 1. Produce populates PluginState.
	require.NoError(t, scorer.Produce(ctx, request, endpoints))

	// 2. Simulate scheduling selecting pod-a.
	scorer.PreRequest(ctx, request, defaultProfileResult(endpoints))

	// 3. Speculative entry must exist in the index.
	index := scorer.kvCacheIndexer.KVBlockIndex()
	keyToPods, err := index.Lookup(ctx, blockKeys, sets.New[string]())
	require.NoError(t, err)

	found := false
	for _, pod := range keyToPods[blockKeys[0]] {
		if pod.PodIdentifier == "10.0.0.1:8080" && pod.Speculative {
			found = true
			break
		}
	}
	assert.True(t, found, "speculative entry for pod-a should be present")

	// 4. TTL cache must track the entry for later eviction.
	item := scorer.speculativeCache.Get(request.RequestID)
	require.NotNil(t, item)
	assert.Equal(t, len(blockKeys), len(item.Value().blockKeys))
}

// TestSpeculativeEntriesEvictOnTTL verifies speculative entries are evicted
// from the index when their TTL expires.
func TestSpeculativeEntriesEvictOnTTL(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer, blockKeys := newSpeculativeFakeScorer(t, 200*time.Millisecond)

	endpoints := []scheduling.Endpoint{testEndpoint("pod-a", "10.0.0.1")}

	request := speculativeRequest("test-ttl-evict")
	require.NoError(t, scorer.Produce(ctx, request, endpoints))

	scorer.PreRequest(ctx, request, defaultProfileResult(endpoints))

	index := scorer.kvCacheIndexer.KVBlockIndex()
	keyToPods, err := index.Lookup(ctx, blockKeys[:1], sets.New[string]())
	require.NoError(t, err)
	require.Greater(t, len(keyToPods[blockKeys[0]]), 0,
		"index should hold the speculative entry before TTL expires")

	// ttlcache v3 runs OnEviction in a separate goroutine, so the index
	// Evict call lands asynchronously; poll with Eventually.
	time.Sleep(500 * time.Millisecond)
	scorer.speculativeCache.DeleteExpired()

	require.Eventually(t, func() bool {
		keyToPods, err := index.Lookup(ctx, blockKeys[:1], sets.New[string]())
		if err != nil {
			return false
		}
		for _, pod := range keyToPods[blockKeys[0]] {
			if pod.Speculative {
				return false
			}
		}
		return true
	}, 2*time.Second, 20*time.Millisecond,
		"speculative entries should be evicted after TTL")
}

// TestKVBlockScorerIntegration guards the kv-cache wiring used by Produce.
// The scoring semantics themselves are owned upstream.
func TestKVBlockScorerIntegration(t *testing.T) {
	scorer, err := kvcache.NewKVBlockScorer(kvcache.DefaultKVBlockScorerConfig())
	require.NoError(t, err)

	blockKeys := []kvblock.BlockHash{
		kvblock.BlockHash(111),
		kvblock.BlockHash(222),
		kvblock.BlockHash(333),
	}

	t.Run("contiguous prefix", func(t *testing.T) {
		keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{
			blockKeys[0]: {{PodIdentifier: "pod-a"}, {PodIdentifier: "pod-b"}},
			blockKeys[1]: {{PodIdentifier: "pod-a"}},
			blockKeys[2]: {{PodIdentifier: "pod-a"}},
		}

		scores, err := scorer.Score(t.Context(), blockKeys, keyToPods)
		require.NoError(t, err)
		assert.Equal(t, float64(3), scores["pod-a"])
		assert.Equal(t, float64(1), scores["pod-b"]) // only first block
	})

	t.Run("prefix breaks at gap", func(t *testing.T) {
		keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{
			blockKeys[0]: {{PodIdentifier: "pod-a"}},
			// blockKeys[1] missing -> prefix chain breaks
			blockKeys[2]: {{PodIdentifier: "pod-a"}},
		}

		scores, err := scorer.Score(t.Context(), blockKeys, keyToPods)
		require.NoError(t, err)
		assert.Equal(t, float64(1), scores["pod-a"]) // only first block counted
	})

	t.Run("empty index", func(t *testing.T) {
		keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{}
		scores, err := scorer.Score(t.Context(), blockKeys, keyToPods)
		require.NoError(t, err)
		assert.Empty(t, scores)
	})
}
