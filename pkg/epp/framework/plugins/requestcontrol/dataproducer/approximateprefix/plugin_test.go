/*
Copyright 2026 The Kubernetes Authors.

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
	"testing"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func testHandle() plugin.Handle {
	return plugin.NewEppHandle(context.Background(), nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
}

// disableMinBlockSizeClamp lowers the runtime block-size floor to 1 for tests
// that exercise prefix-match logic with deliberately small block sizes. The
// production default (64) would clamp tiny test inputs, hiding the behavior
// under test. Restores the previous value on cleanup.
func disableMinBlockSizeClamp(t *testing.T) {
	t.Helper()
	prev := minBlockSizeTokens
	minBlockSizeTokens = 1
	t.Cleanup(func() { minBlockSizeTokens = prev })
}

// tokenizedBody returns a request body carrying only a tokenized prompt.
func tokenizedBody(tokenIDs []uint32) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{
		TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: tokenIDs}}},
	}
}

func TestProduce(t *testing.T) {
	disableMinBlockSizeClamp(t)
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	// Test the "initialize if nil" pattern
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())
	assert.NoError(t, err)
	assert.NotNil(t, p.PluginState())

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1, endpoint2}

	// First request to populate cache.
	req1 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model1",
		Body:        tokenizedBody([]uint32{1, 2}),
	}

	// We need to simulate the PreRequest logic since Produce only reads from the indexer.
	// But first let's see if Produce correctly handles an empty indexer.
	err = p.Produce(context.Background(), req1, endpoints)
	assert.NoError(t, err)

	// Verify state was written to PluginState
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.NotNil(t, state)
	assert.Equal(t, 2, len(state.PerPromptHashes[0])) // 2 token IDs at blockSize 1 -> 2 blocks

	// Verify pod match info was set (should be 0 match since indexer is empty)
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ApproxPrefixCachePluginType).String()
	for _, ep := range endpoints {
		info, ok := ep.Get(key)
		assert.True(t, ok)
		prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
		assert.Equal(t, 0, prefixInfo.MatchBlocks())
		assert.Equal(t, 2, prefixInfo.TotalBlocks())
	}
}

func TestPreRequest(t *testing.T) {
	disableMinBlockSizeClamp(t)
	t.Run("Basic cache update", func(t *testing.T) {
		config := config{
			BlockSizeTokens:        1,
			MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
			LRUCapacityPerServer:   defaultLRUCapacityPerServer,
		}
		p, _ := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())

		endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
		req1 := &fwksched.InferenceRequest{
			RequestID:   uuid.NewString(),
			TargetModel: "test-model1",
			Body:        tokenizedBody([]uint32{1, 2}),
		}

		// 1. Produce data (this saves state)
		_ = p.Produce(context.Background(), req1, []fwksched.Endpoint{endpoint1})

		// 2. Simulate scheduling result
		res := &fwksched.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {
					TargetEndpoints: []fwksched.Endpoint{endpoint1},
				},
			},
		}

		// 3. Call PreRequest
		p.PreRequest(context.Background(), req1, res)

		// Wait for async update
		p.wg.Wait()

		// 4. Verify indexer was updated
		perPromptHashes := getBlockHashes(context.Background(), req1, config.BlockSizeTokens, defaultMaxPrefixBlocks)
		for _, promptHashes := range perPromptHashes {
			for _, hash := range promptHashes {
				pods := p.indexer().Get(hash)
				assert.Contains(t, pods, ServerID(endpoint1.GetMetadata().NamespacedName))
			}
		}
	})

	t.Run("Respects LRUCapacityPerServer config", func(t *testing.T) {
		config := config{
			AutoTune:               false,
			BlockSizeTokens:        1,
			MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
			LRUCapacityPerServer:   2,
		}
		p, _ := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())

		endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())

		// Three requests with distinct token IDs generate distinct hashes.
		// BlockSizeTokens is 1, so each single-token request yields one block.
		tokenSets := [][]uint32{{1}, {2}, {3}}
		allHashes := make([][]blockHash, 0, len(tokenSets))

		for _, tokenIDs := range tokenSets {
			req := &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "test-model1",
				Body:        tokenizedBody(tokenIDs),
			}
			_ = p.Produce(context.Background(), req, []fwksched.Endpoint{endpoint1})

			res := &fwksched.SchedulingResult{
				PrimaryProfileName: "default",
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"default": {
						TargetEndpoints: []fwksched.Endpoint{endpoint1},
					},
				},
			}
			p.PreRequest(context.Background(), req, res)
			p.wg.Wait()

			perPromptHashes := getBlockHashes(context.Background(), req, config.BlockSizeTokens, defaultMaxPrefixBlocks)
			allHashes = append(allHashes, perPromptHashes[0])
		}

		// Since capacity is 2, the first request's hash should have been evicted.
		// The latter two should still be present.
		assert.Empty(t, p.indexer().Get(allHashes[0][0]))
		assert.NotEmpty(t, p.indexer().Get(allHashes[1][0]))
		assert.NotEmpty(t, p.indexer().Get(allHashes[2][0]))
	})
}

func TestDataProducerValidation(t *testing.T) {
	validConfigs := []config{{
		AutoTune:        false,
		BlockSizeTokens: 1,
	}, {
		AutoTune:        false,
		BlockSize:       1,
		BlockSizeTokens: 1,
	}, {
		AutoTune:        true,
		BlockSizeTokens: 0,
	}}
	invalidConfigs := []config{{
		AutoTune:  false,
		BlockSize: 1,
	}, {
		AutoTune:        false,
		BlockSizeTokens: 0,
	}}

	for _, config := range validConfigs {
		_, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())
		assert.NoError(t, err)
	}

	for _, config := range invalidConfigs {
		_, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())
		assert.Error(t, err)
	}
}

func TestPrefixPluginPartialPrefixMatch(t *testing.T) {
	disableMinBlockSizeClamp(t)
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint3 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1, endpoint2, endpoint3}

	// First request: tokens [1, 2].
	req1 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model1",
		Body:        tokenizedBody([]uint32{1, 2}),
	}
	_ = p.Produce(context.Background(), req1, endpoints)
	state, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.Equal(t, 2, len(state.PerPromptHashes[0]))

	// Simulate pod1 was picked and pod3 was picked as a prefill node.
	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default":                         {TargetEndpoints: []fwksched.Endpoint{endpoint1}},
			experimentalDefaultPrefillProfile: {TargetEndpoints: []fwksched.Endpoint{endpoint3}},
		},
	}
	p.PreRequest(context.Background(), req1, schedulingResult)
	p.wg.Wait()

	// Second request shares the first token but diverges on the second.
	req3 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model1",
		Body:        tokenizedBody([]uint32{1, 3}),
	}
	_ = p.Produce(context.Background(), req3, endpoints)

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ApproxPrefixCachePluginType).String()
	// Verify pod1 has the correct prefix match info
	info1, _ := endpoint1.Get(key)
	prefixInfo1 := info1.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 1, prefixInfo1.MatchBlocks()) // one block (token 1) matches
	assert.Equal(t, 2, prefixInfo1.TotalBlocks()) // [1, 3] -> 2 blocks

	// Verify pod3 (prefill node) also has the match
	info3, _ := endpoint3.Get(key)
	prefixInfo3 := info3.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 1, prefixInfo3.MatchBlocks())

	// Verify pod2 has no match info
	info2, _ := endpoint2.Get(key)
	prefixInfo2 := info2.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 0, prefixInfo2.MatchBlocks())
}

func TestPrefixPluginPrefixGrowth(t *testing.T) {
	disableMinBlockSizeClamp(t)
	config := config{
		BlockSizeTokens:        2,
		AutoTune:               false,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1}

	// First request with an initial token prefix.
	req1 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model1",
		Body:        tokenizedBody([]uint32{1, 2, 3, 4, 5, 6}),
	}
	_ = p.Produce(context.Background(), req1, endpoints)
	state1, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	initialHashCount := len(state1.PerPromptHashes[0])
	assert.Greater(t, initialHashCount, 0)

	// Simulate pod1 was picked
	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint1}},
		},
	}
	p.PreRequest(context.Background(), req1, schedulingResult)
	p.wg.Wait()

	// Second request extends the first one's token prefix.
	req2 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model1",
		Body:        tokenizedBody([]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
	}
	_ = p.Produce(context.Background(), req2, endpoints)
	state2, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req2.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	extendedHashCount := len(state2.PerPromptHashes[0])
	assert.Greater(t, extendedHashCount, initialHashCount)

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ApproxPrefixCachePluginType).String()
	info, _ := endpoint1.Get(key)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
	assert.Greater(t, prefixInfo.MatchBlocks(), 0, "should have prefix cache hit")
	assert.Equal(t, extendedHashCount, prefixInfo.TotalBlocks())
}

func TestPrefixPluginAutoTune(t *testing.T) {
	podName := "pod-autotune"
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: podName}},
		&fwkdl.Metrics{
			// Pod reports a block size above minBlockSizeTokens so the autotune
			// path passes the metric through unclamped. (Metric values below the
			// minimum are clamped per #1158; see TestGetBlockSize_AutotuneClampsBelowMinimum.)
			CacheBlockSize: 128,  // 128 tokens per block
			CacheNumBlocks: 1000, // 1000 blocks capacity
		}, fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint}

	// 192 token IDs at the pod's block size of 128 -> 2 blocks.
	tokenIDs := make([]uint32, 192)
	for i := range tokenIDs {
		tokenIDs[i] = uint32(i + 1)
	}
	req := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        tokenizedBody(tokenIDs),
	}

	config := config{
		AutoTune:               true,
		BlockSizeTokens:        256, // Should be ignored in favor of pod metrics (128)
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   1,
	}
	p, _ := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())

	_ = p.Produce(context.Background(), req, endpoints)
	state, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	// 192 tokens / 128 tokens per block = 2 blocks.
	assert.Equal(t, 2, len(state.PerPromptHashes[0]), "Should use pod block size (128 tokens) -> 2 blocks")

	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
	p.PreRequest(context.Background(), req, schedulingResult)
	p.wg.Wait()

	// Check indexer state - should be in tracked pods
	assert.Contains(t, p.indexer().Pods(), ServerID(endpoint.GetMetadata().NamespacedName))
}

func TestMaxPrefixTokensToMatch(t *testing.T) {
	disableMinBlockSizeClamp(t)
	// BlockSizeTokens=1, MaxPrefixTokensToMatch=2 -> maxBlocks = 2/1 = 2.
	// Only the first 2 token blocks of the prompt should be hashed.
	cfg := config{
		BlockSizeTokens:        1,
		MaxPrefixTokensToMatch: 2,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	assert.NoError(t, err)

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)

	// 4 token IDs = 4 blocks at blockSize 1, but should be capped to 2.
	req := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        tokenizedBody([]uint32{1, 2, 3, 4}),
	}

	err = p.Produce(context.Background(), req, []fwksched.Endpoint{endpoint})
	assert.NoError(t, err)

	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.Equal(t, 2, len(state.PerPromptHashes[0]), "should cap at MaxPrefixTokensToMatch/BlockSizeTokens = 2 blocks")

	// When MaxPrefixTokensToMatch is 0 (unset), fall back to MaxPrefixBlocksToMatch.
	cfg2 := config{
		BlockSizeTokens:        1,
		MaxPrefixTokensToMatch: 0,
		MaxPrefixBlocksToMatch: 3,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p2, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg2, testHandle())
	assert.NoError(t, err)

	req2 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        tokenizedBody([]uint32{1, 2, 3, 4}),
	}

	err = p2.Produce(context.Background(), req2, []fwksched.Endpoint{endpoint})
	assert.NoError(t, err)

	state2, err := plugin.ReadPluginStateKey[*SchedulingContextState](p2.PluginState(), req2.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.Equal(t, 3, len(state2.PerPromptHashes[0]), "should fall back to MaxPrefixBlocksToMatch when MaxPrefixTokensToMatch is 0")
}

// TestGetBlockSize_AutotuneClampsBelowMinimum verifies that when AutoTune is on and
// the endpoint reports a small CacheBlockSize, GetBlockSize floors the result at
// minBlockSizeTokens to bound EPP indexer memory. See issue #1158.
func TestGetBlockSize_AutotuneClampsBelowMinimum(t *testing.T) {
	cfg := config{
		AutoTune:        true,
		BlockSizeTokens: 16, // also small; metric should override but clamp wins
	}
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	assert.NoError(t, err)

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
		&fwkdl.Metrics{CacheBlockSize: 16}, // model server uses small blocks
		fwkdl.NewAttributes(),
	)

	got := p.GetBlockSize([]fwksched.Endpoint{endpoint})
	assert.Equal(t, minBlockSizeTokens, got,
		"autotuned block size below the minimum should be clamped to minBlockSizeTokens")
}

// TestGetBlockSize_AutotuneAboveMinimumPassesThrough verifies the clamp is one-sided —
// metric values at or above the minimum are returned unchanged.
func TestGetBlockSize_AutotuneAboveMinimumPassesThrough(t *testing.T) {
	cfg := config{AutoTune: true, BlockSizeTokens: 16}
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	assert.NoError(t, err)

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
		&fwkdl.Metrics{CacheBlockSize: 128},
		fwkdl.NewAttributes(),
	)

	got := p.GetBlockSize([]fwksched.Endpoint{endpoint})
	assert.Equal(t, 128, got, "autotuned block size at or above minimum should not be clamped")
}

// TestGetBlockSize_ManualConfigClampedBelowMinimum verifies that the floor
// applies to manual configuration as well — configured BlockSizeTokens below
// minBlockSizeTokens is silently raised so the indexer memory bound holds
// across both manual and autotune paths.
func TestGetBlockSize_ManualConfigClampedBelowMinimum(t *testing.T) {
	cfg := config{AutoTune: false, BlockSizeTokens: 32}
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	assert.NoError(t, err)

	got := p.GetBlockSize(nil)
	assert.Equal(t, minBlockSizeTokens, got,
		"manual BlockSizeTokens below the minimum should be clamped to the floor")
}

// TestGetBlockSize_AutotuneFallbackClampsLowConfig verifies that the floor applies
// to the autotune fallback path too — when AutoTune is on but no endpoint metric is
// available, the configured BlockSizeTokens still gets clamped. This is the path the
// default config (AutoTune=true, BlockSizeTokens=16) would land on if endpoint
// metrics are missing.
func TestGetBlockSize_AutotuneFallbackClampsLowConfig(t *testing.T) {
	cfg := config{AutoTune: true, BlockSizeTokens: 16} // default config shape
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	assert.NoError(t, err)

	// No endpoints passed — exercise the autotune fallback path.
	got := p.GetBlockSize(nil)
	assert.Equal(t, minBlockSizeTokens, got,
		"autotune fallback should clamp BlockSizeTokens to the floor when below minimum")
}

// TestNewDataProducer_AcceptsLowManualBlockSize verifies that a low
// BlockSizeTokens with AutoTune off does not cause initialization to fail —
// GetBlockSize clamps the effective value to minBlockSizeTokens at request
// time, but the producer construction itself accepts any positive
// BlockSizeTokens.
func TestNewDataProducer_AcceptsLowManualBlockSize(t *testing.T) {
	cfg := config{AutoTune: false, BlockSizeTokens: 32}
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	assert.NoError(t, err, "low manual BlockSizeTokens should not error at init")
	assert.NotNil(t, p)
}

// BenchmarkPrefixPluginStress is a stress test using token prompts of increasing length.
func BenchmarkPrefixPluginStress(b *testing.B) {
	config := config{
		BlockSizeTokens:        16,
		MaxPrefixBlocksToMatch: 50000,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())

	promptLen := []int{1024, 4096, 10000, 50000}

	for _, v := range promptLen {
		b.Run(fmt.Sprintf("length_%d", v), func(b *testing.B) {
			b.ReportAllocs()
			tokenIDs := make([]uint32, v)
			for i := range tokenIDs {
				tokenIDs[i] = uint32(i)
			}
			endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod1"},
			}, nil, fwkdl.NewAttributes())
			endpoints := []fwksched.Endpoint{endpoint}
			req := &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "model-stress",
				Body:        tokenizedBody(tokenIDs),
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = p.Produce(context.Background(), req, endpoints)
				p.PluginState().Delete(req.RequestID)
			}
		})
	}
}

// TestFactory_RejectsUnknownField verifies that strict JSON parsing rejects
// unknown fields in the plugin config. Encodes the strict-parsing policy
// from issue #1068.
func TestFactory_RejectsUnknownField(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	handle := plugin.NewEppHandle(ctx, func() []k8stypes.NamespacedName { return nil }, plugin.WithMetricsRecorder(prometheus.NewRegistry()))

	dec := plugin.StrictDecoder(json.RawMessage(`{"unknownField": "value"}`))
	_, err := ApproxPrefixCacheFactory("test", dec, handle)

	assert.Error(t, err, "factory must reject unknown config fields")
	if err != nil {
		assert.Contains(t, err.Error(), "unknownField",
			"error message should name the offending field")
	}
}

// TestFactory_DeprecatedBlockSizeMapped verifies that the deprecated
// 'blockSize' field is accepted (with a warning) and maps to
// 'blockSizeTokens'. Encodes the two-pass deprecation policy from #1068:
// deprecated fields are valid configuration, not errors.
func TestFactory_DeprecatedBlockSizeMapped(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	handle := plugin.NewEppHandle(ctx, func() []k8stypes.NamespacedName { return nil }, plugin.WithMetricsRecorder(prometheus.NewRegistry()))

	// User supplies only the deprecated field. AutoTune=false forces
	// BlockSizeTokens to be a meaningful value (no auto-tune fallback).
	dec := plugin.StrictDecoder(json.RawMessage(`{"autoTune": false, "blockSize": 24}`))
	p, err := ApproxPrefixCacheFactory("test", dec, handle)

	assert.NoError(t, err, "deprecated blockSize should be accepted, not rejected")
	if err != nil {
		return
	}

	dp, ok := p.(*dataProducer)
	if !ok {
		t.Fatalf("expected *dataProducer, got %T", p)
	}

	assert.Equal(t, 24, dp.config.BlockSizeTokens,
		"deprecated 'blockSize' should map to BlockSizeTokens")
	assert.Equal(t, 0, dp.config.BlockSize,
		"deprecated 'blockSize' should be cleared after mapping")
}

func TestProduce_MultiPrompt(t *testing.T) {
	disableMinBlockSizeClamp(t)
	cfg := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	assert.NoError(t, err)

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)
	endpoints := []fwksched.Endpoint{endpoint}

	req := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}, {TokenIDs: []uint32{4, 5}}},
			},
		},
	}

	err = p.Produce(context.Background(), req, endpoints)
	assert.NoError(t, err)

	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.Equal(t, 2, len(state.PerPromptHashes), "should have hashes for 2 prompts")
	assert.Equal(t, 3, len(state.PerPromptHashes[0]), "first prompt: 3 tokens at blockSize 1")
	assert.Equal(t, 2, len(state.PerPromptHashes[1]), "second prompt: 2 tokens at blockSize 1")

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ApproxPrefixCachePluginType).String()
	info, ok := endpoint.Get(key)
	assert.True(t, ok)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 0, prefixInfo.MatchBlocks(), "empty indexer -> no match")
	assert.Equal(t, 5, prefixInfo.TotalBlocks(), "total blocks = 3 + 2")
}

func TestMultiPromptMatchAggregation(t *testing.T) {
	disableMinBlockSizeClamp(t)
	cfg := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)
	endpoints := []fwksched.Endpoint{endpoint}

	// Seed the indexer with a multi-prompt request.
	req1 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}, {TokenIDs: []uint32{4, 5}}},
			},
		},
	}
	_ = p.Produce(context.Background(), req1, endpoints)
	p.PreRequest(context.Background(), req1, &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: endpoints},
		},
	})
	p.wg.Wait()

	// Second request with the same two prompts — all blocks should match.
	req2 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}, {TokenIDs: []uint32{4, 5}}},
			},
		},
	}
	_ = p.Produce(context.Background(), req2, endpoints)

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ApproxPrefixCachePluginType).String()
	info, _ := endpoint.Get(key)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 5, prefixInfo.MatchBlocks(), "all 5 blocks (3+2) should match")
	assert.Equal(t, 5, prefixInfo.TotalBlocks())
}

func TestMultiPromptPartialMatch(t *testing.T) {
	disableMinBlockSizeClamp(t)
	cfg := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)
	endpoints := []fwksched.Endpoint{endpoint}

	// Seed with two prompts: [1,2] and [3,4].
	req1 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2}}, {TokenIDs: []uint32{3, 4}}},
			},
		},
	}
	_ = p.Produce(context.Background(), req1, endpoints)
	p.PreRequest(context.Background(), req1, &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: endpoints},
		},
	})
	p.wg.Wait()

	// Query with [1,2] (matches) and [5,6] (no match).
	req2 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2}}, {TokenIDs: []uint32{5, 6}}},
			},
		},
	}
	_ = p.Produce(context.Background(), req2, endpoints)

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ApproxPrefixCachePluginType).String()
	info, _ := endpoint.Get(key)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 2, prefixInfo.MatchBlocks(), "only first prompt's 2 blocks should match")
	assert.Equal(t, 4, prefixInfo.TotalBlocks(), "total blocks = 2 + 2")
}

func TestPrefixPluginTokenizedRequest(t *testing.T) {
	disableMinBlockSizeClamp(t)
	cfg := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())
	assert.NoError(t, err)

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)
	endpoints := []fwksched.Endpoint{endpoint}

	// 4 token IDs -> 4 blocks at blockSize 1.
	req := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        tokenizedBody([]uint32{10, 20, 30, 40}),
	}

	err = p.Produce(context.Background(), req, endpoints)
	assert.NoError(t, err)

	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.NotNil(t, state)
	assert.Equal(t, 4, len(state.PerPromptHashes[0]))

	// Verify match info was set on the endpoint (0 match since indexer is empty).
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ApproxPrefixCachePluginType).String()
	info, ok := endpoint.Get(key)
	assert.True(t, ok)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 0, prefixInfo.MatchBlocks())
	assert.Equal(t, 4, prefixInfo.TotalBlocks())
}

func TestPrefixPluginMatchesSameTokens(t *testing.T) {
	// Two requests with identical token IDs should produce the same hashes.
	cfg := config{
		BlockSizeTokens:      1,
		LRUCapacityPerServer: defaultLRUCapacityPerServer,
	}
	p, _ := newDataProducer(context.Background(), ApproxPrefixCachePluginType, cfg, testHandle())

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)
	endpoints := []fwksched.Endpoint{endpoint}

	tokenIDs := []uint32{1, 2, 3, 4, 5, 6, 7, 8}

	req1 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        tokenizedBody(tokenIDs),
	}
	req2 := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        tokenizedBody(tokenIDs),
	}

	_ = p.Produce(context.Background(), req1, endpoints)
	state1, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))

	_ = p.Produce(context.Background(), req2, endpoints)
	state2, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req2.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))

	assert.Equal(t, state1.PerPromptHashes, state2.PerPromptHashes, "identical token IDs must produce identical hashes")
}
