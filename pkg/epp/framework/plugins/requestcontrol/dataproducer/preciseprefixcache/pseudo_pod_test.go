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

package preciseprefixcache

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/test/utils"
)

func endpointOnNode(name, address, node string) scheduling.Endpoint {
	return scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:       k8stypes.NamespacedName{Name: name},
		Address:  address,
		Port:     "8080",
		NodeName: node,
	}, nil, nil)
}

func matchInfo(t *testing.T, ep scheduling.Endpoint) *attrprefix.PrefixCacheMatchInfo {
	t.Helper()
	raw, ok := ep.Get(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test"))
	require.True(t, ok)
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	require.True(t, ok)
	return info
}

// A node:<n> index entry credits only the candidate endpoints on node n; a
// pool:<name> entry credits every candidate endpoint. The lookup filter
// carries the node pseudo-pods of all candidate nodes.
func TestProduce_PseudoPodEntriesCreditEndpoints(t *testing.T) {
	ctx := utils.NewTestContext(t)
	endpoints := []scheduling.Endpoint{
		endpointOnNode("pod-a", "10.0.0.1", "node-n"),
		endpointOnNode("pod-b", "10.0.0.2", "node-n"),
		endpointOnNode("pod-c", "10.0.0.3", "node-m"),
	}

	prompt := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	keys := []kvblock.BlockHash{0xA1, 0xA2}
	var seenFilter sets.Set[string]

	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(_ context.Context, _ []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return keys, nil
		},
		index: &fakeKVBlockIndex{
			lookup: func(_ context.Context, _ []kvblock.BlockHash, podSet sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
				seenFilter = podSet
				return map[kvblock.BlockHash][]kvblock.PodEntry{
					keys[0]: {{PodIdentifier: kvblock.NodePseudoPod("node-n"), DeviceTier: "lmcache-l1"}},
					keys[1]: {{PodIdentifier: kvblock.PoolPseudoPod("fs"), DeviceTier: "lmcache-l2-fs"}},
				}, nil
			},
		},
	}
	scorer, err := kvcache.NewKVBlockScorer(kvcache.DefaultKVBlockScorerConfig())
	require.NoError(t, err)

	p := newProducerWithIndexer(ctx, idx, scorer)
	req := &scheduling.InferenceRequest{
		RequestID:   "req-pseudo-pod",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: prompt}}},
		},
	}
	require.NoError(t, p.Produce(ctx, req, endpoints))

	assert.True(t, seenFilter.HasAll("10.0.0.1:8080", "10.0.0.2:8080", "10.0.0.3:8080",
		kvblock.NodePseudoPod("node-n"), kvblock.NodePseudoPod("node-m")))

	for _, ep := range endpoints[:2] {
		info := matchInfo(t, ep)
		assert.Equal(t, 2, info.CachedBlockCount(), ep.GetMetadata().Name)
		// Per-tier counts are contiguous from block 0; lmcache-l2-fs starts at block 1.
		assert.Equal(t, map[string]int{"lmcache-l1": 1}, info.CachedBlocksByTier())
		// lmcache-l1 weight 0.8 + unknown lmcache-l2-fs at default weight 1.0.
		assert.Equal(t, 1, info.MatchBlocks())
	}

	// pod-c is on another node: the node:node-n entry at block 0 breaks its
	// prefix chain, so the pool entry at block 1 does not count.
	info := matchInfo(t, endpoints[2])
	assert.Equal(t, 0, info.CachedBlockCount())
	assert.Empty(t, info.CachedBlocksByTier())
	assert.Equal(t, 0, info.MatchBlocks())
}
