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

package contextlengthaware

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	concretedatalayer "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/p2psource"
)

const (
	migrationBlockSize         = 64
	migrationInitialTokens     = 102400
	migrationFollowUpTokens    = 103424
	migrationWorkRangeLabel    = "llm-d.ai/prefill-work-range"
	migrationPrefixProducer    = "precise-prefix"
	migrationP2PProducer       = "session-migration-p2p"
	migrationLongSourceAddress = "10.0.0.10"
)

func TestP2PReusablePrefixDependency(t *testing.T) {
	p2p := p2psource.New(migrationP2PProducer, p2psource.Config{MinCachedTokenDelta: 1})
	workRouter := NewContextLengthAware("prefill-work-router", &contextLengthAwareParameters{
		Label:                      migrationWorkRangeLabel,
		ReusableTokensProducerName: migrationP2PProducer,
	})
	key := p2psource.ReusablePrefixTokensDataKey.WithNonEmptyProducerName(migrationP2PProducer)

	produced, producesKey := p2p.Produces()[key]
	consumed, consumesKey := workRouter.Consumes().Required[key]
	require.True(t, producesKey)
	require.True(t, consumesKey)
	assert.IsType(t, produced, consumed)

	ordered, err := concretedatalayer.ValidateAndOrderDataDependencies([]fwkplugin.Plugin{workRouter, p2p})
	require.NoError(t, err)
	assert.Less(t, pluginOrder(ordered, p2p.TypedName().String()),
		pluginOrder(ordered, workRouter.TypedName().String()))
}

func TestP2PReusablePrefixRoutesFollowUpToShortPrefiller(t *testing.T) {
	ctx := context.Background()
	prefixDataKey := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(migrationPrefixProducer)
	p2p := p2psource.New(migrationP2PProducer, p2psource.Config{
		PrefixMatchInfoProducerName: migrationPrefixProducer,
		MinCachedTokenDelta:         1,
		PrefillProfileName:          "prefill",
	})
	workRouter := NewContextLengthAware("prefill-work-router", &contextLengthAwareParameters{
		Label:                      migrationWorkRangeLabel,
		EnableFiltering:            true,
		ReusableTokensProducerName: migrationP2PProducer,
	})
	reusableTokensKey := p2psource.ReusablePrefixTokensDataKey.WithNonEmptyProducerName(migrationP2PProducer)

	// A cold 100K-token request has no transferable source, so its full prompt
	// is prefill work and only the long-prefill worker matches.
	coldLong := migrationPrefiller("p-long", migrationLongSourceAddress, "8193-131072",
		prefixDataKey, 0, migrationInitialTokens/migrationBlockSize)
	coldShort := migrationPrefiller("p-short", "10.0.0.20", "0-8192",
		prefixDataKey, 0, migrationInitialTokens/migrationBlockSize)
	coldRequest := migrationTokenizedRequest("cold-100k", migrationInitialTokens)

	require.NoError(t, p2p.Produce(ctx, coldRequest, []scheduling.Endpoint{coldLong, coldShort}))
	_, hasReusableTokens := scheduling.ReadRequestAttribute[p2psource.ReusablePrefixTokens](coldRequest, reusableTokensKey)
	assert.False(t, hasReusableTokens)
	coldCandidates := workRouter.Filter(ctx, coldRequest, []scheduling.Endpoint{coldLong, coldShort})
	require.Len(t, coldCandidates, 1)
	assert.Equal(t, "p-long", coldCandidates[0].GetMetadata().ID.Name)
	require.NoError(t, p2p.PreRequest(ctx, coldRequest, migrationPrefillResult(coldCandidates[0])))
	assert.NotContains(t, coldRequest.Headers, routing.KVCacheSourceHeader)

	// The next turn has 102400 CPU-tier tokens on P-long and 1024 new tokens.
	// Produce publishes that request-wide reusable floor before filtering, so
	// the work-range filter selects P-short and PreRequest points it at P-long.
	warmLong := migrationPrefiller("p-long", migrationLongSourceAddress, "8193-131072",
		prefixDataKey, migrationInitialTokens/migrationBlockSize, migrationFollowUpTokens/migrationBlockSize)
	warmShort := migrationPrefiller("p-short", "10.0.0.20", "0-8192",
		prefixDataKey, 0, migrationFollowUpTokens/migrationBlockSize)
	followUpRequest := migrationTokenizedRequest("follow-up-101k", migrationFollowUpTokens)

	require.NoError(t, p2p.Produce(ctx, followUpRequest, []scheduling.Endpoint{warmLong, warmShort}))
	reusableTokens, ok := scheduling.ReadRequestAttribute[p2psource.ReusablePrefixTokens](followUpRequest, reusableTokensKey)
	require.True(t, ok)
	assert.Equal(t, p2psource.ReusablePrefixTokens(migrationInitialTokens), reusableTokens)
	warmCandidates := workRouter.Filter(ctx, followUpRequest, []scheduling.Endpoint{warmLong, warmShort})
	require.Len(t, warmCandidates, 1)
	assert.Equal(t, "p-short", warmCandidates[0].GetMetadata().ID.Name)
	require.NoError(t, p2p.PreRequest(ctx, followUpRequest, migrationPrefillResult(warmCandidates[0])))
	assert.Equal(t, migrationLongSourceAddress+":8080", followUpRequest.Headers[routing.KVCacheSourceHeader])
}

func migrationPrefiller(
	name, address, workRange string,
	prefixDataKey fwkplugin.DataKey,
	cachedBlocks, totalBlocks int,
) scheduling.Endpoint {
	endpoint := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      k8stypes.NamespacedName{Namespace: "default", Name: name},
		Name:    name,
		Address: address,
		Port:    "8080",
		Labels:  map[string]string{migrationWorkRangeLabel: workRange},
	}, nil, nil)
	endpoint.Put(prefixDataKey,
		attrprefix.NewPrefixCacheMatchInfo(cachedBlocks, totalBlocks, migrationBlockSize).
			WithCachedBlockCount(cachedBlocks).
			WithCachedBlocksByTier(map[string]int{"cpu": cachedBlocks}))
	return endpoint
}

func migrationTokenizedRequest(requestID string, tokenCount int) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		RequestID: requestID,
		Headers:   map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, tokenCount)}},
			},
		},
	}
}

func migrationPrefillResult(endpoint scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"prefill": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
}

func pluginOrder(plugins []string, name string) int {
	for i, pluginName := range plugins {
		if pluginName == name {
			return i
		}
	}
	return len(plugins)
}
