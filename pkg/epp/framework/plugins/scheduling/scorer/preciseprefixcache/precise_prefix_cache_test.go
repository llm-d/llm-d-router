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
	"encoding/json"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	preciseproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache"
	"github.com/llm-d/llm-d-router/test/utils"
)

func TestAnyMMHit(t *testing.T) {
	const producerName = "test-producer"
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName)
	makeEndpoint := func(name string, info *attrprefix.PrefixCacheMatchInfo) scheduling.Endpoint {
		ep := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}}, fwkdl.NewMetrics(), nil)
		if info != nil {
			ep.Put(key, info)
		}
		return ep
	}

	tests := []struct {
		name        string
		endpoints   []scheduling.Endpoint
		wantHit     bool
		wantTracked bool
	}{
		{name: "no endpoints", endpoints: nil},
		{
			name: "no match info attached",
			endpoints: []scheduling.Endpoint{
				makeEndpoint("a", nil),
				makeEndpoint("b", nil),
			},
		},
		{
			name: "no producer tracked mm",
			endpoints: []scheduling.Endpoint{
				makeEndpoint("a", attrprefix.NewPrefixCacheMatchInfo(5, 10, 16)),
				makeEndpoint("b", attrprefix.NewPrefixCacheMatchInfo(8, 10, 16)),
			},
		},
		{
			name: "tracked but zero mm matches",
			endpoints: []scheduling.Endpoint{
				makeEndpoint("a", attrprefix.NewPrefixCacheMatchInfo(5, 10, 16)),
				makeEndpoint("b", attrprefix.NewPrefixCacheMatchInfo(8, 10, 16).WithMM(attrprefix.MMMatchInfo{MatchBlocks: 0})),
			},
			wantTracked: true,
		},
		{
			name: "one endpoint reports mm match",
			endpoints: []scheduling.Endpoint{
				makeEndpoint("a", attrprefix.NewPrefixCacheMatchInfo(5, 10, 16)),
				makeEndpoint("b", attrprefix.NewPrefixCacheMatchInfo(8, 10, 16).WithMM(attrprefix.MMMatchInfo{MatchBlocks: 2})),
			},
			wantHit: true, wantTracked: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hit, tracked := anyMMHit(tt.endpoints, key)
			assert.Equal(t, tt.wantHit, hit, "hit")
			assert.Equal(t, tt.wantTracked, tracked, "tracked")
		})
	}
}

// In self-host mode the plugin satisfies Scorer, DataProducer, PreRequest,
// and EndpointExtractor.
func TestPluginFactory_SelfHostInterfaces(t *testing.T) {
	handle := fwkplugin.NewEppHandle(utils.NewTestContext(t), nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	plg, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(`{}`)), handle)
	require.NoError(t, err)

	_, ok := plg.(scheduling.Scorer)
	assert.True(t, ok, "plugin must be a Scorer")
	_, ok = plg.(requestcontrol.DataProducer)
	assert.True(t, ok, "plugin must be a DataProducer")
	_, ok = plg.(requestcontrol.PreRequest)
	assert.True(t, ok, "plugin must be a PreRequest")
	_, ok = plg.(fwkdl.EndpointExtractor)
	assert.True(t, ok, "plugin must be an EndpointExtractor")
}

// The inner scorer's Consumes set must include every key the inner producer
// Produces, so the data-layer DAG links them.
func TestPluginFactory_InnerScorerConsumesProducerKeys(t *testing.T) {
	handle := fwkplugin.NewEppHandle(utils.NewTestContext(t), nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	plg, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(`{}`)), handle)
	require.NoError(t, err)
	p := plg.(*Plugin)

	produces := p.Produces()
	require.Len(t, produces, 1)
	consumes := p.Consumes()
	for k := range produces {
		_, inRequired := consumes.Required[k]
		_, inOptional := consumes.Optional[k]
		assert.True(t, inRequired || inOptional, "Consumes must include produced key %s", k.String())
	}
}

// With a precise-prefix-cache-producer pre-registered, the factory returns
// a Scorer-only plugin pointed at it.
func TestPluginFactory_DefersToExistingProducer(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := fwkplugin.NewEppHandle(ctx, nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	existing, err := preciseproducer.PluginFactory("my-precise", nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(existing.TypedName().Name, existing)

	plg, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(`{"speculativeIndexing":true}`)), handle)
	require.NoError(t, err)

	_, isScorer := plg.(scheduling.Scorer)
	assert.True(t, isScorer)
	_, isProducer := plg.(requestcontrol.DataProducer)
	assert.False(t, isProducer, "defer-mode plugin must not be a DataProducer")
}

// Two precise-prefix-cache-producer instances leave the legacy plugin unable
// to choose; the factory errors instead of picking one non-deterministically.
func TestPluginFactory_RejectsMultipleExistingProducers(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := fwkplugin.NewEppHandle(ctx, nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	first, err := preciseproducer.PluginFactory("first", nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(first.TypedName().Name, first)

	second, err := preciseproducer.PluginFactory("second", nil, handle)
	require.NoError(t, err)
	handle.AddPlugin(second.TypedName().Name, second)

	_, err = PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(`{}`)), handle)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "multiple precise-prefix-cache-producer instances")
}

// The self-hosted path forwards indexerConfig verbatim to the producer, whose
// indexer has no tokenization pool, so a config still carrying one is rejected
// by strict decoding rather than silently ignored.
func TestPluginFactory_RejectsTokenizersPoolConfig(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := fwkplugin.NewEppHandle(ctx, nil,
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	raw := json.RawMessage(`{"indexerConfig":{"tokenizersPoolConfig":{"modelName":"x"}}}`)

	_, err := PluginFactory("test", fwkplugin.StrictDecoder(raw), handle)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `unknown field "tokenizersPoolConfig"`)
}
