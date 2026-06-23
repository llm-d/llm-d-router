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
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// modelLenEndpoint builds an endpoint whose /v1/models attribute reports the
// given max_model_len values, one per model entry. With no values, the attribute
// is left unset.
func modelLenEndpoint(maxModelLens ...int) fwksched.Endpoint {
	attrs := fwkdl.NewAttributes()
	if maxModelLens != nil {
		models := make(attrmodels.ModelDataCollection, len(maxModelLens))
		for i, ml := range maxModelLens {
			models[i] = attrmodels.ModelData{ID: "m", MaxModelLen: ml}
		}
		attrs.Put(attrmodels.ModelsAttributeKey.String(), models)
	}
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), attrs)
}

// TestProduceAutoTunesFromModelLen verifies the prefix cap is derived from the
// endpoint's max_model_len attribute when auto-tuning, and falls back otherwise.
func TestProduceAutoTunesFromModelLen(t *testing.T) {
	disableMinBlockSizeClamp(t)

	// A 20-block prompt at block size 1.
	tokens := make([]uint32, 20)
	for i := range tokens {
		tokens[i] = uint32(i + 1)
	}

	produceBlocks := func(t *testing.T, p *dataProducer, ep fwksched.Endpoint) int {
		t.Helper()
		req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "m", Body: tokenizedBody(tokens)}
		assert.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{ep}))
		state, err := plugin.ReadPluginStateKey[*SchedulingContextState](
			p.PluginState(), req.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
		assert.NoError(t, err)
		return len(state.PerPromptHashes[0])
	}

	autoTuneConfig := config{
		AutoTune:               true,
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		MaxPrefixTokensToMatch: defaultMaxPrefixTokens, // left at default -> auto-tune on
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}

	t.Run("caps at max_model_len when known", func(t *testing.T) {
		p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, autoTuneConfig, testHandle())
		assert.NoError(t, err)
		assert.True(t, p.config.autoTuneMaxPrefixTokens)
		// max_model_len = 10 -> cap = 10/blockSize(1) = 10 blocks.
		assert.Equal(t, 10, produceBlocks(t, p, modelLenEndpoint(10)))
	})

	t.Run("uses largest across models", func(t *testing.T) {
		p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, autoTuneConfig, testHandle())
		assert.NoError(t, err)
		// Largest wins regardless of position (first entry here).
		assert.Equal(t, 15, produceBlocks(t, p, modelLenEndpoint(15, 8, 12)))
	})

	t.Run("falls back on negative max_model_len", func(t *testing.T) {
		p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, autoTuneConfig, testHandle())
		assert.NoError(t, err)
		assert.Equal(t, 20, produceBlocks(t, p, modelLenEndpoint(-5)))
	})

	t.Run("falls back on empty models collection", func(t *testing.T) {
		p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, autoTuneConfig, testHandle())
		assert.NoError(t, err)
		// Attribute present but no model entries.
		attrs := fwkdl.NewAttributes()
		attrs.Put(attrmodels.ModelsAttributeKey.String(), attrmodels.ModelDataCollection{})
		ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), attrs)
		assert.Equal(t, 20, produceBlocks(t, p, ep))
	})

	t.Run("falls back to default when attribute absent", func(t *testing.T) {
		p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, autoTuneConfig, testHandle())
		assert.NoError(t, err)
		// No models attribute -> default cap (131072) far exceeds 20 blocks.
		assert.Equal(t, 20, produceBlocks(t, p, modelLenEndpoint()))
	})

	t.Run("falls back when max_model_len is zero", func(t *testing.T) {
		p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, autoTuneConfig, testHandle())
		assert.NoError(t, err)
		// Attribute present but server did not report the field.
		assert.Equal(t, 20, produceBlocks(t, p, modelLenEndpoint(0)))
	})

	t.Run("explicit cap is not auto-tuned", func(t *testing.T) {
		explicit := autoTuneConfig
		explicit.MaxPrefixTokensToMatch = 5 // operator-pinned, non-default
		p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, explicit, testHandle())
		assert.NoError(t, err)
		assert.False(t, p.config.autoTuneMaxPrefixTokens)
		// Even with a known model length, the explicit cap (5) wins.
		assert.Equal(t, 5, produceBlocks(t, p, modelLenEndpoint(10)))
	})

	t.Run("empty pods uses default cap", func(t *testing.T) {
		p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, autoTuneConfig, testHandle())
		assert.NoError(t, err)

		req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "m", Body: tokenizedBody(tokens)}
		assert.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{}))
		state, err := plugin.ReadPluginStateKey[*SchedulingContextState](
			p.PluginState(), req.RequestID, plugin.StateKey(ApproxPrefixCachePluginType))
		assert.NoError(t, err)
		assert.Equal(t, 20, len(state.PerPromptHashes[0]))
	})
}

// TestConsumesDeclaresModelsDependency verifies the producer requires the
// /v1/models attribute when auto-tuning, so its default once-per-endpoint
// producer is wired in, and omits the dependency when not auto-tuning.
func TestConsumesDeclaresModelsDependency(t *testing.T) {
	// Auto-tuning on (default config): the attribute is a required dependency.
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, defaultConfig, testHandle())
	assert.NoError(t, err)
	assert.True(t, p.config.autoTuneMaxPrefixTokens)
	_, ok := p.Consumes().Required[attrmodels.ModelsAttributeKey]
	assert.True(t, ok, "models attribute must be a required dependency when auto-tuning")

	// An explicit cap disables auto-tuning, so the attribute is not requested.
	explicit := defaultConfig
	explicit.MaxPrefixTokensToMatch = 5
	p, err = newDataProducer(context.Background(), ApproxPrefixCachePluginType, explicit, testHandle())
	assert.NoError(t, err)
	assert.False(t, p.config.autoTuneMaxPrefixTokens)
	_, ok = p.Consumes().Required[attrmodels.ModelsAttributeKey]
	assert.False(t, ok, "models attribute must not be requested without auto-tuning")
}

// TestMaxModelLenHelper covers the attribute reader's non-collection and
// missing-attribute paths.
func TestMaxModelLenHelper(t *testing.T) {
	// Missing attribute -> 0.
	assert.Equal(t, 0, maxModelLen(modelLenEndpoint()))

	// Wrong type stored under the key -> 0.
	attrs := fwkdl.NewAttributes()
	attrs.Put(attrmodels.ModelsAttributeKey.String(), attrprefix.NewPrefixCacheMatchInfo(0, 0, 0))
	ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), attrs)
	assert.Equal(t, 0, maxModelLen(ep))
}
