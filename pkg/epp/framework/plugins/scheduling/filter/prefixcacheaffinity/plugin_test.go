/*
Copyright 2025 The Kubernetes Authors.

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

package prefixcacheaffinity

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// makeEndpoint creates a test endpoint with the given prefix cache match ratio
// (prefixMatch out of 100 total blocks), predicted TTFT, and in-flight tokens.
func makeEndpoint(name string, prefixMatch int, ttft float64, tokens int64) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
	}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if prefixMatch >= 0 {
		ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(), attrprefix.NewPrefixCacheMatchInfo(prefixMatch, 100, 16))
	}
	if ttft >= 0 {
		ep.Put(attrlatency.LatencyPredictionInfoDataKey.String(), attrlatency.NewLatencyPredictionInfo(true, true, 0, 0, ttft, 0, 0))
	}
	if tokens >= 0 {
		ep.Put(attrconcurrency.InFlightLoadDataKey.String(), &attrconcurrency.InFlightLoad{Tokens: tokens})
	}
	return ep
}

func newTestPlugin(config Config) *Plugin {
	return &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: "test"},
		config:                       config,
		prefixMatchDataKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
	}
}

func TestFilter_AffinityThresholdDisabled(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 0, 10, 0),
		makeEndpoint("b", 90, 20, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "affinityThreshold=0 should return all")
}

func TestFilter_SingleEndpoint(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80})
	endpoints := []fwksched.Endpoint{makeEndpoint("a", 90, 10, 0)}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 1, len(result), "single endpoint should always pass")
}

func TestFilter_NoStickyEndpoints(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 10, 10, 0),
		makeEndpoint("b", 20, 20, 0),
		makeEndpoint("c", 50, 30, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 3, len(result), "no sticky endpoints should return all")
}

func TestFilter_NarrowToSticky(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 5000, UseLatencyPredictor: true})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 100, 0),
		makeEndpoint("b", 85, 120, 0),
		makeEndpoint("c", 10, 50, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "should narrow to sticky endpoints")
}

func TestFilter_TTFTPenaltyBreaksStickiness(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 100, UseLatencyPredictor: true})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 500, 0),
		makeEndpoint("b", 10, 50, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "TTFT penalty should break stickiness")
}

// With PeakPrefillThroughput=1000 tokens/sec, in-flight tokens map to TTFT as
// tokens/1000*1000 = tokens ms: endpoint "a" -> 500ms, "b" -> 50ms.
func TestFilter_ThroughputTTFTBreaksStickiness(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 100, PeakPrefillThroughput: 1000})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 10, 500),
		makeEndpoint("b", 10, 10, 50),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "throughput-derived TTFT penalty should break stickiness")
}

func TestFilter_ThroughputTTFTWithinThreshold(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 1000, PeakPrefillThroughput: 1000})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 10, 500),
		makeEndpoint("b", 10, 10, 50),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 1, len(result), "throughput-derived TTFT within threshold should NOT break stickiness")
	assert.Equal(t, "a", result[0].GetMetadata().NamespacedName.Name)
}

func TestFilter_TTFTPenaltyDisabled(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 0, PeakPrefillThroughput: 1000})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 10, 5000), // Huge load
		makeEndpoint("b", 10, 10, 50),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 1, len(result), "maxTTFTPenaltyMs=0 should NOT break stickiness")
	assert.Equal(t, "a", result[0].GetMetadata().NamespacedName.Name)
}

func TestFilter_ExplorationProbability(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 1.0})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 100, 0),
		makeEndpoint("b", 10, 50, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "epsilon=1.0 should always skip gate")
}

func TestConsumes_ConditionalAttributes(t *testing.T) {
	// Gate disabled: neither TTFT source is consumed.
	p := newTestPlugin(Config{MaxTTFTPenaltyMs: 0})
	consumed := p.Consumes()
	_, ok := consumed.Required[p.inFlightLoadDataKey]
	assert.False(t, ok, "InFlightLoadDataKey should not be consumed when the gate is disabled")
	_, ok = consumed.Required[p.latencyPredictionInfoDataKey]
	assert.False(t, ok, "LatencyPredictionInfoDataKey should not be consumed when the gate is disabled")

	// Gate using the latency predictor.
	p = newTestPlugin(Config{MaxTTFTPenaltyMs: 5000, UseLatencyPredictor: true})
	consumed = p.Consumes()
	_, ok = consumed.Required[p.latencyPredictionInfoDataKey]
	assert.True(t, ok)
	_, ok = consumed.Required[p.inFlightLoadDataKey]
	assert.False(t, ok)

	// Gate using peak prefill throughput.
	p = newTestPlugin(Config{MaxTTFTPenaltyMs: 5000, UseLatencyPredictor: false, PeakPrefillThroughput: 1000})
	consumed = p.Consumes()
	_, ok = consumed.Required[p.inFlightLoadDataKey]
	assert.True(t, ok)
	_, ok = consumed.Required[p.latencyPredictionInfoDataKey]
	assert.False(t, ok)
}

func TestFactory_ValidConfig(t *testing.T) {
	plugin, err := Factory("test", fwkplugin.StrictDecoder(nil), nil)
	assert.NoError(t, err)
	assert.NotNil(t, plugin)
	assert.Equal(t, PluginType, plugin.TypedName().Type)
}

func TestFactory_PartialConfigPreservesDefaults(t *testing.T) {
	// Setting only affinityThreshold should preserve defaults for other params.
	plugin, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"affinityThreshold": 0.95}`)), nil)
	assert.NoError(t, err)
	p := plugin.(*Plugin)
	assert.Equal(t, 0.95, p.config.AffinityThreshold)
	assert.Equal(t, DefaultConfig.ExplorationProbability, p.config.ExplorationProbability)
	assert.Equal(t, DefaultConfig.MaxTTFTPenaltyMs, p.config.MaxTTFTPenaltyMs)

	// Setting only explorationProbability should preserve defaults for other params.
	plugin, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"explorationProbability": 0.05}`)), nil)
	assert.NoError(t, err)
	p = plugin.(*Plugin)
	assert.Equal(t, DefaultConfig.AffinityThreshold, p.config.AffinityThreshold)
	assert.Equal(t, 0.05, p.config.ExplorationProbability)
	assert.Equal(t, DefaultConfig.MaxTTFTPenaltyMs, p.config.MaxTTFTPenaltyMs)

	// Setting only maxTTFTPenaltyMs should preserve defaults for other params.
	plugin, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"maxTTFTPenaltyMs": 10000}`)), nil)
	assert.NoError(t, err)
	p = plugin.(*Plugin)
	assert.Equal(t, DefaultConfig.AffinityThreshold, p.config.AffinityThreshold)
	assert.Equal(t, DefaultConfig.ExplorationProbability, p.config.ExplorationProbability)
	assert.Equal(t, float64(10000), p.config.MaxTTFTPenaltyMs)
	assert.Equal(t, DefaultConfig.UseLatencyPredictor, p.config.UseLatencyPredictor)
	assert.Equal(t, DefaultConfig.PeakPrefillThroughput, p.config.PeakPrefillThroughput)
}

func TestFactory_InvalidAffinityThreshold(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"affinityThreshold": 1.5}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "affinityThreshold must be <= 1.0")
}

func TestFactory_InvalidExplorationProbability(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"explorationProbability": -0.1}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "explorationProbability must be in [0, 1]")
}

func TestFactory_InvalidPeakPrefillThroughput(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"peakPrefillThroughput": -1}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "peakPrefillThroughput must be >= 0")
}

// The throughput TTFT source needs a non-zero divisor: with the gate enabled
// (maxTTFTPenaltyMs defaults to 5000) and useLatencyPredictor=false,
// peakPrefillThroughput=0 must be rejected.
func TestFactory_ThroughputModeRequiresPeakPrefillThroughput(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"useLatencyPredictor": false, "peakPrefillThroughput": 0}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "peakPrefillThroughput must be > 0 when useLatencyPredictor is false")
}

func TestFactory_ThroughputModeValid(t *testing.T) {
	plugin, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"useLatencyPredictor": false, "peakPrefillThroughput": 1000}`)), nil)
	assert.NoError(t, err)
	p := plugin.(*Plugin)
	assert.False(t, p.config.UseLatencyPredictor)
	assert.Equal(t, float64(1000), p.config.PeakPrefillThroughput)
}

// peakPrefillThroughput=0 is valid as long as the throughput source is unused:
// either the gate is disabled (maxTTFTPenaltyMs=0) or the latency predictor
// supplies TTFT.
func TestFactory_ZeroPeakPrefillThroughputAllowedWhenUnused(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"maxTTFTPenaltyMs": 0, "useLatencyPredictor": false, "peakPrefillThroughput": 0}`)), nil)
	assert.NoError(t, err, "throughput source unused when the gate is disabled")

	_, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"useLatencyPredictor": true, "peakPrefillThroughput": 0}`)), nil)
	assert.NoError(t, err, "throughput source unused when the latency predictor supplies TTFT")
}
