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

package metrics

import (
	"math"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

const (
	sampleCustomKey      = "custom_gauge"
	sampleCustomMetric   = "vllm:custom_gauge"
	sampleOptionalKey    = "optional_gauge"
	sampleOptionalMetric = "vllm:optional_gauge"
)

func newSampleExtractor(t *testing.T) *Extractor {
	t.Helper()
	registry := NewMappingRegistry()
	mapping, err := NewMappingFromConfig(MappingConfig{
		Queue:   defaultTotalQueuedRequestsMetric,
		Running: defaultTotalRunningRequestsMetric,
		KVUsage: defaultKvCacheUsagePercentageMetric,
		CustomMetrics: []CustomMetric{
			{AttributeKey: sampleCustomKey, Spec: mustSpec(t, sampleCustomMetric)},
			{AttributeKey: sampleOptionalKey, Spec: mustSpec(t, sampleOptionalMetric), Optional: true},
		},
	})
	require.NoError(t, err)
	require.NoError(t, registry.Register(DefaultEngineType, mapping))
	extractor, err := NewCoreMetricsExtractor(registry, "")
	require.NoError(t, err)
	return extractor
}

// extractAfterTick runs one scrape after the wall clock has advanced, so
// consecutive scrapes have distinct times.
func extractAfterTick(t *testing.T, extractor *Extractor, ep fwkdl.Endpoint, data sourcemetrics.PrometheusMetricMap) error {
	t.Helper()
	if prev := ep.GetMetrics().UpdateTime; !prev.IsZero() {
		for !time.Now().After(prev) {
			time.Sleep(time.Millisecond)
		}
	}
	return extractor.Extract(t.Context(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep})
}

func scalarSample(t *testing.T, ep fwkdl.Endpoint, field fwkdl.ScalarField) fwkdl.ScalarSample {
	t.Helper()
	sample, ok := ep.GetMetrics().ScalarSample(field)
	require.True(t, ok, "expected a sample for %v", field)
	return sample
}

func customSample(t *testing.T, ep fwkdl.Endpoint, attributeKey string) fwkdl.ScalarSample {
	t.Helper()
	sample, ok := ep.GetMetrics().CustomScalarSample(attributeKey)
	require.True(t, ok, "expected a custom sample for %q", attributeKey)
	return sample
}

func TestExtractorRecordsScalarSamples(t *testing.T) {
	extractor := newSampleExtractor(t)
	ep := fwkdl.NewEndpoint(nil, nil)
	full := sourcemetrics.PrometheusMetricMap{
		defaultTotalQueuedRequestsMetric:    gaugeFamily(5),
		defaultTotalRunningRequestsMetric:   gaugeFamily(2),
		defaultKvCacheUsagePercentageMetric: gaugeFamily(0.4),
		sampleCustomMetric:                  gaugeFamily(9),
	}

	// Present: each observed metric records its value and the scrape time.
	require.NoError(t, extractAfterTick(t, extractor, ep, full))
	first := ep.GetMetrics().UpdateTime
	require.False(t, first.IsZero())
	assert.Equal(t, fwkdl.ScalarSample{Value: 5, ScrapeTime: first}, scalarSample(t, ep, fwkdl.ScalarFieldWaitingQueueSize))
	assert.Equal(t, fwkdl.ScalarSample{Value: 2, ScrapeTime: first}, scalarSample(t, ep, fwkdl.ScalarFieldRunningRequestsSize))
	assert.Equal(t, fwkdl.ScalarSample{Value: 0.4, ScrapeTime: first}, scalarSample(t, ep, fwkdl.ScalarFieldKVCacheUsagePercent))
	assert.Equal(t, fwkdl.ScalarSample{Value: 9, ScrapeTime: first}, customSample(t, ep, sampleCustomKey))
	_, ok := ep.GetMetrics().CustomScalarSample(sampleOptionalKey)
	assert.False(t, ok, "a configured metric never observed has no sample")

	// Unchanged but present: the value is the same and the scrape time advances.
	require.NoError(t, extractAfterTick(t, extractor, ep, full))
	second := ep.GetMetrics().UpdateTime
	require.True(t, second.After(first))
	assert.Equal(t, fwkdl.ScalarSample{Value: 5, ScrapeTime: second}, scalarSample(t, ep, fwkdl.ScalarFieldWaitingQueueSize))
	assert.Equal(t, fwkdl.ScalarSample{Value: 9, ScrapeTime: second}, customSample(t, ep, sampleCustomKey))

	// Partial scrape: the queue and custom metrics are missing and KV usage is
	// NaN. Only the running-request sample advances.
	partial := sourcemetrics.PrometheusMetricMap{
		defaultTotalRunningRequestsMetric:   gaugeFamily(3),
		defaultKvCacheUsagePercentageMetric: gaugeFamily(math.NaN()),
	}
	require.Error(t, extractAfterTick(t, extractor, ep, partial))
	third := ep.GetMetrics().UpdateTime
	require.True(t, third.After(second))
	queue := scalarSample(t, ep, fwkdl.ScalarFieldWaitingQueueSize)
	assert.Equal(t, fwkdl.ScalarSample{Value: 5, ScrapeTime: second}, queue)
	assert.True(t, queue.ScrapeTime.Before(third), "a sample older than UpdateTime shows a later scrape lacked it")
	assert.Equal(t, fwkdl.ScalarSample{Value: 3, ScrapeTime: third}, scalarSample(t, ep, fwkdl.ScalarFieldRunningRequestsSize))
	assert.Equal(t, fwkdl.ScalarSample{Value: 0.4, ScrapeTime: second}, scalarSample(t, ep, fwkdl.ScalarFieldKVCacheUsagePercent))
	assert.Equal(t, fwkdl.ScalarSample{Value: 9, ScrapeTime: second}, customSample(t, ep, sampleCustomKey))
	// Existing fields and attributes keep their last-known value.
	assert.Equal(t, 5, ep.GetMetrics().WaitingQueueSize)
	assertScalarMetric(t, ep, WaitingQueueSizeKey, 5)
	assertScalarMetric(t, ep, sampleCustomKey, 9)

	// Invalid values: infinities and a family with no series leave every
	// sample, and UpdateTime, unchanged.
	before := ep.GetMetrics().ScalarSamples
	invalid := sourcemetrics.PrometheusMetricMap{
		defaultTotalQueuedRequestsMetric:    {Type: dto.MetricType_GAUGE.Enum()},
		defaultTotalRunningRequestsMetric:   gaugeFamily(math.Inf(1)),
		defaultKvCacheUsagePercentageMetric: gaugeFamily(math.Inf(-1)),
		sampleCustomMetric:                  gaugeFamily(math.NaN()),
	}
	require.Error(t, extractAfterTick(t, extractor, ep, invalid))
	assert.Equal(t, third, ep.GetMetrics().UpdateTime)
	assert.Same(t, before, ep.GetMetrics().ScalarSamples)

	// Missing after present and failed polls age out the same way: neither
	// advances the queue sample, so it stops being fresh once maxAge passes.
	maxAge := 20 * time.Millisecond
	time.Sleep(maxAge)
	queue = scalarSample(t, ep, fwkdl.ScalarFieldWaitingQueueSize)
	assert.False(t, queue.FreshWithin(maxAge))
	assert.True(t, queue.FreshWithin(time.Hour))
	running := scalarSample(t, ep, fwkdl.ScalarFieldRunningRequestsSize)
	assert.Equal(t, third, running.ScrapeTime)
	assert.False(t, running.FreshWithin(maxAge))
}

func TestExtractorDiscardsScalarSamplesOnMappingChange(t *testing.T) {
	registry := NewMappingRegistry()
	vllm, err := NewMapping(defaultTotalQueuedRequestsMetric, defaultTotalRunningRequestsMetric, "", "", "")
	require.NoError(t, err)
	require.NoError(t, registry.Register(DefaultEngineType, vllm))
	sglang, err := NewMapping("sglang:num_queue_reqs", "sglang:num_running_reqs", "", "", "")
	require.NoError(t, err)
	require.NoError(t, registry.Register("sglang", sglang))
	extractor, err := NewCoreMetricsExtractor(registry, "")
	require.NoError(t, err)

	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{DefaultEngineTypeLabelKey: "vllm"},
	}, nil)
	vllmData := sourcemetrics.PrometheusMetricMap{
		defaultTotalQueuedRequestsMetric:  gaugeFamily(4),
		defaultTotalRunningRequestsMetric: gaugeFamily(1),
	}
	require.NoError(t, extractAfterTick(t, extractor, ep, vllmData))
	scalarSample(t, ep, fwkdl.ScalarFieldWaitingQueueSize)

	// An engine type without a mapping falls back to the same default mapping,
	// so its samples are kept.
	ep.UpdateMetadata(&fwkdl.EndpointMetadata{Labels: map[string]string{DefaultEngineTypeLabelKey: "unknown"}})
	require.Error(t, extractAfterTick(t, extractor, ep, sourcemetrics.PrometheusMetricMap{}))
	scalarSample(t, ep, fwkdl.ScalarFieldWaitingQueueSize)

	// A different mapping discards the samples even when its scrape observes
	// nothing, and leaves UpdateTime unchanged.
	updated := ep.GetMetrics().UpdateTime
	ep.UpdateMetadata(&fwkdl.EndpointMetadata{Labels: map[string]string{DefaultEngineTypeLabelKey: "sglang"}})
	require.Error(t, extractAfterTick(t, extractor, ep, vllmData))
	assert.Nil(t, ep.GetMetrics().ScalarSamples)
	assert.Equal(t, updated, ep.GetMetrics().UpdateTime)

	require.Error(t, extractAfterTick(t, extractor, ep, sourcemetrics.PrometheusMetricMap{
		"sglang:num_queue_reqs": gaugeFamily(7),
	}))
	assert.Equal(t, 7.0, scalarSample(t, ep, fwkdl.ScalarFieldWaitingQueueSize).Value)
	_, ok := ep.GetMetrics().ScalarSample(fwkdl.ScalarFieldRunningRequestsSize)
	assert.False(t, ok, "a sample recorded under the previous mapping is gone")
}

func TestExtractorKeepsStandardAndCustomSamplesApart(t *testing.T) {
	// A customMetrics entry that reuses a built-in attributeKey owns the
	// attribute, while the built-in spec keeps its own sample.
	registry := NewMappingRegistry()
	mapping, err := NewMappingFromConfig(MappingConfig{
		Queue: defaultTotalQueuedRequestsMetric,
		CustomMetrics: []CustomMetric{
			{AttributeKey: WaitingQueueSizeKey, Spec: mustSpec(t, sampleCustomMetric)},
		},
	})
	require.NoError(t, err)
	require.NoError(t, registry.Register(DefaultEngineType, mapping))
	extractor, err := NewCoreMetricsExtractor(registry, "")
	require.NoError(t, err)

	ep := fwkdl.NewEndpoint(nil, nil)
	require.NoError(t, extractAfterTick(t, extractor, ep, sourcemetrics.PrometheusMetricMap{
		defaultTotalQueuedRequestsMetric: gaugeFamily(3),
		sampleCustomMetric:               gaugeFamily(11),
	}))
	assert.Equal(t, 3.0, scalarSample(t, ep, fwkdl.ScalarFieldWaitingQueueSize).Value)
	assert.Equal(t, 11.0, customSample(t, ep, WaitingQueueSizeKey).Value)
	assert.NotEqual(t, attrmetrics.ScalarSampleDataKey(fwkdl.ScalarFieldWaitingQueueSize),
		attrmetrics.CustomScalarSampleDataKey(WaitingQueueSizeKey))
}

func TestExtractorProducesScalarSampleKeys(t *testing.T) {
	produced := newSampleExtractor(t).Produces()
	for _, field := range []fwkdl.ScalarField{
		fwkdl.ScalarFieldWaitingQueueSize,
		fwkdl.ScalarFieldRunningRequestsSize,
		fwkdl.ScalarFieldKVCacheUsagePercent,
	} {
		assert.IsType(t, fwkdl.ScalarSample{}, produced[attrmetrics.ScalarSampleDataKey(field)], field.String())
	}
	for _, key := range []string{sampleCustomKey, sampleOptionalKey} {
		assert.IsType(t, fwkdl.ScalarSample{}, produced[attrmetrics.CustomScalarSampleDataKey(key)], key)
		assert.IsType(t, attrmetrics.ScalarMetricValue(0), produced[attrmetrics.ScalarMetricDataKey(key)], key)
	}
}
