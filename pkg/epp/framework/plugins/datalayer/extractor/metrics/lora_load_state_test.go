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
	"context"
	"testing"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

const (
	loraLoadedMetric    = "vllm:lora_adapter_loaded"
	loraGPULoadedMetric = "vllm:num_gpu_loaded_lora_adapters"
	loraGPUSlotsMetric  = "vllm:max_gpu_lora_adapters"
)

func loadedSeries(engine, adapter, level, pinned string, value float64) *dto.Metric {
	return &dto.Metric{
		Label: []*dto.LabelPair{
			{Name: proto.String("engine"), Value: proto.String(engine)},
			{Name: proto.String(LoraLoadedAdapterNameLabel), Value: proto.String(adapter)},
			{Name: proto.String(LoraLoadedLevelLabel), Value: proto.String(level)},
			{Name: proto.String(LoraLoadedPinnedLabel), Value: proto.String(pinned)},
		},
		Gauge: &dto.Gauge{Value: ptr.To(value)},
	}
}

func rankedSeries(engine, adapter, level, rank string) *dto.Metric {
	m := loadedSeries(engine, adapter, level, "false", 1)
	m.Label = append(m.Label, &dto.LabelPair{Name: proto.String(LoraLoadedRankLabel), Value: proto.String(rank)})
	return m
}

func countSeries(engine string, value float64) *dto.Metric {
	return &dto.Metric{
		Label: []*dto.LabelPair{
			{Name: proto.String("engine"), Value: proto.String(engine)},
			{Name: proto.String(LoraModelNameLabel), Value: proto.String("base")},
		},
		Gauge: &dto.Gauge{Value: ptr.To(value)},
	}
}

func gaugeFamily(series ...*dto.Metric) *dto.MetricFamily {
	return &dto.MetricFamily{Type: dto.MetricType_GAUGE.Enum(), Metric: series}
}

func residencyExtractor(t *testing.T, loadedSpec, countSpec string) *Extractor {
	t.Helper()
	mapping, err := NewMappingFromConfig(MappingConfig{LoraLoaded: loadedSpec, LoraGPULoaded: countSpec})
	require.NoError(t, err)
	registry := NewMappingRegistry()
	require.NoError(t, registry.Register(DefaultEngineType, mapping))
	extractor, err := NewCoreMetricsExtractor(registry, "")
	require.NoError(t, err)
	return extractor
}

func TestExtractorLoraLoadState(t *testing.T) {
	gpu := fwkdl.LoraLoadState{Level: fwkdl.LoraLoadLevelGPU}
	cpu := fwkdl.LoraLoadState{Level: fwkdl.LoraLoadLevelCPU}

	tests := []struct {
		name        string
		loadedSpec  string
		countSpec   string
		families    sourcemetrics.PrometheusMetricMap
		wantLoaded  map[string]fwkdl.LoraLoadState
		wantGPU     int
		wantBase    string
		wantUpdated bool
	}{
		{
			name:       "two engines union, gpu outranks cpu, zeroed series skipped, count is the max",
			loadedSpec: loraLoadedMetric,
			countSpec:  loraGPULoadedMetric,
			families: sourcemetrics.PrometheusMetricMap{
				loraLoadedMetric: gaugeFamily(
					loadedSeries("0", "alice", "gpu", "false", 1),
					loadedSeries("0", "bob", "cpu", "true", 1),
					loadedSeries("0", "dave", "gpu", "false", 0),
					loadedSeries("1", "alice", "cpu", "false", 1),
					loadedSeries("1", "carol", "gpu", "false", 1),
				),
				loraGPULoadedMetric: gaugeFamily(countSeries("0", 1), countSeries("1", 2)),
			},
			wantLoaded: map[string]fwkdl.LoraLoadState{
				"alice": gpu,
				"bob":   {Level: fwkdl.LoraLoadLevelCPU, Pinned: true},
				"carol": gpu,
			},
			wantGPU:     2,
			wantBase:    "base",
			wantUpdated: true,
		},
		{
			name:       "count gauge at zero with no loaded series reports an empty resident set",
			loadedSpec: loraLoadedMetric,
			countSpec:  loraGPULoadedMetric,
			families: sourcemetrics.PrometheusMetricMap{
				loraGPULoadedMetric: gaugeFamily(countSeries("0", 0)),
			},
			wantLoaded:  map[string]fwkdl.LoraLoadState{},
			wantGPU:     0,
			wantBase:    "base",
			wantUpdated: true,
		},
		{
			name:        "neither family leaves residency unreported",
			loadedSpec:  loraLoadedMetric,
			countSpec:   loraGPULoadedMetric,
			families:    sourcemetrics.PrometheusMetricMap{},
			wantLoaded:  nil,
			wantGPU:     0,
			wantUpdated: false,
		},
		{
			name:       "count derived from the resident set when the count gauge is not configured",
			loadedSpec: loraLoadedMetric,
			countSpec:  "",
			families: sourcemetrics.PrometheusMetricMap{
				loraLoadedMetric: gaugeFamily(
					loadedSeries("0", "alice", "gpu", "false", 1),
					loadedSeries("0", "bob", "cpu", "false", 1),
					loadedSeries("0", "carol", "gpu", "false", 1),
				),
			},
			wantLoaded:  map[string]fwkdl.LoraLoadState{"alice": gpu, "bob": cpu, "carol": gpu},
			wantGPU:     2,
			wantUpdated: true,
		},
		{
			name:       "spec label matchers select one engine",
			loadedSpec: loraLoadedMetric + "{engine=1}",
			countSpec:  loraGPULoadedMetric + "{engine=1}",
			families: sourcemetrics.PrometheusMetricMap{
				loraLoadedMetric: gaugeFamily(
					loadedSeries("0", "alice", "gpu", "false", 1),
					loadedSeries("1", "carol", "gpu", "false", 1),
				),
				loraGPULoadedMetric: gaugeFamily(countSeries("0", 5), countSeries("1", 1)),
			},
			wantLoaded:  map[string]fwkdl.LoraLoadState{"carol": gpu},
			wantGPU:     1,
			wantBase:    "base",
			wantUpdated: true,
		},
		{
			name:       "rank label is kept and survives the multi-engine union",
			loadedSpec: loraLoadedMetric,
			countSpec:  loraGPULoadedMetric,
			families: sourcemetrics.PrometheusMetricMap{
				loraLoadedMetric: gaugeFamily(
					rankedSeries("0", "alice", "gpu", "64"),
					rankedSeries("1", "alice", "cpu", "64"),
					rankedSeries("0", "bob", "gpu", "not-a-number"),
				),
				loraGPULoadedMetric: gaugeFamily(countSeries("0", 2)),
			},
			wantLoaded: map[string]fwkdl.LoraLoadState{
				"alice": {Level: fwkdl.LoraLoadLevelGPU, Rank: 64},
				"bob":   gpu,
			},
			wantGPU:     2,
			wantBase:    "base",
			wantUpdated: true,
		},
		{
			name:       "series without an adapter name is ignored",
			loadedSpec: loraLoadedMetric,
			countSpec:  loraGPULoadedMetric,
			families: sourcemetrics.PrometheusMetricMap{
				loraLoadedMetric:    gaugeFamily(&dto.Metric{Gauge: &dto.Gauge{Value: ptr.To(1.0)}}),
				loraGPULoadedMetric: gaugeFamily(countSeries("0", 1)),
			},
			wantLoaded:  map[string]fwkdl.LoraLoadState{},
			wantGPU:     1,
			wantBase:    "base",
			wantUpdated: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extractor := residencyExtractor(t, tt.loadedSpec, tt.countSpec)
			ep := fwkdl.NewEndpoint(nil, nil)
			before := ep.GetMetrics().Clone()

			err := extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: tt.families, Endpoint: ep})
			require.NoError(t, err)

			after := ep.GetMetrics()
			if tt.wantUpdated {
				assert.NotEqual(t, before.UpdateTime, after.UpdateTime, "expected an update")
			} else {
				assert.Equal(t, before.UpdateTime, after.UpdateTime, "expected no update")
			}
			assert.Equal(t, tt.wantLoaded, after.LoadedModels)
			assert.Equal(t, tt.wantGPU, after.GPULoadedModels)
			assert.Equal(t, tt.wantBase, after.BaseModel)
		})
	}
}

func TestExtractorLoraLoadStateResetsWhenResidencyDisappears(t *testing.T) {
	extractor := residencyExtractor(t, loraLoadedMetric, loraGPULoadedMetric)
	ep := fwkdl.NewEndpoint(nil, nil)

	families := sourcemetrics.PrometheusMetricMap{
		loraLoadedMetric:    gaugeFamily(loadedSeries("0", "alice", "gpu", "false", 1)),
		loraGPULoadedMetric: gaugeFamily(countSeries("0", 1)),
	}
	require.NoError(t, extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: families, Endpoint: ep}))
	require.Len(t, ep.GetMetrics().LoadedModels, 1)

	// A later scrape carrying only the count gauge means the adapter was evicted.
	families = sourcemetrics.PrometheusMetricMap{loraGPULoadedMetric: gaugeFamily(countSeries("0", 0))}
	require.NoError(t, extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: families, Endpoint: ep}))
	assert.Equal(t, map[string]fwkdl.LoraLoadState{}, ep.GetMetrics().LoadedModels)
	assert.Equal(t, 0, ep.GetMetrics().GPULoadedModels)
}

func TestExtractorLoraGPUSlotsOverridesInfoLabel(t *testing.T) {
	mapping, err := NewMappingFromConfig(MappingConfig{Lora: "vllm:lora_requests_info", LoraGPUSlots: loraGPUSlotsMetric})
	require.NoError(t, err)
	registry := NewMappingRegistry()
	require.NoError(t, registry.Register(DefaultEngineType, mapping))
	extractor, err := NewCoreMetricsExtractor(registry, "")
	require.NoError(t, err)

	info := &dto.MetricFamily{Type: dto.MetricType_GAUGE.Enum(), Metric: []*dto.Metric{{
		Label: []*dto.LabelPair{{Name: proto.String(LoraInfoMaxAdaptersMetricName), Value: proto.String("2")}},
		Gauge: &dto.Gauge{Value: ptr.To(1.0)},
	}}}

	// Slot gauge present: it wins over the label, largest engine counts.
	ep := fwkdl.NewEndpoint(nil, nil)
	families := sourcemetrics.PrometheusMetricMap{
		"vllm:lora_requests_info": info,
		loraGPUSlotsMetric:        gaugeFamily(countSeries("0", 4), countSeries("1", 3)),
	}
	require.NoError(t, extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: families, Endpoint: ep}))
	assert.Equal(t, 4, ep.GetMetrics().MaxActiveModels)

	// Slot gauge alone, before any adapter has served: capacity is still known.
	ep = fwkdl.NewEndpoint(nil, nil)
	families = sourcemetrics.PrometheusMetricMap{loraGPUSlotsMetric: gaugeFamily(countSeries("0", 4))}
	require.NoError(t, extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: families, Endpoint: ep}))
	assert.Equal(t, 4, ep.GetMetrics().MaxActiveModels)

	// Slot gauge absent: the label still works.
	ep = fwkdl.NewEndpoint(nil, nil)
	families = sourcemetrics.PrometheusMetricMap{"vllm:lora_requests_info": info}
	require.NoError(t, extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: families, Endpoint: ep}))
	assert.Equal(t, 2, ep.GetMetrics().MaxActiveModels)
}

func TestVLLMDefaultsIncludeLoraResidency(t *testing.T) {
	extractor, err := newCoreMetricsExtractorPlugin(context.Background(), "test", nil)
	require.NoError(t, err)

	mapping, ok := extractor.registry.Get("vllm")
	require.True(t, ok)
	require.NotNil(t, mapping.LoraLoaded)
	require.NotNil(t, mapping.LoraGPULoaded)
	assert.Equal(t, loraLoadedMetric, mapping.LoraLoaded.Name)
	assert.Equal(t, loraGPULoadedMetric, mapping.LoraGPULoaded.Name)
	require.NotNil(t, mapping.LoraGPUSlots)
	assert.Equal(t, loraGPUSlotsMetric, mapping.LoraGPUSlots.Name)
	assert.Contains(t, mapping.MetricNames(), loraLoadedMetric)
	assert.Contains(t, mapping.MetricNames(), loraGPULoadedMetric)
	assert.Contains(t, mapping.MetricNames(), loraGPUSlotsMetric)

	for _, engine := range []string{"sglang", "trtllm-serve", "triton-tensorrt-llm", "triton"} {
		mapping, ok := extractor.registry.Get(engine)
		require.True(t, ok, engine)
		assert.Nil(t, mapping.LoraLoaded, engine)
		assert.Nil(t, mapping.LoraGPULoaded, engine)
		assert.Nil(t, mapping.LoraGPUSlots, engine)
	}
}

func histogramSeries(engine, transition string, count uint64, sum float64) *dto.Metric {
	return &dto.Metric{
		Label: []*dto.LabelPair{
			{Name: proto.String("engine"), Value: proto.String(engine)},
			{Name: proto.String(LoraLoadTransitionLabel), Value: proto.String(transition)},
		},
		Histogram: &dto.Histogram{SampleCount: proto.Uint64(count), SampleSum: proto.Float64(sum)},
	}
}

func TestExtractorLoraTransitionTimes(t *testing.T) {
	mapping, err := NewMappingFromConfig(MappingConfig{LoraLoadSeconds: "vllm:lora_adapter_load_seconds"})
	require.NoError(t, err)
	registry := NewMappingRegistry()
	require.NoError(t, registry.Register(DefaultEngineType, mapping))
	extractor, err := NewCoreMetricsExtractor(registry, "")
	require.NoError(t, err)

	families := sourcemetrics.PrometheusMetricMap{
		"vllm:lora_adapter_load_seconds": &dto.MetricFamily{
			Type: dto.MetricType_HISTOGRAM.Enum(),
			Metric: []*dto.Metric{
				histogramSeries("0", "load", 2, 0.8),
				histogramSeries("1", "load", 2, 1.2), // pooled with engine 0: 2.0 over 4
				histogramSeries("0", "activate", 4, 0.4),
				histogramSeries("0", "unknown", 1, 9),
			},
		},
	}
	ep := fwkdl.NewEndpoint(nil, nil)
	require.NoError(t, extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: families, Endpoint: ep}))
	assert.InDelta(t, 0.5, ep.GetMetrics().LoraLoadSeconds, 1e-9)
	assert.InDelta(t, 0.1, ep.GetMetrics().LoraActivateSeconds, 1e-9)

	// No samples yet: both stay 0.
	empty := sourcemetrics.PrometheusMetricMap{
		"vllm:lora_adapter_load_seconds": &dto.MetricFamily{
			Type:   dto.MetricType_HISTOGRAM.Enum(),
			Metric: []*dto.Metric{histogramSeries("0", "load", 0, 0)},
		},
	}
	ep = fwkdl.NewEndpoint(nil, nil)
	require.NoError(t, extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: empty, Endpoint: ep}))
	assert.Equal(t, 0.0, ep.GetMetrics().LoraLoadSeconds)
	assert.Equal(t, 0.0, ep.GetMetrics().LoraActivateSeconds)
}
