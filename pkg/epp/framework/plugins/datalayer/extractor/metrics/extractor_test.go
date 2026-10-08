/*
Copyright 2025 The Kubernetes Authors.
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
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/proto"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

const (
	// use hardcoded values - importing causes cycle
	defaultTotalQueuedRequestsMetric    = "vllm:num_requests_waiting"
	defaultTotalRunningRequestsMetric   = "vllm:num_requests_running"
	defaultKvCacheUsagePercentageMetric = "vllm:kv_cache_usage_perc"
	defaultLoraInfoMetric               = "vllm:lora_requests_info"
	defaultCacheInfoMetric              = "vllm:cache_config_info"
	customMetricsConfigKey              = "customMetrics"
)

func TestExtractorExtract(t *testing.T) {
	ctx := context.Background()

	if _, err := NewCoreMetricsExtractor(nil, ""); err == nil {
		t.Error("expected to fail to create extractor with nil registry")
	}

	registry := NewMappingRegistry()
	mapping, err := NewMapping(defaultTotalQueuedRequestsMetric, defaultTotalRunningRequestsMetric,
		defaultKvCacheUsagePercentageMetric, defaultLoraInfoMetric, defaultCacheInfoMetric)
	if err != nil {
		t.Fatalf("failed to create mapping: %v", err)
	}
	if err := registry.Register(DefaultEngineType, mapping); err != nil {
		t.Fatalf("failed to register mapping: %v", err)
	}

	extractor, err := NewCoreMetricsExtractor(registry, "")
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}

	if exType := extractor.TypedName().Type; exType == "" {
		t.Error("empty extractor type")
	}

	if exName := extractor.TypedName().Name; exName == "" {
		t.Error("empty extractor name")
	}

	ep := fwkdl.NewEndpoint(nil, nil)
	if ep == nil {
		t.Fatal("expected non-nil endpoint")
	}

	tests := []struct {
		name    string
		data    sourcemetrics.PrometheusMetricMap
		wantErr bool
		updated bool // whether metrics are expected to change
	}{
		{
			name:    "nil data",
			data:    nil,
			wantErr: true,
			updated: false,
		},
		{
			name:    "empty PrometheusMetricMap",
			data:    sourcemetrics.PrometheusMetricMap{},
			wantErr: true,  // errors when metrics are missing
			updated: false, // and also not updated...
		},
		{
			name: "single valid metric",
			data: sourcemetrics.PrometheusMetricMap{
				defaultTotalQueuedRequestsMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{Value: ptr.To(5.0)},
						},
					},
				},
			},
			wantErr: true, // missing metrics can return an error
			updated: true, // but should still update
		},
		{
			name: "multiple valid metrics",
			data: sourcemetrics.PrometheusMetricMap{
				defaultTotalQueuedRequestsMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{Value: ptr.To(5.0)},
						},
					},
				},
				defaultTotalRunningRequestsMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{Value: ptr.To(1.0)},
						},
					},
				},
				defaultKvCacheUsagePercentageMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Gauge: &dto.Gauge{Value: ptr.To(0.5)},
						},
					},
				},
				defaultLoraInfoMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Label: []*dto.LabelPair{
								{
									Name:  proto.String(LoraInfoRunningAdaptersMetricName),
									Value: proto.String("lora1"),
								},
								{
									Name:  proto.String(LoraInfoWaitingAdaptersMetricName),
									Value: proto.String("lora2"),
								},
								{
									Name:  proto.String(LoraInfoMaxAdaptersMetricName),
									Value: proto.String("1"),
								},
							},
						},
					},
				},
				defaultCacheInfoMetric: &dto.MetricFamily{
					Type: dto.MetricType_GAUGE.Enum(),
					Metric: []*dto.Metric{
						{
							Label: []*dto.LabelPair{
								{
									Name:  proto.String(CacheConfigBlockSizeInfoMetricName),
									Value: proto.String("16"),
								},
								{
									Name:  proto.String(CacheConfigNumGPUBlocksMetricName),
									Value: proto.String("1024"),
								},
							},
							Gauge: &dto.Gauge{Value: ptr.To(1.0)},
						},
					},
				},
			},
			wantErr: false,
			updated: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Extract panicked: %v", r)
				}
			}()

			before := ep.GetMetrics().Clone()
			err := extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: tt.data, Endpoint: ep})
			after := ep.GetMetrics()

			if tt.wantErr && err == nil {
				t.Errorf("expected error but got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if tt.updated {
				if diff := cmp.Diff(before, after); diff == "" {
					t.Errorf("expected metrics to be updated, but no change detected")
				}
			} else {
				if diff := cmp.Diff(before, after); diff != "" {
					t.Errorf("expected no metrics update, but got changes:\n%s", diff)
				}
			}
		})
	}
}

func TestExtractorMultiEngine(t *testing.T) {
	ctx := context.Background()

	registry := NewMappingRegistry()
	// Default mapping (vllm)
	mDef, _ := NewMapping("vllm:num_requests_waiting", "vllm:num_requests_running", "", "", "")
	_ = registry.Register(DefaultEngineType, mDef)
	// SGLang mapping
	mSgl, _ := NewMapping("sglang:num_queue_reqs", "sglang:num_running_reqs", "", "", "")
	_ = registry.Register("sglang", mSgl)

	extractor, _ := NewCoreMetricsExtractor(registry, "")

	// Sample metric data
	data := sourcemetrics.PrometheusMetricMap{
		"vllm:num_requests_waiting": &dto.MetricFamily{
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{
					Gauge: &dto.Gauge{Value: ptr.To(10.0)},
				},
			},
		},
		"sglang:num_queue_reqs": &dto.MetricFamily{
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{
					Gauge: &dto.Gauge{Value: ptr.To(20.0)},
				},
			},
		},
	}

	// Case 1: Engine = vllm (uses default)
	epVllm := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{DefaultEngineTypeLabelKey: "vllm"},
	}, nil)
	_ = extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: epVllm})
	if epVllm.GetMetrics().WaitingQueueSize != 10 {
		t.Errorf("vllm: expected queue size 10, got %v", epVllm.GetMetrics().WaitingQueueSize)
	}

	// Case 2: Engine = sglang (uses specific)
	epSgl := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{DefaultEngineTypeLabelKey: "sglang"},
	}, nil)
	_ = extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: epSgl})
	if epSgl.GetMetrics().WaitingQueueSize != 20 {
		t.Errorf("sglang: expected queue size 20, got %v", epSgl.GetMetrics().WaitingQueueSize)
	}
}

func TestBuiltinAtomEngine(t *testing.T) {
	ctx := context.Background()

	extractor, err := newCoreMetricsExtractorPlugin(ctx, "atom-test", &modelServerExtractorParams{
		DefaultEngine: "atom",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gauge := func(v float64) *dto.MetricFamily {
		return &dto.MetricFamily{
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{Gauge: &dto.Gauge{Value: ptr.To(v)}},
			},
		}
	}

	data := sourcemetrics.PrometheusMetricMap{
		"atom:requests_waiting":     gauge(7),
		"atom:requests_running":     gauge(3),
		"atom:kv_cache_usage_ratio": gauge(0.25),
		"atom:cache_config_info": {
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{
					Label: []*dto.LabelPair{
						{Name: proto.String("block_size"), Value: proto.String("16")},
						{Name: proto.String("num_gpu_blocks"), Value: proto.String("138302")},
					},
					Gauge: &dto.Gauge{Value: ptr.To(1.0)},
				},
			},
		},
	}

	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{DefaultEngineTypeLabelKey: "atom"},
	}, nil)

	if err := extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{
		Payload:  data,
		Endpoint: ep,
	}); err != nil {
		t.Fatalf("extract: %v", err)
	}

	metrics := ep.GetMetrics()
	if metrics.WaitingQueueSize != 7 {
		t.Errorf("WaitingQueueSize = %v, want 7", metrics.WaitingQueueSize)
	}
	if metrics.RunningRequestsSize != 3 {
		t.Errorf("RunningRequestsSize = %v, want 3", metrics.RunningRequestsSize)
	}
	if metrics.KVCacheUsagePercent != 0.25 {
		t.Errorf("KVCacheUsagePercent = %v, want 0.25", metrics.KVCacheUsagePercent)
	}
	if metrics.CacheBlockSize != 16 {
		t.Errorf("CacheBlockSize = %v, want 16", metrics.CacheBlockSize)
	}
	if metrics.CacheNumBlocks != 138302 {
		t.Errorf("CacheNumBlocks = %v, want 138302", metrics.CacheNumBlocks)
	}
}

func TestExtractorReadsEveryEngineOfDataParallelPod(t *testing.T) {
	ctx := context.Background()

	registry := NewMappingRegistry()
	mapping, err := NewMapping(defaultTotalQueuedRequestsMetric, defaultTotalRunningRequestsMetric,
		defaultKvCacheUsagePercentageMetric, "", "")
	if err != nil {
		t.Fatalf("failed to create mapping: %v", err)
	}
	if err := registry.Register(DefaultEngineType, mapping); err != nil {
		t.Fatalf("failed to register mapping: %v", err)
	}
	extractor, err := NewCoreMetricsExtractor(registry, "")
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}

	perEngine := func(values ...float64) *dto.MetricFamily {
		series := make([]*dto.Metric, 0, len(values))
		for i, v := range values {
			series = append(series, &dto.Metric{
				Label: []*dto.LabelPair{{Name: proto.String("engine"), Value: proto.String(strconv.Itoa(i))}},
				Gauge: &dto.Gauge{Value: ptr.To(v)},
			})
		}
		return &dto.MetricFamily{Type: dto.MetricType_GAUGE.Enum(), Metric: series}
	}

	fourEngines := sourcemetrics.PrometheusMetricMap{
		defaultTotalQueuedRequestsMetric:    perEngine(6, 10, 11, 7),
		defaultTotalRunningRequestsMetric:   perEngine(40, 38, 41, 37),
		defaultKvCacheUsagePercentageMetric: perEngine(0.52, 0.91, 0.78, 0.60),
	}

	tests := []struct {
		name string
		// previous, when set, is extracted first so the test can check which fields a failed scrape keeps.
		previous    sourcemetrics.PrometheusMetricMap
		data        sourcemetrics.PrometheusMetricMap
		wantErr     bool
		wantWaiting int
		wantRunning int
		wantKV      float64
	}{
		{
			name:        "four engines",
			data:        fourEngines,
			wantWaiting: 34,
			wantRunning: 156,
			wantKV:      0.91,
		},
		{
			name: "single engine",
			data: sourcemetrics.PrometheusMetricMap{
				defaultTotalQueuedRequestsMetric:    perEngine(6),
				defaultTotalRunningRequestsMetric:   perEngine(40),
				defaultKvCacheUsagePercentageMetric: perEngine(0.52),
			},
			wantWaiting: 6,
			wantRunning: 40,
			wantKV:      0.52,
		},
		{
			name:     "non-finite engine keeps the previous value",
			previous: fourEngines,
			data: sourcemetrics.PrometheusMetricMap{
				defaultTotalQueuedRequestsMetric:    perEngine(1, math.NaN(), 2, 3),
				defaultTotalRunningRequestsMetric:   perEngine(10, 20, 30, 40),
				defaultKvCacheUsagePercentageMetric: perEngine(0.1, 0.2, 0.3, 0.4),
			},
			wantErr:     true,
			wantWaiting: 34,
			wantRunning: 100,
			wantKV:      0.4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
			if tt.previous != nil {
				if err := extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: tt.previous, Endpoint: ep}); err != nil {
					t.Fatalf("unexpected error on previous scrape: %v", err)
				}
			}
			err := extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: tt.data, Endpoint: ep})
			if gotErr := err != nil; gotErr != tt.wantErr {
				t.Fatalf("Extract error: want error %v, got %v", tt.wantErr, err)
			}
			got := ep.GetMetrics()
			if got.WaitingQueueSize != tt.wantWaiting {
				t.Errorf("WaitingQueueSize: want %d, got %d", tt.wantWaiting, got.WaitingQueueSize)
			}
			if got.RunningRequestsSize != tt.wantRunning {
				t.Errorf("RunningRequestsSize: want %d, got %d", tt.wantRunning, got.RunningRequestsSize)
			}
			if got.KVCacheUsagePercent != tt.wantKV {
				t.Errorf("KVCacheUsagePercent: want %v, got %v", tt.wantKV, got.KVCacheUsagePercent)
			}
		})
	}
}

func TestBackwardCompatibility(t *testing.T) {
	ctx := context.Background()

	registry := NewMappingRegistry()
	// Default mapping (legacy behavior)
	mDef, _ := NewMapping("vllm:num_requests_waiting", "", "", "", "")
	_ = registry.Register(DefaultEngineType, mDef)

	extractor, _ := NewCoreMetricsExtractor(registry, "")

	data := sourcemetrics.PrometheusMetricMap{
		"vllm:num_requests_waiting": &dto.MetricFamily{
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{
					Gauge: &dto.Gauge{Value: ptr.To(100.0)},
				},
			},
		},
	}

	// Case 1: No labels at all
	epNone := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Labels: nil}, nil)
	_ = extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: epNone})
	if epNone.GetMetrics().WaitingQueueSize != 100 {
		t.Errorf("no labels: expected 100, got %v", epNone.GetMetrics().WaitingQueueSize)
	}

	// Case 2: Different label key or unknown value
	epUnknown := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{DefaultEngineTypeLabelKey: "unknown-engine"},
	}, nil)
	_ = extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: epUnknown})
	if epUnknown.GetMetrics().WaitingQueueSize != 100 {
		t.Errorf("unknown label: expected 100, got %v", epUnknown.GetMetrics().WaitingQueueSize)
	}
}

func TestCacheInfoLabelAliasing(t *testing.T) {
	ctx := context.Background()

	// An engine config aliases the cache-info label names when the engine
	// carries block size and block count under something other than
	// "block_size" and "num_gpu_blocks".
	registry := NewMappingRegistry()
	mapping, err := NewMappingFromConfig(MappingConfig{
		Queue:               "myengine:num_queue_reqs",
		Running:             "myengine:num_running_reqs",
		KVUsage:             "myengine:token_usage",
		CacheInfo:           "myengine:cache_config_info",
		CacheBlockSizeLabel: "page_size",
		CacheNumBlocksLabel: "num_pages",
	})
	if err != nil {
		t.Fatalf("failed to create mapping: %v", err)
	}
	if err := registry.Register(DefaultEngineType, mapping); err != nil {
		t.Fatalf("failed to register mapping: %v", err)
	}

	extractor, _ := NewCoreMetricsExtractor(registry, "")

	data := sourcemetrics.PrometheusMetricMap{
		"myengine:cache_config_info": &dto.MetricFamily{
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{
					Label: []*dto.LabelPair{
						{
							Name:  proto.String("page_size"),
							Value: proto.String("64"),
						},
						{
							Name:  proto.String("num_pages"),
							Value: proto.String("11147"),
						},
					},
					Gauge: &dto.Gauge{Value: ptr.To(1.0)},
				},
			},
		},
	}

	ep := fwkdl.NewEndpoint(nil, nil)
	_ = extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep})

	if ep.GetMetrics().CacheBlockSize != 64 {
		t.Errorf("expected CacheBlockSize 64, got %d", ep.GetMetrics().CacheBlockSize)
	}
	if ep.GetMetrics().CacheNumBlocks != 11147 {
		t.Errorf("expected CacheNumBlocks 11147, got %d", ep.GetMetrics().CacheNumBlocks)
	}
}

func TestDirectGaugeSpecExtraction(t *testing.T) {
	ctx := context.Background()

	// Triton exposes block size and num blocks as separate gauge values
	registry := NewMappingRegistry()
	mapping, err := NewMappingFromConfig(MappingConfig{
		Queue:          "nv_trt_llm_request_metrics{request_type=waiting}",
		Running:        "nv_trt_llm_request_metrics{request_type=scheduled}",
		KVUsage:        "nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=fraction}",
		CacheBlockSize: "nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=tokens_per}",
		CacheNumBlocks: "nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=max}",
	})
	if err != nil {
		t.Fatalf("failed to create mapping: %v", err)
	}
	if err := registry.Register(DefaultEngineType, mapping); err != nil {
		t.Fatalf("failed to register mapping: %v", err)
	}

	extractor, _ := NewCoreMetricsExtractor(registry, "")

	data := sourcemetrics.PrometheusMetricMap{
		"nv_trt_llm_kv_cache_block_metrics": &dto.MetricFamily{
			Type: dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{
				{
					Label: []*dto.LabelPair{
						{
							Name:  proto.String("kv_cache_block_type"),
							Value: proto.String("tokens_per"),
						},
					},
					Gauge: &dto.Gauge{Value: ptr.To(64.0)},
				},
				{
					Label: []*dto.LabelPair{
						{
							Name:  proto.String("kv_cache_block_type"),
							Value: proto.String("max"),
						},
					},
					Gauge: &dto.Gauge{Value: ptr.To(6239.0)},
				},
				{
					Label: []*dto.LabelPair{
						{
							Name:  proto.String("kv_cache_block_type"),
							Value: proto.String("fraction"),
						},
					},
					Gauge: &dto.Gauge{Value: ptr.To(0.42)},
				},
			},
		},
	}

	ep := fwkdl.NewEndpoint(nil, nil)
	_ = extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep})

	if ep.GetMetrics().CacheBlockSize != 64 {
		t.Errorf("expected CacheBlockSize 64, got %d", ep.GetMetrics().CacheBlockSize)
	}
	if ep.GetMetrics().CacheNumBlocks != 6239 {
		t.Errorf("expected CacheNumBlocks 6239, got %d", ep.GetMetrics().CacheNumBlocks)
	}
	if ep.GetMetrics().KVCacheUsagePercent != 0.42 {
		t.Errorf("expected KVCacheUsagePercent 0.42, got %f", ep.GetMetrics().KVCacheUsagePercent)
	}
}

func TestCoreMetricsExtractorFactoryDefaultEngine(t *testing.T) {
	tests := []struct {
		name         string
		params       map[string]any
		wantErr      bool
		errContains  string
		checkDefault string // engine name that should be default
	}{
		{
			name:         "no params uses vllm as default",
			params:       nil,
			wantErr:      false,
			checkDefault: "vllm",
		},
		{
			name: "defaultEngine sglang",
			params: map[string]any{
				"defaultEngine": "sglang",
			},
			wantErr:      false,
			checkDefault: "sglang",
		},
		{
			name: "defaultEngine atom",
			params: map[string]any{
				"defaultEngine": "atom",
			},
			wantErr:      false,
			checkDefault: "atom",
		},
		{
			name: "defaultEngine vllm explicit",
			params: map[string]any{
				"defaultEngine": "vllm",
			},
			wantErr:      false,
			checkDefault: "vllm",
		},
		{
			name: "defaultEngine trtllm-serve",
			params: map[string]any{
				"defaultEngine": "trtllm-serve",
			},
			wantErr:      false,
			checkDefault: "trtllm-serve",
		},
		{
			name: "defaultEngine not found",
			params: map[string]any{
				"defaultEngine": "unknown-engine",
			},
			wantErr:     true,
			errContains: "not found in engineConfigs",
		},
		{
			name: "engine config name is reserved default",
			params: map[string]any{
				"defaultEngine": "default",
				"engineConfigs": []map[string]any{
					{
						"name":               "default",
						"queuedRequestsSpec": "test:metric",
					},
				},
			},
			wantErr:     true,
			errContains: "reserved",
		},
		{
			name: "custom engineConfigs with defaultEngine",
			params: map[string]any{
				"defaultEngine": "custom",
				"engineConfigs": []map[string]any{
					{
						"name":               "custom",
						"queuedRequestsSpec": "test:metric",
					},
				},
			},
			wantErr:      false,
			checkDefault: "custom",
		},
		{
			name: "custom engineConfigs with custom metrics",
			params: map[string]any{
				"defaultEngine": "custom",
				"engineConfigs": []map[string]any{
					{
						"name": "custom",
						customMetricsConfigKey: []map[string]any{
							{
								"attributeKey": "custom.queue_depth",
								"metricSpec":   "custom_queue_depth{tier=gold}",
							},
						},
					},
				},
			},
			wantErr:      false,
			checkDefault: "custom",
		},
		{
			name: "custom metric attributeKey is required",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name": "custom",
						customMetricsConfigKey: []map[string]any{
							{
								"metricSpec": "custom_queue_depth",
							},
						},
					},
				},
			},
			wantErr:     true,
			errContains: "attributeKey cannot be empty",
		},
		{
			name: "custom metric spec is required",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name": "custom",
						customMetricsConfigKey: []map[string]any{
							{
								"attributeKey": "custom.queue_depth",
							},
						},
					},
				},
			},
			wantErr:     true,
			errContains: "spec cannot be empty",
		},
		{
			name: "custom engineConfigs auto-appends vllm sglang trtllm-serve and triton-tensorrt-llm",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name":               "custom-engine",
						"queuedRequestsSpec": "custom:waiting",
					},
				},
			},
			wantErr:      false,
			checkDefault: "vllm", // vllm is auto-appended and becomes default
		},
		{
			name: "defaultEngine triton-tensorrt-llm",
			params: map[string]any{
				"defaultEngine": "triton-tensorrt-llm",
			},
			wantErr:      false,
			checkDefault: "triton-tensorrt-llm",
		},
		{
			name: "custom engineConfigs with custom vllm preserves user config",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name":               "vllm",
						"queuedRequestsSpec": "custom:vllm_waiting",
					},
				},
			},
			wantErr:      false,
			checkDefault: "vllm", // user's vllm config is used, sglang is auto-appended
		},
		{
			name: "empty engineConfigs uses defaults",
			params: map[string]any{
				"engineConfigs": []map[string]any{},
			},
			wantErr:      false,
			checkDefault: "vllm", // all built-in engines are auto-appended
		},
		{
			name: "defaultEngine triton with triton defined",
			params: map[string]any{
				"defaultEngine": "triton",
				"engineConfigs": []map[string]any{
					{
						"name":               "triton",
						"queuedRequestsSpec": "nv_trt_llm:waiting",
					},
				},
			},
			wantErr:      false,
			checkDefault: "triton", // triton is default, vllm/sglang/trtllm-serve auto-appended
		},
		{
			name: "both vllm and sglang custom defined",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name":               "vllm",
						"queuedRequestsSpec": "custom:vllm_metric",
					},
					{
						"name":               "sglang",
						"queuedRequestsSpec": "custom:sglang_metric",
					},
				},
			},
			wantErr:      false,
			checkDefault: "vllm", // no auto-append, user's configs used
		},
		{
			name: "duplicate engine names",
			params: map[string]any{
				"engineConfigs": []map[string]any{
					{
						"name":               "vllm",
						"queuedRequestsSpec": "custom:metric1",
					},
					{
						"name":               "vllm",
						"queuedRequestsSpec": "custom:metric2",
					},
				},
			},
			wantErr:     true,
			errContains: "already exists",
		},
		{
			name: "empty engineLabelKey uses default",
			params: map[string]any{
				"engineLabelKey": "",
			},
			wantErr:      false,
			checkDefault: "vllm",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var params json.RawMessage
			if tt.params != nil {
				var err error
				params, err = json.Marshal(tt.params)
				if err != nil {
					t.Fatalf("failed to marshal params: %v", err)
				}
			}

			plugin, err := CoreMetricsExtractorFactory("test", fwkplugin.StrictDecoder(params), nil)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error but got nil")
				} else if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errContains)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if plugin == nil {
				t.Fatal("expected non-nil plugin")
			}

			// Verify the correct default is set
			if tt.checkDefault != "" {
				extractor, ok := plugin.(*Extractor)
				if !ok {
					t.Fatal("plugin is not an Extractor")
				}

				// Verify EngineLabelKey is defaulted correctly
				if extractor.engineLabelKey != DefaultEngineTypeLabelKey {
					t.Errorf("engineLabelKey = %q, want %q", extractor.engineLabelKey, DefaultEngineTypeLabelKey)
				}

				// Check that the default mapping exists and matches expected engine
				defaultMapping, found := extractor.registry.Get(DefaultEngineType)
				if !found {
					t.Fatal("default mapping not found in registry")
				}

				engineMapping, found := extractor.registry.Get(tt.checkDefault)
				if !found {
					t.Fatalf("mapping for %q not found in registry", tt.checkDefault)
				}

				// The default mapping should be the same as the expected engine's mapping
				if defaultMapping != engineMapping {
					t.Errorf("default mapping does not match %q engine mapping", tt.checkDefault)
				}
			}
		})
	}
}

func TestGetEngineTypeFromEndpoint(t *testing.T) {
	tests := []struct {
		name     string
		labels   map[string]string
		labelKey string
		want     string
	}{
		{
			name:     "new label key",
			labels:   map[string]string{DefaultEngineTypeLabelKey: "vllm"},
			labelKey: DefaultEngineTypeLabelKey,
			want:     "vllm",
		},
		{
			name:     "legacy GAIE label key is ignored",
			labels:   map[string]string{"inference.networking.k8s.io/engine-type": "sglang"},
			labelKey: DefaultEngineTypeLabelKey,
			want:     DefaultEngineType,
		},
		{
			name:     "no labels returns default",
			labels:   map[string]string{},
			labelKey: DefaultEngineTypeLabelKey,
			want:     DefaultEngineType,
		},
		{
			name:     "nil labels returns default",
			labels:   nil,
			labelKey: DefaultEngineTypeLabelKey,
			want:     DefaultEngineType,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Labels: tt.labels}, nil)
			got := getEngineTypeFromEndpoint(ep, tt.labelKey)
			if got != tt.want {
				t.Errorf("getEngineTypeFromEndpoint() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestExtractorMetricFamiliesCoverEveryEngineSpec(t *testing.T) {
	ext, err := newCoreMetricsExtractorPlugin(context.Background(), "core", nil)
	if err != nil {
		t.Fatalf("new extractor: %v", err)
	}
	declared := map[string]bool{}
	for _, name := range ext.MetricFamilies() {
		declared[name] = true
	}
	for _, engine := range defaultEngineConfigs {
		for _, raw := range []string{
			engine.QueuedRequestsSpec, engine.RunningRequestsSpec, engine.KVUsageSpec, engine.LoRASpec,
			engine.CacheInfoSpec, engine.CacheBlockSizeSpec, engine.CacheNumBlocksSpec,
		} {
			spec, err := parseStringToSpec(raw)
			if err != nil {
				t.Fatalf("engine %s spec %q: %v", engine.Name, raw, err)
			}
			if spec != nil && !declared[spec.Name] {
				t.Errorf("engine %s reads %s, which MetricFamilies does not declare", engine.Name, spec.Name)
			}
		}
	}
}

func TestExtractorRejectsNonFiniteMetrics(t *testing.T) {
	ctx := context.Background()

	registry := NewMappingRegistry()
	mapping, err := NewMapping(
		defaultTotalQueuedRequestsMetric,
		defaultTotalRunningRequestsMetric,
		defaultKvCacheUsagePercentageMetric,
		"",
		"",
	)
	if err != nil {
		t.Fatalf("failed to create mapping: %v", err)
	}
	if err := registry.Register(DefaultEngineType, mapping); err != nil {
		t.Fatalf("failed to register mapping: %v", err)
	}
	extractor, err := NewCoreMetricsExtractor(registry, "")
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}

	ep := fwkdl.NewEndpoint(nil, nil)

	// Seed valid baseline metrics.
	validData := sourcemetrics.PrometheusMetricMap{
		defaultTotalQueuedRequestsMetric: &dto.MetricFamily{
			Type:   dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(4.0)}}},
		},
		defaultTotalRunningRequestsMetric: &dto.MetricFamily{
			Type:   dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(2.0)}}},
		},
		defaultKvCacheUsagePercentageMetric: &dto.MetricFamily{
			Type:   dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(0.35)}}},
		},
	}
	if err := extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: validData, Endpoint: ep}); err != nil {
		t.Fatalf("unexpected baseline extract error: %v", err)
	}
	baseline := ep.GetMetrics().Clone()

	// Scrape where all gauges are non-finite (NaN / +Inf / -Inf): must error and keep baseline unchanged.
	nonFiniteData := sourcemetrics.PrometheusMetricMap{
		defaultTotalQueuedRequestsMetric: &dto.MetricFamily{
			Type:   dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(math.NaN())}}},
		},
		defaultTotalRunningRequestsMetric: &dto.MetricFamily{
			Type:   dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(math.Inf(1))}}},
		},
		defaultKvCacheUsagePercentageMetric: &dto.MetricFamily{
			Type:   dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(math.Inf(-1))}}},
		},
	}
	if err := extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: nonFiniteData, Endpoint: ep}); err == nil {
		t.Fatal("expected error for non-finite metrics, got nil")
	}
	if diff := cmp.Diff(baseline, ep.GetMetrics()); diff != "" {
		t.Errorf("expected baseline metrics to be preserved when all samples are non-finite (-want +got):\n%s", diff)
	}

	// Partial scrape: valid queue depth updates, while NaN KV cache usage preserves previous KV usage and returns error.
	partialData := sourcemetrics.PrometheusMetricMap{
		defaultTotalQueuedRequestsMetric: &dto.MetricFamily{
			Type:   dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(7.0)}}},
		},
		defaultTotalRunningRequestsMetric: &dto.MetricFamily{
			Type:   dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(2.0)}}},
		},
		defaultKvCacheUsagePercentageMetric: &dto.MetricFamily{
			Type:   dto.MetricType_GAUGE.Enum(),
			Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(math.NaN())}}},
		},
	}
	if err := extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: partialData, Endpoint: ep}); err == nil {
		t.Fatal("expected error when KV cache gauge is NaN, got nil")
	}
	afterPartial := ep.GetMetrics()
	if afterPartial.WaitingQueueSize != 7 {
		t.Errorf("WaitingQueueSize = %d, want 7", afterPartial.WaitingQueueSize)
	}
	if afterPartial.KVCacheUsagePercent != 0.35 {
		t.Errorf("KVCacheUsagePercent = %v, want 0.35", afterPartial.KVCacheUsagePercent)
	}
}
