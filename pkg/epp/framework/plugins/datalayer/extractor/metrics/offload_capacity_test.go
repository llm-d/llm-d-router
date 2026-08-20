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

package metrics

import (
	"context"
	"testing"

	dto "github.com/prometheus/client_model/go"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

func gaugeFamily(value float64, labels map[string]string) *dto.MetricFamily {
	metric := &dto.Metric{Gauge: &dto.Gauge{Value: ptr.To(value)}}
	for name, val := range labels {
		metric.Label = append(metric.Label, &dto.LabelPair{Name: ptr.To(name), Value: ptr.To(val)})
	}
	return &dto.MetricFamily{
		Type:   dto.MetricType_GAUGE.Enum(),
		Metric: []*dto.Metric{metric},
	}
}

// sglangMapping builds a mapping mirroring the built-in sglang engine config.
func sglangMapping(t *testing.T) *Mapping {
	t.Helper()
	mapping, err := NewMappingFromConfig(MappingConfig{
		TotalCacheTokens: []string{
			"sglang:hicache_host_total_tokens",
			"sglang:max_total_num_tokens",
		},
	})
	if err != nil {
		t.Fatalf("failed to create sglang mapping: %v", err)
	}
	return mapping
}

// vllmMapping builds a mapping mirroring the built-in vllm engine config.
func vllmMapping(t *testing.T) *Mapping {
	t.Helper()
	mapping, err := NewMappingFromConfig(MappingConfig{
		CacheInfo: "vllm:cache_config_info",
		OffloadDetection: []string{
			"vllm:kv_offload_cpu_cache_usage_perc",
			"vllm:kv_offload_load_bytes",
			"vllm:kv_offload_store_bytes",
			"vllm:kv_offload_total_bytes",
		},
		OffloadSizeLabel: "kv_offloading_size",
	})
	if err != nil {
		t.Fatalf("failed to create vllm mapping: %v", err)
	}
	return mapping
}

func extractWith(t *testing.T, mapping *Mapping, data sourcemetrics.PrometheusMetricMap) *fwkdl.Metrics {
	t.Helper()
	registry := NewMappingRegistry()
	if err := registry.Register(DefaultEngineType, mapping); err != nil {
		t.Fatalf("failed to register mapping: %v", err)
	}
	extractor, err := NewCoreMetricsExtractor(registry, "")
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}
	ep := fwkdl.NewEndpoint(nil, nil)
	// Errors are expected for specs whose metrics are absent from the scrape;
	// extraction of present metrics still proceeds (matches Extract semantics).
	_ = extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep})
	return ep.GetMetrics()
}

func TestExtractTotalCacheTokens(t *testing.T) {
	tests := []struct {
		name              string
		data              sourcemetrics.PrometheusMetricMap
		wantTokenCapacity int
	}{
		{
			name: "hicache host gauge present, host larger than device (inclusive tiers)",
			data: sourcemetrics.PrometheusMetricMap{
				"sglang:hicache_host_total_tokens": gaugeFamily(800000, nil),
				"sglang:max_total_num_tokens":      gaugeFamily(400000, nil),
			},
			wantTokenCapacity: 800000,
		},
		{
			name: "host smaller than device keeps device capacity (max semantics)",
			data: sourcemetrics.PrometheusMetricMap{
				"sglang:hicache_host_total_tokens": gaugeFamily(300000, nil),
				"sglang:max_total_num_tokens":      gaugeFamily(400000, nil),
			},
			wantTokenCapacity: 400000,
		},
		{
			name: "old engine version without hicache gauge falls back to device gauge",
			data: sourcemetrics.PrometheusMetricMap{
				"sglang:max_total_num_tokens": gaugeFamily(400000, nil),
			},
			wantTokenCapacity: 400000,
		},
		{
			name:              "no capacity gauges leaves token capacity unset",
			data:              sourcemetrics.PrometheusMetricMap{},
			wantTokenCapacity: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics := extractWith(t, sglangMapping(t), tt.data)
			if metrics.KvCacheMaxTokenCapacity != tt.wantTokenCapacity {
				t.Errorf("KvCacheMaxTokenCapacity = %d, want %d", metrics.KvCacheMaxTokenCapacity, tt.wantTokenCapacity)
			}
		})
	}
}

func TestDetectOffload(t *testing.T) {
	cacheInfoLabels := func(offloadSize string) map[string]string {
		return map[string]string{
			"num_gpu_blocks":     "1570",
			"block_size":         "256",
			"kv_offloading_size": offloadSize,
		}
	}

	tests := []struct {
		name         string
		data         sourcemetrics.PrometheusMetricMap
		wantDetected bool
	}{
		{
			name: "offload connector runtime metric present",
			data: sourcemetrics.PrometheusMetricMap{
				"vllm:cache_config_info":     gaugeFamily(1, cacheInfoLabels("None")),
				"vllm:kv_offload_load_bytes": gaugeFamily(0, nil),
			},
			wantDetected: true,
		},
		{
			name: "kv_offloading_size label set via built-in flag",
			data: sourcemetrics.PrometheusMetricMap{
				"vllm:cache_config_info": gaugeFamily(1, cacheInfoLabels("4.0")),
			},
			wantDetected: true,
		},
		{
			name: "no offload: label None and no connector metrics",
			data: sourcemetrics.PrometheusMetricMap{
				"vllm:cache_config_info": gaugeFamily(1, cacheInfoLabels("None")),
			},
			wantDetected: false,
		},
		{
			name:         "no offload: empty scrape",
			data:         sourcemetrics.PrometheusMetricMap{},
			wantDetected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics := extractWith(t, vllmMapping(t), tt.data)
			if metrics.KvCacheOffloadDetected != tt.wantDetected {
				t.Errorf("KvCacheOffloadDetected = %v, want %v", metrics.KvCacheOffloadDetected, tt.wantDetected)
			}
		})
	}
}

// TestDefaultEngineConfigsIncludeOffloadSpecs guards the built-in engine
// mappings: sglang carries the tier capacity gauges and vllm carries offload
// detection, so offload-extended capacity flows without operator config.
func TestDefaultEngineConfigsIncludeOffloadSpecs(t *testing.T) {
	extractor, err := newCoreMetricsExtractorPlugin(context.Background(), "test", nil)
	if err != nil {
		t.Fatalf("failed to build extractor from default configs: %v", err)
	}
	sglang, ok := extractor.registry.Get("sglang")
	if !ok {
		t.Fatal("no sglang mapping registered")
	}
	if len(sglang.TotalCacheTokens) != 2 {
		t.Errorf("sglang TotalCacheTokens specs = %d, want 2", len(sglang.TotalCacheTokens))
	}
	vllm, ok := extractor.registry.Get("vllm")
	if !ok {
		t.Fatal("no vllm mapping registered")
	}
	if len(vllm.OffloadDetection) == 0 {
		t.Error("vllm mapping has no offload detection specs")
	}
	if vllm.OffloadSizeLabel != "kv_offloading_size" {
		t.Errorf("vllm OffloadSizeLabel = %q, want kv_offloading_size", vllm.OffloadSizeLabel)
	}
}
