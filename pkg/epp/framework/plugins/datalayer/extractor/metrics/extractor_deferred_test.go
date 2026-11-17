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
	"testing"

	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/proto"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// TestExtractorPublishesDeferredWaitingRequests covers a deferredRequestsSpec
// that selects one stage of vllm:num_requests_kv_fetch_by_stage: that stage is
// published as its own attribute, WaitingQueueSize stays the queue total, and
// a scrape without the series reports 0 deferred requests.
func TestExtractorPublishesDeferredWaitingRequests(t *testing.T) {
	ctx := context.Background()
	config := `{"engineConfigs": [{
		"name": "vllm",
		"queuedRequestsSpec": "vllm:num_requests_waiting",
		"deferredRequestsSpec": "vllm:num_requests_kv_fetch_by_stage{stage=\"in_progress\"}",
		"runningRequestsSpec": "vllm:num_requests_running",
		"kvUsageSpec": "vllm:kv_cache_usage_perc"
	}]}`
	var params modelServerExtractorParams
	if err := json.Unmarshal([]byte(config), &params); err != nil {
		t.Fatalf("decode config: %v", err)
	}
	ext, err := newCoreMetricsExtractorPlugin(ctx, "core", &params)
	if err != nil {
		t.Fatalf("new extractor: %v", err)
	}

	scrape := func(byStage *dto.MetricFamily, waiting ...*dto.Metric) sourcemetrics.PrometheusMetricMap {
		data := sourcemetrics.PrometheusMetricMap{
			"vllm:num_requests_waiting": familyOf(waiting...),
			"vllm:num_requests_running": familyOf(labeledGauge(0)),
			"vllm:kv_cache_usage_perc":  familyOf(labeledGauge(0)),
		}
		if byStage != nil {
			data["vllm:num_requests_kv_fetch_by_stage"] = byStage
		}
		return data
	}
	stages := familyOf(labeledGauge(2, "stage", "in_progress"), labeledGauge(3, "stage", "completed_waiting"))

	tests := []struct {
		name string
		// previous, when set, is extracted first so the test can check what a later scrape leaves.
		previous     sourcemetrics.PrometheusMetricMap
		data         sourcemetrics.PrometheusMetricMap
		wantWaiting  int
		wantDeferred float64
	}{
		{
			name:         "stage series present",
			data:         scrape(stages, labeledGauge(6)),
			wantWaiting:  6,
			wantDeferred: 2,
		},
		{
			name:         "stage series absent",
			data:         scrape(nil, labeledGauge(6)),
			wantWaiting:  6,
			wantDeferred: 0,
		},
		{
			name:         "stage series absent after a scrape that had it",
			previous:     scrape(stages, labeledGauge(6)),
			data:         scrape(nil, labeledGauge(6)),
			wantWaiting:  6,
			wantDeferred: 0,
		},
		{
			name:         "non-finite deferred series counts as not reported",
			data:         scrape(familyOf(labeledGauge(math.NaN(), "stage", "in_progress"), labeledGauge(3, "stage", "completed_waiting")), labeledGauge(6)),
			wantWaiting:  6,
			wantDeferred: 0,
		},
		{
			name: "deferred series per engine of a data-parallel pod",
			data: scrape(familyOf(
				labeledGauge(4, "engine", "0", "stage", "in_progress"), labeledGauge(1, "engine", "0", "stage", "completed_waiting"),
				labeledGauge(0, "engine", "1", "stage", "in_progress"), labeledGauge(2, "engine", "1", "stage", "completed_waiting")),
				labeledGauge(5, "engine", "0"), labeledGauge(3, "engine", "1")),
			wantWaiting:  8,
			wantDeferred: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
			if tt.previous != nil {
				if err := ext.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: tt.previous, Endpoint: ep}); err != nil {
					t.Fatalf("Extract (previous): %v", err)
				}
			}
			if err := ext.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: tt.data, Endpoint: ep}); err != nil {
				t.Fatalf("Extract: %v", err)
			}
			if got := ep.GetMetrics().WaitingQueueSize; got != tt.wantWaiting {
				t.Errorf("WaitingQueueSize: want %d, got %d", tt.wantWaiting, got)
			}
			assertScalarMetric(t, ep, "WaitingQueueSize", float64(tt.wantWaiting))
			assertScalarMetric(t, ep, "DeferredQueueSize", tt.wantDeferred)
		})
	}
}

// TestExtractorSkipsDeferredWithoutWaitingFamily: a scrape that has the
// stage family but not the waiting family publishes no DeferredQueueSize.
func TestExtractorSkipsDeferredWithoutWaitingFamily(t *testing.T) {
	ctx := context.Background()
	ext, err := newCoreMetricsExtractorPlugin(ctx, "core", nil)
	if err != nil {
		t.Fatalf("new extractor: %v", err)
	}
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
	data := sourcemetrics.PrometheusMetricMap{
		"vllm:num_requests_kv_fetch_by_stage": familyOf(labeledGauge(2, "stage", "in_progress")),
		"vllm:num_requests_running":           familyOf(labeledGauge(0)),
		"vllm:kv_cache_usage_perc":            familyOf(labeledGauge(0)),
	}
	if err := ext.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep}); err == nil {
		t.Fatal("Extract: want an error for the missing waiting family")
	}
	assertNoScalarMetric(t, ep, "WaitingQueueSize")
	assertNoScalarMetric(t, ep, "DeferredQueueSize")
}

// TestExtractorWithoutDeferredSpecLeavesAttributeUnset pins the engines that
// name no deferredRequestsSpec: they publish no DeferredQueueSize attribute.
func TestExtractorWithoutDeferredSpecLeavesAttributeUnset(t *testing.T) {
	ctx := context.Background()
	registry := NewMappingRegistry()
	mapping, err := NewMapping("vllm:num_requests_waiting", "vllm:num_requests_running", "", "", "")
	if err != nil {
		t.Fatalf("new mapping: %v", err)
	}
	if err := registry.Register(DefaultEngineType, mapping); err != nil {
		t.Fatalf("register mapping: %v", err)
	}
	ext, err := NewCoreMetricsExtractor(registry, "")
	if err != nil {
		t.Fatalf("new extractor: %v", err)
	}
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
	data := sourcemetrics.PrometheusMetricMap{
		"vllm:num_requests_waiting": {Type: dto.MetricType_GAUGE.Enum(), Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(6.0)}}}},
		"vllm:num_requests_running": {Type: dto.MetricType_GAUGE.Enum(), Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(0.0)}}}},
	}
	if err := ext.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep}); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	assertScalarMetric(t, ep, "WaitingQueueSize", 6)
	if val, ok := ep.GetAttributes().Get(attrmetrics.ScalarMetricDataKey("DeferredQueueSize")); ok {
		t.Errorf("DeferredQueueSize: want no attribute, got %v", val)
	}
}

// labeledGauge builds a gauge sample with the given label pairs.
func labeledGauge(value float64, labels ...string) *dto.Metric {
	m := &dto.Metric{Gauge: &dto.Gauge{Value: ptr.To(value)}}
	for i := 0; i+1 < len(labels); i += 2 {
		m.Label = append(m.Label, &dto.LabelPair{Name: proto.String(labels[i]), Value: proto.String(labels[i+1])})
	}
	return m
}

// familyOf builds a gauge family from its samples.
func familyOf(series ...*dto.Metric) *dto.MetricFamily {
	return &dto.MetricFamily{Type: dto.MetricType_GAUGE.Enum(), Metric: series}
}

// TestBuiltinVLLMConfigReadsKVTransfersInProgress pins the built-in vllm
// engine config: with no configuration DeferredQueueSize is the in_progress
// stage of vllm:num_requests_kv_fetch_by_stage. A request whose transfer has
// finished (completed_waiting) still carries the deferred reason and is not
// counted.
func TestBuiltinVLLMConfigReadsKVTransfersInProgress(t *testing.T) {
	ctx := context.Background()
	ext, err := newCoreMetricsExtractorPlugin(ctx, "core", nil)
	if err != nil {
		t.Fatalf("new extractor: %v", err)
	}
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
	data := sourcemetrics.PrometheusMetricMap{
		"vllm:num_requests_waiting":           familyOf(labeledGauge(6)),
		"vllm:num_requests_waiting_by_reason": familyOf(labeledGauge(3, "reason", "capacity"), labeledGauge(3, "reason", "deferred")),
		"vllm:num_requests_kv_fetch_by_stage": familyOf(
			labeledGauge(0, "stage", "waiting_to_start"), labeledGauge(1, "stage", "in_progress"), labeledGauge(2, "stage", "completed_waiting")),
		"vllm:num_requests_running": familyOf(labeledGauge(0)),
		"vllm:kv_cache_usage_perc":  familyOf(labeledGauge(0)),
		"vllm:lora_requests_info":   familyOf(labeledGauge(1, "running_lora_adapters", "", "waiting_lora_adapters", "", "max_lora", "0")),
		"vllm:cache_config_info":    familyOf(labeledGauge(1, "block_size", "16", "num_gpu_blocks", "1000")),
	}
	if err := ext.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep}); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	assertScalarMetric(t, ep, "WaitingQueueSize", 6)
	assertScalarMetric(t, ep, "DeferredQueueSize", 1)
}
