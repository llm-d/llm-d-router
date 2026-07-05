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

package dcgm

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/proto"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrgpu "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/gpu"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

func makeGauge(values ...float64) *dto.MetricFamily {
	gaugeType := dto.MetricType_GAUGE
	fam := &dto.MetricFamily{
		Name: proto.String(gpuUtilMetricName),
		Type: &gaugeType,
	}
	for _, v := range values {
		fam.Metric = append(fam.Metric, &dto.Metric{
			Gauge: &dto.Gauge{Value: &v},
		})
	}
	return fam
}

func TestExtractorExtract(t *testing.T) {
	ctx := context.Background()

	extPlugin, err := DCGMExtractorFactory("test-extractor", nil, nil)
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}
	extractor := extPlugin.(*Extractor)

	if exType := extPlugin.TypedName().Type; exType == "" {
		t.Error("empty extractor type")
	}
	if exName := extPlugin.TypedName().Name; exName == "" {
		t.Error("empty extractor name")
	}

	key := attrgpu.GPUUtilizationDataKey.WithNonEmptyProducerName(attrgpu.DCGMExtractorType).String()
	gaugeType := dto.MetricType_GAUGE

	tests := []struct {
		name        string
		data        sourcemetrics.PrometheusMetricMap
		wantErr     bool
		errContains string
		updated     bool
		wantUtil    float64
	}{
		{
			name:        "missing metric",
			data:        sourcemetrics.PrometheusMetricMap{},
			wantErr:     true,
			errContains: gpuUtilMetricName,
		},
		{
			name: "empty samples",
			data: sourcemetrics.PrometheusMetricMap{
				gpuUtilMetricName: &dto.MetricFamily{
					Name:   proto.String(gpuUtilMetricName),
					Type:   &gaugeType,
					Metric: []*dto.Metric{},
				},
			},
			wantErr:     true,
			errContains: "no samples",
		},
		{
			name: "single GPU",
			data: sourcemetrics.PrometheusMetricMap{
				gpuUtilMetricName: makeGauge(73),
			},
			updated:  true,
			wantUtil: 0.73,
		},
		{
			name: "multi-GPU takes max",
			data: sourcemetrics.PrometheusMetricMap{
				gpuUtilMetricName: makeGauge(73, 81, 65),
			},
			updated:  true,
			wantUtil: 0.81,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Extract panicked: %v", r)
				}
			}()

			ep := fwkdl.NewEndpoint(nil, nil)
			attr := ep.GetAttributes()
			before, _ := attr.Get(key)

			err := extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{
				Payload: tt.data, Endpoint: ep,
			})
			after, _ := attr.Get(key)

			if tt.wantErr && err == nil {
				t.Error("expected error but got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.wantErr && err != nil && tt.errContains != "" {
				if !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errContains)
				}
			}

			if tt.updated {
				if diff := cmp.Diff(before, after); diff == "" {
					t.Error("expected attribute to be updated, but no change detected")
				}
				val, ok := attrgpu.ReadGPUUtilization(attr, key)
				if !ok {
					t.Fatal("GPU utilization not found in attributes after extract")
				}
				if math.Abs(float64(val)-tt.wantUtil) > 0.001 {
					t.Errorf("expected utilization %.3f, got %.3f", tt.wantUtil, float64(val))
				}
			} else {
				if diff := cmp.Diff(before, after); diff != "" {
					t.Errorf("expected no attribute update, but got changes:\n%s", diff)
				}
			}
		})
	}
}

func TestFactory_SetsName(t *testing.T) {
	p, err := DCGMExtractorFactory("my-dcgm", nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := p.TypedName().Type; got != attrgpu.DCGMExtractorType {
		t.Errorf("Type = %q, want %q", got, attrgpu.DCGMExtractorType)
	}
	if got := p.TypedName().Name; got != "my-dcgm" {
		t.Errorf("Name = %q, want %q", got, "my-dcgm")
	}
}

func TestProduces_DeclaresGPUUtilization(t *testing.T) {
	ext := NewDCGMExtractor()
	produced := ext.Produces()
	if _, ok := produced[attrgpu.GPUUtilizationDataKey]; !ok {
		t.Error("Produces must declare GPUUtilizationDataKey")
	}
}
