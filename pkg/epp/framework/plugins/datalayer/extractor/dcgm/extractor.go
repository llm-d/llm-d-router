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
	"encoding/json"
	"fmt"

	dto "github.com/prometheus/client_model/go"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrgpu "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/gpu"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

const gpuUtilMetricName = "DCGM_FI_DEV_GPU_UTIL"

var _ fwkplugin.ProducerPlugin = &Extractor{}

var _ fwkdl.PollingExtractor[sourcemetrics.PrometheusMetricMap] = &Extractor{}

// Extractor reads DCGM Exporter metrics and stores aggregated GPU
// utilization in the endpoint's AttributeMap.
type Extractor struct {
	typedName fwkplugin.TypedName
	dk        fwkplugin.DataKey
}

// NewDCGMExtractor returns a new DCGM metrics extractor.
func NewDCGMExtractor() *Extractor {
	return &Extractor{
		typedName: fwkplugin.TypedName{
			Type: attrgpu.DCGMExtractorType,
			Name: attrgpu.DCGMExtractorType,
		},
		dk: attrgpu.GPUUtilizationDataKey,
	}
}

func (e *Extractor) TypedName() fwkplugin.TypedName {
	return e.typedName
}

// DCGMExtractorFactory instantiates a dcgm-extractor plugin from configuration.
func DCGMExtractorFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	ext := NewDCGMExtractor()
	ext.typedName.Name = name
	return ext, nil
}

// Extract parses DCGM_FI_DEV_GPU_UTIL from the Prometheus families,
// aggregates across GPUs (max), normalizes to [0.0, 1.0], and stores
// the result in the endpoint's AttributeMap.
func (e *Extractor) Extract(_ context.Context, in fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]) error {
	family, ok := in.Payload[gpuUtilMetricName]
	if !ok {
		return fmt.Errorf("metric %q not found in DCGM response", gpuUtilMetricName)
	}

	maxUtil := 0.0
	found := false
	for _, m := range family.GetMetric() {
		val := gaugeValue(m)
		if val > maxUtil {
			maxUtil = val
		}
		found = true
	}
	if !found {
		return fmt.Errorf("metric %q present but has no samples", gpuUtilMetricName)
	}

	normalized := attrgpu.GPUUtilization(maxUtil / 100.0)
	in.Endpoint.GetAttributes().Put(e.dk.String(), normalized)
	return nil
}

// Produces declares the data key this extractor publishes.
func (e *Extractor) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{e.dk: attrgpu.GPUUtilization(0)}
}

func gaugeValue(m *dto.Metric) float64 {
	if g := m.GetGauge(); g != nil {
		return g.GetValue()
	}
	return 0
}
