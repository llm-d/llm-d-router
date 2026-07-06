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

package gpuutilization

import (
	"context"
	"encoding/json"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrgpu "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/gpu"
)

const (
	PluginType = "gpu-utilization-scorer"
)

var _ fwksched.Scorer = &Scorer{}

// Scorer scores endpoints inversely to their GPU utilization: lower
// utilization yields a higher score, spreading load toward less-busy GPUs.
type Scorer struct {
	typedName      fwkplugin.TypedName
	gpuUtilDataKey fwkplugin.DataKey
}

// Factory creates a gpu-utilization-scorer plugin from configuration.
func Factory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return &Scorer{
		typedName:      fwkplugin.TypedName{Type: PluginType, Name: name},
		gpuUtilDataKey: attrgpu.GPUUtilizationDataKey,
	}, nil
}

func (s *Scorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns Distribution: the scorer spreads requests toward endpoints
// with lower GPU utilization.
func (s *Scorer) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

// Score returns 1 - GPUUtilization for each endpoint. Endpoints without GPU
// data receive score 0 (prefer endpoints with known-low utilization).
func (s *Scorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		raw, ok := ep.Get(s.gpuUtilDataKey.String())
		if !ok {
			scores[ep] = 0
			continue
		}
		util := raw.(attrgpu.GPUUtilization)
		scores[ep] = 1.0 - float64(util)
	}
	return scores
}

// Consumes declares a Required dependency on GPUUtilization so the framework
// validates at init that a producer (dcgm-extractor) is configured.
func (s *Scorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{s.gpuUtilDataKey: attrgpu.GPUUtilization(0)},
	}
}
