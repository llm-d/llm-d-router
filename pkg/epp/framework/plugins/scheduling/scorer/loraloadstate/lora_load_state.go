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

package loraloadstate

import (
	"context"
	"encoding/json"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

const (
	LoraLoadStateScorerType = "lora-load-state-scorer"

	scoreGPUResident = 1.0
	scoreCPUResident = 0.8
	scoreFreeSlot    = 0.6
	scoreSaturated   = 0.0
)

// compile-time type assertion
var (
	_ fwksched.Scorer          = &LoraLoadStateScorer{}
	_ fwkplugin.ConsumerPlugin = &LoraLoadStateScorer{}
)

// LoraLoadStateScorerFactory defines the factory function for LoraLoadStateScorer.
func LoraLoadStateScorerFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewLoraLoadStateScorer().WithName(name), nil
}

// NewLoraLoadStateScorer initializes a new LoraLoadStateScorer and returns its pointer.
func NewLoraLoadStateScorer() *LoraLoadStateScorer {
	return &LoraLoadStateScorer{
		typedName: fwkplugin.TypedName{Type: LoraLoadStateScorerType, Name: LoraLoadStateScorerType},
	}
}

// LoraLoadStateScorer scores candidate endpoints by where the requested LoRA
// adapter's weights currently live on each model server: a GPU slot, the host
// cache, or nowhere. It reads adapter residency rather than request activity,
// so an idle adapter still attracts its own traffic.
type LoraLoadStateScorer struct {
	typedName fwkplugin.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *LoraLoadStateScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *LoraLoadStateScorer) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

// Consumes declares the scorer reads the per-pod resident adapter set and GPU
// slot occupancy from the endpoint's Metrics struct, published by the
// core-metrics-extractor.
func (s *LoraLoadStateScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.LoadedModelsKey, metrics.MetricsExtractorType):    map[string]fwkdl.LoraLoadState{},
			fwkplugin.NewDataKey(metrics.GPULoadedModelsKey, metrics.MetricsExtractorType): int(0),
		},
	}
}

// WithName sets the name of the scorer.
func (s *LoraLoadStateScorer) WithName(name string) *LoraLoadStateScorer {
	s.typedName.Name = name
	return s
}

// Score ranks endpoints by the cost of serving the target adapter there.
// Endpoints whose model server reports no residency (LoadedModels nil) fall
// through to the capacity tiers, which score the same for every such
// endpoint and so leave the decision to the other scorers.
func (s *LoraLoadStateScorer) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))

	for _, endpoint := range endpoints {
		m := endpoint.GetMetrics()
		state, resident := m.LoadedModels[request.TargetModel]

		switch {
		case resident && state.Level == fwkdl.LoraLoadLevelGPU:
			scores[endpoint] = scoreGPUResident
		case resident:
			scores[endpoint] = scoreCPUResident
		case m.GPULoadedModels < m.MaxActiveModels:
			scores[endpoint] = scoreFreeSlot
		default:
			scores[endpoint] = scoreSaturated
		}
	}

	return scores
}
