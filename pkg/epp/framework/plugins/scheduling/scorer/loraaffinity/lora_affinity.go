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

package loraaffinity

import (
	"context"
	"encoding/json"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

const (
	LoraAffinityScorerType = "lora-affinity-scorer"
)

// compile-time type assertion
var (
	_ fwksched.Scorer          = &LoraAffinityScorer{}
	_ fwkplugin.ConsumerPlugin = &LoraAffinityScorer{}
)

// LoraAffinityScorerFactory defines the factory function for LoraAffinityScorer.
func LoraAffinityScorerFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewLoraAffinityScorer().WithName(name), nil
}

// NewLoraAffinityScorer initializes a new LoraAffinityScorer and returns its pointer.
func NewLoraAffinityScorer() *LoraAffinityScorer {
	return &LoraAffinityScorer{
		typedName: fwkplugin.TypedName{Type: LoraAffinityScorerType, Name: LoraAffinityScorerType},
	}
}

// LoraAffinityScorer scores list of candidate pods based on Lora affinity and availability.
type LoraAffinityScorer struct {
	typedName fwkplugin.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *LoraAffinityScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *LoraAffinityScorer) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

// Consumes declares the scorer reads the per-pod active and waiting model
// sets from the endpoint's Metrics struct, published by the core-metrics-
// extractor.
func (s *LoraAffinityScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.ActiveModelsKey, metrics.MetricsExtractorType):  map[string]int{},
			fwkplugin.NewDataKey(metrics.WaitingModelsKey, metrics.MetricsExtractorType): map[string]int{},
		},
	}
}

// WithName sets the name of the scorer.
func (s *LoraAffinityScorer) WithName(name string) *LoraAffinityScorer {
	s.typedName.Name = name
	return s
}

func (s *LoraAffinityScorer) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))

	for _, endpoint := range endpoints {
		m := endpoint.GetMetrics()
		_, active := m.ActiveModels[request.TargetModel]
		_, waiting := m.WaitingModels[request.TargetModel]

		// ActiveModels and WaitingModels share the same source in current vLLM,
		// so take the union to count each adapter once. This may change later if
		// vLLM adds native support.
		unionCount := len(m.ActiveModels)
		for k := range m.WaitingModels {
			if _, ok := m.ActiveModels[k]; !ok {
				unionCount++
			}
		}

		switch {
		case active:
			scores[endpoint] = 1.0
		case unionCount < m.MaxActiveModels:
			scores[endpoint] = 0.8
		// Unreachable against current vLLM (waiting implies active), but
		// reachable with the simulator and future backends.
		case waiting:
			scores[endpoint] = 0.6
		default:
			scores[endpoint] = 0.0
		}
	}

	return scores
}
