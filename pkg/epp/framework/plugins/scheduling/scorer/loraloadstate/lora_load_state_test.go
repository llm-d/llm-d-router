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
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

func endpoint(name string, m *fwkdl.Metrics) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}}, m, nil)
}

func TestLoraLoadStateScorer(t *testing.T) {
	gpu := fwkdl.LoraLoadState{Level: fwkdl.LoraLoadLevelGPU}
	cpu := fwkdl.LoraLoadState{Level: fwkdl.LoraLoadLevelCPU}

	tests := []struct {
		name      string
		request   *fwksched.InferenceRequest
		endpoints []fwksched.Endpoint
		expected  map[string]float64
	}{
		{
			name:    "every tier at once",
			request: &fwksched.InferenceRequest{TargetModel: "target"},
			endpoints: []fwksched.Endpoint{
				endpoint("gpu-resident", &fwkdl.Metrics{
					LoadedModels:    map[string]fwkdl.LoraLoadState{"target": gpu, "other": gpu},
					GPULoadedModels: 2, MaxActiveModels: 2,
				}),
				endpoint("cpu-resident-full", &fwkdl.Metrics{
					LoadedModels:    map[string]fwkdl.LoraLoadState{"target": cpu, "a": gpu, "b": gpu},
					GPULoadedModels: 2, MaxActiveModels: 2,
				}),
				endpoint("free-slot", &fwkdl.Metrics{
					LoadedModels:    map[string]fwkdl.LoraLoadState{"other": gpu},
					GPULoadedModels: 1, MaxActiveModels: 2,
				}),
				endpoint("saturated", &fwkdl.Metrics{
					LoadedModels:    map[string]fwkdl.LoraLoadState{"a": gpu, "b": gpu},
					GPULoadedModels: 2, MaxActiveModels: 2,
				}),
			},
			expected: map[string]float64{
				"gpu-resident":      1.0,
				"cpu-resident-full": 0.8,
				"free-slot":         0.6,
				"saturated":         0.0,
			},
		},
		{
			name:    "idle resident adapter still wins over a busy one",
			request: &fwksched.InferenceRequest{TargetModel: "target"},
			endpoints: []fwksched.Endpoint{
				endpoint("idle-holder", &fwkdl.Metrics{
					ActiveModels:    map[string]int{},
					LoadedModels:    map[string]fwkdl.LoraLoadState{"target": gpu},
					GPULoadedModels: 1, MaxActiveModels: 4,
				}),
				endpoint("busy-elsewhere", &fwkdl.Metrics{
					ActiveModels:    map[string]int{"other": 1},
					LoadedModels:    map[string]fwkdl.LoraLoadState{"other": gpu},
					GPULoadedModels: 1, MaxActiveModels: 4,
				}),
			},
			expected: map[string]float64{"idle-holder": 1.0, "busy-elsewhere": 0.6},
		},
		{
			name:    "cpu-resident beats a cold free slot",
			request: &fwksched.InferenceRequest{TargetModel: "target"},
			endpoints: []fwksched.Endpoint{
				endpoint("cpu-cached", &fwkdl.Metrics{
					LoadedModels:    map[string]fwkdl.LoraLoadState{"target": cpu},
					GPULoadedModels: 0, MaxActiveModels: 2,
				}),
				endpoint("cold", &fwkdl.Metrics{
					LoadedModels:    map[string]fwkdl.LoraLoadState{},
					GPULoadedModels: 0, MaxActiveModels: 2,
				}),
			},
			expected: map[string]float64{"cpu-cached": 0.8, "cold": 0.6},
		},
		{
			name:    "residency not reported scores by capacity only",
			request: &fwksched.InferenceRequest{TargetModel: "target"},
			endpoints: []fwksched.Endpoint{
				endpoint("legacy-with-capacity", &fwkdl.Metrics{
					ActiveModels: map[string]int{"target": 1}, MaxActiveModels: 2,
				}),
				endpoint("legacy-unknown-capacity", &fwkdl.Metrics{
					ActiveModels: map[string]int{"target": 1},
				}),
			},
			expected: map[string]float64{"legacy-with-capacity": 0.6, "legacy-unknown-capacity": 0.0},
		},
		{
			name:    "pinned does not change the tier",
			request: &fwksched.InferenceRequest{TargetModel: "target"},
			endpoints: []fwksched.Endpoint{
				endpoint("pinned", &fwkdl.Metrics{
					LoadedModels:    map[string]fwkdl.LoraLoadState{"target": {Level: fwkdl.LoraLoadLevelGPU, Pinned: true}},
					GPULoadedModels: 1, MaxActiveModels: 1,
				}),
			},
			expected: map[string]float64{"pinned": 1.0},
		},
		{
			name:      "no endpoints",
			request:   &fwksched.InferenceRequest{TargetModel: "target"},
			endpoints: []fwksched.Endpoint{},
			expected:  map[string]float64{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scores := NewLoraLoadStateScorer().Score(context.Background(), test.request, test.endpoints)
			assert.Len(t, scores, len(test.expected))
			for _, ep := range test.endpoints {
				name := ep.GetMetadata().ID.Name
				want, ok := test.expected[name]
				if !ok {
					t.Fatalf("no expected score for endpoint %s", name)
				}
				assert.InDelta(t, want, scores[ep], 0.0001, "endpoint %s", name)
			}
		})
	}
}

func TestLoraLoadStateScorerPlugin(t *testing.T) {
	plugin, err := LoraLoadStateScorerFactory("my-lora", nil, nil)
	assert.NoError(t, err)

	scorer, ok := plugin.(*LoraLoadStateScorer)
	assert.True(t, ok)
	assert.Equal(t, fwkplugin.TypedName{Type: LoraLoadStateScorerType, Name: "my-lora"}, scorer.TypedName())
	assert.Equal(t, fwksched.Affinity, scorer.Category())

	deps := scorer.Consumes()
	assert.Empty(t, deps.Optional)
	assert.Contains(t, deps.Required, fwkplugin.NewDataKey(metrics.LoadedModelsKey, metrics.MetricsExtractorType))
	assert.Contains(t, deps.Required, fwkplugin.NewDataKey(metrics.GPULoadedModelsKey, metrics.MetricsExtractorType))
}
