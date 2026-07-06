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
	"math"
	"testing"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrgpu "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/gpu"
)

func makeEndpoint(util float64, hasData bool) fwksched.Endpoint {
	ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if hasData {
		ep.Put(attrgpu.GPUUtilizationDataKey.String(), attrgpu.GPUUtilization(util))
	}
	return ep
}

func TestScore(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name       string
		endpoints  []fwksched.Endpoint
		wantScores []float64
	}{
		{
			name: "different utilizations",
			endpoints: []fwksched.Endpoint{
				makeEndpoint(0.80, true),
				makeEndpoint(0.50, true),
				makeEndpoint(0.00, true),
			},
			wantScores: []float64{0.20, 0.50, 1.00},
		},
		{
			name: "same utilization",
			endpoints: []fwksched.Endpoint{
				makeEndpoint(0.60, true),
				makeEndpoint(0.60, true),
			},
			wantScores: []float64{0.40, 0.40},
		},
		{
			name: "full utilization",
			endpoints: []fwksched.Endpoint{
				makeEndpoint(1.00, true),
				makeEndpoint(0.50, true),
			},
			wantScores: []float64{0.00, 0.50},
		},
		{
			name: "zero utilization",
			endpoints: []fwksched.Endpoint{
				makeEndpoint(0.00, true),
				makeEndpoint(0.00, true),
			},
			wantScores: []float64{1.00, 1.00},
		},
		{
			name: "no GPU data scores zero",
			endpoints: []fwksched.Endpoint{
				makeEndpoint(0.30, true),
				makeEndpoint(0, false),
			},
			wantScores: []float64{0.70, 0.00},
		},
	}

	scorer := &Scorer{
		typedName:      fwkplugin.TypedName{Type: PluginType, Name: "test"},
		gpuUtilDataKey: attrgpu.GPUUtilizationDataKey,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Score panicked: %v", r)
				}
			}()

			scores := scorer.Score(ctx, nil, tt.endpoints)

			for i, ep := range tt.endpoints {
				got := scores[ep]
				want := tt.wantScores[i]
				if math.Abs(got-want) > 0.0001 {
					t.Errorf("endpoint[%d]: score = %.4f, want %.4f", i, got, want)
				}
			}
		})
	}
}

func TestCategory(t *testing.T) {
	s := &Scorer{
		typedName:      fwkplugin.TypedName{Type: PluginType, Name: "test"},
		gpuUtilDataKey: attrgpu.GPUUtilizationDataKey,
	}
	if got := s.Category(); got != fwksched.Distribution {
		t.Errorf("Category() = %v, want Distribution", got)
	}
}

func TestFactory_SetsName(t *testing.T) {
	p, err := Factory("my-scorer", nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := p.TypedName().Type; got != PluginType {
		t.Errorf("Type = %q, want %q", got, PluginType)
	}
	if got := p.TypedName().Name; got != "my-scorer" {
		t.Errorf("Name = %q, want %q", got, "my-scorer")
	}
}

func TestConsumes_DeclaresRequired(t *testing.T) {
	s := &Scorer{
		typedName:      fwkplugin.TypedName{Type: PluginType, Name: "test"},
		gpuUtilDataKey: attrgpu.GPUUtilizationDataKey,
	}
	deps := s.Consumes()
	if len(deps.Required) == 0 {
		t.Fatal("expected Required dependencies")
	}
	if _, ok := deps.Required[attrgpu.GPUUtilizationDataKey]; !ok {
		t.Error("expected GPUUtilizationDataKey in Required")
	}
}
