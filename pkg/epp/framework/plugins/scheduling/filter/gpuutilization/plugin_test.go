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
	"testing"

	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrgpu "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/gpu"
)

func makeEndpoint(name string, util float64, hasData bool) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
	}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if hasData {
		ep.Put(attrgpu.GPUUtilizationDataKey.String(), attrgpu.GPUUtilization(util))
	}
	return ep
}

func newTestPlugin(threshold float64) *Plugin {
	return &Plugin{
		typedName:      fwkplugin.TypedName{Type: PluginType, Name: "test"},
		config:         Config{Threshold: threshold},
		gpuUtilDataKey: attrgpu.GPUUtilizationDataKey,
	}
}

func endpointNames(eps []fwksched.Endpoint) []string {
	names := make([]string, len(eps))
	for i, ep := range eps {
		names[i] = ep.GetMetadata().NamespacedName.Name
	}
	return names
}

func TestFilter(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		threshold float64
		endpoints []fwksched.Endpoint
		wantNames []string
	}{
		{
			name:      "single endpoint returned as-is",
			threshold: 0.90,
			endpoints: []fwksched.Endpoint{
				makeEndpoint("a", 0.95, true),
			},
			wantNames: []string{"a"},
		},
		{
			name:      "below threshold passes",
			threshold: 0.90,
			endpoints: []fwksched.Endpoint{
				makeEndpoint("low", 0.45, true),
				makeEndpoint("high", 0.92, true),
				makeEndpoint("mid", 0.71, true),
			},
			wantNames: []string{"low", "mid"},
		},
		{
			name:      "at threshold passes",
			threshold: 0.90,
			endpoints: []fwksched.Endpoint{
				makeEndpoint("exact", 0.90, true),
				makeEndpoint("over", 0.91, true),
			},
			wantNames: []string{"exact"},
		},
		{
			name:      "all above threshold returns all (graceful fallback)",
			threshold: 0.90,
			endpoints: []fwksched.Endpoint{
				makeEndpoint("a", 0.95, true),
				makeEndpoint("b", 0.92, true),
				makeEndpoint("c", 0.98, true),
			},
			wantNames: []string{"a", "b", "c"},
		},
		{
			name:      "no GPU data passes through",
			threshold: 0.90,
			endpoints: []fwksched.Endpoint{
				makeEndpoint("with-data", 0.92, true),
				makeEndpoint("no-data", 0, false),
			},
			wantNames: []string{"no-data"},
		},
		{
			name:      "all without data keeps all",
			threshold: 0.90,
			endpoints: []fwksched.Endpoint{
				makeEndpoint("a", 0, false),
				makeEndpoint("b", 0, false),
			},
			wantNames: []string{"a", "b"},
		},
		{
			name:      "mixed: below threshold and no data both pass",
			threshold: 0.90,
			endpoints: []fwksched.Endpoint{
				makeEndpoint("low", 0.30, true),
				makeEndpoint("high", 0.95, true),
				makeEndpoint("unknown", 0, false),
			},
			wantNames: []string{"low", "unknown"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Filter panicked: %v", r)
				}
			}()

			p := newTestPlugin(tt.threshold)
			got := p.Filter(ctx, nil, tt.endpoints)
			gotNames := endpointNames(got)

			if len(gotNames) != len(tt.wantNames) {
				t.Fatalf("got %d endpoints %v, want %d %v", len(gotNames), gotNames, len(tt.wantNames), tt.wantNames)
			}
			for i, name := range tt.wantNames {
				if gotNames[i] != name {
					t.Errorf("endpoint[%d] = %q, want %q", i, gotNames[i], name)
				}
			}
		})
	}
}

func TestFactory_ValidConfig(t *testing.T) {
	plugin, err := Factory("test", fwkplugin.StrictDecoder(nil), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plugin == nil {
		t.Fatal("expected non-nil plugin")
	}
	if got := plugin.TypedName().Type; got != PluginType {
		t.Errorf("Type = %q, want %q", got, PluginType)
	}
}

func TestFactory_CustomThreshold(t *testing.T) {
	plugin, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"threshold": 0.75}`)), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	p := plugin.(*Plugin)
	if p.config.Threshold != 0.75 {
		t.Errorf("Threshold = %f, want 0.75", p.config.Threshold)
	}
}

func TestFactory_InvalidThreshold(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"threshold": 1.5}`)), nil)
	if err == nil {
		t.Fatal("expected error for threshold > 1")
	}
}

func TestConsumes_DeclaresRequired(t *testing.T) {
	p := newTestPlugin(DefaultThreshold)
	deps := p.Consumes()
	if len(deps.Required) == 0 {
		t.Fatal("expected Required dependencies")
	}
	if _, ok := deps.Required[attrgpu.GPUUtilizationDataKey]; !ok {
		t.Error("expected GPUUtilizationDataKey in Required")
	}
}
