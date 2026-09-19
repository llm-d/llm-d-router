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

package modelaffinity

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type fakeEndpoint struct {
	meta *fwkdl.EndpointMetadata
}

func (f *fakeEndpoint) GetMetadata() *fwkdl.EndpointMetadata { return f.meta }
func (f *fakeEndpoint) GetMetrics() *fwkdl.Metrics           { return nil }
func (f *fakeEndpoint) String() string                       { return f.meta.Name }
func (f *fakeEndpoint) Get(string) (fwkdl.Cloneable, bool)   { return nil, false }
func (f *fakeEndpoint) Put(string, fwkdl.Cloneable)          {}
func (f *fakeEndpoint) Keys() []string                       { return nil }
func (f *fakeEndpoint) Clone() fwkdl.AttributeMap            { return nil }

func ep(name string, labels map[string]string) scheduling.Endpoint {
	return &fakeEndpoint{meta: &fwkdl.EndpointMetadata{Name: name, Labels: labels}}
}

func TestFactory(t *testing.T) {
	t.Run("nil parameters uses defaults", func(t *testing.T) {
		p, err := Factory("test", nil, nil)
		require.NoError(t, err)
		f := p.(*ModelAffinityFilter)
		assert.Equal(t, DefaultLabelKey, f.labelKey)
		assert.Equal(t, DefaultModelHeader, f.modelHeader)
	})

	t.Run("empty name returns error", func(t *testing.T) {
		_, err := Factory("", nil, nil)
		require.Error(t, err)
	})
}

func TestFilter(t *testing.T) {
	endpoints := []scheduling.Endpoint{
		ep("spoke1", map[string]string{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}),
		ep("spoke2", map[string]string{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}),
		ep("spoke3", map[string]string{"model": "Qwen/Qwen2.5-0.5B-Instruct"}),
	}

	tests := []struct {
		name         string
		request      *scheduling.InferenceRequest
		expected     []string
		labelKey     string
		modelHeader  string
	}{
		{
			name: "filters by header",
			request: &scheduling.InferenceRequest{
				Headers: map[string]string{"x-gateway-model-name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
			},
			expected: []string{"spoke1", "spoke2"},
		},
		{
			name: "filters by TargetModel fallback",
			request: &scheduling.InferenceRequest{
				TargetModel: "Qwen/Qwen2.5-0.5B-Instruct",
			},
			expected: []string{"spoke3"},
		},
		{
			name:     "nil request passes all through",
			request:  nil,
			expected: []string{"spoke1", "spoke2", "spoke3"},
		},
		{
			name: "no matching model returns empty",
			request: &scheduling.InferenceRequest{
				TargetModel: "nonexistent-model",
			},
			expected: []string{},
		},
		{
			name: "header takes priority over TargetModel",
			request: &scheduling.InferenceRequest{
				Headers:     map[string]string{"x-gateway-model-name": "Qwen/Qwen2.5-0.5B-Instruct"},
				TargetModel: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
			},
			expected: []string{"spoke3"},
		},
		{
			name: "custom label key",
			request: &scheduling.InferenceRequest{
				TargetModel: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
			},
			labelKey: "served-model",
			expected: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			labelKey := tt.labelKey
			if labelKey == "" {
				labelKey = DefaultLabelKey
			}
			modelHeader := tt.modelHeader
			if modelHeader == "" {
				modelHeader = DefaultModelHeader
			}

			f, err := New("test", parameters{LabelKey: labelKey, ModelHeader: modelHeader})
			require.NoError(t, err)

			result := f.Filter(context.Background(), tt.request, endpoints)

			names := make([]string, 0, len(result))
			for _, ep := range result {
				names = append(names, ep.GetMetadata().Name)
			}
			assert.Equal(t, tt.expected, names)
		})
	}
}
