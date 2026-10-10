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

package servedmodel

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
)

func TestServedModelFilterFactory(t *testing.T) {
	tests := []struct {
		name                string
		pluginName          string
		parameters          string
		wantName            string
		wantPassOnMissing   bool
		wantFallbackOnEmpty bool
		wantErr             string
	}{
		{name: "no parameters", pluginName: "test-filter", wantName: "test-filter", wantPassOnMissing: true},
		{name: "default name", wantName: ServedModelFilterType, wantPassOnMissing: true},
		{name: "empty object", pluginName: "test-filter", parameters: `{}`, wantName: "test-filter", wantPassOnMissing: true},
		{name: "fail on missing", pluginName: "test-filter", parameters: `{"onMissing": "Fail"}`, wantName: "test-filter"},
		{name: "pass on missing with fallback", pluginName: "test-filter", parameters: `{"onMissing": "Pass", "fallbackOnEmpty": true}`,
			wantName: "test-filter", wantPassOnMissing: true, wantFallbackOnEmpty: true},
		{name: "invalid onMissing", pluginName: "test-filter", parameters: `{"onMissing": "Maybe"}`, wantErr: "onMissing"},
		{name: "malformed", pluginName: "test-filter", parameters: `{"onMissing": 3}`, wantErr: "failed to parse"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var decoder *json.Decoder
			if test.parameters != "" {
				decoder = json.NewDecoder(strings.NewReader(test.parameters))
			}
			p, err := ServedModelFilterFactory(test.pluginName, decoder, nil)
			if test.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), test.wantErr)
				return
			}
			require.NoError(t, err)
			f, ok := p.(*ServedModelFilter)
			require.True(t, ok)
			assert.Equal(t, plugin.TypedName{Type: ServedModelFilterType, Name: test.wantName}, f.TypedName())
			assert.Equal(t, test.wantPassOnMissing, f.passOnMissing)
			assert.Equal(t, test.wantFallbackOnEmpty, f.fallbackOnEmpty)
		})
	}
}

func TestServedModelFilterConsumesModels(t *testing.T) {
	f, err := NewServedModelFilter("", parameters{})
	require.NoError(t, err)
	deps := f.Consumes()
	require.Contains(t, deps.Required, attrmodels.ModelsAttributeKey, "models attribute must be a required dependency")
	assert.IsType(t, attrmodels.ModelDataCollection{}, deps.Required[attrmodels.ModelsAttributeKey])
}

func TestServedModelFilter(t *testing.T) {
	base := "llama"
	withBase := newEndpoint("p1", base)                // base only
	withAdapter := newEndpoint("p2", base, "sql-v3")   // base and adapter
	withOther := newEndpoint("p3", base, "support-v1") // base and another adapter
	noList := newEndpointWithoutModels("p4")           // not polled yet, or every poll failed
	emptyList := newEndpoint("p5")                     // polled, lists nothing
	noList2 := newEndpointWithoutModels("p6")

	all := []scheduling.Endpoint{withBase, withAdapter, withOther, noList, emptyList}

	tests := []struct {
		name       string
		params     parameters
		model      string
		candidates []scheduling.Endpoint // defaults to all
		want       []scheduling.Endpoint
	}{
		{
			name:  "adapter request skips endpoints without a list when a listed endpoint serves it",
			model: "sql-v3",
			want:  []scheduling.Endpoint{withAdapter},
		},
		{
			name:  "base model request keeps endpoints without a list",
			model: base,
			want:  []scheduling.Endpoint{withBase, withAdapter, withOther, noList},
		},
		{
			name:  "endpoints without a list are used when no endpoint lists the model",
			model: "nobody-has-this",
			want:  []scheduling.Endpoint{noList},
		},
		{
			name:       "no endpoint has a list",
			model:      base,
			candidates: []scheduling.Endpoint{noList, noList2},
			want:       []scheduling.Endpoint{noList, noList2},
		},
		{
			name:   "endpoints without a list take precedence over fallback",
			params: parameters{FallbackOnEmpty: true},
			model:  "nobody-has-this",
			want:   []scheduling.Endpoint{noList},
		},
		{
			name:       "fallback when no endpoint lacks a list",
			params:     parameters{FallbackOnEmpty: true},
			model:      "nobody-has-this",
			candidates: []scheduling.Endpoint{withBase, emptyList},
			want:       []scheduling.Endpoint{withBase, emptyList},
		},
		{
			name:   "fail on missing drops endpoints without a list for a base model",
			params: parameters{OnMissing: onMissingFail},
			model:  base,
			want:   []scheduling.Endpoint{withBase, withAdapter, withOther},
		},
		{
			name:   "fail on missing yields no endpoint for an unlisted model",
			params: parameters{OnMissing: onMissingFail},
			model:  "nobody-has-this",
			want:   nil,
		},
		{
			name:       "fail on missing with no list anywhere yields no endpoint",
			params:     parameters{OnMissing: onMissingFail},
			model:      base,
			candidates: []scheduling.Endpoint{noList, noList2},
			want:       nil,
		},
		{
			name:   "fallback returns everything when no endpoint qualifies",
			params: parameters{OnMissing: onMissingFail, FallbackOnEmpty: true},
			model:  "nobody-has-this",
			want:   all,
		},
		{
			name:  "empty target model leaves the set alone",
			model: "",
			want:  all,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			f, err := NewServedModelFilter("", test.params)
			require.NoError(t, err)
			candidates := test.candidates
			if candidates == nil {
				candidates = all
			}
			got := f.Filter(t.Context(), &scheduling.InferenceRequest{TargetModel: test.model}, candidates)
			assert.ElementsMatch(t, test.want, got)
		})
	}
}

func TestServedModelFilterNilRequest(t *testing.T) {
	f, err := NewServedModelFilter("", parameters{})
	require.NoError(t, err)
	all := []scheduling.Endpoint{newEndpoint("p1", "llama")}
	assert.Equal(t, all, f.Filter(t.Context(), nil, all))
}

func newEndpoint(name string, models ...string) scheduling.Endpoint {
	collection := make(attrmodels.ModelDataCollection, 0, len(models))
	for i, id := range models {
		md := attrmodels.ModelData{ID: id, Object: "model"}
		if i > 0 {
			md.Parent = models[0]
		}
		collection = append(collection, md)
	}
	attrs := fwkdl.NewAttributes()
	attrs.Put(attrmodels.ModelsAttributeKey, collection)
	return scheduling.NewEndpoint(&fwkdl.EndpointMetadata{Name: name}, &fwkdl.Metrics{}, attrs)
}

func newEndpointWithoutModels(name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(&fwkdl.EndpointMetadata{Name: name}, &fwkdl.Metrics{}, nil)
}
