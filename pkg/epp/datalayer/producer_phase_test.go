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

package datalayer

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// mockScreener is a Screener that consumes the given keys.
type mockScreener struct {
	name     string
	consumes map[fwkplugin.DataKey]any
	optional map[fwkplugin.DataKey]any
}

func (m *mockScreener) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: m.name, Type: "screener"}
}

func (m *mockScreener) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Required: m.consumes, Optional: m.optional}
}

func (m *mockScreener) Screen(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	return endpoints
}

// mockResponseHeaderScreener is a Screener that also implements
// ResponseHeaderProcessor, which places it in the request control layer
// unless screeners are ordered first.
type mockResponseHeaderScreener struct {
	mockScreener
}

func (m *mockResponseHeaderScreener) ResponseHeader(context.Context, *fwksched.InferenceRequest, *fwkrc.Response, *fwkdl.EndpointMetadata) {
}

func TestResolveDataProducerPhases(t *testing.T) {
	tokens := fwkplugin.NewDataKey("tokens", "mock")
	prefix := fwkplugin.NewDataKey("prefix", "mock")
	load := fwkplugin.NewDataKey("load", "mock")
	latency := fwkplugin.NewDataKey("latency", "mock")

	tokenizer := &mockDataProducerP{name: "tokenizer", produces: map[fwkplugin.DataKey]any{tokens: nil}}
	prefixCache := &mockDataProducerP{name: "prefix", produces: map[fwkplugin.DataKey]any{prefix: nil},
		consumes: map[fwkplugin.DataKey]any{tokens: nil}}
	inflight := &mockDataProducerP{name: "inflight", produces: map[fwkplugin.DataKey]any{load: nil}}
	predictor := &mockDataProducerP{name: "predictor", produces: map[fwkplugin.DataKey]any{latency: nil},
		consumes: map[fwkplugin.DataKey]any{tokens: nil}}

	tests := []struct {
		name     string
		plugins  []fwkplugin.Plugin
		explicit map[string]fwkrc.DataProducerPhase
		want     map[string]fwkrc.DataProducerPhase
	}{
		{
			name:    "no screener consumes producer data",
			plugins: []fwkplugin.Plugin{tokenizer, prefixCache, inflight, &mockScreener{name: "s"}},
			want: map[string]fwkrc.DataProducerPhase{
				"tokenizer": fwkrc.DataProducerPhaseScheduling,
				"prefix":    fwkrc.DataProducerPhaseScheduling,
				"inflight":  fwkrc.DataProducerPhaseScheduling,
			},
		},
		{
			name: "screener requirement promotes the producer and its inputs",
			plugins: []fwkplugin.Plugin{tokenizer, prefixCache, inflight, predictor,
				&mockScreener{name: "s", consumes: map[fwkplugin.DataKey]any{prefix: nil}}},
			want: map[string]fwkrc.DataProducerPhase{
				"tokenizer": fwkrc.DataProducerPhasePreScreen,
				"prefix":    fwkrc.DataProducerPhasePreScreen,
				"inflight":  fwkrc.DataProducerPhaseScheduling,
				"predictor": fwkrc.DataProducerPhaseScheduling,
			},
		},
		{
			name: "optional screener dependency promotes",
			plugins: []fwkplugin.Plugin{inflight,
				&mockScreener{name: "s", optional: map[fwkplugin.DataKey]any{load: nil}}},
			want: map[string]fwkrc.DataProducerPhase{"inflight": fwkrc.DataProducerPhasePreScreen},
		},
		{
			name: "screener implementing ResponseHeaderProcessor promotes",
			plugins: []fwkplugin.Plugin{inflight,
				&mockResponseHeaderScreener{mockScreener{name: "s", consumes: map[fwkplugin.DataKey]any{load: nil}}}},
			want: map[string]fwkrc.DataProducerPhase{"inflight": fwkrc.DataProducerPhasePreScreen},
		},
		{
			name:     "explicit PreScreen producer promotes its inputs",
			plugins:  []fwkplugin.Plugin{tokenizer, prefixCache},
			explicit: map[string]fwkrc.DataProducerPhase{"prefix": fwkrc.DataProducerPhasePreScreen},
			want: map[string]fwkrc.DataProducerPhase{
				"tokenizer": fwkrc.DataProducerPhasePreScreen,
				"prefix":    fwkrc.DataProducerPhasePreScreen,
			},
		},
		{
			name:     "explicit PreScreen without consumers",
			plugins:  []fwkplugin.Plugin{inflight},
			explicit: map[string]fwkrc.DataProducerPhase{"inflight": fwkrc.DataProducerPhasePreScreen},
			want:     map[string]fwkrc.DataProducerPhase{"inflight": fwkrc.DataProducerPhasePreScreen},
		},
		{
			name: "explicit Scheduling is not overridden",
			plugins: []fwkplugin.Plugin{tokenizer, prefixCache,
				&mockScreener{name: "s", consumes: map[fwkplugin.DataKey]any{prefix: nil}}},
			explicit: map[string]fwkrc.DataProducerPhase{"prefix": fwkrc.DataProducerPhaseScheduling},
			want: map[string]fwkrc.DataProducerPhase{
				"tokenizer": fwkrc.DataProducerPhaseScheduling,
				"prefix":    fwkrc.DataProducerPhaseScheduling,
			},
		},
		{
			name:     "explicit entry for a non-producer is ignored",
			plugins:  []fwkplugin.Plugin{inflight, &mockScreener{name: "s"}},
			explicit: map[string]fwkrc.DataProducerPhase{"s": fwkrc.DataProducerPhasePreScreen},
			want:     map[string]fwkrc.DataProducerPhase{"inflight": fwkrc.DataProducerPhaseScheduling},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, ResolveDataProducerPhases(tt.plugins, tt.explicit))
		})
	}
}

func TestValidateAndOrderDataDependenciesProducerPhases(t *testing.T) {
	tokens := fwkplugin.NewDataKey("tokens", "mock")
	prefix := fwkplugin.NewDataKey("prefix", "mock")
	tokenizer := &mockDataProducerP{name: "tokenizer", produces: map[fwkplugin.DataKey]any{tokens: nil}}
	prefixCache := &mockDataProducerP{name: "prefix", produces: map[fwkplugin.DataKey]any{prefix: nil},
		consumes: map[fwkplugin.DataKey]any{tokens: nil}}
	screener := &mockScreener{name: "s", consumes: map[fwkplugin.DataKey]any{prefix: nil}}
	responseHeaderScreener := &mockResponseHeaderScreener{mockScreener{name: "rh", consumes: map[fwkplugin.DataKey]any{prefix: nil}}}
	preScreen := map[string]fwkrc.DataProducerPhase{
		"tokenizer": fwkrc.DataProducerPhasePreScreen,
		"prefix":    fwkrc.DataProducerPhasePreScreen,
	}

	t.Run("screener consumes PreScreen producers", func(t *testing.T) {
		ordered, err := ValidateAndOrderDataDependencies([]fwkplugin.Plugin{screener, prefixCache, tokenizer},
			WithDataProducerPhases(preScreen))
		require.NoError(t, err)
		assert.Equal(t, []string{"tokenizer/mock", "prefix/mock", "s/screener"}, ordered)
	})

	t.Run("screener consuming a Scheduling producer fails with a hint", func(t *testing.T) {
		_, err := ValidateAndOrderDataDependencies([]fwkplugin.Plugin{screener, prefixCache, tokenizer})
		assert.ErrorContains(t, err, "producer prefix/mock needs to be executed before consumer s/screener")
		assert.ErrorContains(t, err, "run the producer in the PreScreen phase")
	})

	t.Run("screener implementing ResponseHeaderProcessor is ordered as a screener", func(t *testing.T) {
		_, err := ValidateAndOrderDataDependencies([]fwkplugin.Plugin{responseHeaderScreener, prefixCache, tokenizer})
		assert.ErrorContains(t, err, "needs to be executed before consumer rh/screener")

		_, err = ValidateAndOrderDataDependencies([]fwkplugin.Plugin{responseHeaderScreener, prefixCache, tokenizer},
			WithDataProducerPhases(preScreen))
		assert.NoError(t, err)
	})

	t.Run("PreScreen producer consuming a Scheduling producer fails", func(t *testing.T) {
		_, err := ValidateAndOrderDataDependencies([]fwkplugin.Plugin{prefixCache, tokenizer},
			WithDataProducerPhases(map[string]fwkrc.DataProducerPhase{"prefix": fwkrc.DataProducerPhasePreScreen}))
		assert.ErrorContains(t, err, "producer tokenizer/mock needs to be executed before consumer prefix/mock")
		assert.ErrorContains(t, err, "run the producer in the PreScreen phase")
	})

	t.Run("resolved phases validate", func(t *testing.T) {
		plugins := []fwkplugin.Plugin{screener, prefixCache, tokenizer}
		_, err := ValidateAndOrderDataDependencies(plugins, WithDataProducerPhases(ResolveDataProducerPhases(plugins, nil)))
		assert.NoError(t, err)
	})
}
