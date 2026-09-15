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

package prefix

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const mmProducerName = "precise-prefix-cache-producer"

var mmMatchInfoKey = attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(mmProducerName)

// mmEndpoint builds an endpoint carrying info, or carrying no match info at
// all when info is nil.
func mmEndpoint(name string, info *attrprefix.PrefixCacheMatchInfo) fwksched.Endpoint {
	ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}}, fwkdl.NewMetrics(), nil)
	if info != nil {
		ep.Put(mmMatchInfoKey, info)
	}
	return ep
}

func requestWithMMFeatures(features ...fwkrh.MultiModalFeature) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{
				Prompts: []fwkrh.PromptTokens{{MultiModalFeatures: features}},
			},
		},
	}
}

func TestAnyMMHit(t *testing.T) {
	tests := []struct {
		name        string
		endpoints   []fwksched.Endpoint
		wantHit     bool
		wantTracked bool
	}{
		{name: "no endpoints", endpoints: nil},
		{
			name: "no match info attached",
			endpoints: []fwksched.Endpoint{
				mmEndpoint("a", nil),
				mmEndpoint("b", nil),
			},
		},
		{
			name: "no producer tracked mm",
			endpoints: []fwksched.Endpoint{
				mmEndpoint("a", attrprefix.NewPrefixCacheMatchInfo(5, 10, 16)),
				mmEndpoint("b", attrprefix.NewPrefixCacheMatchInfo(8, 10, 16)),
			},
		},
		{
			name: "tracked but zero mm matches",
			endpoints: []fwksched.Endpoint{
				mmEndpoint("a", attrprefix.NewPrefixCacheMatchInfo(5, 10, 16)),
				mmEndpoint("b", attrprefix.NewPrefixCacheMatchInfo(8, 10, 16).WithMM(attrprefix.MMMatchInfo{MatchBlocks: 0})),
			},
			wantTracked: true,
		},
		{
			name: "one endpoint reports mm match",
			endpoints: []fwksched.Endpoint{
				mmEndpoint("a", attrprefix.NewPrefixCacheMatchInfo(5, 10, 16)),
				mmEndpoint("b", attrprefix.NewPrefixCacheMatchInfo(8, 10, 16).WithMM(attrprefix.MMMatchInfo{MatchBlocks: 2})),
			},
			wantHit: true, wantTracked: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hit, tracked := anyMMHit(tt.endpoints, mmMatchInfoKey)
			assert.Equal(t, tt.wantHit, hit, "hit")
			assert.Equal(t, tt.wantTracked, tracked, "tracked")
		})
	}
}

// scoreWithRecordedSpan runs Score inside a recording span and returns that
// span's attributes, mirroring how the scheduler wraps each scorer. It swaps
// the global tracer provider for the duration of the test.
func scoreWithRecordedSpan(t *testing.T, p *Plugin, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[string]any {
	t.Helper()

	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	origTP := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(origTP) })

	ctx, span := tp.Tracer("test").Start(context.Background(), "scorer."+PrefixCacheScorerPluginType)
	p.Score(ctx, request, endpoints)
	span.End()

	ended := recorder.Ended()
	require.Len(t, ended, 1)

	attrs := map[string]any{}
	for _, kv := range ended[0].Attributes() {
		attrs[string(kv.Key)] = kv.Value.AsInterface()
	}
	return attrs
}

func TestPrefixPluginScoreEmitsMultimodalAttributes(t *testing.T) {
	tests := []struct {
		name          string
		request       *fwksched.InferenceRequest
		endpoints     []fwksched.Endpoint
		wantModality  string
		wantHashCount int64
		wantHitAttr   bool
		wantHit       bool
	}{
		{
			name:          "text-only request omits mm.hit",
			request:       requestWithMMFeatures(),
			endpoints:     []fwksched.Endpoint{mmEndpoint("pod1", attrprefix.NewPrefixCacheMatchInfo(5, 10, 16))},
			wantModality:  "none",
			wantHashCount: 0,
		},
		{
			name: "multimodal miss reports false",
			request: requestWithMMFeatures(
				fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage, Hash: "h1"},
			),
			endpoints: []fwksched.Endpoint{
				mmEndpoint("pod1", attrprefix.NewPrefixCacheMatchInfo(5, 10, 16).WithMM(attrprefix.MMMatchInfo{MatchBlocks: 0})),
			},
			wantModality:  "image",
			wantHashCount: 1,
			wantHitAttr:   true,
		},
		{
			name: "multimodal hit reports true",
			request: requestWithMMFeatures(
				fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage, Hash: "h1"},
				fwkrh.MultiModalFeature{Modality: fwkrh.ModalityAudio, Hash: "h2"},
			),
			endpoints: []fwksched.Endpoint{
				mmEndpoint("pod1", attrprefix.NewPrefixCacheMatchInfo(5, 10, 16).WithMM(attrprefix.MMMatchInfo{MatchBlocks: 3})),
			},
			wantModality:  "audio,image",
			wantHashCount: 2,
			wantHitAttr:   true, wantHit: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := New(context.Background(), PrefixCacheScorerPluginType, mmProducerName)
			require.NoError(t, err)

			attrs := scoreWithRecordedSpan(t, p, tt.request, tt.endpoints)

			assert.Equal(t, tt.wantModality, attrs["mm.modality"])
			assert.Equal(t, tt.wantHashCount, attrs["mm.hash_count"])

			got, ok := attrs["mm.hit"]
			require.Equal(t, tt.wantHitAttr, ok, "mm.hit presence")
			if tt.wantHitAttr {
				assert.Equal(t, tt.wantHit, got)
			}
		})
	}
}
