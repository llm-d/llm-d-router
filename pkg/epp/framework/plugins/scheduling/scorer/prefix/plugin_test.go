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

package prefix

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

type capturedLogCall struct {
	level int
	msg   string
	kv    []any
}

type captureLogSink struct {
	infos  []capturedLogCall
	errors []capturedLogCall
}

func (s *captureLogSink) Init(_ logr.RuntimeInfo) {}
func (s *captureLogSink) Enabled(_ int) bool      { return true }
func (s *captureLogSink) Info(level int, msg string, kv ...any) {
	s.infos = append(s.infos, capturedLogCall{level: level, msg: msg, kv: kv})
}
func (s *captureLogSink) Error(_ error, msg string, kv ...any) {
	s.errors = append(s.errors, capturedLogCall{msg: msg, kv: kv})
}
func (s *captureLogSink) WithValues(_ ...any) logr.LogSink { return s }
func (s *captureLogSink) WithName(_ string) logr.LogSink   { return s }

func logValue(call capturedLogCall, key string) any {
	for i := 0; i+1 < len(call.kv); i += 2 {
		if call.kv[i] == key {
			return call.kv[i+1]
		}
	}
	return nil
}

func TestPrefixPluginScore(t *testing.T) {
	producerName := "approx-prefix-cache-producer"
	p, _ := New(context.Background(), PrefixCacheScorerPluginType, producerName)

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), nil)
	endpoint1.Put(key, attrprefix.NewPrefixCacheMatchInfo(5, 10, 1))

	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), nil)
	endpoint2.Put(key, attrprefix.NewPrefixCacheMatchInfo(2, 10, 1))

	endpoints := []fwksched.Endpoint{endpoint1, endpoint2}
	scores := p.Score(context.Background(), nil, endpoints)

	assert.Equal(t, 0.5, scores[endpoint1])
	assert.Equal(t, 0.2, scores[endpoint2])
}

func TestPrefixPluginScoreAggregatesMissingMatchInfo(t *testing.T) {
	producerName := "precise-prefix-cache-producer"
	p, err := New(context.Background(), PrefixCacheScorerPluginType, producerName)
	require.NoError(t, err)

	endpoints := make([]fwksched.Endpoint, 3)
	for i := range endpoints {
		endpoints[i] = fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
			ID: k8stypes.NamespacedName{Name: "pod" + string(rune('1'+i))},
		}, fwkdl.NewMetrics(), nil)
	}

	sink := &captureLogSink{}
	ctx := log.IntoContext(t.Context(), logr.New(sink))
	scores := p.Score(ctx, nil, endpoints)

	require.Len(t, scores, len(endpoints))
	for _, endpoint := range endpoints {
		assert.Zero(t, scores[endpoint])
	}
	require.Len(t, sink.infos, 1)
	assert.Empty(t, sink.errors)
	assert.Equal(t, logging.DEFAULT, sink.infos[0].level)
	assert.Equal(t, "PrefixCacheMatchInfo not found for endpoints, assigning score 0", sink.infos[0].msg)
	assert.Equal(t, len(endpoints), logValue(sink.infos[0], "count"))
}

// Endpoints missing match info are counted on their own: the aggregate record
// reports only those, while endpoints carrying match info still score normally.
func TestPrefixPluginScoreMixedMissingMatchInfo(t *testing.T) {
	producerName := "precise-prefix-cache-producer"
	p, err := New(context.Background(), PrefixCacheScorerPluginType, producerName)
	require.NoError(t, err)
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName)

	endpoints := make([]fwksched.Endpoint, 3)
	for i := range endpoints {
		endpoints[i] = fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
			ID: k8stypes.NamespacedName{Name: "pod" + string(rune('1'+i))},
		}, fwkdl.NewMetrics(), nil)
	}
	// match 5 of 10 blocks, block size 1 -> matchRatio 0.5
	endpoints[0].Put(key, attrprefix.NewPrefixCacheMatchInfo(5, 10, 1))

	sink := &captureLogSink{}
	ctx := log.IntoContext(t.Context(), logr.New(sink))
	scores := p.Score(ctx, nil, endpoints)

	require.Len(t, scores, len(endpoints))
	assert.InDelta(t, 0.5, scores[endpoints[0]], 0.0001, "endpoint with match info scores normally")
	assert.Zero(t, scores[endpoints[1]])
	assert.Zero(t, scores[endpoints[2]])

	require.Len(t, sink.infos, 1, "one aggregate record per scoring call")
	assert.Empty(t, sink.errors)
	assert.Equal(t, logging.DEFAULT, sink.infos[0].level)
	assert.Equal(t, 2, logValue(sink.infos[0], "count"), "count covers only endpoints missing match info")
}

func TestPrefixPluginScoreWithWeights(t *testing.T) {
	producerName := "approx-prefix-cache-producer"
	// matchLengthWeight = 0.5, matchLengthScaleTokens = 100
	p, _ := New(context.Background(), PrefixCacheScorerPluginType, producerName)
	p.matchLengthWeight = 0.5
	p.matchLengthScaleTokens = 100

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName)

	// Endpoint 1: match 5, total 10, block size 1
	// matchRatio = 5/10 = 0.5
	// matchLengthRatio = min(1.0, 5*1/100) = 0.05 -> squared = 0.0025
	// score = 0.5 * 0.0025 + 0.5 * 0.5 = 0.25125
	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), nil)
	endpoint1.Put(key, attrprefix.NewPrefixCacheMatchInfo(5, 10, 1))

	// Endpoint 2: match 50, total 100, block size 1
	// matchRatio = 50/100 = 0.5
	// matchLengthRatio = min(1.0, 50*1/100) = 0.5 -> squared = 0.25
	// score = 0.5 * 0.25 + 0.5 * 0.5 = 0.375
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), nil)
	endpoint2.Put(key, attrprefix.NewPrefixCacheMatchInfo(50, 100, 1))

	endpoints := []fwksched.Endpoint{endpoint1, endpoint2}
	scores := p.Score(context.Background(), nil, endpoints)

	// matchRatio is the same but we still give longer request higher score
	assert.InDelta(t, 0.25125, scores[endpoint1], 1e-6)
	assert.InDelta(t, 0.375, scores[endpoint2], 1e-6)
}

func TestPrefixPluginFactoryValidation(t *testing.T) {
	tests := []struct {
		name                  string
		config                string
		expectErr             bool
		wantMatchLengthWeight float64
		wantMatchLengthScale  int
	}{
		{
			name:                  "valid config with defaults",
			config:                `{}`,
			expectErr:             false,
			wantMatchLengthWeight: defaultMatchLengthWeight,
			wantMatchLengthScale:  defaultMatchLengthScaleTokens,
		},
		{
			name:                  "valid config with custom values",
			config:                `{"matchLengthWeight": 0.5, "matchLengthScaleTokens": 100}`,
			expectErr:             false,
			wantMatchLengthWeight: 0.5,
			wantMatchLengthScale:  100,
		},
		{
			name:      "invalid matchLengthWeight < 0",
			config:    `{"matchLengthWeight": -0.1, "matchLengthScaleTokens": 100}`,
			expectErr: true,
		},
		{
			name:      "invalid matchLengthWeight > 1",
			config:    `{"matchLengthWeight": 1.1, "matchLengthScaleTokens": 100}`,
			expectErr: true,
		},
		{
			name:      "invalid matchLengthScaleTokens <= 0",
			config:    `{"matchLengthWeight": 0.5, "matchLengthScaleTokens": 0}`,
			expectErr: true,
		},
		{
			name:                  "missing matchLengthScaleTokens when matchLengthWeight > 0 uses default",
			config:                `{"matchLengthWeight": 0.5}`,
			expectErr:             false,
			wantMatchLengthWeight: 0.5,
			wantMatchLengthScale:  defaultMatchLengthScaleTokens,
		},
		{
			name:                  "zero matchLengthWeight doesn't require matchLengthScaleTokens",
			config:                `{"matchLengthWeight": 0.0}`,
			expectErr:             false,
			wantMatchLengthWeight: 0.0,
			wantMatchLengthScale:  defaultMatchLengthScaleTokens,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handle := plugin.NewEppHandle(context.Background(), nil)
			var decoder *json.Decoder
			if tt.config != "" {
				decoder = json.NewDecoder(strings.NewReader(tt.config))
			}
			p, err := PrefixCachePluginFactory("test", decoder, handle)
			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, p)
			} else {
				assert.NoError(t, err)
				prefixPlugin, ok := p.(*Plugin)
				if assert.True(t, ok, "plugin must be of type *Plugin") {
					assert.Equal(t, tt.wantMatchLengthWeight, prefixPlugin.matchLengthWeight)
					assert.Equal(t, tt.wantMatchLengthScale, prefixPlugin.matchLengthScaleTokens)
				}
			}
		})
	}
}
