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

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestPrefixPluginScore(t *testing.T) {
	producerName := "approx-prefix-cache-producer"
	p, _ := New(context.Background(), PrefixCacheScorerPluginType, producerName)

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName).String()

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), nil)
	endpoint1.Put(key, attrprefix.NewPrefixCacheMatchInfo(5, 10, 1))

	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), nil)
	endpoint2.Put(key, attrprefix.NewPrefixCacheMatchInfo(2, 10, 1))

	endpoints := []fwksched.Endpoint{endpoint1, endpoint2}
	scores := p.Score(context.Background(), nil, endpoints)

	assert.Equal(t, 0.5, scores[endpoint1])
	assert.Equal(t, 0.2, scores[endpoint2])
}

func TestPrefixPluginScoreWithWeights(t *testing.T) {
	producerName := "approx-prefix-cache-producer"
	// prefixLengthWeight = 0.5, maxModelLen = 100
	p, _ := New(context.Background(), PrefixCacheScorerPluginType, producerName)
	p.prefixLengthWeight = 0.5
	p.maxModelLen = 100

	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName).String()

	// Endpoint 1: match 50, total 80, block size 1
	// matchRatio = 50/80 = 0.625
	// matchLengthRatio = 80*1/100 = 0.8 -> squared = 0.64
	// score = 0.5 * 0.64 + 0.5 * 0.625 = 0.6325
	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), nil)
	endpoint1.Put(key, attrprefix.NewPrefixCacheMatchInfo(50, 80, 1))

	// Endpoint 2: match 10, total 200, block size 1 (exceeds maxModelLen)
	// matchRatio = 10/200 = 0.05
	// matchLengthRatio = 200*1/100 = 2.0 -> squared = 4.0 -> capped at 1.0
	// score = 0.5 * 1.0 + 0.5 * 0.05 = 0.525
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), nil)
	endpoint2.Put(key, attrprefix.NewPrefixCacheMatchInfo(10, 200, 1))

	endpoints := []fwksched.Endpoint{endpoint1, endpoint2}
	scores := p.Score(context.Background(), nil, endpoints)

	assert.InDelta(t, 0.6325, scores[endpoint1], 1e-6)
	assert.InDelta(t, 0.525, scores[endpoint2], 1e-6)
}

func TestPrefixPluginFactoryValidation(t *testing.T) {
	tests := []struct {
		name      string
		config    string
		expectErr bool
	}{
		{
			name:      "valid config with defaults",
			config:    `{}`,
			expectErr: false,
		},
		{
			name:      "valid config with custom values",
			config:    `{"prefixLengthWeight": 0.5, "maxModelLen": 100}`,
			expectErr: false,
		},
		{
			name:      "invalid prefixLengthWeight < 0",
			config:    `{"prefixLengthWeight": -0.1, "maxModelLen": 100}`,
			expectErr: true,
		},
		{
			name:      "invalid prefixLengthWeight > 1",
			config:    `{"prefixLengthWeight": 1.1, "maxModelLen": 100}`,
			expectErr: true,
		},
		{
			name:      "invalid maxModelLen <= 0",
			config:    `{"prefixLengthWeight": 0.5, "maxModelLen": 0}`,
			expectErr: true,
		},
		{
			name:      "missing maxModelLen when prefixLengthWeight > 0",
			config:    `{"prefixLengthWeight": 0.5}`,
			expectErr: true,
		},
		{
			name:      "zero prefixLengthWeight doesn't require maxModelLen",
			config:    `{"prefixLengthWeight": 0.0}`,
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handle := plugin.NewEppHandle(context.Background(), nil)
			var decoder *json.Decoder
			if tt.config != "" {
				decoder = json.NewDecoder(strings.NewReader(tt.config))
			}
			_, err := PrefixCachePluginFactory("test", decoder, handle)
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
