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

package attributeweight

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
)

const testAttributeKey = "gpu.product"

func createEndpoint(name, value string) scheduling.Endpoint {
	attrs := fwkdl.NewAttributes()
	if value != "" {
		attrs.Put(testAttributeKey, attrstring.Value(value))
	}
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: name},
		},
		nil,
		attrs,
	)
}

func TestFactory(t *testing.T) {
	tests := []struct {
		name       string
		pluginName string
		jsonParams string
		wantErr    string
	}{
		{
			name:       "valid configuration",
			pluginName: "gpu-weight",
			jsonParams: `{"attributeKey": "gpu.product", "weights": {"H100": 4, "A100": 2}}`,
		},
		{
			name:       "missing attribute key",
			pluginName: "gpu-weight",
			jsonParams: `{"weights": {"H100": 4}}`,
			wantErr:    "attributeKey",
		},
		{
			name:       "empty weights",
			pluginName: "gpu-weight",
			jsonParams: `{"attributeKey": "gpu.product", "weights": {}}`,
			wantErr:    "weights",
		},
		{
			name:       "zero weight",
			pluginName: "gpu-weight",
			jsonParams: `{"attributeKey": "gpu.product", "weights": {"L40S": 0}}`,
			wantErr:    "positive",
		},
		{
			name:       "negative weight",
			pluginName: "gpu-weight",
			jsonParams: `{"attributeKey": "gpu.product", "weights": {"L40S": -1}}`,
			wantErr:    "positive",
		},
		{
			name:       "malformed JSON",
			pluginName: "gpu-weight",
			jsonParams: `{"attributeKey": "test"`,
			wantErr:    "failed to parse",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plugin, err := Factory(tt.pluginName, fwkplugin.StrictDecoder(json.RawMessage(tt.jsonParams)), nil)

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				assert.Nil(t, plugin)
			} else {
				require.NoError(t, err)
				require.NotNil(t, plugin)
			}
		})
	}
}

func TestFactory_DefaultsNameToType(t *testing.T) {
	plugin, err := Factory("", fwkplugin.StrictDecoder(json.RawMessage(`{"attributeKey": "l", "weights": {"a": 1}}`)), nil)

	require.NoError(t, err)
	assert.Equal(t, EndpointAttributeWeightScorerType, plugin.TypedName().Name)
}

func TestScore(t *testing.T) {
	scorer, err := NewEndpointAttributeWeightScorer("gpu-weight", parameters{
		AttributeKey: testAttributeKey,
		Weights: map[string]float64{
			"H100": 4,
			"A100": 2,
			"L40S": 1,
		},
	})
	require.NoError(t, err)

	endpoints := []scheduling.Endpoint{
		createEndpoint("h100-pod", "H100"),
		createEndpoint("a100-pod", "A100"),
		createEndpoint("l40s-pod", "L40S"),
		createEndpoint("unknown-value-pod", "V100"),
		createEndpoint("no-attribute-pod", ""),
	}

	scores := scorer.Score(context.Background(), &scheduling.InferenceRequest{}, endpoints)

	assert.Equal(t, 1.0, scores[endpoints[0]], "highest configured weight normalizes to 1.0")
	assert.Equal(t, 0.5, scores[endpoints[1]], "half the max weight normalizes to 0.5")
	assert.Equal(t, 0.25, scores[endpoints[2]], "quarter the max weight normalizes to 0.25")
	assert.Equal(t, 0.25, scores[endpoints[3]], "unrecognized attribute value falls back to the lowest configured weight")
	assert.Equal(t, 0.25, scores[endpoints[4]], "missing attribute falls back to the lowest configured weight")

	assert.LessOrEqual(t, scores[endpoints[3]], scores[endpoints[2]],
		"an unrecognized value must never outscore the configured lowest tier")
	assert.LessOrEqual(t, scores[endpoints[4]], scores[endpoints[2]],
		"a missing attribute must never outscore the configured lowest tier")
	assert.Greater(t, scores[endpoints[2]], 0.0, "the lowest configured weight never scores 0")
}

func TestScore_SingleConfiguredValueNormalizesToOne(t *testing.T) {
	scorer, err := NewEndpointAttributeWeightScorer("gpu-weight", parameters{
		AttributeKey: testAttributeKey,
		Weights:      map[string]float64{"H100": 4},
	})
	require.NoError(t, err)

	endpoints := []scheduling.Endpoint{
		createEndpoint("h100-pod", "H100"),
		createEndpoint("no-attribute-pod", ""),
	}

	scores := scorer.Score(context.Background(), &scheduling.InferenceRequest{}, endpoints)

	assert.Equal(t, 1.0, scores[endpoints[0]])
	assert.Equal(t, 1.0, scores[endpoints[1]], "with a single configured value, the fallback equals it too")
}

func TestCategory(t *testing.T) {
	scorer, err := NewEndpointAttributeWeightScorer("gpu-weight", parameters{
		AttributeKey: testAttributeKey,
		Weights:      map[string]float64{"H100": 4},
	})
	require.NoError(t, err)

	assert.Equal(t, scheduling.Affinity, scorer.Category())
}
