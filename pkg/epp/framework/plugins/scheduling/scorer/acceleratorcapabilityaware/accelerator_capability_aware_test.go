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

package acceleratorcapabilityaware

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

func createEndpoint(nsn k8stypes.NamespacedName, labels map[string]string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: nsn,
			Address:        "10.0.0.1",
			Labels:         labels,
		},
		nil,
		nil,
	)
}

func createRequest(tokenCount int) *scheduling.InferenceRequest {
	tokenIDs := make([]uint32, tokenCount)
	for i := range tokenIDs {
		tokenIDs[i] = uint32(i + 1)
	}
	return &scheduling.InferenceRequest{
		RequestID: "test-request",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{tokenIDs}},
		},
	}
}

func TestFactory(t *testing.T) {
	tests := []struct {
		name       string
		jsonParams string
		expectErr  bool
	}{
		{
			name:       "valid configuration with defaults",
			jsonParams: `{}`,
		},
		{
			name:       "empty label should error",
			jsonParams: `{"label": ""}`,
			expectErr:  true,
		},
		{
			name:       "malformed JSON should error",
			jsonParams: `{"label": "test"`,
			expectErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plugin, err := Factory("accel-aware", fwkplugin.StrictDecoder(json.RawMessage(tt.jsonParams)), nil)
			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, plugin)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, plugin)
			}
		})
	}
}

func TestFilter(t *testing.T) {
	ctx := utils.NewTestContext(t)
	endpoints := []scheduling.Endpoint{
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "small-mig"},
			map[string]string{DefaultLabel: "0-1000"}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "medium-mig"},
			map[string]string{DefaultLabel: "1000-4000"}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "full-gpu"},
			map[string]string{DefaultLabel: "4000-32000"}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "unlabeled"},
			map[string]string{}),
	}

	plugin := New("test-filter", &parameters{Label: DefaultLabel, EnableFiltering: true})
	filteredEndpoints := plugin.Filter(ctx, createRequest(2500), endpoints)

	gotNames := make([]string, len(filteredEndpoints))
	for i, endpoint := range filteredEndpoints {
		gotNames[i] = endpoint.GetMetadata().NamespacedName.Name
	}
	assert.ElementsMatch(t, []string{"medium-mig", "unlabeled"}, gotNames)
}

func TestScore(t *testing.T) {
	ctx := utils.NewTestContext(t)
	endpoints := []scheduling.Endpoint{
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "tight-match"},
			map[string]string{DefaultLabel: "2000-3000"}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "wide-match"},
			map[string]string{DefaultLabel: "0-32000"}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "too-small"},
			map[string]string{DefaultLabel: "0-1000"}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "unlabeled"},
			map[string]string{}),
	}

	plugin := New("test-scorer", &parameters{Label: DefaultLabel})
	scores := plugin.Score(ctx, createRequest(2500), endpoints)

	assert.Greater(t, scores[endpoints[0]], scores[endpoints[1]], "tighter matching range should score higher")
	assert.Greater(t, scores[endpoints[0]], 0.3, "in-range score must be above fallback tier")
	assert.Greater(t, scores[endpoints[1]], 0.3, "wide in-range score must be above fallback tier")
	assert.Greater(t, scores[endpoints[2]], 0.0, "out-of-range endpoints should get a proximity fallback score")
	assert.Less(t, scores[endpoints[2]], 0.3, "out-of-range fallback must stay below in-range tier")
	assert.Equal(t, 0.5, scores[endpoints[3]], "unlabeled endpoints should score neutral")
}

func TestParseCapabilityRange(t *testing.T) {
	tests := []struct {
		name      string
		rangeStr  string
		expected  capabilityRange
		expectErr bool
	}{
		{
			name:     "valid range",
			rangeStr: "0-100",
			expected: capabilityRange{min: 0, max: 100},
		},
		{
			name:      "empty string",
			rangeStr:  "",
			expectErr: true,
		},
		{
			name:      "invalid format with three parts",
			rangeStr:  "0-100-200",
			expectErr: true,
		},
		{
			name:      "min greater than max",
			rangeStr:  "100-50",
			expectErr: true,
		},
		{
			name:      "non-numeric value",
			rangeStr:  "abc-100",
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := parseCapabilityRange(tt.rangeStr)
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expected, r)
			}
		})
	}
}

func TestNilTokenizedPromptIsUnknownSize(t *testing.T) {
	ctx := utils.NewTestContext(t)
	endpoints := []scheduling.Endpoint{
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "zero-range"},
			map[string]string{DefaultLabel: "0-100"}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "larger-range"},
			map[string]string{DefaultLabel: "100-1000"}),
	}

	plugin := New("test-no-tokens", &parameters{Label: DefaultLabel, EnableFiltering: true})
	request := &scheduling.InferenceRequest{RequestID: "test-request", Body: &fwkrh.InferenceRequestBody{}}

	filteredEndpoints := plugin.Filter(ctx, request, endpoints)
	require.Len(t, filteredEndpoints, 1)
	assert.Equal(t, "zero-range", filteredEndpoints[0].GetMetadata().NamespacedName.Name)
}
