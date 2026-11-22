/*
Copyright 2025 The llm-d Authors.

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

package metadata

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHeaderAliases(t *testing.T) {
	t.Parallel()

	assert.Equal(t, []string{ObjectiveKey, OldObjectiveKey}, HeaderNames(ObjectiveKey))
	assert.Equal(t, []string{ObjectiveKey, OldObjectiveKey}, HeaderNames("X-LLM-D-Inference-Objective"))
	assert.Equal(t, []string{TTFTSLOHeaderKey, OldTTFTSLOHeaderKey}, HeaderNames(TTFTSLOHeaderKey))
	assert.Equal(t, []string{DestinationEndpointKey}, HeaderNames(DestinationEndpointKey))
	assert.Equal(t, []string{"x-user-header"}, HeaderNames("X-User-Header"))
}

func TestGetLowerCaseHeaderValuePrefersCurrentName(t *testing.T) {
	t.Parallel()

	headers := map[string]string{
		OldObjectiveKey: "old-objective",
		ObjectiveKey:    "new-objective",
	}
	got, ok := GetLowerCaseHeaderValue(headers, ObjectiveKey)
	assert.True(t, ok)
	assert.Equal(t, "new-objective", got)

	headers = map[string]string{
		OldObjectiveKey: "old-objective",
	}
	got, ok = GetLowerCaseHeaderValue(headers, ObjectiveKey)
	assert.True(t, ok)
	assert.Equal(t, "old-objective", got)

	headers = map[string]string{
		"x-user-header": "user-value",
	}
	got, ok = GetLowerCaseHeaderValue(headers, "X-User-Header")
	assert.True(t, ok)
	assert.Equal(t, "user-value", got)
}

func TestIsFiniteFloat(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		value float64
		want  bool
	}{
		{"zero", 0, true},
		{"positive", 12.5, true},
		{"negative", -1, true},
		{"maximum", math.MaxFloat64, true},
		{"NaN", math.NaN(), false},
		{"positive infinity", math.Inf(1), false},
		{"negative infinity", math.Inf(-1), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tc.want, IsFiniteFloat(tc.value))
		})
	}
}

func TestGetLowerCaseHeaderValueAliases(t *testing.T) {
	t.Parallel()

	for key, names := range headerAliases {
		if len(names) < 2 {
			continue
		}
		current, deprecated := names[0], names[1]
		tests := []struct {
			name    string
			headers map[string]string
			want    string
			wantOK  bool
		}{
			{name: "deprecated name only", headers: map[string]string{deprecated: "old"}, want: "old", wantOK: true},
			{name: "both names", headers: map[string]string{current: "new", deprecated: "old"}, want: "new", wantOK: true},
			{name: "current name empty", headers: map[string]string{current: "", deprecated: "old"}, want: "old", wantOK: true},
			{name: "current name empty and deprecated absent", headers: map[string]string{current: ""}, want: "", wantOK: true},
			{name: "both names empty", headers: map[string]string{current: "", deprecated: ""}, want: "", wantOK: true},
			{name: "neither name", headers: map[string]string{}, want: "", wantOK: false},
		}
		for _, test := range tests {
			t.Run(key+"/"+test.name, func(t *testing.T) {
				got, ok := GetLowerCaseHeaderValue(test.headers, key)
				assert.Equal(t, test.wantOK, ok)
				assert.Equal(t, test.want, got)
			})
		}
	}
}

func TestServedEndpoint(t *testing.T) {
	tests := []struct {
		name        string
		reqMetadata map[string]any
		want        string
		wantOK      bool
	}{
		{
			name:        "served endpoint present",
			reqMetadata: map[string]any{DestinationEndpointNamespace: map[string]any{DestinationEndpointServedKey: "10.0.0.7:8000"}},
			want:        "10.0.0.7:8000",
			wantOK:      true,
		},
		{
			name:        "namespace without the served key",
			reqMetadata: map[string]any{DestinationEndpointNamespace: map[string]any{DestinationEndpointKey: "10.0.0.7:8000"}},
		},
		{
			name:        "served key with a non-string value",
			reqMetadata: map[string]any{DestinationEndpointNamespace: map[string]any{DestinationEndpointServedKey: 1}},
		},
		{
			name:        "no namespace",
			reqMetadata: map[string]any{},
		},
		{
			name: "nil metadata",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ServedEndpoint(tt.reqMetadata)
			if got != tt.want || ok != tt.wantOK {
				t.Errorf("ServedEndpoint() = (%q, %v), want (%q, %v)", got, ok, tt.want, tt.wantOK)
			}
		})
	}
}
