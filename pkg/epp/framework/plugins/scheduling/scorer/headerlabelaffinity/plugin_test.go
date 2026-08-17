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

package headerlabelaffinity

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestFactory(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{
			name:       "valid",
			parameters: `{"headerName":"X-Disagg-Slice","labelKey":"disaggregatedset.x-k8s.io/slice"}`,
		},
		{
			name:       "missing parameters",
			parameters: "",
			wantErr:    "requires parameters",
		},
		{
			name:       "missing header name",
			parameters: `{"labelKey":"disaggregatedset.x-k8s.io/slice"}`,
			wantErr:    "headerName is required",
		},
		{
			name:       "missing label key",
			parameters: `{"headerName":"x-disagg-slice"}`,
			wantErr:    "labelKey is required",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var decoder *json.Decoder
			if test.parameters != "" {
				decoder = json.NewDecoder(strings.NewReader(test.parameters))
			}
			plugin, err := Factory("slice-affinity", decoder, nil)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			scorer := plugin.(*Scorer)
			assert.Equal(t, PluginType, scorer.TypedName().Type)
			assert.Equal(t, "slice-affinity", scorer.TypedName().Name)
			assert.Equal(t, "x-disagg-slice", scorer.headerName)
		})
	}
}

func TestScore(t *testing.T) {
	matching := endpoint(map[string]string{"slice": "slice-a"})
	nonMatching := endpoint(map[string]string{"slice": "slice-b"})
	missingLabel := endpoint(nil)
	nilMetadata := fwksched.NewEndpoint(nil, nil, nil)
	endpoints := []fwksched.Endpoint{matching, nonMatching, missingLabel, nilMetadata}
	scorer := &Scorer{headerName: "x-slice", labelKey: "slice"}

	tests := []struct {
		name    string
		request *fwksched.InferenceRequest
		want    []float64
	}{
		{
			name:    "matching label receives affinity",
			request: &fwksched.InferenceRequest{Headers: map[string]string{"x-slice": "slice-a"}},
			want:    []float64{1, 0, 0, 0},
		},
		{
			name:    "unknown value is a soft miss",
			request: &fwksched.InferenceRequest{Headers: map[string]string{"x-slice": "unknown"}},
			want:    []float64{0, 0, 0, 0},
		},
		{
			name:    "missing header contributes no score",
			request: &fwksched.InferenceRequest{Headers: map[string]string{}},
			want:    []float64{0, 0, 0, 0},
		},
		{
			name: "nil request contributes no score",
			want: []float64{0, 0, 0, 0},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scores := scorer.Score(context.Background(), test.request, endpoints)
			for index, endpoint := range endpoints {
				assert.Equal(t, test.want[index], scores[endpoint])
			}
		})
	}
}

func endpoint(labels map[string]string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Labels: labels}, nil, nil)
}
