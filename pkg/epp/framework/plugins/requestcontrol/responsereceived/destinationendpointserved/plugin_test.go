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

package destinationendpointserved

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
)

func TestPlugin_ResponseHeader(t *testing.T) {
	testCases := []struct {
		name           string
		targetEndpoint *fwkdl.EndpointMetadata
		responseIn     *requestcontrol.Response
		wantHeader     string
	}{
		{
			name: "writes endpoint Name from populated targetEndpoint",
			targetEndpoint: &fwkdl.EndpointMetadata{
				ID:      types.NamespacedName{Namespace: "pricing-reversal-cw-a", Name: "spoke-gemma-a"},
				Name:    "spoke-gemma-a",
				Address: "10.16.1.169",
				Port:    "80",
			},
			responseIn: &requestcontrol.Response{Headers: map[string]string{}},
			wantHeader: "spoke-gemma-a",
		},
		{
			name:           "writes failure string when targetEndpoint is nil",
			targetEndpoint: nil,
			responseIn:     &requestcontrol.Response{Headers: map[string]string{}},
			wantHeader:     FailureNoEndpoint,
		},
		{
			name: "initialises Headers map when nil",
			targetEndpoint: &fwkdl.EndpointMetadata{
				Name: "spoke-gemma-b",
			},
			responseIn: &requestcontrol.Response{Headers: nil},
			wantHeader: "spoke-gemma-b",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			p := New().WithName("test-plugin")

			p.ResponseHeader(context.Background(), nil, tc.responseIn, tc.targetEndpoint)

			got, ok := tc.responseIn.Headers[ResultHeader]
			require.True(t, ok, "expected header %s to be set", ResultHeader)
			require.Equal(t, tc.wantHeader, got)
		})
	}
}

func TestPlugin_TypedName(t *testing.T) {
	p := New().WithName("costguard-attribution")
	tn := p.TypedName()
	require.Equal(t, PluginType, tn.Type)
	require.Equal(t, "costguard-attribution", tn.Name)
}
