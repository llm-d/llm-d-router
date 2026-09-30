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

package prefillpin

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func endpoint(ip, port string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: ip, Port: port}, nil, nil)
}

func TestScreen(t *testing.T) {
	p1 := endpoint("10.0.3.7", "8000")
	p1Rank1 := endpoint("10.0.3.7", "8001")
	p2 := endpoint("10.0.3.8", "8000")
	v6 := endpoint("fd00::7", "8000")
	all := []fwksched.Endpoint{p1, p1Rank1, p2, v6}

	tests := []struct {
		name      string
		request   *fwksched.InferenceRequest
		endpoints []fwksched.Endpoint
		want      []fwksched.Endpoint
	}{
		{name: "nil request keeps every endpoint", endpoints: all, want: all},
		{name: "no pin keeps every endpoint", request: &fwksched.InferenceRequest{Headers: map[string]string{}}, endpoints: all, want: all},
		{name: "empty pin keeps every endpoint", request: pinned(""), endpoints: all, want: all},
		{name: "pin keeps only the matching endpoint", request: pinned("10.0.3.7:8000"), endpoints: all, want: []fwksched.Endpoint{p1}},
		{name: "pin matches the port as well as the address", request: pinned("10.0.3.7:8001"), endpoints: all, want: []fwksched.Endpoint{p1Rank1}},
		{name: "pin matches a bracketed IPv6 address", request: pinned("[fd00::7]:8000"), endpoints: all, want: []fwksched.Endpoint{v6}},
		{name: "pin to a missing endpoint keeps nothing", request: pinned("10.0.9.9:8000"), endpoints: all, want: nil},
		{name: "pin with no candidates keeps nothing", request: pinned("10.0.3.7:8000"), endpoints: nil, want: nil},
		{name: "address without port does not match", request: pinned("10.0.3.7"), endpoints: all, want: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := New().Screen(context.Background(), tt.request, tt.endpoints)
			assert.Equal(t, tt.want, got)
		})
	}
}

func pinned(pin string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{Headers: map[string]string{routing.PrefillPinHeader: pin}}
}

func TestFactory(t *testing.T) {
	p, err := Factory("my-pin", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, p.TypedName().Type)
	assert.Equal(t, "my-pin", p.TypedName().Name)
}
