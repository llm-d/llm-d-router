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

package reserveendpoint

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func endpoint(ip string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: ip, Port: "8000"}, nil, nil)
}

func TestPreRequest(t *testing.T) {
	reserve := map[string]string{routing.PreferHeader: routing.PreferReserveEndpoint}
	picked := &fwksched.SchedulingResult{
		PrimaryProfileName: "prefill",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{endpoint("10.0.3.7"), endpoint("10.0.3.8")}},
			"decode":  {TargetEndpoints: []fwksched.Endpoint{endpoint("10.0.4.1")}},
		},
	}
	ipv6 := &fwksched.SchedulingResult{
		PrimaryProfileName: "prefill",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{endpoint("fd00::7")}},
		},
	}
	noPrimary := &fwksched.SchedulingResult{
		PrimaryProfileName: "prefill",
		ProfileResults:     map[string]*fwksched.ProfileRunResult{"prefill": {}},
	}

	tests := []struct {
		name      string
		headers   map[string]string
		result    *fwksched.SchedulingResult
		wantClaim string
	}{
		{name: "no preference is a no-op", headers: map[string]string{}, result: picked},
		{name: "other preference is a no-op", headers: map[string]string{routing.PreferHeader: routing.PreferIfAvailable}, result: picked},
		{name: "claims the first primary endpoint", headers: reserve, result: picked, wantClaim: "10.0.3.7:8000"},
		{name: "IPv6 address is bracketed", headers: reserve, result: ipv6, wantClaim: "[fd00::7]:8000"},
		{name: "no primary endpoint is left unclaimed", headers: reserve, result: noPrimary},
		{name: "nil result is left unclaimed", headers: reserve},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &fwksched.InferenceRequest{Headers: tt.headers}
			require.NoError(t, New().PreRequest(context.Background(), req, tt.result))

			got, ok := fwksched.ReadRequestAttribute[string](req, fwkrc.ReservedEndpointAttributeKey)
			if tt.wantClaim == "" {
				assert.False(t, ok, "request must stay unclaimed")
				return
			}
			assert.True(t, ok)
			assert.Equal(t, tt.wantClaim, got)
		})
	}
}

func TestPreRequestNilRequest(t *testing.T) {
	assert.NoError(t, New().PreRequest(context.Background(), nil, nil))
}

func TestFactory(t *testing.T) {
	p, err := Factory("my-reserve", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, p.TypedName().Type)
	assert.Equal(t, "my-reserve", p.TypedName().Name)
}
