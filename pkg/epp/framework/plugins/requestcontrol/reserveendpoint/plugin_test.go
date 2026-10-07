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

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func endpoint(ip string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Address: ip, Port: "8000"}, nil, nil)
}

func result(primary string, endpoints ...fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: primary,
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			primary:  {TargetEndpoints: endpoints},
			"decode": {TargetEndpoints: []fwksched.Endpoint{endpoint("10.0.4.1")}},
		},
	}
}

func TestPreRequest(t *testing.T) {
	reserve := map[string]string{routing.PreferHeader: routing.PreferReserveEndpoint}
	picked := result("prefill", endpoint("10.0.3.7"), endpoint("10.0.3.8"))

	tests := []struct {
		name    string
		headers map[string]string
		result  *fwksched.SchedulingResult
		// wantEndpoint is the answered <ip:port>; empty means no answer.
		wantEndpoint string
	}{
		{name: "no preference is a no-op", headers: map[string]string{}, result: picked},
		{name: "other preference is a no-op", headers: map[string]string{routing.PreferHeader: routing.PreferIfAvailable}, result: picked},
		{name: "answers with the first primary endpoint", headers: reserve, result: picked, wantEndpoint: "10.0.3.7:8000"},
		{name: "answers among other tokens", headers: map[string]string{routing.PreferHeader: "respond-async, Reserve-Endpoint;x=1"}, result: picked, wantEndpoint: "10.0.3.7:8000"},
		{name: "IPv6 address is bracketed", headers: reserve, result: result("prefill", endpoint("fd00::7")), wantEndpoint: "[fd00::7]:8000"},
		// A per-phase prefill EPP runs its only profile as "default".
		{name: "answer does not depend on the profile name", headers: reserve, result: result("default", endpoint("10.0.3.7")), wantEndpoint: "10.0.3.7:8000"},
		{name: "no primary endpoint is left to the director", headers: reserve, result: result("prefill")},
		{name: "nil result is left to the director", headers: reserve},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := New().PreRequest(context.Background(), &fwksched.InferenceRequest{Headers: tt.headers}, tt.result)

			if tt.wantEndpoint == "" {
				assert.NoError(t, err)
				return
			}
			var answer errcommon.Error
			require.ErrorAs(t, err, &answer)
			assert.Equal(t, errcommon.NoContent, answer.Code)
			assert.Equal(t, map[string]string{
				routing.ReservedEndpointHeader:  tt.wantEndpoint,
				routing.PreferenceAppliedHeader: routing.PreferReserveEndpoint,
			}, answer.Headers)
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
