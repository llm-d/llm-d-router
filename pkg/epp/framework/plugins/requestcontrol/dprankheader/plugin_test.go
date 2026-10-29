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

package dprankheader

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func newHandler(t *testing.T, params string) *Handler {
	t.Helper()
	var dec *json.Decoder
	if params != "" {
		dec = json.NewDecoder(strings.NewReader(params))
	}
	p, err := Factory("test", dec, nil)
	require.NoError(t, err)
	return p.(*Handler)
}

func makeEndpoint(rank, ranksPerPort int) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		ID:           types.NamespacedName{Name: "pod1-rank-1", Namespace: "default"},
		Address:      "10.0.0.1",
		Port:         "8000",
		RankIndex:    rank,
		RanksPerPort: ranksPerPort,
	}
	return fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
}

func schedulingResult(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func TestPreRequest_SetsSelectedRank(t *testing.T) {
	h := newHandler(t, "")
	request := &fwksched.InferenceRequest{}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(makeEndpoint(1, 2))))
	assert.Equal(t, "1", request.Headers[routing.DataParallelRankHeader])
}

func TestPreRequest_OverwritesClientValue(t *testing.T) {
	h := newHandler(t, "")
	request := &fwksched.InferenceRequest{Headers: map[string]string{routing.DataParallelRankHeader: "7"}}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(makeEndpoint(0, 2))))
	assert.Equal(t, "0", request.Headers[routing.DataParallelRankHeader],
		"the selected rank wins over a client-supplied value")
}

func TestPreRequest_ConfiguredHeaderName(t *testing.T) {
	h := newHandler(t, `{"headerName": "X-Engine-Rank"}`)
	request := &fwksched.InferenceRequest{}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(makeEndpoint(3, 4))))
	assert.Equal(t, "3", request.Headers["x-engine-rank"])
	assert.NotContains(t, request.Headers, routing.DataParallelRankHeader)
}

func TestPreRequest_EndpointWithOneRankPerPortUnchanged(t *testing.T) {
	h := newHandler(t, "")
	request := &fwksched.InferenceRequest{Headers: map[string]string{routing.DataParallelRankHeader: "5"}}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(makeEndpoint(1, 0))))
	assert.Equal(t, map[string]string{routing.DataParallelRankHeader: "5"}, request.Headers,
		"an endpoint that is the only rank on its port leaves the headers untouched")
}

func TestPreRequest_NoSelectedEndpoint(t *testing.T) {
	h := newHandler(t, "")
	request := &fwksched.InferenceRequest{}
	result := &fwksched.SchedulingResult{PrimaryProfileName: "missing"}

	require.NoError(t, h.PreRequest(context.Background(), request, result))
	assert.Empty(t, request.Headers)
}

func TestFactory_RejectsEPPOwnedHeaderNames(t *testing.T) {
	for _, headerName := range []string{"x-llm-d-inference-objective", routing.PrefillEndpointHeader} {
		t.Run(headerName, func(t *testing.T) {
			params := `{"headerName": "` + headerName + `"}`
			_, err := Factory("test", json.NewDecoder(strings.NewReader(params)), nil)
			require.Error(t, err)
		})
	}
}
