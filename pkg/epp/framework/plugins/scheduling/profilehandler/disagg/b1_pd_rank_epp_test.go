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

package disagg

import (
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

// B1-U04 (EPP side): the P/D handler publishes the selected prefill endpoint as
// <ip:port> only. The endpoint's rank identity, which a shared-frontend SGLang
// deployment needs in order to translate the selection into the engine's
// routed_dp_rank, does not survive into the request.
//
// This pins the shape of the gap rather than the fix: EndpointMetadata already
// carries RankIndex, so the missing piece is the transport of that index to the
// ingress, not a new schema.
func TestHandler_PreRequest_PrefillHeaderCarriesOnlyHostPort(t *testing.T) {
	ctx := utils.NewTestContext(t)
	h := NewDisaggProfileHandler("decode", "prefill", "", nil, nil)

	selected := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		Address:   "10.0.0.1",
		Port:      "8000",
		RankIndex: 2,
	}, nil, nil)

	request := &scheduling.InferenceRequest{Headers: map[string]string{}}
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"prefill": {TargetEndpoints: []scheduling.Endpoint{selected}},
		},
	}

	require.NoError(t, h.PreRequest(ctx, request, result))

	hostPort := request.Headers[routing.PrefillEndpointHeader]
	require.NotEmpty(t, hostPort)

	// The selected rank is not recoverable from what the request carries.
	assert.Equal(t, net.JoinHostPort("10.0.0.1", "8000"), hostPort)
	assert.Equal(t, 2, selected.GetMetadata().GetRankIndex(),
		"the selected endpoint does carry its rank in the datalayer")
	assert.NotContains(t, hostPort, "2",
		"the prefill selection is indistinguishable from rank 0 of the same serving endpoint")
}

// Two prefill candidates that differ only by rank produce the same header: at
// this hop the router cannot tell which internal rank the scheduler chose.
func TestHandler_PreRequest_RankCandidatesCollapseToTheSameHeader(t *testing.T) {
	ctx := utils.NewTestContext(t)
	h := NewDisaggProfileHandler("decode", "prefill", "", nil, nil)

	headersFor := func(rank int) string {
		request := &scheduling.InferenceRequest{Headers: map[string]string{}}
		endpoint := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
			Address:   "10.0.0.1",
			Port:      "8000",
			RankIndex: rank,
		}, nil, nil)
		require.NoError(t, h.PreRequest(ctx, request, &scheduling.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults: map[string]*scheduling.ProfileRunResult{
				"prefill": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
			},
		}))
		return request.Headers[routing.PrefillEndpointHeader]
	}

	assert.Equal(t, headersFor(0), headersFor(1),
		"shared-frontend ranks are indistinguishable at the P/D handoff")
}
