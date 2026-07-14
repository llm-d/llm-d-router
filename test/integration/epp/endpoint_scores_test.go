/*
Copyright 2025 The Kubernetes Authors.

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

package epp

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	integration "github.com/llm-d/llm-d-router/test/integration"
)

// TestEndpointScoresMetadata verifies the opt-in request-path dynamic metadata contract: with
// --emit-endpoint-scores, the envoy.lb namespace carries an x-gateway-destination-endpoint-scores
// struct mapping every endpoint listed in x-gateway-destination-endpoint to its scheduler score,
// while the endpoint key itself is unchanged.
func TestEndpointScoresMetadata(t *testing.T) {
	h := NewTestHarness(t.Context(), t, WithStandardMode(), WithEmitEndpointScores()).WithBaseResources()

	pods := []PodState{P(0, 0, 0.1, modelMyModelTarget), P(1, 5, 0.5, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)

	envoyLb := requestHeaderEnvoyLbMetadata(t, h)

	endpointValue, ok := envoyLb.Fields[metadata.DestinationEndpointKey]
	require.True(t, ok, "expected destination endpoint in envoy.lb namespace")
	endpoints := strings.Split(endpointValue.GetStringValue(), ",")
	require.NotEmpty(t, endpoints, "expected at least one destination endpoint")

	scoresValue, ok := envoyLb.Fields[metadata.DestinationEndpointScoresKey]
	require.True(t, ok, "expected destination endpoint scores in envoy.lb namespace")
	scoreFields := scoresValue.GetStructValue().Fields
	require.Len(t, scoreFields, len(endpoints), "expected exactly one score per destination endpoint")
	for _, endpoint := range endpoints {
		score, ok := scoreFields[endpoint]
		require.True(t, ok, "expected a score for destination endpoint %s", endpoint)
		require.GreaterOrEqual(t, score.GetNumberValue(), 0.0, "expected a non-negative score for endpoint %s", endpoint)
	}
}

// TestEndpointScoresMetadataOffByDefault verifies that without --emit-endpoint-scores the
// request-path dynamic metadata carries only the destination endpoint key.
func TestEndpointScoresMetadataOffByDefault(t *testing.T) {
	h := NewTestHarness(t.Context(), t, WithStandardMode()).WithBaseResources()

	pods := []PodState{P(0, 0, 0.1, modelMyModelTarget), P(1, 5, 0.5, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)

	envoyLb := requestHeaderEnvoyLbMetadata(t, h)

	_, ok := envoyLb.Fields[metadata.DestinationEndpointKey]
	require.True(t, ok, "expected destination endpoint in envoy.lb namespace")
	_, ok = envoyLb.Fields[metadata.DestinationEndpointScoresKey]
	require.False(t, ok, "expected destination endpoint scores to be absent from envoy.lb namespace by default")
}

// requestHeaderEnvoyLbMetadata drives one LLM request through the EPP and returns the envoy.lb
// dynamic metadata struct attached to the request-headers ext_proc response.
func requestHeaderEnvoyLbMetadata(t *testing.T, h *TestHarness) *structpb.Struct {
	t.Helper()

	requests := integration.ReqLLM(logger, "hello", modelMyModel, modelMyModelTarget)

	// RequestHeaders + RequestBody -> request header response and request body response.
	responses, err := integration.StreamedRequest(t, h.Client, requests, 2)
	require.NoError(t, err)
	require.Len(t, responses, 2)

	res := responses[0]
	require.NotNil(t, res.DynamicMetadata, "expected DynamicMetadata in the request headers ext_proc response")
	envoyLb, ok := res.DynamicMetadata.Fields[metadata.DestinationEndpointNamespace]
	require.True(t, ok, "expected envoy.lb namespace in DynamicMetadata")
	return envoyLb.GetStructValue()
}
