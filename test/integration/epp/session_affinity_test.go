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

package epp

import (
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	integration "github.com/llm-d/llm-d-router/test/integration"
)

// TestSessionAffinityFilter_RequestFlow validates the end-to-end session
// affinity flow against an EPP initialized from text config: the first request
// (no session header) is routed and gets a session token on the response, and
// the second request (carrying that token) is pinned to the same endpoint.
func TestSessionAffinityFilter_RequestFlow(t *testing.T) {
	configText := `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: queue-scorer
  - type: session-affinity-filter
  - type: mock-metrics-source
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
      - pluginRef: session-affinity-filter
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
`

	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()

	// Two ready pods so the filter has a real choice to pin to.
	pods := []PodState{
		P(0, 0, 0.1, modelMyModelTarget),
		P(1, 0, 0.1, modelMyModelTarget),
	}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	// First request: no session header. Read the routed endpoint and the
	// session token the plugin writes on the response.
	firstEndpoint, token := sendSessionRequest(t, h, "")
	require.NotEmpty(t, token, "first response must carry a session token")

	// Second request: echo the token back. It must pin to the same endpoint.
	secondEndpoint, _ := sendSessionRequest(t, h, token)
	require.Equal(t, firstEndpoint, secondEndpoint,
		"second request carrying the session token must be pinned to the first request's endpoint")
}

// sendSessionRequest drives one full request/response transaction through the
// EPP and returns the routed destination endpoint and the session token set on
// the response headers. When sessionToken is non-empty it is sent as the
// session header on the request.
func sendSessionRequest(t *testing.T, h *TestHarness, sessionToken string) (endpoint, token string) {
	t.Helper()

	reqHeaders := map[string]string{
		metadata.ObjectiveKey:        modelMyModel,
		metadata.ModelNameRewriteKey: modelMyModelTarget,
		reqcommon.RequestIDHeaderKey: "session-req",
	}
	if sessionToken != "" {
		reqHeaders[sessionutil.DefaultHeader] = sessionToken
	}

	requests := integration.ReqRaw(reqHeaders, `{"model":"`+modelMyModel+`","prompt":"hello","max_tokens":10,"temperature":0}`)
	requests = append(requests, ReqResponseOnly(
		map[string]string{"content-type": "application/json", "status": "200"},
		`{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hi","role":"assistant"}}],"model":"`+modelMyModelTarget+`","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}`,
	)...)

	// Each request is a distinct HTTP transaction, so open a fresh ext_proc stream.
	client, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(t.Context())
	require.NoError(t, err)

	// RequestHeaders + RequestBody + ResponseHeaders + ResponseBody -> 4 responses.
	responses, err := integration.StreamedRequest(t, client, requests, 4)
	require.NoError(t, err)
	require.Len(t, responses, 4)

	endpoint = headerValue(responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey)
	require.NotEmpty(t, endpoint, "request headers response must set the destination endpoint")

	token = headerValue(responses[2].GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), sessionutil.DefaultHeader)
	return endpoint, token
}

// headerValue returns the value set for key among the given header mutations, or
// "" when absent.
func headerValue(setHeaders []*configPb.HeaderValueOption, key string) string {
	for _, h := range setHeaders {
		if h.GetHeader().GetKey() == key {
			if raw := h.GetHeader().GetRawValue(); len(raw) > 0 {
				return string(raw)
			}
			return h.GetHeader().GetValue()
		}
	}
	return ""
}
