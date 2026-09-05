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
	"fmt"
	"sync/atomic"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
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

// TestSessionAffinityScorer_RequestFlow is the scorer analogue of
// TestSessionAffinityFilter_RequestFlow: the encoded scorer feeds max-score-picker
// rather than hard-narrowing candidates, but the affinity contract is the same.
// The first request is routed and gets a session token; the second, carrying
// that token, pins to the same endpoint.
func TestSessionAffinityScorer_RequestFlow(t *testing.T) {
	configText := `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: session-affinity-scorer
  - type: max-score-picker
  - type: mock-metrics-source
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: session-affinity-scorer
      - pluginRef: max-score-picker
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
`

	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()

	pods := []PodState{
		P(0, 0, 0.1, modelMyModelTarget),
		P(1, 0, 0.1, modelMyModelTarget),
	}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	firstEndpoint, token := sendSessionRequest(t, h, "")
	require.NotEmpty(t, token, "first response must carry a session token")

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

// pdConfigBase is the shared EPP config skeleton for PD session-affinity tests.
// The caller must supply the disagg-profile-handler parameters block (%s).
const pdConfigBase = `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: always-disagg-pd-decider
  - type: disagg-profile-handler
    parameters:
%s
  - type: prefill-filter
  - type: decode-filter
  - type: queue-scorer
  - type: max-score-picker
  - type: session-affinity-filter
    name: session-affinity-decode
  - type: session-affinity-filter
    name: session-affinity-prefill
    parameters:
      profileName: prefill
      encodedEndpointHeaderConfig:
        header: x-session-token-prefill
  - type: mock-metrics-source
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
schedulingProfiles:
  - name: prefill
    plugins:
    - pluginRef: prefill-filter
    - pluginRef: queue-scorer
    - pluginRef: max-score-picker
    - pluginRef: session-affinity-prefill
  - name: decode
    plugins:
    - pluginRef: decode-filter
    - pluginRef: queue-scorer
    - pluginRef: max-score-picker
    - pluginRef: session-affinity-decode
`

// setupPDHarness creates a test harness with prefill and decode pods for PD
// session-affinity tests.
func setupPDHarness(t *testing.T, configText string) *TestHarness {
	t.Helper()
	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()

	metricsMap := make(map[types.NamespacedName]*fwkdl.Metrics)
	type podDef struct {
		index int
		role  string
	}
	pods := []podDef{
		{0, bylabel.RolePrefill},
		{1, bylabel.RolePrefill},
		{2, bylabel.RoleDecode},
		{3, bylabel.RoleDecode},
	}
	for _, p := range pods {
		name := fmt.Sprintf("pod-%d", p.index)
		metricsMap[types.NamespacedName{Namespace: h.Namespace, Name: name + "-rank-0"}] = &fwkdl.Metrics{
			ActiveModels:  map[string]int{modelMyModelTarget: 1},
			WaitingModels: make(map[string]int),
		}
		pod := testutil.MakePod(name).
			Namespace(h.Namespace).
			ReadyCondition().
			Labels(map[string]string{
				"app":             testPoolName,
				bylabel.RoleLabel: p.role,
			}).
			IP(fmt.Sprintf("192.168.1.%d", p.index+1)).
			Complete().
			ObjRef()

		intendedStatus := pod.Status
		require.NoError(t, k8sClient.Create(ctx, pod))
		pod.Status = intendedStatus
		require.NoError(t, k8sClient.Status().Update(ctx, pod))
	}
	h.metricsBackend.SetPodMetrics(metricsMap)
	h.WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))
	return h
}

// TestSessionAffinityFilter_PDDisaggregated validates the session affinity flow
// in a prefill/decode disaggregated setup where every request is disaggregated.
// Two session-affinity-filter instances emit separate tokens for the decode and
// prefill endpoints. The second request pins both profiles to the same pods.
func TestSessionAffinityFilter_PDDisaggregated(t *testing.T) {
	const prefillHeader = "x-session-token-prefill"

	configText := fmt.Sprintf(pdConfigBase, "      deciders:\n        prefill: always-disagg-pd-decider")
	h := setupPDHarness(t, configText)

	// First request: no session headers.
	firstDecodeEP, decodeToken := sendSessionRequest(t, h, "")
	require.NotEmpty(t, decodeToken, "first response must carry a decode session token")

	firstPrefillToken := sendSessionRequestGetHeader(t, h, "", "", prefillHeader)
	require.NotEmpty(t, firstPrefillToken, "first response must carry a prefill session token")

	// Second request: echo both tokens back.
	secondDecodeEP, _ := sendSessionRequest(t, h, decodeToken)
	require.Equal(t, firstDecodeEP, secondDecodeEP,
		"second request must pin to the same decode endpoint")

	secondPrefillToken := sendSessionRequestGetHeader(t, h, decodeToken, firstPrefillToken, prefillHeader)
	require.NotEmpty(t, secondPrefillToken, "second response must carry a prefill session token")
}

// TestSessionAffinityFilter_DecodeOnly validates that when the disagg profile
// handler has no prefill decider configured, prefill is skipped and the
// profile-scoped session-affinity-filter does NOT emit a prefill session token
// (i.e. it does not fall back to the decode pod).
func TestSessionAffinityFilter_DecodeOnly(t *testing.T) {
	const prefillHeader = "x-session-token-prefill"

	// No deciders block: disagg-profile-handler skips prefill entirely.
	configText := fmt.Sprintf(pdConfigBase, "      {}")
	h := setupPDHarness(t, configText)

	// Request routed to decode only; decode token must be present, prefill
	// token must be absent.
	_, decodeToken := sendSessionRequest(t, h, "")
	require.NotEmpty(t, decodeToken, "response must carry a decode session token")

	prefillToken := sendSessionRequestGetHeader(t, h, "", "", prefillHeader)
	require.Empty(t, prefillToken,
		"decode-only response must NOT carry a prefill session token")
}

// sendSessionRequestGetHeader drives a request through the EPP and returns the
// value of the named response header. It extends sendSessionRequest to support
// the prefill session token header.
func sendSessionRequestGetHeader(t *testing.T, h *TestHarness, decodeToken, prefillToken, headerKey string) string {
	t.Helper()

	reqHeaders := map[string]string{
		metadata.ObjectiveKey:        modelMyModel,
		metadata.ModelNameRewriteKey: modelMyModelTarget,
		reqcommon.RequestIDHeaderKey: "session-pd-req",
	}
	if decodeToken != "" {
		reqHeaders[sessionutil.DefaultHeader] = decodeToken
	}
	if prefillToken != "" {
		reqHeaders["x-session-token-prefill"] = prefillToken
	}

	requests := integration.ReqRaw(reqHeaders, `{"model":"`+modelMyModel+`","prompt":"hello","max_tokens":10,"temperature":0}`)
	requests = append(requests, ReqResponseOnly(
		map[string]string{"content-type": "application/json", "status": "200"},
		`{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hi","role":"assistant"}}],"model":"`+modelMyModelTarget+`","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}`,
	)...)

	client, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(t.Context())
	require.NoError(t, err)

	responses, err := integration.StreamedRequest(t, client, requests, 4)
	require.NoError(t, err)
	require.Len(t, responses, 4)

	return headerValue(responses[2].GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), headerKey)
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

// TestSessionAffinityFilter_SessionIDHeaderStrategy exercises the session_id
// strategy end to end with an ordered source list [header x-session-id, attribute
// agent-identity]. It proves: (1) the header source binds and pins a session; (2)
// a request carrying only an agent header resolves via the agent-identity plugin's
// published attribute and pins (the cross-plugin handoff); (3) the header wins
// over the attribute when both are present.
func TestSessionAffinityFilter_SessionIDHeaderStrategy(t *testing.T) {
	configText := `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: agent-identity
  - type: session-affinity-filter
    parameters:
      strategy: session_id
      sessionIdConfig:
        sources:
          - header: x-session-id
          - attribute: agent-identity
  - type: mock-metrics-source
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: session-affinity-filter
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
`
	runSessionIDHeaderStrategyChecks(t, configText)
}

// TestSessionAffinityScorer_SessionIDHeaderStrategy is the scorer analogue: the
// same source list and cross-plugin handoff, but the scorer feeds max-score-picker
// rather than hard-narrowing candidates.
func TestSessionAffinityScorer_SessionIDHeaderStrategy(t *testing.T) {
	configText := `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: agent-identity
  - type: session-affinity-scorer
    parameters:
      strategy: session_id
      sessionIdConfig:
        sources:
          - header: x-session-id
          - attribute: agent-identity
  - type: max-score-picker
  - type: mock-metrics-source
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: session-affinity-scorer
      - pluginRef: max-score-picker
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
`
	runSessionIDHeaderStrategyChecks(t, configText)
}

// runSessionIDHeaderStrategyChecks drives the shared session_id assertions
// against a plugin (filter or scorer) configured with the ordered
// [header x-session-id, attribute agent-identity] source list. Affinity is
// asserted by same-session-same-endpoint, since session_id writes no token
// back and the plugin does not place a session on a fixed pod.
func runSessionIDHeaderStrategyChecks(t *testing.T, configText string) {
	const claudeHeader = "x-claude-code-session-id"

	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()
	pods := []PodState{
		P(0, 0, 0.1, modelMyModelTarget),
		P(1, 0, 0.1, modelMyModelTarget),
	}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	// repeats amplifies the same-session assertions: the plugin places nothing,
	// so the first request lands via the picker's random tie-break, and every
	// subsequent request for that identifier must pin to it. An abstaining or
	// no-op plugin would draw independently each time and fail within a few
	// repeats (~(1/2)^repeats), so this distinguishes a real binding from chance
	// without the flaky cross-session comparison it replaces.
	const repeats = 10

	// (1) Header source: the same x-session-id pins to one endpoint across
	// repeated requests.
	headerFirst := sendRoutedRequest(t, h, map[string]string{"x-session-id": "sess-A"})
	for i := 0; i < repeats; i++ {
		require.Equal(t, headerFirst, sendRoutedRequest(t, h, map[string]string{"x-session-id": "sess-A"}),
			"same x-session-id must pin to the same endpoint")
	}

	// (2) Attribute fallback: no header, agent-identity publishes the attribute
	// from the agent header, and the session pins across repeated requests. That
	// the pin holds across every repeat (not by chance) is itself proof the
	// attribute source resolved a real identifier: an unresolved source would
	// abstain, binding nothing, and each request would land on a random pod.
	agentFirst := sendRoutedRequest(t, h, map[string]string{claudeHeader: "agent-B"})
	for i := 0; i < repeats; i++ {
		require.Equal(t, agentFirst, sendRoutedRequest(t, h, map[string]string{claudeHeader: "agent-B"}),
			"same agent identity resolved via the attribute source must pin to the same endpoint")
	}

	// (3) Priority: header wins over attribute. A request carrying both the
	// x-session-id header of session A and a different agent header resolves as
	// session A, so it pins to session A's endpoint.
	both := sendRoutedRequest(t, h, map[string]string{"x-session-id": "sess-A", claudeHeader: "agent-B"})
	require.Equal(t, headerFirst, both,
		"the header source must win over the attribute source when both are present")
}

// routedRequestSeq gives each sendRoutedRequest call a distinct RequestID.
var routedRequestSeq int64

// sendRoutedRequest drives one full request/response transaction with the given
// extra request headers and returns the routed destination endpoint. Unlike
// sendSessionRequest it reads no response token: session_id writes nothing
// back, so affinity is asserted purely by the endpoint the request is routed to.
func sendRoutedRequest(t *testing.T, h *TestHarness, extraHeaders map[string]string) string {
	t.Helper()

	// Unique per call: the Choose->PreRequest boundPodPresence handoff is keyed by
	// RequestID, so reusing one across sequential requests would couple their state.
	reqHeaders := map[string]string{
		metadata.ObjectiveKey:        modelMyModel,
		metadata.ModelNameRewriteKey: modelMyModelTarget,
		reqcommon.RequestIDHeaderKey: fmt.Sprintf("session-req-%d", atomic.AddInt64(&routedRequestSeq, 1)),
	}
	for k, v := range extraHeaders {
		reqHeaders[k] = v
	}

	requests := integration.ReqRaw(reqHeaders, `{"model":"`+modelMyModel+`","prompt":"hello","max_tokens":10,"temperature":0}`)
	requests = append(requests, ReqResponseOnly(
		map[string]string{"content-type": "application/json", "status": "200"},
		`{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hi","role":"assistant"}}],"model":"`+modelMyModelTarget+`","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}`,
	)...)

	client, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(t.Context())
	require.NoError(t, err)

	responses, err := integration.StreamedRequest(t, client, requests, 4)
	require.NoError(t, err)
	require.Len(t, responses, 4)

	endpoint := headerValue(responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey)
	require.NotEmpty(t, endpoint, "request headers response must set the destination endpoint")
	return endpoint
}
