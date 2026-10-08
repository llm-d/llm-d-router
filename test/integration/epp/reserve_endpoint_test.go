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

package epp

import (
	"fmt"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// reserveConfigBase is the EPP config of a per-phase prefill EPP: one profile,
// run under the default profile handler. The slot holds the reserve-endpoint
// plugin line, or nothing.
const reserveConfigBase = `
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
%s
  - type: prefill-filter
  - type: queue-scorer
  - type: max-score-picker
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
`

var (
	reserveConfig   = fmt.Sprintf(reserveConfigBase, "  - type: reserve-endpoint")
	noReserveConfig = fmt.Sprintf(reserveConfigBase, "")

	// The prefill pods that setupPDHarness creates.
	reservePrefillEndpoints = []string{"192.168.1.1:8000", "192.168.1.2:8000"}
)

// reservePDConfig is the EPP config of a single EPP that schedules prefill and
// decode in one pass, as for the sidecar. The disaggregation profile handler
// sets x-prefiller-host-port on the request, and the primary profile is decode.
const reservePDConfig = `
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: reserve-endpoint
  - type: always-disagg-pd-decider
  - type: disagg-profile-handler
    parameters:
      deciders:
        prefill: always-disagg-pd-decider
  - type: prefill-filter
  - type: decode-filter
  - type: queue-scorer
  - type: max-score-picker
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
  - name: decode
    plugins:
    - pluginRef: decode-filter
    - pluginRef: queue-scorer
    - pluginRef: max-score-picker
`

// The decode pods that setupPDHarness creates.
var reserveDecodeEndpoints = []string{"192.168.1.3:8000", "192.168.1.4:8000"}

// TestReserveEndpoint_Answered sends the ask of the coordinator: it is answered
// with 204 and a prefill endpoint on the response headers, and is not routed.
// A request without the preference is still routed.
func TestReserveEndpoint_Answered(t *testing.T) {
	h := setupPDHarness(t, reserveConfig)

	answer := sendAnsweredRequest(t, h, map[string]string{routing.PreferHeader: routing.PreferReserveEndpoint})
	require.Equal(t, envoyTypePb.StatusCode_NoContent, answer.GetStatus().GetCode())
	require.Empty(t, answer.GetBody(), "a 204 answer carries no body")
	setHeaders := answer.GetHeaders().GetSetHeaders()
	require.Equal(t, routing.PreferReserveEndpoint, headerValue(setHeaders, routing.PreferenceAppliedHeader))
	require.Contains(t, reservePrefillEndpoints, headerValue(setHeaders, routing.ReservedEndpointHeader),
		"the ask must be answered with a prefill endpoint")
	require.Empty(t, headerValue(setHeaders, routing.PrefillEndpointHeader),
		"a single-profile EPP sets no routing header, so the answer carries none")

	routed := sendRoutedRequest(t, h, map[string]string{})
	require.Contains(t, reservePrefillEndpoints, routed, "a request without the preference is routed")
}

// TestReserveEndpoint_RoutingHeadersOnAnswer sends the ask to an EPP whose
// profile handler sets an internal routing header on the request. The answer
// carries that header next to the reserved endpoint, as the forwarded request
// would have carried it to the sidecar.
func TestReserveEndpoint_RoutingHeadersOnAnswer(t *testing.T) {
	h := setupPDHarness(t, reservePDConfig)

	answer := sendAnsweredRequest(t, h, map[string]string{routing.PreferHeader: routing.PreferReserveEndpoint})
	require.Equal(t, envoyTypePb.StatusCode_NoContent, answer.GetStatus().GetCode())
	setHeaders := answer.GetHeaders().GetSetHeaders()
	require.Contains(t, reserveDecodeEndpoints, headerValue(setHeaders, routing.ReservedEndpointHeader),
		"the reserved endpoint is the primary profile's pick, a decode pod")
	require.Contains(t, reservePrefillEndpoints, headerValue(setHeaders, routing.PrefillEndpointHeader),
		"the prefill routing header set by the profile handler travels on the answer")
}

// TestReserveEndpoint_PluginNotConfigured sends the ask to an EPP without the
// reserve-endpoint plugin. The EPP must reject it and not route it to a pod.
func TestReserveEndpoint_PluginNotConfigured(t *testing.T) {
	h := setupPDHarness(t, noReserveConfig)

	answer := sendAnsweredRequest(t, h, map[string]string{routing.PreferHeader: routing.PreferReserveEndpoint})
	require.Equal(t, envoyTypePb.StatusCode_InternalServerError, answer.GetStatus().GetCode())
	require.Contains(t, string(answer.GetBody()), "no plugin handled it")
	require.Empty(t, headerValue(answer.GetHeaders().GetSetHeaders(), routing.ReservedEndpointHeader))
}

// sendAnsweredRequest sends one request with the given extra headers and
// returns the immediate response that the EPP answers it with. The test fails
// when the EPP routes the request.
func sendAnsweredRequest(t *testing.T, h *TestHarness, extraHeaders map[string]string) *extProcPb.ImmediateResponse {
	t.Helper()

	responses := streamRequests(t, h, newRoutedRequest(extraHeaders), 1)
	answer := responses[0].GetImmediateResponse()
	require.NotNil(t, answer, "the EPP must answer the request and not route it: %v", responses[0])
	return answer
}
