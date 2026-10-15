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
	"net"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// headerCopyConfigBase is a PD config whose disagg-profile-handler sets the
// prefill endpoint header in PreRequest and whose request-header-copy-handler
// copies it onto the response. The caller supplies the disagg-profile-handler
// parameters block (%s).
const headerCopyConfigBase = `
apiVersion: llm-d.ai/v1
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
  - type: request-header-copy-handler
    parameters:
      headers:
        - x-llm-d-prefiller-host-port
  - type: mock-metrics-source
  - type: core-metrics-extractor
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
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

// TestRequestHeaderCopyHandler_CopiesPreRequestHeader validates that a header a
// PreRequest plugin sets on the request reaches the ext_proc response header
// mutation: disagg-profile-handler sets the prefill pod's address, and
// request-header-copy-handler copies it onto the response.
func TestRequestHeaderCopyHandler_CopiesPreRequestHeader(t *testing.T) {
	configText := fmt.Sprintf(headerCopyConfigBase, "      deciders:\n        prefill: always-disagg-pd-decider")
	h := setupPDHarness(t, configText)

	value := sendRequestGetResponseHeader(t, h, nil, routing.PrefillEndpointHeader)
	host, port, err := net.SplitHostPort(value)
	require.NoError(t, err, "response must carry the prefill endpoint as host:port, got %q", value)
	require.Contains(t, []string{"192.168.1.1", "192.168.1.2"}, host, "the copied address must be a prefill pod")
	require.Equal(t, "8000", port)
}

// TestRequestHeaderCopyHandler_DoesNotEchoClientValue validates that a
// configured header no plugin set is absent from the response, even when the
// client sent it: EPP drops the client value on ingress, so the handler has
// nothing to copy.
func TestRequestHeaderCopyHandler_DoesNotEchoClientValue(t *testing.T) {
	// No deciders block: disagg-profile-handler skips prefill and sets no
	// prefill endpoint header.
	configText := fmt.Sprintf(headerCopyConfigBase, "      {}")
	h := setupPDHarness(t, configText)

	value := sendRequestGetResponseHeader(t, h,
		map[string]string{routing.PrefillEndpointHeader: "10.0.0.1:9999"}, routing.PrefillEndpointHeader)
	require.Empty(t, value, "a client-sent value must not be copied onto the response")
}

// sendRequestGetResponseHeader drives one full request/response transaction
// with the given extra request headers and returns the value of the named
// header in the response headers mutation, or "" when absent.
func sendRequestGetResponseHeader(t *testing.T, h *TestHarness, extraHeaders map[string]string, key string) string {
	t.Helper()

	requests := append(newRoutedRequest(extraHeaders), ReqResponseOnly(
		map[string]string{"content-type": "application/json", "status": "200"},
		`{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hi","role":"assistant"}}],"model":"`+modelMyModelTarget+`","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}`,
	)...)
	responses := streamRequests(t, h, requests, 4)
	return headerValue(responses[2].GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), key)
}
