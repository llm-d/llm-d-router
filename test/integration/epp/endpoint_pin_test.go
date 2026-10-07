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

	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

const (
	// profileHeader is the header that header-profile-handler reads.
	profileHeader  = "epp-profile"
	prefillProfile = "prefill"
)

// pinConfigBase is the EPP config of a coordinator deployment: one scheduling
// call per profile, selected by header. The slot holds the screener line, or
// nothing.
const pinConfigBase = `
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
%s
  - type: header-profile-handler
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

var (
	pinConfig        = fmt.Sprintf(pinConfigBase, "  - type: endpoint-pin-screener")
	noScreenerConfig = fmt.Sprintf(pinConfigBase, "")

	// The pods that setupPDHarness creates.
	pinPrefillEndpoints = []string{"192.168.1.1:8000", "192.168.1.2:8000"}
	pinDecodeEndpoint   = "192.168.1.3:8000"
)

// TestEndpointPin_RoutesToThePinnedEndpoint pins several requests to each
// prefill endpoint in turn. Without the pin, the picker breaks the tie between
// the two equal pods at random.
func TestEndpointPin_RoutesToThePinnedEndpoint(t *testing.T) {
	h := setupPDHarness(t, pinConfig)

	const repeats = 5
	for _, pin := range pinPrefillEndpoints {
		for range repeats {
			routed := sendRoutedRequest(t, h, map[string]string{
				profileHeader:             prefillProfile,
				routing.EndpointPinHeader: pin,
			})
			require.Equal(t, pin, routed)
		}
	}
}

// TestEndpointPin_NoEndpoint covers pins that leave no endpoint to schedule on.
// The request is rejected; it is not routed to another pod.
func TestEndpointPin_NoEndpoint(t *testing.T) {
	h := setupPDHarness(t, pinConfig)

	const (
		screenedMsg = "inference error: ServiceUnavailable - screeners eliminated all endpoint candidates"
		filteredMsg = "inference error: ServiceUnavailable - no endpoints available for the given request"
	)
	tests := []struct {
		name    string
		pin     string
		wantMsg string
	}{
		{name: "endpoint is not in the pool", pin: "192.168.1.200:8000", wantMsg: screenedMsg},
		{name: "address of a pool pod with another port", pin: "192.168.1.1:9999", wantMsg: screenedMsg},
		{name: "address without a port", pin: "192.168.1.1", wantMsg: screenedMsg},
		// The screener keeps the pod and the prefill profile's filter removes it.
		{name: "endpoint is a decode pod", pin: pinDecodeEndpoint, wantMsg: filteredMsg},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			responses := streamRequests(t, h, newRoutedRequest(map[string]string{
				profileHeader:             prefillProfile,
				routing.EndpointPinHeader: tt.pin,
			}), 1)
			want := ExpectRejectWithDropReason(envoyTypePb.StatusCode_ServiceUnavailable, tt.wantMsg, errcommon.RequestDroppedReasonNoEndpoints)
			if diff := cmp.Diff(want, responses, protocmp.Transform()); diff != "" {
				t.Fatalf("the EPP must answer the request and not route it (-want +got):\n%s", diff)
			}
		})
	}
}

// TestEndpointPin_ScreenerNotConfigured sends a pinned request to an EPP
// without the screener. The header has no effect, and the request is routed.
func TestEndpointPin_ScreenerNotConfigured(t *testing.T) {
	h := setupPDHarness(t, noScreenerConfig)

	routed := sendRoutedRequest(t, h, map[string]string{
		profileHeader:             prefillProfile,
		routing.EndpointPinHeader: "192.168.1.200:8000",
	})
	require.Contains(t, pinPrefillEndpoints, routed)
}
