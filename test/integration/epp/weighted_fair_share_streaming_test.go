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

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/weightedfairshare"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
)

func TestWeightedFairShareStreamingUsage(t *testing.T) {
	configText := fmt.Sprintf(`
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: openai-parser
- type: mock-metrics-source
- type: random-picker
- type: %[1]s
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  sources:
  - pluginRef: mock-metrics-source
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
flowControl:
  priorityBands:
  - priority: 2
    fairnessPolicyRef: %[1]s
`, weightedfairshare.WeightedFairShareFairnessPolicyType)

	for _, tc := range []struct {
		name        string
		usage       string
		wantService float64
	}{
		{name: "without_usage", wantService: 6},
		{
			name:        "reported_usage",
			usage:       `,"usage":{"prompt_tokens":0,"completion_tokens":10,"total_tokens":10}`,
			wantService: 20,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Isolate the policy's per-instance collectors between harness runs.
			registry := crmetrics.Registry
			crmetrics.Registry = prometheus.NewRegistry()
			t.Cleanup(func() { crmetrics.Registry = registry })

			h := NewTestHarness(t.Context(), t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()
			pods := []PodState{P(0, 0, 0.1, modelMyModelTarget)}
			h.WithPods(pods).WaitForSync(len(pods), modelMyModel).WaitForMetricsDelivery()

			requests := fwkepp.ReqRaw(map[string]string{
				":method":                    "POST",
				":path":                      "/v1/chat/completions",
				"content-type":               "application/json",
				reqcommon.RequestIDHeaderKey: tc.name,
				metadata.ObjectiveKey:        modelMyModel,
				metadata.FlowFairnessIDKey:   "alpha",
			}, fmt.Sprintf(`{"model":%q,"stream":true,"messages":[{"role":"user","content":"hello"}]}`, modelMyModel))
			requests = append(requests, ReqResponseOnly(
				map[string]string{"content-type": "text/event-stream", ":status": "200"},
				"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n",
				"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n",
				fmt.Sprintf("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]%s}\n\n", tc.usage),
				"data: [DONE]\n\n",
			)...)

			responses, err := fwkepp.StreamedRequest(t, h.Client, requests, len(requests))
			require.NoError(t, err)
			require.Len(t, responses, len(requests))

			h.ExpectMetrics(map[string]string{
				"llm_d_epp_weighted_fair_share_virtual_service": fmt.Sprintf(`
# HELP llm_d_epp_weighted_fair_share_virtual_service [ALPHA] Virtual service per fairness flow: charged cost divided by weight.
# TYPE llm_d_epp_weighted_fair_share_virtual_service gauge
llm_d_epp_weighted_fair_share_virtual_service{flow_id="alpha",policy="%s",priority="2"} %g
`, weightedfairshare.WeightedFairShareFairnessPolicyType, tc.wantService),
			})
		})
	}
}
