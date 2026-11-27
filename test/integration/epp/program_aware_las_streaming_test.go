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
	programaware "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/program-aware"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

func TestProgramAwareLASStreamingUsage(t *testing.T) {
	// A half-life of 0 keeps the gauge at the exact charged cost.
	configText := `
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: openai-parser
- type: mock-metrics-source
- type: random-picker
- type: %[1]s
  parameters:
    lasHalfLifeSeconds: 0%[2]s
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
`

	for _, tc := range []struct {
		name        string
		plugins     string
		usage       string
		wantService float64
	}{
		{name: "without_usage", wantService: 6},
		{
			// inflight-load-producer consumes TokenizedPrompt, so the estimate token producer runs
			// and packs "user" + "hello" into 3 tokens, which survive to the response phase.
			name:        "without_usage_tokenized_prompt",
			plugins:     "\n- type: inflight-load-producer",
			wantService: 9,
		},
		{
			name:        "reported_usage",
			usage:       `,"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6}`,
			wantService: 9,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Isolate the plugin's collectors between harness runs.
			registry := crmetrics.Registry
			crmetrics.Registry = prometheus.NewRegistry()
			t.Cleanup(func() { crmetrics.Registry = registry })

			config := fmt.Sprintf(configText, programaware.ProgramAwarePluginType, tc.plugins)
			h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode(), eppharness.WithConfigText(config)).WithBaseResources()
			pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget)}
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
				"llm_d_epp_program_aware_attained_service_tokens": fmt.Sprintf(`
# HELP llm_d_epp_program_aware_attained_service_tokens [ALPHA] Time-decayed attained service (weighted tokens consumed) per program.
# TYPE llm_d_epp_program_aware_attained_service_tokens gauge
llm_d_epp_program_aware_attained_service_tokens{policy=%q,program_id="alpha"} %g
`, programaware.ProgramAwarePluginType, tc.wantService),
			})
		})
	}
}
