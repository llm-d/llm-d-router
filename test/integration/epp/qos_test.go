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
	_ "embed"
	"testing"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/require"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/qoscontract"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

//go:embed testdata/qos-config.yaml
var qosTestConfig string

// withHeader appends a request header to the headers frame of an LLM request set.
func withHeader(requests []*extProcPb.ProcessingRequest, key, value string) []*extProcPb.ProcessingRequest {
	hm := requests[0].GetRequestHeaders().GetHeaders()
	hm.Headers = append(hm.Headers, &envoyCorev3.HeaderValue{Key: key, Value: value})
	return requests
}

func responseHeader(resp *extProcPb.ProcessingResponse, key string) (string, bool) {
	for _, h := range resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
		if h.GetHeader().GetKey() == key {
			return string(h.GetHeader().GetRawValue()), true
		}
	}
	return "", false
}

func immediateHeader(resp *extProcPb.ProcessingResponse, key string) (string, bool) {
	for _, h := range resp.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		if h.GetHeader().GetKey() == key {
			return string(h.GetHeader().GetRawValue()), true
		}
	}
	return "", false
}

// requestTotalPriorities returns the priority label of every llm_d_epp_request_total series for
// the given fairness ID.
func requestTotalPriorities(t *testing.T, fairnessID string) []string {
	t.Helper()
	families, err := crmetrics.Registry.Gather()
	require.NoError(t, err)
	var priorities []string
	for _, family := range families {
		if family.GetName() != "llm_d_epp_request_total" {
			continue
		}
		for _, m := range family.GetMetric() {
			labels := map[string]string{}
			for _, l := range m.GetLabel() {
				labels[l.GetName()] = l.GetValue()
			}
			if labels["fairness_id"] == fairnessID {
				priorities = append(priorities, labels["priority"])
			}
		}
	}
	return priorities
}

func TestQoSContract(t *testing.T) {
	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode(), eppharness.WithConfigText(qosTestConfig)).WithBaseResources()
	pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	t.Run("over-commit critical traffic is served at the next class", func(t *testing.T) {
		requests := withHeader(fwkepp.ReqLLM(reqLogger, "hello", "modelName", "modelName"), metadata.FlowFairnessIDKey, "team-a")
		requests = append(requests, ReqResponseOnly(
			map[string]string{"content-type": "application/json", ":status": "200"},
			`{"choices":[{"finish_reason":"stop","index":0,"text":"hi"}],"model":"modelName","object":"text_completion","usage":{"completion_tokens":1,"prompt_tokens":2,"total_tokens":3}}`,
		)...)

		responses, err := fwkepp.StreamedRequest(t, h.Client, requests, 4)
		require.NoError(t, err)
		require.Len(t, responses, 4)
		class, ok := responseHeader(responses[2], qoscontract.ClassHeader)
		require.True(t, ok, "the served class is reported on the response")
		require.Equal(t, "batch", class)
		_, ok = responseHeader(responses[2], qoscontract.ConcurrencyRemainingHeader)
		require.True(t, ok)
		// The director, not the plugin, records the request at the priority it was served at.
		require.Equal(t, []string{"-10"}, requestTotalPriorities(t, "team-a"))
	})

	t.Run("over-commit traffic in the lowest class is rejected with 429", func(t *testing.T) {
		requests := withHeader(fwkepp.ReqLLM(reqLogger, "hello", "modelName", "modelName"), metadata.FlowFairnessIDKey, "team-b")

		responses, err := fwkepp.StreamedRequest(t, h.Client, requests, 1)
		require.NoError(t, err)
		require.Len(t, responses, 1)
		ir := responses[0].GetImmediateResponse()
		require.NotNil(t, ir, "want an immediate response")
		require.Equal(t, envoyTypePb.StatusCode_TooManyRequests, ir.GetStatus().GetCode())
		reason, _ := immediateHeader(responses[0], errcommon.RequestDroppedReasonHeaderKey)
		require.Equal(t, string(errcommon.RequestDroppedReasonOverQuota), reason)
		retry, ok := immediateHeader(responses[0], qoscontract.RetryAfterHeader)
		require.True(t, ok)
		require.NotEmpty(t, retry)
	})
}
