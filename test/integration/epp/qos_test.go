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

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/qosadmission"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/test/integration"
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

func TestQoSAdmission(t *testing.T) {
	h := NewTestHarness(t.Context(), t, WithStandardMode(), WithConfigText(qosTestConfig)).WithBaseResources()
	pods := []PodState{P(0, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	t.Run("over-commit critical traffic is served at the next class", func(t *testing.T) {
		requests := withHeader(integration.ReqLLM(reqLogger, "hello", "modelName", "modelName"), metadata.FlowFairnessIDKey, "team-a")
		requests = append(requests, ReqResponseOnly(
			map[string]string{"content-type": "application/json", ":status": "200"},
			`{"choices":[{"finish_reason":"stop","index":0,"text":"hi"}],"model":"modelName","object":"text_completion","usage":{"completion_tokens":1,"prompt_tokens":2,"total_tokens":3}}`,
		)...)

		responses, err := integration.StreamedRequest(t, h.Client, requests, 4)
		require.NoError(t, err)
		require.Len(t, responses, 4)
		class, ok := responseHeader(responses[2], qosadmission.ClassHeader)
		require.True(t, ok, "the served class is reported on the response")
		require.Equal(t, "batch", class)
		_, ok = responseHeader(responses[2], qosadmission.ConcurrencyRemainingHeader)
		require.True(t, ok)
	})

	t.Run("over-commit traffic in the lowest class is rejected with 429", func(t *testing.T) {
		requests := withHeader(integration.ReqLLM(reqLogger, "hello", "modelName", "modelName"), metadata.FlowFairnessIDKey, "team-b")

		responses, err := integration.StreamedRequest(t, h.Client, requests, 1)
		require.NoError(t, err)
		require.Len(t, responses, 1)
		ir := responses[0].GetImmediateResponse()
		require.NotNil(t, ir, "want an immediate response")
		require.Equal(t, envoyTypePb.StatusCode_TooManyRequests, ir.GetStatus().GetCode())
		reason, _ := immediateHeader(responses[0], errcommon.RequestDroppedReasonHeaderKey)
		require.Equal(t, string(errcommon.RequestDroppedReasonOverQuota), reason)
		retry, ok := immediateHeader(responses[0], qosadmission.RetryAfterHeader)
		require.True(t, ok)
		require.NotEmpty(t, retry)
	})
}
