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
	"context"
	"testing"
	"time"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

const passthroughGRPCStreamConfig = `
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: passthrough-parser
- type: mock-metrics-source
- name: constant-cost-reporter
  type: request-attribute-reporter
  parameters:
    attributes:
    - key:
        namespace: envoy.lb
        name: x-gateway-inference-request-cost
      expression: "1"
requestHandler:
  parsers:
  - pluginRef: passthrough-parser
dataLayer:
  injectDefaults: false
  sources:
  - pluginRef: mock-metrics-source
schedulingProfiles:
- name: default
`

// TestPassthroughGRPCServerStreamForwardsEachMessage runs a server-streaming gRPC response through the configured EPP.
// Message 2 is sent only after message 1 comes back, so an EPP that holds the response until the trailers fails here.
func TestPassthroughGRPCServerStreamForwardsEachMessage(t *testing.T) {
	ctx := t.Context()
	h := eppharness.NewTestHarness(ctx, t, eppharness.WithStandardMode(), eppharness.WithConfigText(passthroughGRPCStreamConfig)).WithBaseResources()
	pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	streamCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	client, err := extProcPb.NewExternalProcessorClient(h.GRPCConn()).Process(streamCtx)
	require.NoError(t, err)

	frame := func(payload string) []byte {
		f, err := grpcframe.FramePayload([]byte(payload))
		require.NoError(t, err)
		return f
	}
	send := func(req *extProcPb.ProcessingRequest) {
		t.Helper()
		require.NoError(t, client.Send(req))
	}
	headers := func(kv map[string]string, eos bool) *extProcPb.HttpHeaders {
		return &extProcPb.HttpHeaders{Headers: &envoyCorev3.HeaderMap{Headers: buildEnvoyHeaders(kv)}, EndOfStream: eos}
	}

	send(&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: headers(map[string]string{
		":path":               "/grpc.testing.TestService/StreamingOutputCall",
		"content-type":        "application/grpc",
		metadata.ObjectiveKey: modelMyModel,
	}, false)}})
	send(&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{Body: frame("request"), EndOfStream: true}}})
	resp, err := client.Recv()
	require.NoError(t, err)
	require.NotNil(t, resp.GetRequestHeaders())
	resp, err = client.Recv()
	require.NoError(t, err)
	require.NotNil(t, resp.GetRequestBody())

	send(&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: headers(map[string]string{
		":status": "200", "content-type": "application/grpc",
	}, false)}})
	resp, err = client.Recv()
	require.NoError(t, err)
	require.NotNil(t, resp.GetResponseHeaders())

	for _, message := range []string{"message 1", "message 2"} {
		body := frame(message)
		send(&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{Body: body}}})
		resp, err = client.Recv()
		require.NoError(t, err, "waiting for %s", message)
		require.Equal(t, body, resp.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse().GetBody())
	}

	send(&extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseTrailers{ResponseTrailers: &extProcPb.HttpTrailers{
		Trailers: &envoyCorev3.HeaderMap{Headers: buildEnvoyHeaders(map[string]string{"grpc-status": "0"})},
	}}})
	resp, err = client.Recv()
	require.NoError(t, err)
	streamed := resp.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.Empty(t, streamed.GetBody())
	require.False(t, streamed.GetEndOfStream())
	cost := resp.GetDynamicMetadata().GetFields()["envoy.lb"].GetStructValue().GetFields()["x-gateway-inference-request-cost"]
	require.NotNil(t, cost, "final dynamic metadata precedes the trailers")
	require.Equal(t, 1.0, cost.GetNumberValue())
	resp, err = client.Recv()
	require.NoError(t, err)
	require.NotNil(t, resp.GetResponseTrailers())
}
