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
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	integration "github.com/llm-d/llm-d-router/test/integration"
)

func TestDynamicAttributes_Concurrency(t *testing.T) {
	configText := `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: passthrough-parser
- type: round-robin-fairness-policy
- type: fcfs-ordering-policy
- type: concurrency-detector
  parameters:
    maxConcurrency: 1
    concurrencyMode: requests
    headroom: 0.0
- type: mock-metrics-source
requestHandler:
  parsers:
  - pluginRef: passthrough-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
    weight: 2
flowControl:
  saturationDetector:
    pluginRef: concurrency-detector
  maxRequests: "100"
  defaultRequestTTL: "60s"
  priorityBands:
  - priority: 0
    maxRequests: "100"
    fairnessPolicyRef: round-robin-fairness-policy
    orderingPolicyRef: fcfs-ordering-policy
`

	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithConfigText(configText), WithStandardMode())
	h = h.WithBaseResources()

	// Add one pod. We need it to support modelMyModelTarget.
	pods := []PodState{
		P(0, 0, 0.0, modelMyModelTarget),
	}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	// Stream 1: Request 1
	client1, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(ctx)
	require.NoError(t, err)

	// Send Request 1 Headers
	req1Headers := integration.ReqRaw(
		map[string]string{
			"hi":                         "mom",
			metadata.ObjectiveKey:        modelMyModel,
			metadata.ModelNameRewriteKey: modelMyModelTarget,
			reqcommon.RequestIDHeaderKey: "req-1",
		},
	)
	require.Len(t, req1Headers, 1)
	err = client1.Send(req1Headers[0])
	require.NoError(t, err)

	// Send Request 1 Body
	req1Body := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{
				Body:        []byte("passthrough-body-1"),
				EndOfStream: true,
			},
		},
	}
	err = client1.Send(req1Body)
	require.NoError(t, err)

	// Receive Request 1 Headers Response
	resp1Headers, err := client1.Recv()
	require.NoError(t, err)
	require.NotNil(t, resp1Headers.GetRequestHeaders())

	// Receive Request 1 Body Response
	resp1Body, err := client1.Recv()
	require.NoError(t, err)
	require.NotNil(t, resp1Body.GetRequestBody())

	// At this point, Request 1 is routed and in-flight.
	// Load should be 1.

	// Stream 2: Request 2
	client2, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(ctx)
	require.NoError(t, err)

	// Send Request 2 Headers
	req2Headers := integration.ReqRaw(
		map[string]string{
			"hi":                         "mom",
			metadata.ObjectiveKey:        modelMyModel,
			metadata.ModelNameRewriteKey: modelMyModelTarget,
			reqcommon.RequestIDHeaderKey: "req-2",
		},
	)
	require.Len(t, req2Headers, 1)
	err = client2.Send(req2Headers[0])
	require.NoError(t, err)

	// Send Request 2 Body
	req2Body := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{
				Body:        []byte("passthrough-body-2"),
				EndOfStream: true,
			},
		},
	}
	err = client2.Send(req2Body)
	require.NoError(t, err)

	// Verify Request 2 is blocked (no response for 1 second)
	type recvResult struct {
		res *extProcPb.ProcessingResponse
		err error
	}
	client2RecvChan := make(chan recvResult, 1)
	go func() {
		res, err := client2.Recv()
		client2RecvChan <- recvResult{res, err}
	}()

	select {
	case result := <-client2RecvChan:
		t.Fatalf("Expected Request 2 to be blocked, but received response: %v, err: %v", result.res, result.err)
	case <-time.After(1 * time.Second):
		// Success, timed out without response.
	}

	// Complete Request 1
	// Send Response Headers
	respHeadersReq := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{
				Headers: &configPb.HeaderMap{
					Headers: []*configPb.HeaderValue{
						{Key: "status", Value: "200"},
					},
				},
			},
		},
	}
	err = client1.Send(respHeadersReq)
	require.NoError(t, err)

	respHeadersResp, err := client1.Recv()
	require.NoError(t, err)
	require.NotNil(t, respHeadersResp.GetResponseHeaders())

	// Send Response Body with EndOfStream=true
	respBodyReq := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseBody{
			ResponseBody: &extProcPb.HttpBody{
				Body:        []byte("response-body-1"),
				EndOfStream: true,
			},
		},
	}
	err = client1.Send(respBodyReq)
	require.NoError(t, err)

	respBodyResp, err := client1.Recv()
	require.NoError(t, err)
	require.NotNil(t, respBodyResp.GetResponseBody())

	// Request 1 is now fully complete. Load should be 0.
	// Request 2 should be released.
	select {
	case result := <-client2RecvChan:
		require.NoError(t, result.err)
		require.NotNil(t, result.res.GetRequestHeaders())
	case <-time.After(5 * time.Second):
		t.Fatal("Timed out waiting for Request 2 to be released after Request 1 completed")
	}

	// Clean up client2 stream by receiving the body response too.
	resp2Body, err := client2.Recv()
	require.NoError(t, err)
	require.NotNil(t, resp2Body.GetRequestBody())
}
