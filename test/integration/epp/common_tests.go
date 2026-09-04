/*
Copyright 2025 The Kubernetes Authors.

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
	"encoding/json"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

// Model name constants shared across test suites.
const (
	modelMyModel       = "my-model"
	modelMyModelTarget = "my-model-12345"
	modelSQLLora       = "sql-lora"
	modelSQLLoraTarget = "sql-lora-1fdg2"
)

// --- Domain Request Builders ---
// buildEnvoyHeaders is a helper to convert a map of strings into Envoy HeaderValues.
func buildEnvoyHeaders(headers map[string]string) []*envoyCorev3.HeaderValue {
	hList := make([]*envoyCorev3.HeaderValue, 0, len(headers))
	for k, v := range headers {
		hList = append(hList, &envoyCorev3.HeaderValue{Key: k, RawValue: []byte(v)})
	}
	return hList
}

// ReqSubset creates a request sequence with Envoy Endpoint Metadata.
// This simulates the "Subset Load Balancing" flow where EPP picks a specific pod IP.
func ReqSubset(prompt, model, target string, subsets ...string) []*extProcPb.ProcessingRequest {
	// Uses the shared low-level generator which handles the metadata construction
	return fwkepp.GenerateStreamedRequestSet(eppharness.Logger(), prompt, model, target, subsets)
}

// ReqResponseOnly creates a sequence simulating only the response phase from Envoy.
// It skips the RequestHeaders phase entirely.
func ReqResponseOnly(
	respHeaders map[string]string,
	bodyChunks ...string,
) []*extProcPb.ProcessingRequest {
	reqs := make([]*extProcPb.ProcessingRequest, 0, 1+len(bodyChunks))

	// 1. Response Headers
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: buildEnvoyHeaders(respHeaders)},
			},
		},
	})

	// 2. Response Body Chunks
	for i, chunk := range bodyChunks {
		reqs = append(reqs, &extProcPb.ProcessingRequest{
			Request: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body:        []byte(chunk),
					EndOfStream: i == len(bodyChunks)-1,
				},
			},
		})
	}
	return reqs
}

// ReqRequestHeadersAndResponseGRPC creates a sequence that starts with request headers (to resolve parser)
// followed by response phase.
func ReqRequestHeadersAndResponseGRPC(
	reqHeaders map[string]string,
	respHeaders map[string]string,
	bodyChunks ...[]byte,
) []*extProcPb.ProcessingRequest {
	reqs := make([]*extProcPb.ProcessingRequest, 0, 2+len(bodyChunks))

	// 1. Request Headers
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: buildEnvoyHeaders(reqHeaders)},
			},
		},
	})

	// 2. Response Headers
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: buildEnvoyHeaders(respHeaders)},
			},
		},
	})

	// 3. Response Body Chunks
	for _, chunk := range bodyChunks {
		reqs = append(reqs, &extProcPb.ProcessingRequest{
			Request: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body:        chunk,
					EndOfStream: false,
				},
			},
		})
	}

	// 4. Response Trailer
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_ResponseTrailers{
			ResponseTrailers: &extProcPb.HttpTrailers{},
		},
	})
	return reqs
}

// --- Response Expectations ---

func buildRouteResponse(endpoint, targetModel, prompt string, stream bool) []*extProcPb.ProcessingResponse {
	bodyMap := map[string]any{
		"max_tokens": 100, "model": targetModel, "prompt": prompt, "temperature": 0,
	}
	if stream {
		bodyMap["stream"] = true
	}
	j, _ := json.Marshal(bodyMap)

	return fwkepp.NewRequestBufferedResponse(
		endpoint, j,
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "hi", RawValue: []byte("mom")}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      reqcommon.RequestIDHeaderKey,
			RawValue: []byte("test-request-id"),
		}},
	)
}

func buildGRPCRouteResponse(endpoint, prompt, methodName string, stream bool) []*extProcPb.ProcessingResponse {
	j, _ := fwkepp.CreateGrpcPayload(fwkepp.GRPCRequestProto(prompt, methodName, stream))

	return fwkepp.NewRequestBufferedResponse(
		endpoint, j,
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "hi", RawValue: []byte("mom")}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      reqcommon.RequestIDHeaderKey,
			RawValue: []byte("test-request-id"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: ":path", RawValue: []byte(methodName)}},
	)
}

// ExpectRouteTo asserts that the request was successfully routed to the specified endpoint and that the body was
// rewritten to match the target model.
func ExpectRouteTo(endpoint, targetModel, prompt string) []*extProcPb.ProcessingResponse {
	return buildRouteResponse(endpoint, targetModel, prompt, false)
}

// ExpectRouteToWithStream asserts that the request was successfully routed with streaming enabled.
func ExpectRouteToWithStream(endpoint, targetModel, prompt string) []*extProcPb.ProcessingResponse {
	return buildRouteResponse(endpoint, targetModel, prompt, true)
}

// ExpectPassthroughRouteTo asserts that the request was successfully routed to the specified endpoint and that the body was
// not parsed.
func ExpectPassthroughRouteTo(endpoint string, rawBytes []byte) []*extProcPb.ProcessingResponse {
	return fwkepp.NewRequestBufferedResponse(
		endpoint, rawBytes,
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "hi", RawValue: []byte("mom")}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      reqcommon.RequestIDHeaderKey,
			RawValue: []byte("test-request-id"),
		}},
	)
}

// ExpectGRPCRouteTo asserts that the request was successfully routed to the specified endpoint.
func ExpectGRPCRouteTo(endpoint, prompt, method string) []*extProcPb.ProcessingResponse {
	return buildGRPCRouteResponse(endpoint, prompt, method, false)
}

// ExpectGRPCRouteToWithStream asserts that the request was successfully routed with streaming enabled.
func ExpectGRPCRouteToWithStream(endpoint, prompt, method string) []*extProcPb.ProcessingResponse {
	return buildGRPCRouteResponse(endpoint, prompt, method, true)
}

// ExpectReject asserts that the EPP immediately rejected the request with the given code and message.
func ExpectReject(code envoyTypePb.StatusCode, msg string) []*extProcPb.ProcessingResponse {
	return fwkepp.NewImmediateErrorResponse(code, msg)
}

// ExpectRejectWithDropReason asserts that the EPP immediately rejected the request with the given code
// and message, and that the response carries the drop-reason header.
func ExpectRejectWithDropReason(code envoyTypePb.StatusCode, msg string, reason errcommon.RequestDroppedReason) []*extProcPb.ProcessingResponse {
	return fwkepp.NewImmediateErrorResponse(code, msg, &envoyCorev3.HeaderValueOption{
		Header: &envoyCorev3.HeaderValue{
			Key:      errcommon.RequestDroppedReasonHeaderKey,
			RawValue: []byte(reason),
		},
	})
}

// ExpectBufferResp asserts that the EPP buffers the response and rewrites the body.
// This uses the shared primitive but adds EPP-specific headers we expect.
func ExpectBufferResp(body string, contentType string) []*extProcPb.ProcessingResponse {
	return fwkepp.NewResponseBufferedResponse(
		body,
		contentType != "application/grpc", // For gRPC, the EoS will not be set to true but it will return a trailer.
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "x-went-into-resp-headers",
			RawValue: []byte("true"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "content-type",
			RawValue: []byte(contentType),
		}},
	)
}

// ExpectStreamResp asserts that the EPP streams the response chunks (pass-through).
// It constructs a sequence of:
// 1. ResponseHeaders (with "x-went-into-resp-headers" and "text/event-stream")
// 2. ResponseBody chunks (with EndOfStream=true on the final chunk)
func ExpectStreamResp(chunks ...string) []*extProcPb.ProcessingResponse {
	res := make([]*extProcPb.ProcessingResponse, 0, 1+len(chunks))
	res = append(res, fwkepp.NewResponseHeaders(
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "x-went-into-resp-headers",
			RawValue: []byte("true"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "content-type",
			RawValue: []byte("text/event-stream"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{Key: "status", RawValue: []byte("200")}},
	))

	// 2. The Body Chunk Frames
	for i, chunk := range chunks {
		res = append(res, fwkepp.NewResponseStreamChunk(chunk, i == len(chunks)-1))
	}
	return res
}

// ExpectGRPCStreamResp asserts that the EPP streams the gRPC response chunks.
func ExpectGRPCStreamResp(chunks ...string) []*extProcPb.ProcessingResponse {
	res := make([]*extProcPb.ProcessingResponse, 0, 1+len(chunks))
	res = append(res, fwkepp.NewResponseHeaders(
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "x-went-into-resp-headers",
			RawValue: []byte("true"),
		}},
		&envoyCorev3.HeaderValueOption{Header: &envoyCorev3.HeaderValue{
			Key:      "content-type",
			RawValue: []byte("application/grpc"),
		}},
	))

	for _, chunk := range chunks {
		res = append(res, fwkepp.NewResponseStreamChunk(chunk, false))
	}
	return res
}

// --- Shared Test Cases ---

// testCase defines a single integration test scenario that can be shared across test suites.
type testCase struct {
	name          string
	requests      []*extProcPb.ProcessingRequest
	pods          []eppharness.PodState
	configText    string
	wantResponses []*extProcPb.ProcessingResponse
	wantMetrics   map[string]string
	waitForModel  string
	// requiresCRDs indicates that this test case relies on specific Gateway API CRD features (like
	// InferenceModelRewrite) which are not available in Standalone mode without CRD.
	requiresCRDs bool
	// wantSpans lists the span names expected to be recorded (hermetic tests only).
	wantSpans []string
}

// commonTestCases returns the test cases shared between the standard and data layer test suites.
// prio adjusts expected priority label values based on execution context (e.g. 0 in NoCRD mode).
func commonTestCases(prio func(int) int) []testCase {
	return []testCase{
		{
			name:     "select lower queue and kv cache",
			requests: fwkepp.ReqLLM(eppharness.Logger(), "test1", modelMyModel, modelMyModelTarget),
			pods: []eppharness.PodState{
				eppharness.P(0, 3, 0.2),
				eppharness.P(1, 0, 0.1), // Winner (Low Queue, Low KV)
				eppharness.P(2, 10, 0.2),
			},
			wantResponses: ExpectRouteTo("192.168.1.2:8000", modelMyModelTarget, "test1"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": eppharness.CleanMetric(eppharness.MetricReqTotal(modelMyModel, modelMyModelTarget, prio(2))),
				"inference_pool_ready_pods":         eppharness.CleanMetric(eppharness.MetricReadyPods(3)),
			},
			wantSpans: []string{"request", "request_orchestration"},
		},
		{
			name:     "select active lora, low queue",
			requests: fwkepp.ReqLLM(eppharness.Logger(), "test2", modelSQLLora, modelSQLLoraTarget),
			pods: []eppharness.PodState{
				eppharness.P(0, 0, 0.2, "foo", "bar"),
				eppharness.P(1, 0, 0.1, "foo", modelSQLLoraTarget), // Winner (Has LoRA)
				eppharness.P(2, 10, 0.2, "foo", "bar"),
			},
			wantResponses: ExpectRouteTo("192.168.1.2:8000", modelSQLLoraTarget, "test2"),
			wantMetrics: map[string]string{
				"inference_objective_request_total": eppharness.CleanMetric(eppharness.MetricReqTotal(modelSQLLora, modelSQLLoraTarget, prio(2))),
			},
		},
		{
			name:     "no backend pods available",
			requests: fwkepp.ReqHeaderOnly(map[string]string{"content-type": "application/json"}),
			pods:     nil,
			wantResponses: ExpectReject(envoyTypePb.StatusCode_InternalServerError,
				"inference error: Internal - no pods available in datastore"),
		},
		{
			name: "request missing model field",
			requests: fwkepp.ReqRaw(
				map[string]string{"content-type": "application/json"},
				`{"prompt":"hello world"}`,
			),
			wantResponses: ExpectReject(envoyTypePb.StatusCode_BadRequest,
				"inference error: BadRequest - model not found in request body"),
		},
	}
}
