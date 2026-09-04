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

// Package epp provides ext_proc request and response builders and gRPC stream executors
// for tests that drive an EPP ext_proc server.
//
// It may import the R1/R2 framework packages, leaf pkg/epp packages (metadata, the
// vllmgrpc generated API) and pkg/common; it must never import pkg/epp/server or cmd/.
package epp

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	pb "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	headerKeyContentLength       = "Content-Length"
	extprocConnSetupTimeout      = 10 * time.Second
	extPorcConnSetupPollInterval = 50 * time.Millisecond
	GenerateGRPCMethodName       = "/vllm.grpc.engine.VllmEngine/Generate"
	EmbedGRPCMethodName          = "/vllm.grpc.engine.VllmEngine/Embed"
)

// --- Request Builders (Protocol Level) ---

// ReqLLM creates a sequence of gRPC messages representing a standard, streamed LLM inference request.
// It generates:
//  1. A RequestHeaders message containing standard inference headers (Objective, Model Rewrite, Request ID).
//  2. A RequestBody message containing the JSON payload with EndOfStream=true.
//
// Use this for the majority of "Happy Path" EPP and BBR streaming tests.
func ReqLLM(logger logr.Logger, prompt, model, targetModel string) []*extProcPb.ProcessingRequest {
	return GenerateStreamedRequestSet(logger, prompt, model, targetModel, nil)
}

func ReqGRPCLLM(logger logr.Logger, prompt, inferenceObjective, methodName string) []*extProcPb.ProcessingRequest {
	return GenerateStreamedGRPCRequestSet(logger, prompt, inferenceObjective, nil, methodName)
}

// ReqRaw creates a custom sequence of gRPC messages with specific headers and arbitrary body chunks.
// This is a lower-level helper useful for testing edge cases, such as:
//   - Invalid JSON bodies (to test error handling).
//   - Fragmentation (split bodies) to ensure the processor handles accumulation correctly.
//   - Protocol attacks (e.g., missing headers).
func ReqRaw(headers map[string]string, bodyChunks ...string) []*extProcPb.ProcessingRequest {
	reqs := make([]*extProcPb.ProcessingRequest, 0, 1+len(bodyChunks))

	// 1. Headers Phase
	hList := make([]*envoyCorev3.HeaderValue, 0, len(headers))
	for k, v := range headers {
		hList = append(hList, &envoyCorev3.HeaderValue{Key: k, Value: v})
	}
	reqs = append(reqs, &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: hList},
			},
		},
	})

	// 2. Body Phase (Chunks)
	for i, chunk := range bodyChunks {
		reqs = append(reqs, &extProcPb.ProcessingRequest{
			Request: &extProcPb.ProcessingRequest_RequestBody{
				RequestBody: &extProcPb.HttpBody{
					Body:        []byte(chunk),
					EndOfStream: i == len(bodyChunks)-1,
				},
			},
		})
	}
	return reqs
}

// ReqHeaderOnly creates a request sequence consisting solely of headers, with no body.
// It sets `EndOfStream: true` on the headers frame.
//
// Use this for testing non-inference traffic, such as GET requests, health checks, or requests that should bypass the
// inference processor logic.
func ReqHeaderOnly(headers map[string]string) []*extProcPb.ProcessingRequest {
	hList := make([]*envoyCorev3.HeaderValue, 0, len(headers))
	for k, v := range headers {
		hList = append(hList, &envoyCorev3.HeaderValue{Key: k, Value: v})
	}
	return []*extProcPb.ProcessingRequest{{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers:     &envoyCorev3.HeaderMap{Headers: hList},
				EndOfStream: true,
			},
		},
	}}
}

// --- Request Builders (Low-Level Generators) ---

// GenerateRequest constructs a `ProcessingRequest` containing a JSON-formatted LLM payload.
// It accepts a filterMetadata slice to inject Envoy Dynamic Metadata (used for subset load balancing).
func GenerateRequest(logger logr.Logger, prompt, model string, filterMetadata []string) *extProcPb.ProcessingRequest {
	j := map[string]any{
		"prompt":      prompt,
		"max_tokens":  100,
		"temperature": 0,
	}
	if model != "" {
		j["model"] = model
	}

	// Panic on marshal failure is acceptable in test helpers as it implies a bug in the test code itself.
	llmReq, err := json.Marshal(j)
	if err != nil {
		panic(fmt.Errorf("failed to marshal LLM request: %w", err))
	}

	return generateRequestFromBytes(llmReq, filterMetadata)
}

func GenerateGRPCRequest(logger logr.Logger, prompt, methodName string, stream bool, filterMetadata []string) *extProcPb.ProcessingRequest {
	req := GRPCRequestProto(prompt, methodName, stream)
	// Panic on marshal failure is acceptable in test helpers as it implies a bug in the test code itself.
	payload, err := CreateGrpcPayload(req)
	if err != nil {
		panic(fmt.Errorf("failed to marshal LLM request: %w", err))
	}
	return generateRequestFromBytes(payload, filterMetadata)
}

func ReqGRPCLLMWithStream(logger logr.Logger, prompt, inferenceObjective, methodName string) []*extProcPb.ProcessingRequest {
	requests := make([]*extProcPb.ProcessingRequest, 0, 2)
	requests = append(requests, generateHeaders(inferenceObjective, "", nil, map[string]string{":path": methodName}))
	requests = append(requests, GenerateGRPCRequest(logger, prompt, methodName, true, nil))
	return requests
}

func GRPCRequestProto(prompt, methodName string, stream bool) proto.Message {
	var req proto.Message
	switch methodName {
	case GenerateGRPCMethodName:
		req = &pb.GenerateRequest{
			Input: &pb.GenerateRequest_Text{
				Text: prompt,
			},
			Stream: stream,
		}
	case EmbedGRPCMethodName:
		req = &pb.EmbedRequest{
			Tokenized: &pb.TokenizedInput{
				OriginalText: prompt,
			},
		}
	}
	return req
}

func generateRequestFromBytes(payload []byte, filterMetadata []string) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: payload, EndOfStream: true},
		},
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: GenerateRequestMetadata(filterMetadata),
		},
	}
}

// helper function to simulate the gRPC payload framing
// [1 byte compression flag] [4 bytes message length] [message bytes...]
func CreateGrpcPayload(msg proto.Message) ([]byte, error) {
	b, err := proto.Marshal(msg)
	if err != nil {
		return nil, err
	}

	payload := make([]byte, 5+len(b))
	payload[0] = 0 // 0 = uncompressed
	binary.BigEndian.PutUint32(payload[1:5], uint32(len(b)))
	copy(payload[5:], b)
	return payload, nil
}

// GenerateStreamedRequestSet creates a slice of requests simulating an Envoy stream:
// 1. A Headers frame with standard Inference Extension headers.
// 2. A Body frame with the JSON payload.
func GenerateStreamedRequestSet(
	logger logr.Logger,
	prompt, model, targetModel string,
	filterMetadata []string,
) []*extProcPb.ProcessingRequest {
	requests := make([]*extProcPb.ProcessingRequest, 0, 2)

	// Headers
	requests = append(requests, generateHeaders(model, targetModel, filterMetadata, nil))

	// Body
	requests = append(requests, GenerateRequest(logger, prompt, model, filterMetadata))
	return requests
}

// GenerateStreamedRequestSet creates a slice of requests simulating an Envoy stream:
// 1. A Headers frame with standard Inference Extension headers.
// 2. A Body frame with the gRPC payload.
func GenerateStreamedGRPCRequestSet(
	logger logr.Logger,
	prompt string,
	inferenceObjective string, // Set to non-empty to set x-llm-d-inference-objective value
	filterMetadata []string,
	methodName string,
) []*extProcPb.ProcessingRequest {
	requests := make([]*extProcPb.ProcessingRequest, 0, 2)

	// Headers
	requests = append(requests, generateHeaders(inferenceObjective, "", filterMetadata, map[string]string{":path": methodName})) // GRPC payload does not need model and dose not support TargetModel.

	// Body
	requests = append(requests, GenerateGRPCRequest(logger, prompt, methodName, false, filterMetadata))
	return requests
}

func generateHeaders(inferenceObjective, targetModel string, filterMetadata []string, customHeaders map[string]string) *extProcPb.ProcessingRequest {
	headers := []*envoyCorev3.HeaderValue{
		{Key: "hi", Value: "mom"},
		{Key: reqcommon.RequestIDHeaderKey, Value: "test-request-id"},
	}
	if inferenceObjective != "" {
		headers = append(headers, &envoyCorev3.HeaderValue{Key: metadata.ObjectiveKey, Value: inferenceObjective})
	}
	if targetModel != "" {
		headers = append(headers, &envoyCorev3.HeaderValue{Key: metadata.ModelNameRewriteKey, Value: targetModel})
	}
	for k, v := range customHeaders {
		headers = append(headers, &envoyCorev3.HeaderValue{Key: k, Value: v})
	}

	return &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{Headers: headers},
			},
		},
		MetadataContext: &envoyCorev3.Metadata{
			FilterMetadata: GenerateRequestMetadata(filterMetadata),
		},
	}
}

// GenerateRequestMetadata constructs the Envoy Dynamic Metadata structure.
// This is primarily used to inject "envoy.lb" subset keys for testing logic that depends on specific backend subsets.
func GenerateRequestMetadata(filterMetadata []string) map[string]*structpb.Struct {
	requestMetadata := make(map[string]*structpb.Struct)
	interfaceList := make([]any, len(filterMetadata))
	for i, val := range filterMetadata {
		interfaceList[i] = val
	}
	if filterMetadata != nil {
		structVal, _ := structpb.NewStruct(map[string]any{
			metadata.SubsetFilterKey: interfaceList,
		})
		requestMetadata[metadata.SubsetFilterNamespace] = structVal
	}
	return requestMetadata
}

// --- Response Builders (Protocol Level) ---

// NewRequestBufferedResponse creates a complete set of responses for the Request phase.
// It simulates the EPP deciding to:
//  1. Modify headers (e.g., set destination endpoint).
//  2. Replace the entire request body (e.g., rewriting the model name).
//
// It returns two messages: one for the Header response and one for the Body response.
func NewRequestBufferedResponse(
	destinationEndpoint string,
	rewrittenBody []byte,
	otherHeaders ...*envoyCorev3.HeaderValueOption,
) []*extProcPb.ProcessingResponse {
	setHeaders := make([]*envoyCorev3.HeaderValueOption, 0, 2+len(otherHeaders))
	setHeaders = append(setHeaders,
		&envoyCorev3.HeaderValueOption{
			Header: &envoyCorev3.HeaderValue{
				Key:      metadata.DestinationEndpointKey,
				RawValue: []byte(destinationEndpoint),
			},
		},
		&envoyCorev3.HeaderValueOption{
			Header: &envoyCorev3.HeaderValue{
				Key:      headerKeyContentLength,
				RawValue: []byte(strconv.Itoa(len(rewrittenBody))),
			},
		})
	setHeaders = append(setHeaders, otherHeaders...)

	headerResponse := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: setHeaders,
					},
				},
			},
		},
		DynamicMetadata: makeDestinationMetadata(destinationEndpoint),
	}

	bodyResponse := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_StreamedResponse{
							StreamedResponse: &extProcPb.StreamedBodyResponse{
								Body:        rewrittenBody,
								EndOfStream: true,
							},
						},
					},
				},
			},
		},
	}

	return []*extProcPb.ProcessingResponse{headerResponse, bodyResponse}
}

// NewResponseBufferedResponse creates a complete set of responses for the Response phase.
// It simulates the EPP modifying the upstream response before sending it to the client.
// It returns a Header mutation message followed by a Body replacement message.
func NewResponseBufferedResponse(
	rewrittenBody string,
	eos bool,
	headersToSet ...*envoyCorev3.HeaderValueOption,
) []*extProcPb.ProcessingResponse {
	return []*extProcPb.ProcessingResponse{
		NewResponseHeaders(headersToSet...),
		NewResponseStreamChunk(rewrittenBody, eos),
	}
}

// NewResponseHeaders creates a single response message to modify response headers.
// Use this when testing header mutations without body changes, or as the first step in a streamed response test.
func NewResponseHeaders(headersToSet ...*envoyCorev3.HeaderValueOption) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: headersToSet,
					},
				},
			},
		},
	}
}

// NewResponseStreamChunk creates a single gRPC message representing one chunk of a streaming response.
// Use this to verify that EPP correctly passes through chunks (e.g., SSE events) or injects specific chunks.
func NewResponseStreamChunk(body string, endOfStream bool) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_StreamedResponse{
							StreamedResponse: &extProcPb.StreamedBodyResponse{
								Body:        []byte(body),
								EndOfStream: endOfStream,
							},
						},
					},
				},
			},
		},
	}
}

// NewImmediateErrorResponse creates a response that immediately terminates the request with a specific HTTP status code,
// body, and optional response headers.
// Use this for testing Load Shedding (503), Rate Limiting (429), or Bad Request (400) logic.
func NewImmediateErrorResponse(code envoyTypePb.StatusCode, body string, headers ...*envoyCorev3.HeaderValueOption) []*extProcPb.ProcessingResponse {
	immediateResponse := &extProcPb.ImmediateResponse{
		Status: &envoyTypePb.HttpStatus{Code: code},
		Body:   []byte(body),
	}
	if len(headers) > 0 {
		immediateResponse.Headers = &extProcPb.HeaderMutation{SetHeaders: headers}
	}
	return []*extProcPb.ProcessingResponse{{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: immediateResponse,
		},
	}}
}

// --- Internal Helpers ---

// makeDestinationMetadata helper to construct the Envoy dynamic metadata for routing.
func makeDestinationMetadata(endpoint string) *structpb.Struct {
	return &structpb.Struct{
		Fields: map[string]*structpb.Value{
			metadata.DestinationEndpointNamespace: {
				Kind: &structpb.Value_StructValue{
					StructValue: &structpb.Struct{
						Fields: map[string]*structpb.Value{
							metadata.DestinationEndpointKey: {
								Kind: &structpb.Value_StringValue{
									StringValue: endpoint,
								},
							},
						},
					},
				},
			},
		},
	}
}
