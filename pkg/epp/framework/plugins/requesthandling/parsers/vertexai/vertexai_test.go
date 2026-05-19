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

package vertexai

import (
	"context"
	"encoding/binary"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"google.golang.org/genproto/googleapis/api/httpbody"
	"google.golang.org/protobuf/proto"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

func TestParseRequest(t *testing.T) {
	parser := NewVertexAIParser()

	jsonPayload := []byte(`{"messages":[{"role":"user","content":"Hello"}],"stream":true}`)

	reqMsg := &aiplatformpb.ChatCompletionsRequest{
		HttpBody: &httpbody.HttpBody{
			Data: jsonPayload,
		},
	}

	validBody, err := createGrpcFrame(reqMsg)
	if err != nil {
		t.Fatalf("Failed to create gRPC frame: %v", err)
	}

	tests := []struct {
		name       string
		body       []byte
		headers    map[string]string
		wantErr    string
		wantResult *fwkrh.ParseResult
	}{
		{
			name: "Success - ChatCompletions",
			body: validBody,
			headers: map[string]string{
				":path":        "/google.cloud.aiplatform.v1beta1.PredictionService/ChatCompletions",
				"content-type": "application/grpc",
			},
			wantResult: &fwkrh.ParseResult{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{Role: "user", Content: fwkrh.Content{Raw: "Hello"}},
						},
					},
					Stream:  true,
					Payload: fwkrh.PayloadProto{Message: reqMsg},
				},
				Skip: false,
			},
		},
		{
			name:    "Unsupported Path",
			body:    []byte{},
			headers: map[string]string{":path": "/unsupported/path", "content-type": "application/grpc"},
			wantResult: &fwkrh.ParseResult{
				Skip: true,
			},
		},
		{
			name:    "Invalid gRPC frame",
			body:    []byte{0, 0, 0, 0}, // Too short
			headers: map[string]string{":path": "/google.cloud.aiplatform.v1beta1.PredictionService/ChatCompletions", "content-type": "application/grpc"},
			wantErr: "invalid gRPC frame",
		},
		{
			name:    "Invalid proto message",
			body:    []byte{0, 0, 0, 0, 1, 0xFF}, // Valid header, invalid payload
			headers: map[string]string{":path": "/google.cloud.aiplatform.v1beta1.PredictionService/ChatCompletions", "content-type": "application/grpc"},
			wantErr: "unmarshaling ChatCompletionsRequest",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := parser.ParseRequest(context.Background(), tc.body, tc.headers)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatal("Expected error, got nil")
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("Expected error containing %q, got %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseRequest failed: %v", err)
			}

			if diff := cmp.Diff(tc.wantResult, res, protocmp.Transform()); diff != "" {
				t.Errorf("ParseResult mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func createGrpcFrame(msg proto.Message) ([]byte, error) {
	payload, err := proto.Marshal(msg)
	if err != nil {
		return nil, err
	}
	return createGrpcFrameRaw(payload)
}

func TestParseResponse(t *testing.T) {
	parser := NewVertexAIParser()

	jsonPayload := []byte(`{"object":"chat.completion","usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`)
	httpBody := &httpbody.HttpBody{
		Data: jsonPayload,
	}
	httpBodyBytes, err := proto.Marshal(httpBody)
	if err != nil {
		t.Fatalf("Failed to marshal HttpBody: %v", err)
	}
	validBody, err := createGrpcFrameRaw(httpBodyBytes)
	if err != nil {
		t.Fatalf("Failed to create gRPC frame: %v", err)
	}

	invalidProtoBody, _ := createGrpcFrameRaw([]byte{0xFF})

	tests := []struct {
		name       string
		body       []byte
		headers    map[string]string
		wantErr    string
		wantResult *fwkrh.ParsedResponse
	}{
		{
			name:       "Empty body",
			body:       []byte{},
			headers:    nil,
			wantResult: nil,
		},
		{
			name:    "Valid JSON response",
			body:    validBody,
			headers: map[string]string{"content-type": "application/grpc"},
			wantResult: &fwkrh.ParsedResponse{
				Usage: &fwkrh.Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
			},
		},
		{
			name:    "Invalid gRPC frame",
			body:    []byte{0, 0, 0, 0},
			headers: map[string]string{"content-type": "application/grpc"},
			wantErr: "invalid gRPC frame",
		},
		{
			name:    "Invalid proto message",
			body:    invalidProtoBody,
			headers: map[string]string{"content-type": "application/grpc"},
			wantErr: "unmarshaling HttpBody response",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := parser.ParseResponse(context.Background(), tc.body, tc.headers, false)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatal("Expected error, got nil")
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("Expected error containing %q, got %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseResponse failed: %v", err)
			}

			if diff := cmp.Diff(tc.wantResult, resp, protocmp.Transform()); diff != "" {
				t.Errorf("ParsedResponse mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestVertexAIParser_Metadata(t *testing.T) {
	parser := NewVertexAIParser()

	typedName := parser.TypedName()
	if typedName.Type != VertexAIParserType {
		t.Errorf("Expected type %s, got %s", VertexAIParserType, typedName.Type)
	}
	if typedName.Name != VertexAIParserType {
		t.Errorf("Expected name %s, got %s", VertexAIParserType, typedName.Name)
	}

	protocols := parser.SupportedAppProtocols()
	if len(protocols) != 1 || protocols[0] != v1.AppProtocolH2C {
		t.Errorf("Expected protocols [h2c], got %v", protocols)
	}
}

func createGrpcFrameRaw(payload []byte) ([]byte, error) {
	frame := make([]byte, 5+len(payload))
	frame[0] = 0 // uncompressed
	binary.BigEndian.PutUint32(frame[1:], uint32(len(payload)))
	copy(frame[5:], payload)
	return frame, nil
}
