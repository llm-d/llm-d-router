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

package handlers

import (
	"context"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/metadata"
)

func TestHandleRequestHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		headers        []*configPb.HeaderValue
		wantHeaders    map[string]string
		wantFairnessID string
		wantObjective  string
		wantTarget     string
	}{
		{
			name: "Extracts old Fairness ID alias",
			headers: []*configPb.HeaderValue{
				{Key: "x-test", Value: "val"},
				{Key: metadata.OldFlowFairnessIDKey, Value: "user-123"},
			},
			wantHeaders:    map[string]string{"x-test": "val"},
			wantFairnessID: "user-123",
		},
		{
			name: "Prefers RawValue over Value",
			headers: []*configPb.HeaderValue{
				{Key: metadata.FlowFairnessIDKey, RawValue: []byte("binary-id"), Value: "wrong-id"},
			},
			wantFairnessID: "binary-id",
		},
		{
			name: "Prefers new control headers over old aliases",
			headers: []*configPb.HeaderValue{
				{Key: metadata.OldFlowFairnessIDKey, Value: "old-user"},
				{Key: metadata.FlowFairnessIDKey, Value: "new-user"},
				{Key: metadata.OldObjectiveKey, Value: "old-objective"},
				{Key: metadata.ObjectiveKey, Value: "new-objective"},
				{Key: metadata.OldModelNameRewriteKey, Value: "old-model"},
				{Key: metadata.ModelNameRewriteKey, Value: "new-model"},
			},
			wantFairnessID: "new-user",
			wantObjective:  "new-objective",
			wantTarget:     "new-model",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{}
			reqCtx := &RequestContext{
				Request: &Request{Headers: make(map[string]string)},
			}
			req := &extProcPb.ProcessingRequest_RequestHeaders{
				RequestHeaders: &extProcPb.HttpHeaders{
					Headers: &configPb.HeaderMap{Headers: tc.headers},
				},
			}

			err := server.HandleRequestHeaders(context.Background(), reqCtx, req)
			assert.NoError(t, err, "HandleRequestHeaders should not return an error")

			assert.Equal(t, tc.wantFairnessID, reqCtx.FairnessID, "FairnessID should match expected value")
			assert.Equal(t, tc.wantObjective, reqCtx.ObjectiveKey, "ObjectiveKey should match expected value")
			assert.Equal(t, tc.wantTarget, reqCtx.TargetModelName, "TargetModelName should match expected value")

			if tc.wantHeaders != nil {
				for k, v := range tc.wantHeaders {
					assert.Equal(t, v, reqCtx.Request.Headers[k], "Header %q should match expected value", k)
				}
			}
		})
	}
}

func TestGenerateHeaders_Sanitization(t *testing.T) {
	server := &StreamingServer{}
	reqCtx := &RequestContext{
		TargetEndpoint: "1.2.3.4:8080",
		RequestSize:    123,
		Request: &Request{
			Headers: map[string]string{
				"x-user-data":                      "important",                  // should passthrough
				metadata.ObjectiveKey:              "sensitive-objective-id",     // should be stripped
				metadata.OldObjectiveKey:           "old-sensitive-objective-id", // should be stripped
				metadata.DestinationEndpointKey:    "1.1.1.1:666",                // should be stripped
				metadata.OldDestinationEndpointKey: "1.1.1.1:777",                // should be stripped
				"content-length":                   "99999",                      // should be stripped (re-added by logic)
			},
		},
	}

	results := server.generateHeaders(context.Background(), reqCtx)

	gotHeaders := make(map[string]string)
	for _, h := range results {
		gotHeaders[h.Header.Key] = string(h.Header.RawValue)
	}

	assert.Contains(t, gotHeaders, "x-user-data")
	assert.NotContains(t, gotHeaders, metadata.ObjectiveKey)
	assert.NotContains(t, gotHeaders, metadata.OldObjectiveKey)
	assert.Equal(t, "1.2.3.4:8080", gotHeaders[metadata.DestinationEndpointKey])
	assert.Equal(t, "1.2.3.4:8080", gotHeaders[metadata.OldDestinationEndpointKey])
	assert.Equal(t, "123", gotHeaders["Content-Length"])
}

func TestGenerateRequestHeaderResponse_MergeMetadata(t *testing.T) {
	t.Parallel()

	server := &StreamingServer{}
	reqCtx := &RequestContext{
		TargetEndpoint: "1.2.3.4:8080",
		Request: &Request{
			Headers: make(map[string]string),
		},
		Response: &Response{
			DynamicMetadata: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"existing_namespace": {
						Kind: &structpb.Value_StructValue{
							StructValue: &structpb.Struct{
								Fields: map[string]*structpb.Value{
									"existing_key": {Kind: &structpb.Value_StringValue{StringValue: "existing_value"}},
								},
							},
						},
					},
				},
			},
		},
	}

	resp := server.generateRequestHeaderResponse(context.Background(), reqCtx)

	// Check that the existing metadata is preserved
	existingNamespace, ok := resp.DynamicMetadata.Fields["existing_namespace"]
	assert.True(t, ok, "Expected existing_namespace to be in DynamicMetadata")
	existingKey, ok := existingNamespace.GetStructValue().Fields["existing_key"]
	assert.True(t, ok, "Expected existing_key to be in existing_namespace")
	assert.Equal(t, "existing_value", existingKey.GetStringValue(), "Unexpected value for existing_key")

	// Check that the new metadata is added
	endpointNamespace, ok := resp.DynamicMetadata.Fields[metadata.DestinationEndpointNamespace]
	assert.True(t, ok, "Expected DestinationEndpointNamespace to be in DynamicMetadata")
	endpointKey, ok := endpointNamespace.GetStructValue().Fields[metadata.DestinationEndpointKey]
	assert.True(t, ok, "Expected DestinationEndpointKey to be in DestinationEndpointNamespace")
	assert.Equal(t, "1.2.3.4:8080", endpointKey.GetStringValue(), "Unexpected value for DestinationEndpointKey")
	oldEndpointKey, ok := endpointNamespace.GetStructValue().Fields[metadata.OldDestinationEndpointKey]
	assert.True(t, ok, "Expected old DestinationEndpointKey alias to be in DestinationEndpointNamespace")
	assert.Equal(t, "1.2.3.4:8080", oldEndpointKey.GetStringValue(), "Unexpected value for old DestinationEndpointKey alias")
}

func TestFallbackToRandomEndpoint(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		requestSize     int
		wantBodyRespLen int
	}{
		{
			name:            "No body",
			requestSize:     0,
			wantBodyRespLen: 0,
		},
		{
			name:            "With body",
			requestSize:     9,
			wantBodyRespLen: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := &StreamingServer{
				director: &mockDirectorRequest{},
			}
			reqCtx := &RequestContext{
				Request:  &Request{Headers: make(map[string]string), RawBody: []byte("test body")},
				Response: &Response{Headers: make(map[string]string)},
			}

			err := server.fallbackToRandomEndpoint(context.Background(), reqCtx, tc.requestSize)
			assert.NoError(t, err)

			if tc.wantBodyRespLen > 0 {
				assert.NotNil(t, reqCtx.reqBodyResp)
				assert.Len(t, reqCtx.reqBodyResp, tc.wantBodyRespLen)
				bodyResp := reqCtx.reqBodyResp[0].GetRequestBody().GetResponse()
				assert.NotNil(t, bodyResp.BodyMutation)
				streamedResp := bodyResp.BodyMutation.GetStreamedResponse()
				assert.NotNil(t, streamedResp)
				assert.Equal(t, []byte("test body"), streamedResp.Body)
			} else {
				assert.Nil(t, reqCtx.reqBodyResp)
			}
		})
	}
}

type mockDirectorRequest struct {
	Director
}

func (m *mockDirectorRequest) GetRandomEndpoint() *datalayer.EndpointMetadata {
	return &datalayer.EndpointMetadata{
		Address: "1.2.3.4",
		Port:    "80",
	}
}
