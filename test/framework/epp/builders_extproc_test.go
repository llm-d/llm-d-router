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
	"encoding/binary"
	"encoding/json"
	"strconv"
	"testing"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	pb "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func setHeader(key, value string) *envoyCorev3.HeaderValueOption {
	return &envoyCorev3.HeaderValueOption{
		Header: &envoyCorev3.HeaderValue{Key: key, RawValue: []byte(value)},
	}
}

func TestResponseBuilders(t *testing.T) {
	t.Run("NewResponseHeaders", func(t *testing.T) {
		tests := []struct {
			name    string
			headers []*envoyCorev3.HeaderValueOption
		}{
			{name: "empty header list"},
			{name: "single header", headers: []*envoyCorev3.HeaderValueOption{setHeader("x-a", "1")}},
			{
				name:    "multiple headers",
				headers: []*envoyCorev3.HeaderValueOption{setHeader("x-a", "1"), setHeader("x-b", "2")},
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := NewResponseHeaders(tc.headers...)

				resp := got.GetResponseHeaders()
				require.NotNil(t, resp, "builder must populate the ResponseHeaders oneof")
				assert.Equal(t, tc.headers, resp.GetResponse().GetHeaderMutation().GetSetHeaders())
			})
		}
	})

	t.Run("NewResponseStreamChunk", func(t *testing.T) {
		tests := []struct {
			name        string
			body        string
			endOfStream bool
		}{
			{name: "empty body chunk", body: "", endOfStream: false},
			{name: "empty body chunk at end of stream", body: "", endOfStream: true},
			{name: "chunk mid stream", body: "data: hello\n\n", endOfStream: false},
			{name: "chunk at end of stream", body: "data: [DONE]\n\n", endOfStream: true},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := NewResponseStreamChunk(tc.body, tc.endOfStream)

				streamed := got.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
				require.NotNil(t, streamed, "builder must populate the streamed body mutation")
				assert.Equal(t, []byte(tc.body), streamed.GetBody())
				assert.Equal(t, tc.endOfStream, streamed.GetEndOfStream())
			})
		}
	})

	t.Run("NewImmediateErrorResponse", func(t *testing.T) {
		tests := []struct {
			name string
			code envoyTypePb.StatusCode
			body string
		}{
			{name: "service unavailable", code: envoyTypePb.StatusCode_ServiceUnavailable, body: "load shed"},
			{name: "too many requests", code: envoyTypePb.StatusCode_TooManyRequests, body: "rate limited"},
			{name: "bad request with empty body", code: envoyTypePb.StatusCode_BadRequest, body: ""},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := NewImmediateErrorResponse(tc.code, tc.body)

				require.Len(t, got, 1)
				immediate := got[0].GetImmediateResponse()
				require.NotNil(t, immediate, "builder must populate the ImmediateResponse oneof")
				assert.Equal(t, tc.code, immediate.GetStatus().GetCode())
				assert.Equal(t, []byte(tc.body), immediate.GetBody())
			})
		}
	})
}

// headerMap flattens the header frame of a request into key -> value.
func headerMap(t *testing.T, req *extProcPb.ProcessingRequest) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, h := range req.GetRequestHeaders().GetHeaders().GetHeaders() {
		out[h.GetKey()] = h.GetValue()
	}
	return out
}

// bodyJSON decodes the JSON body frame of a request.
func bodyJSON(t *testing.T, req *extProcPb.ProcessingRequest) map[string]any {
	t.Helper()
	out := map[string]any{}
	require.NoError(t, json.Unmarshal(req.GetRequestBody().GetBody(), &out))
	return out
}

func TestRequestBuilders(t *testing.T) {
	logger := logr.Discard()

	t.Run("ReqLLM", func(t *testing.T) {
		tests := []struct {
			name        string
			model       string
			targetModel string
			wantHeaders map[string]string
			wantModel   bool
		}{
			{
				name:        "objective and target model",
				model:       "food-review",
				targetModel: "food-review-v1",
				wantHeaders: map[string]string{
					metadata.ObjectiveKey:        "food-review",
					metadata.ModelNameRewriteKey: "food-review-v1",
				},
				wantModel: true,
			},
			{
				name:        "objective only",
				model:       "food-review",
				wantHeaders: map[string]string{metadata.ObjectiveKey: "food-review"},
				wantModel:   true,
			},
			{name: "no model at all"},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := ReqLLM(logger, "hello", tc.model, tc.targetModel)

				require.Len(t, got, 2, "a headers frame followed by a body frame")
				hdrs := headerMap(t, got[0])
				assert.Equal(t, "test-request-id", hdrs[reqcommon.RequestIDHeaderKey])
				for k, v := range tc.wantHeaders {
					assert.Equal(t, v, hdrs[k])
				}
				if tc.model == "" {
					assert.NotContains(t, hdrs, metadata.ObjectiveKey)
				}
				if tc.targetModel == "" {
					assert.NotContains(t, hdrs, metadata.ModelNameRewriteKey)
				}

				body := bodyJSON(t, got[1])
				assert.Equal(t, "hello", body["prompt"])
				assert.True(t, got[1].GetRequestBody().GetEndOfStream())
				if tc.wantModel {
					assert.Equal(t, tc.model, body["model"])
				} else {
					assert.NotContains(t, body, "model")
				}
			})
		}
	})

	t.Run("ReqRaw", func(t *testing.T) {
		tests := []struct {
			name       string
			headers    map[string]string
			bodyChunks []string
		}{
			{name: "headers only", headers: map[string]string{"x-a": "1"}},
			{name: "no headers, single chunk", bodyChunks: []string{"{}"}},
			{name: "fragmented body", headers: map[string]string{"x-a": "1"}, bodyChunks: []string{`{"pro`, `mpt":"hi"}`}},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := ReqRaw(tc.headers, tc.bodyChunks...)

				require.Len(t, got, 1+len(tc.bodyChunks))
				assert.Len(t, got[0].GetRequestHeaders().GetHeaders().GetHeaders(), len(tc.headers))
				assert.False(t, got[0].GetRequestHeaders().GetEndOfStream())
				for i, chunk := range tc.bodyChunks {
					frame := got[i+1].GetRequestBody()
					assert.Equal(t, []byte(chunk), frame.GetBody())
					assert.Equal(t, i == len(tc.bodyChunks)-1, frame.GetEndOfStream(), "only the last chunk ends the stream")
				}
			})
		}
	})

	t.Run("ReqHeaderOnly", func(t *testing.T) {
		tests := []struct {
			name    string
			headers map[string]string
		}{
			{name: "empty headers"},
			{name: "get request", headers: map[string]string{":method": "GET", ":path": "/healthz"}},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := ReqHeaderOnly(tc.headers)

				require.Len(t, got, 1)
				frame := got[0].GetRequestHeaders()
				assert.True(t, frame.GetEndOfStream(), "a header-only request terminates the stream")
				hdrs := headerMap(t, got[0])
				require.Len(t, hdrs, len(tc.headers), "exactly the requested headers, no extras")
				for k, v := range tc.headers {
					assert.Equal(t, v, hdrs[k], "header %q", k)
				}
			})
		}
	})

	t.Run("gRPC request sets", func(t *testing.T) {
		tests := []struct {
			name       string
			build      func() []*extProcPb.ProcessingRequest
			methodName string
			wantStream bool
		}{
			{
				name:       "unary generate",
				build:      func() []*extProcPb.ProcessingRequest { return ReqGRPCLLM(logger, "hi", "obj", GenerateGRPCMethodName) },
				methodName: GenerateGRPCMethodName,
			},
			{
				name:       "unary embed",
				build:      func() []*extProcPb.ProcessingRequest { return ReqGRPCLLM(logger, "hi", "", EmbedGRPCMethodName) },
				methodName: EmbedGRPCMethodName,
			},
			{
				name: "streamed generate",
				build: func() []*extProcPb.ProcessingRequest {
					return ReqGRPCLLMWithStream(logger, "hi", "obj", GenerateGRPCMethodName)
				},
				methodName: GenerateGRPCMethodName,
				wantStream: true,
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := tc.build()

				require.Len(t, got, 2)
				assert.Equal(t, tc.methodName, headerMap(t, got[0])[":path"])

				payload := got[1].GetRequestBody().GetBody()
				require.Greater(t, len(payload), 5, "gRPC framing prefix plus a message")
				assert.Equal(t, byte(0), payload[0], "uncompressed flag")
				assert.Equal(t, uint32(len(payload)-5), binary.BigEndian.Uint32(payload[1:5]))

				if tc.methodName == GenerateGRPCMethodName {
					msg := &pb.GenerateRequest{}
					require.NoError(t, proto.Unmarshal(payload[5:], msg))
					assert.Equal(t, "hi", msg.GetText())
					assert.Equal(t, tc.wantStream, msg.GetStream())
				}
			})
		}
	})

	t.Run("GRPCRequestProto", func(t *testing.T) {
		tests := []struct {
			name       string
			methodName string
			stream     bool
			want       proto.Message // nil means the builder must produce no message
		}{
			{
				name:       "generate carries the prompt as text",
				methodName: GenerateGRPCMethodName,
				want:       &pb.GenerateRequest{Input: &pb.GenerateRequest_Text{Text: "hi"}},
			},
			{
				name:       "generate propagates the stream flag",
				methodName: GenerateGRPCMethodName,
				stream:     true,
				want:       &pb.GenerateRequest{Input: &pb.GenerateRequest_Text{Text: "hi"}, Stream: true},
			},
			{
				// Embed has no stream field: the flag must not leak into the message.
				name:       "embed carries the prompt as tokenized original text",
				methodName: EmbedGRPCMethodName,
				stream:     true,
				want:       &pb.EmbedRequest{Tokenized: &pb.TokenizedInput{OriginalText: "hi"}},
			},
			{name: "unknown method yields no message", methodName: "/unknown/Method"},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := GRPCRequestProto("hi", tc.methodName, tc.stream)
				if tc.want == nil {
					assert.Nil(t, got)
					return
				}
				require.NotNil(t, got)
				assert.True(t, proto.Equal(tc.want, got), "want %v, got %v", tc.want, got)
			})
		}
	})

	t.Run("CreateGrpcPayload", func(t *testing.T) {
		payload, err := CreateGrpcPayload(&pb.GenerateRequest{Input: &pb.GenerateRequest_Text{Text: "hi"}})

		require.NoError(t, err)
		require.Greater(t, len(payload), 5)
		assert.Equal(t, byte(0), payload[0])
		assert.Equal(t, uint32(len(payload)-5), binary.BigEndian.Uint32(payload[1:5]))
	})

	t.Run("GenerateRequestMetadata", func(t *testing.T) {
		tests := []struct {
			name           string
			filterMetadata []string
			wantSubset     bool
		}{
			{name: "nil metadata yields empty map"},
			{name: "empty non-nil slice still sets the namespace", filterMetadata: []string{}, wantSubset: true},
			{name: "subset endpoints", filterMetadata: []string{"1.2.3.4:8000", "5.6.7.8:8000"}, wantSubset: true},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := GenerateRequestMetadata(tc.filterMetadata)

				if !tc.wantSubset {
					assert.Empty(t, got)
					return
				}
				subset := got[metadata.SubsetFilterNamespace]
				require.NotNil(t, subset)
				assert.Len(t, subset.GetFields()[metadata.SubsetFilterKey].GetListValue().GetValues(), len(tc.filterMetadata))
			})
		}
	})

	t.Run("GenerateStreamedGRPCRequestSet carries filter metadata", func(t *testing.T) {
		got := GenerateStreamedGRPCRequestSet(logger, "hi", "obj", []string{"1.2.3.4:8000"}, GenerateGRPCMethodName)

		require.Len(t, got, 2)
		for _, req := range got {
			assert.Contains(t, req.GetMetadataContext().GetFilterMetadata(), metadata.SubsetFilterNamespace)
		}
	})
}

func TestBufferedResponseBuilders(t *testing.T) {
	t.Run("NewRequestBufferedResponse", func(t *testing.T) {
		tests := []struct {
			name         string
			body         []byte
			otherHeaders []*envoyCorev3.HeaderValueOption
		}{
			{name: "empty body", body: []byte{}},
			{name: "rewritten body", body: []byte(`{"model":"v1"}`)},
			{name: "with extra headers", body: []byte(`{}`), otherHeaders: []*envoyCorev3.HeaderValueOption{setHeader("x-a", "1")}},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := NewRequestBufferedResponse("1.2.3.4:8000", tc.body, tc.otherHeaders...)

				require.Len(t, got, 2, "a header response followed by a body response")

				hdrResp := got[0].GetRequestHeaders().GetResponse()
				assert.True(t, hdrResp.GetClearRouteCache())
				set := hdrResp.GetHeaderMutation().GetSetHeaders()
				require.Len(t, set, 2+len(tc.otherHeaders))
				assert.Equal(t, metadata.DestinationEndpointKey, set[0].GetHeader().GetKey())
				assert.Equal(t, []byte("1.2.3.4:8000"), set[0].GetHeader().GetRawValue())
				assert.Equal(t, []byte(strconv.Itoa(len(tc.body))), set[1].GetHeader().GetRawValue())

				// The routing decision is also mirrored into Envoy dynamic metadata.
				ns := got[0].GetDynamicMetadata().GetFields()[metadata.DestinationEndpointNamespace]
				require.NotNil(t, ns)
				assert.Equal(t, "1.2.3.4:8000",
					ns.GetStructValue().GetFields()[metadata.DestinationEndpointKey].GetStringValue())

				streamed := got[1].GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse()
				assert.Equal(t, tc.body, streamed.GetBody())
				assert.True(t, streamed.GetEndOfStream())
			})
		}
	})

	t.Run("NewResponseBufferedResponse", func(t *testing.T) {
		tests := []struct {
			name string
			eos  bool
		}{
			{name: "mid stream", eos: false},
			{name: "end of stream", eos: true},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := NewResponseBufferedResponse("chunk", tc.eos, setHeader("x-a", "1"))

				require.Len(t, got, 2)
				assert.Len(t, got[0].GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), 1)
				streamed := got[1].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
				assert.Equal(t, []byte("chunk"), streamed.GetBody())
				assert.Equal(t, tc.eos, streamed.GetEndOfStream())
			})
		}
	})
}
