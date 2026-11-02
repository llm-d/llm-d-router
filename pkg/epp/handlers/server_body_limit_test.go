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

package handlers

import (
	"context"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

func TestProcessRejectsOversizedRequestBody(t *testing.T) {
	const body = `{"model":"m","prompt":"hi"}`
	limit := len(body)

	// The director rejects every request it sees with 503, so a 503 proves the body passed
	// the size check and a 413 proves it never reached the director.
	tests := []struct {
		name        string
		path        string
		limit       int
		chunks      []string
		endOfStream bool
		wantStatus  envoyTypePb.StatusCode
		wantBody    string
	}{
		{
			name:        "body at the limit reaches the director",
			path:        "/v1/completions",
			limit:       limit,
			chunks:      []string{body[:10], body[10:]},
			endOfStream: true,
			wantStatus:  envoyTypePb.StatusCode_ServiceUnavailable,
		},
		{
			name:        "body over the limit is rejected before end of stream",
			path:        "/v1/completions",
			limit:       limit,
			chunks:      []string{body[:10], body[10:] + " "},
			endOfStream: false,
			wantStatus:  envoyTypePb.StatusCode_PayloadTooLarge,
			wantBody:    `{"error":{"message":"request body exceeds the 27-byte limit","type":"invalid_request_error","code":413}}`,
		},
		{
			name:        "Messages API uses the Anthropic envelope",
			path:        "/v1/messages",
			limit:       limit,
			chunks:      []string{body + " "},
			endOfStream: true,
			wantStatus:  envoyTypePb.StatusCode_PayloadTooLarge,
			wantBody:    `{"type":"error","error":{"type":"request_too_large","message":"request body exceeds the 27-byte limit"}}`,
		},
		{
			name:        "zero disables the limit",
			path:        "/v1/completions",
			limit:       0,
			chunks:      []string{body + " "},
			endOfStream: true,
			wantStatus:  envoyTypePb.StatusCode_ServiceUnavailable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)

			reqs := []*extProcPb.ProcessingRequest{newRequestHeaders(map[string]string{":path": tt.path})}
			for i, chunk := range tt.chunks {
				reqs = append(reqs, &extProcPb.ProcessingRequest{
					Request: &extProcPb.ProcessingRequest_RequestBody{
						RequestBody: &extProcPb.HttpBody{
							Body:        []byte(chunk),
							EndOfStream: tt.endOfStream && i == len(tt.chunks)-1,
						},
					},
				})
			}
			srv := &replayProcessServer{ctx: logutil.NewTestLoggerIntoContext(context.Background()), reqs: reqs}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			director := &rejectingDirector{err: errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "no endpoints"}}

			s := NewStreamingServer(nil, director, registry, 0)
			s.SetMaxRequestBodySize(tt.limit)
			require.NoError(t, s.Process(srv))

			require.Len(t, srv.sentResponses, 1)
			resp := srv.sentResponses[0].GetImmediateResponse()
			require.Equal(t, tt.wantStatus, resp.GetStatus().GetCode())
			if tt.wantBody != "" {
				require.JSONEq(t, tt.wantBody, string(resp.GetBody()))
			}
			wantCount := 0.0
			if tt.wantStatus == envoyTypePb.StatusCode_PayloadTooLarge {
				wantCount = 1
			}
			require.Equal(t, wantCount, requestErrorCount(t, errcommon.PayloadTooLarge))
		})
	}
}
