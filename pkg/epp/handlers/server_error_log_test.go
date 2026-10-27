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
	"io"
	"strings"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// rejectingDirector fails every request with err.
type rejectingDirector struct {
	mockDirector
	err error
}

func (d *rejectingDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	return reqCtx, d.err
}

// replayProcessServer replays reqs in order, then reports EOF.
type replayProcessServer struct {
	mockProcessServer
	ctx  context.Context
	reqs []*extProcPb.ProcessingRequest
}

func (m *replayProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if len(m.reqs) == 0 {
		return nil, io.EOF
	}
	req := m.reqs[0]
	m.reqs = m.reqs[1:]
	return req, nil
}

func (m *replayProcessServer) Context() context.Context { return m.ctx }

func TestProcessLogsRequestErrorOnce(t *testing.T) {
	const validBody = `{"model":"m","prompt":"hi"}`

	tests := []struct {
		name        string
		path        string
		body        string
		directorErr error
		wantStatus  envoyTypePb.StatusCode
	}{
		{
			name:       "parser not resolved",
			path:       "/v1/unclaimed",
			body:       validBody,
			wantStatus: envoyTypePb.StatusCode_BadRequest,
		},
		{
			name:       "body not parsed",
			path:       "/v1/completions",
			body:       `{`,
			wantStatus: envoyTypePb.StatusCode_BadRequest,
		},
		{
			name: "director rejects request",
			path: "/v1/completions",
			body: validBody,
			directorErr: errcommon.Error{
				Code: errcommon.ServiceUnavailable,
				Msg:  "no endpoints available for the given request",
			},
			wantStatus: envoyTypePb.StatusCode_ServiceUnavailable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var errorLines []string
			capture := funcr.New(func(prefix, args string) {
				if strings.Contains(args, `"error"=`) {
					errorLines = append(errorLines, prefix+" "+args)
				}
			}, funcr.Options{Verbosity: 2})

			srv := &replayProcessServer{
				ctx: log.IntoContext(context.Background(), capture),
				reqs: []*extProcPb.ProcessingRequest{
					newRequestHeaders(map[string]string{":path": tt.path, "x-request-id": "req-error-log"}),
					{
						Request: &extProcPb.ProcessingRequest_RequestBody{
							RequestBody: &extProcPb.HttpBody{Body: []byte(tt.body), EndOfStream: true},
						},
					},
				},
			}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			director := &rejectingDirector{err: tt.directorErr}

			require.NoError(t, NewStreamingServer(nil, director, registry, 0).Process(srv))

			require.Len(t, srv.sentResponses, 1)
			require.Equal(t, tt.wantStatus, srv.sentResponses[0].GetImmediateResponse().GetStatus().GetCode())
			require.Len(t, errorLines, 1, "a failed request must be logged at error level once: %v", errorLines)
		})
	}
}

func TestProcessRequestErrorOmitsBody(t *testing.T) {
	for _, tt := range []struct {
		name      string
		verbosity int
	}{
		{name: "debug", verbosity: logutil.DEBUG},
		{name: "trace", verbosity: logutil.TRACE},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var logged []string
			capture := funcr.New(func(prefix, args string) {
				logged = append(logged, prefix+" "+args)
			}, funcr.Options{Verbosity: tt.verbosity})
			body := []byte(`{"model":"m","messages":[{"role":"user","content":"private_prompt_content"},{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"private_tool_name","arguments":"private_tool_arguments"}}]}],"tools":[{"type":"function","function":{"name":"private_tool_name","parameters":{"type":"object","properties":{"private_schema_property":{"type":"string"}}}}}]}`)
			srv := &replayProcessServer{
				ctx: log.IntoContext(context.Background(), capture),
				reqs: []*extProcPb.ProcessingRequest{
					newRequestHeaders(map[string]string{":path": "/v1/chat/completions", "x-request-id": "req-private-error-log"}),
					{Request: &extProcPb.ProcessingRequest_RequestBody{
						RequestBody: &extProcPb.HttpBody{Body: body, EndOfStream: true},
					}},
				},
			}
			director := &rejectingDirector{err: errcommon.Error{
				Code: errcommon.ServiceUnavailable,
				Msg:  "no endpoints available for the given request",
			}}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())
			require.NoError(t, NewStreamingServer(nil, director, registry, 0).Process(srv))
			require.Len(t, srv.sentResponses, 1)
			require.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, srv.sentResponses[0].GetImmediateResponse().GetStatus().GetCode())

			output := strings.Join(logged, "\n")
			require.Equal(t, 1, strings.Count(output, "Failed to process request"))
			require.Contains(t, output, "no endpoints available for the given request")
			require.Contains(t, output, "req-private-error-log")
			for _, privateValue := range []string{"private_prompt_content", "private_tool_name", "private_tool_arguments", "private_schema_property"} {
				require.NotContains(t, output, privateValue)
			}
			require.NotContains(t, output, `"request"=`)
		})
	}
}
