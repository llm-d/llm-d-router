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
	"bytes"
	"context"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// answeringDirector schedules a pod and then answers the caller with err
// instead of forwarding. It records end-of-stream cleanups.
type answeringDirector struct {
	mockDirector
	err             error
	endOfStreamSeen int
	cause           fwkrc.TerminationCause
}

func (d *answeringDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	reqCtx.TargetPod = &fwkdl.EndpointMetadata{Address: "10.0.3.7", Port: "8000"}
	reqCtx.TargetEndpoint = "10.0.3.7:8000"
	return reqCtx, d.err
}

func (d *answeringDirector) HandleResponseBody(_ context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext {
	if endOfStream {
		d.endOfStreamSeen++
		d.cause = reqCtx.TerminationCause
	}
	return reqCtx
}

// TestProcessAnswersWithoutForwarding covers a request that the director
// answers after it picked a pod: the caller gets the status and headers of the
// error, nothing is forwarded, and the per-request plugin state of the picked
// pod is released. A NoContent answer, as for "Prefer: reserve-endpoint", has
// no body, is logged at info level and is not counted in request_error_total;
// a rejection carries its text, is logged at error level and is counted.
func TestProcessAnswersWithoutForwarding(t *testing.T) {
	tests := []struct {
		name       string
		code       string
		wantStatus envoyTypePb.StatusCode
		wantBody   string
		// wantErrorSeries is the number of request_error_total series.
		wantErrorSeries int
		// wantLogs are fragments the log output must carry; wantNoLogs are
		// fragments it must not. "ERROR" is the level of an error log line.
		wantLogs   []string
		wantNoLogs []string
	}{
		{
			name:       "NoContent answer",
			code:       errcommon.NoContent,
			wantStatus: envoyTypePb.StatusCode_NoContent,
			wantLogs:   []string{"Answered request without forwarding", `"reason": "answered"`},
			wantNoLogs: []string{"ERROR", "inference error"},
		},
		{
			name:            "PreconditionFailed rejection",
			code:            errcommon.PreconditionFailed,
			wantStatus:      envoyTypePb.StatusCode_PreconditionFailed,
			wantBody:        "inference error: PreconditionFailed - answered",
			wantErrorSeries: 1,
			wantLogs:        []string{"ERROR", "Failed to process request"},
			wantNoLogs:      []string{"Answered request without forwarding"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)

			director := &answeringDirector{err: errcommon.Error{
				Code:    tt.code,
				Msg:     "answered",
				Headers: map[string]string{routing.ReservedEndpointHeader: "10.0.3.7:8000"},
			}}
			server := NewStreamingServer(nil, director, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)

			var logs bytes.Buffer
			ctx := log.IntoContext(context.Background(), logutil.NewTestLoggerWithWriter(&logs))
			srv := &scriptedProcessServer{ctx: ctx, reqs: []*extProcPb.ProcessingRequest{
				newRequestHeaders(map[string]string{":path": "/v1/chat/completions", "x-request-id": "req-ask"}),
				{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
					Body:        []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`),
					EndOfStream: true,
				}}},
			}}

			require.NoError(t, server.Process(srv))

			require.Len(t, srv.sentResponses, 1, "the answer is the only response; nothing is forwarded")
			ir := srv.sentResponses[0].GetImmediateResponse()
			require.NotNil(t, ir)
			assert.Equal(t, tt.wantStatus, ir.Status.Code)
			assert.Equal(t, tt.wantBody, string(ir.Body))
			require.NotNil(t, ir.Headers)
			require.Len(t, ir.Headers.SetHeaders, 1)
			assert.Equal(t, routing.ReservedEndpointHeader, ir.Headers.SetHeaders[0].Header.Key)
			assert.Equal(t, "10.0.3.7:8000", string(ir.Headers.SetHeaders[0].Header.RawValue))
			assert.Equal(t, 1, director.endOfStreamSeen, "stream-end cleanup releases per-request plugin state")
			assert.Equal(t, fwkrc.TerminationCauseError, director.cause)
			errorSeries, err := promtestutil.GatherAndCount(ctrlmetrics.Registry, "llm_d_epp_request_error_total")
			require.NoError(t, err)
			assert.Equal(t, tt.wantErrorSeries, errorSeries, "request_error_total series")
			for _, want := range tt.wantLogs {
				assert.Contains(t, logs.String(), want)
			}
			for _, unwanted := range tt.wantNoLogs {
				assert.NotContains(t, logs.String(), unwanted)
			}
		})
	}
}
