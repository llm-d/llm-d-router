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
	"errors"
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

// answeringDirector schedules a pod and then either answers the caller with
// answer or fails with err, instead of forwarding. It records end-of-stream
// cleanups.
type answeringDirector struct {
	mockDirector
	answer          *Answer
	err             error
	endOfStreamSeen int
	cause           fwkrc.TerminationCause
}

func (d *answeringDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	reqCtx.TargetPod = &fwkdl.EndpointMetadata{Address: "10.0.3.7", Port: "8000"}
	reqCtx.TargetEndpoint = "10.0.3.7:8000"
	reqCtx.Answer = d.answer
	return reqCtx, d.err
}

func (d *answeringDirector) HandleResponseBody(_ context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext {
	if endOfStream {
		d.endOfStreamSeen++
		d.cause = reqCtx.TerminationCause
	}
	return reqCtx
}

// TestProcessAnswersWithoutForwarding covers a request the director does not
// forward after it picked a pod. An answer, as for "Prefer: reserve-endpoint",
// reaches the caller with the status and headers the director chose and no
// body, is logged at info level, is counted in neither request_total nor
// request_error_total, and releases the per-request plugin state of the
// picked pod with the answered cause. A rejection reaches the caller as a
// JSON error with its headers, is logged at error level, and is counted in
// request_error_total.
func TestProcessAnswersWithoutForwarding(t *testing.T) {
	answer := map[string]string{
		routing.ReservedEndpointHeader:  "10.0.3.7:8000",
		routing.PreferenceAppliedHeader: routing.PreferReserveEndpoint,
		routing.KVCacheSourceHeader:     "10.0.3.9:8000",
	}
	rejection := map[string]string{routing.ReservedEndpointHeader: "10.0.3.7:8000"}

	tests := []struct {
		name        string
		director    *answeringDirector
		wantStatus  envoyTypePb.StatusCode
		wantBody    string
		wantHeaders map[string]string
		wantCause   fwkrc.TerminationCause
		// wantErrorSeries is the number of request_error_total series.
		wantErrorSeries int
		// wantLogs are fragments the log output must carry; wantNoLogs are
		// fragments it must not. "ERROR" is the level of an error log line.
		wantLogs   []string
		wantNoLogs []string
	}{
		{
			name:        "answer",
			director:    &answeringDirector{answer: &Answer{Status: envoyTypePb.StatusCode_NoContent, Headers: answer}},
			wantStatus:  envoyTypePb.StatusCode_NoContent,
			wantHeaders: answer,
			wantCause:   fwkrc.TerminationCauseAnswered,
			wantLogs:    []string{"Answered request without forwarding", `"targetEndpoint": "10.0.3.7:8000"`},
			wantNoLogs:  []string{"ERROR", "inference error"},
		},
		{
			name:        "answer carries the director's status",
			director:    &answeringDirector{answer: &Answer{Status: envoyTypePb.StatusCode_OK, Headers: answer}},
			wantStatus:  envoyTypePb.StatusCode_OK,
			wantHeaders: answer,
			wantCause:   fwkrc.TerminationCauseAnswered,
			wantLogs:    []string{"Answered request without forwarding"},
			wantNoLogs:  []string{"ERROR"},
		},
		{
			name:        "answer without headers",
			director:    &answeringDirector{answer: &Answer{Status: envoyTypePb.StatusCode_NoContent}},
			wantStatus:  envoyTypePb.StatusCode_NoContent,
			wantHeaders: map[string]string{},
			wantCause:   fwkrc.TerminationCauseAnswered,
			wantLogs:    []string{"Answered request without forwarding"},
			wantNoLogs:  []string{"ERROR"},
		},
		{
			name: "rejection",
			director: &answeringDirector{err: errcommon.Error{
				Code:    errcommon.PreconditionFailed,
				Msg:     "rejected",
				Headers: rejection,
			}},
			wantStatus:      envoyTypePb.StatusCode_PreconditionFailed,
			wantBody:        `{"error":{"message":"rejected","type":"invalid_request_error","code":412}}`,
			wantHeaders:     map[string]string{"content-type": "application/json", routing.ReservedEndpointHeader: "10.0.3.7:8000"},
			wantCause:       fwkrc.TerminationCauseError,
			wantErrorSeries: 1,
			wantLogs:        []string{"ERROR", "Failed to process request"},
			wantNoLogs:      []string{"Answered request without forwarding"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetEPPMetrics(t)
			server := newAnswerServer(tt.director)

			var logs bytes.Buffer
			ctx := log.IntoContext(context.Background(), logutil.NewTestLoggerWithWriter(&logs))
			srv := &scriptedProcessServer{ctx: ctx, reqs: answerRequests()}

			require.NoError(t, server.Process(srv))

			require.Len(t, srv.sentResponses, 1, "the immediate response is the only response; nothing is forwarded")
			ir := srv.sentResponses[0].GetImmediateResponse()
			require.NotNil(t, ir)
			assert.Equal(t, tt.wantStatus, ir.Status.Code)
			assert.Equal(t, tt.wantBody, string(ir.Body))
			assert.Equal(t, tt.wantHeaders, setHeadersMap(ir.GetHeaders().GetSetHeaders()))
			assert.Equal(t, 1, tt.director.endOfStreamSeen, "stream-end cleanup releases per-request plugin state")
			assert.Equal(t, tt.wantCause, tt.director.cause)
			errorSeries, err := promtestutil.GatherAndCount(ctrlmetrics.Registry, "llm_d_epp_request_error_total")
			require.NoError(t, err)
			assert.Equal(t, tt.wantErrorSeries, errorSeries, "request_error_total series")
			totalSeries, err := promtestutil.GatherAndCount(ctrlmetrics.Registry, "llm_d_epp_request_total")
			require.NoError(t, err)
			assert.Equal(t, 0, totalSeries, "request_total does not count a request that was not forwarded")
			for _, want := range tt.wantLogs {
				assert.Contains(t, logs.String(), want)
			}
			for _, unwanted := range tt.wantNoLogs {
				assert.NotContains(t, logs.String(), unwanted)
			}
		})
	}
}

// resetEPPMetrics gives the test an empty metrics registry.
func resetEPPMetrics(t *testing.T) {
	t.Helper()
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)
}

func newAnswerServer(director Director) *StreamingServer {
	return NewStreamingServer(nil, director, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)
}

// answerRequests is the ext_proc script of one request with a body.
func answerRequests() []*extProcPb.ProcessingRequest {
	return []*extProcPb.ProcessingRequest{
		newRequestHeaders(map[string]string{":path": "/v1/chat/completions", "x-request-id": "req-ask"}),
		{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
			Body:        []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`),
			EndOfStream: true,
		}}},
	}
}

// failingSendServer fails every Send, as when Envoy tears the stream down
// while the EPP answers.
type failingSendServer struct {
	scriptedProcessServer
	err error
}

func (m *failingSendServer) Send(*extProcPb.ProcessingResponse) error { return m.err }

// TestProcessAnswerSendFailure covers an answer the caller never received.
// Process returns the send error, request_error_total counts it, and the
// per-request plugin state is released with the error cause rather than the
// answered cause.
func TestProcessAnswerSendFailure(t *testing.T) {
	resetEPPMetrics(t)
	director := &answeringDirector{answer: &Answer{Status: envoyTypePb.StatusCode_NoContent, Headers: map[string]string{routing.ReservedEndpointHeader: "10.0.3.7:8000"}}}
	server := newAnswerServer(director)

	sendErr := errors.New("stream closed")
	ctx := log.IntoContext(context.Background(), logr.Discard())
	srv := &failingSendServer{err: sendErr, scriptedProcessServer: scriptedProcessServer{ctx: ctx, reqs: answerRequests()}}

	err := server.Process(srv)
	require.ErrorContains(t, err, sendErr.Error())
	assert.Equal(t, 1, director.endOfStreamSeen, "stream-end cleanup releases per-request plugin state")
	assert.Equal(t, fwkrc.TerminationCauseError, director.cause)
	errorSeries, gatherErr := promtestutil.GatherAndCount(ctrlmetrics.Registry, "llm_d_epp_request_error_total")
	require.NoError(t, gatherErr)
	assert.Equal(t, 1, errorSeries, "request_error_total counts the failed send")
}
