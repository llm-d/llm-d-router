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

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const outcomeTestBody = `{"model":"m","prompt":"hi"}`

func outcomeRequest(path, body string) []*extProcPb.ProcessingRequest {
	return []*extProcPb.ProcessingRequest{
		newRequestHeaders(map[string]string{":path": path, "x-request-id": "req-outcome"}),
		{Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: []byte(body), EndOfStream: true},
		}},
	}
}

func outcomeResponse(status string) []*extProcPb.ProcessingRequest {
	return []*extProcPb.ProcessingRequest{
		{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
				{Key: ":status", RawValue: []byte(status)},
			}}},
		}},
		{Request: &extProcPb.ProcessingRequest_ResponseBody{
			ResponseBody: &extProcPb.HttpBody{Body: []byte(`{}`), EndOfStream: true},
		}},
	}
}

func outcomeGRPCResponse(grpcStatus string) []*extProcPb.ProcessingRequest {
	return []*extProcPb.ProcessingRequest{
		{Request: &extProcPb.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
				{Key: ":status", RawValue: []byte("200")},
			}}},
		}},
		{Request: &extProcPb.ProcessingRequest_ResponseTrailers{
			ResponseTrailers: &extProcPb.HttpTrailers{Trailers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
				{Key: "grpc-status", RawValue: []byte(grpcStatus)},
			}}},
		}},
	}
}

func droppedError(code string, reason errcommon.RequestDroppedReason) errcommon.Error {
	return errcommon.Error{
		Code:    code,
		Msg:     "dropped",
		Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(reason)},
	}
}

func TestProcessRecordsRequestOutcome(t *testing.T) {
	tests := []struct {
		name        string
		reqs        []*extProcPb.ProcessingRequest
		directorErr error
		wantOutcome string
		wantReason  string
	}{
		{
			name:        "model server answers 200",
			reqs:        append(outcomeRequest("/v1/completions", outcomeTestBody), outcomeResponse("200")...),
			wantOutcome: "completed",
		},
		{
			name:        "model server answers 502",
			reqs:        append(outcomeRequest("/v1/completions", outcomeTestBody), outcomeResponse("502")...),
			wantOutcome: "failed",
			wantReason:  errcommon.ModelServerError,
		},
		{
			name:        "gRPC error in trailers after 200",
			reqs:        append(outcomeRequest("/v1/completions", outcomeTestBody), outcomeGRPCResponse("13")...),
			wantOutcome: "failed",
			wantReason:  errcommon.ModelServerError,
		},
		{
			name:        "gRPC OK in trailers",
			reqs:        append(outcomeRequest("/v1/completions", outcomeTestBody), outcomeGRPCResponse("0")...),
			wantOutcome: "completed",
		},
		{
			name:        "admission sheds the request",
			reqs:        outcomeRequest("/v1/completions", outcomeTestBody),
			directorErr: droppedError(errcommon.ResourceExhausted, errcommon.RequestDroppedReasonSaturated),
			wantOutcome: "rejected",
			wantReason:  string(errcommon.RequestDroppedReasonSaturated),
		},
		{
			name:        "client leaves while queued",
			reqs:        outcomeRequest("/v1/completions", outcomeTestBody),
			directorErr: droppedError(errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonContextCancelled),
			wantOutcome: "rejected",
			wantReason:  string(errcommon.RequestDroppedReasonContextCancelled),
		},
		{
			name:        "scheduling fails without a dropped reason",
			reqs:        outcomeRequest("/v1/completions", outcomeTestBody),
			directorErr: errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "no endpoints"},
			wantOutcome: "failed",
			wantReason:  errcommon.ServiceUnavailable,
		},
		{
			name:        "malformed body",
			reqs:        outcomeRequest("/v1/completions", `{`),
			wantOutcome: "failed",
			wantReason:  errcommon.BadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)

			srv := &replayProcessServer{ctx: logutil.NewTestLoggerIntoContext(context.Background()), reqs: tt.reqs}
			director := &rejectingDirector{err: tt.directorErr}
			server := NewStreamingServer(nil, director, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)

			require.NoError(t, server.Process(srv))

			require.Equal(t, map[[2]string]float64{{tt.wantOutcome, tt.wantReason}: 1}, requestOutcomes(t))
		})
	}
}

// Process sets the evicted state and reason when flow control evicts a dispatched request.
func TestRequestOutcomeOfEvictedRequest(t *testing.T) {
	reqCtx := &RequestContext{
		bodyReceived:         true,
		requestState:         requestEvicted,
		requestDroppedReason: errcommon.RequestDroppedReasonEvictedPriority,
	}

	outcome, reason, ok := requestOutcome(reqCtx, nil, nil)

	require.True(t, ok)
	require.Equal(t, "evicted", outcome)
	require.Equal(t, string(errcommon.RequestDroppedReasonEvictedPriority), reason)
}

// The client disconnects after the request was dispatched and before any response.
func TestProcessRecordsCancelledOutcome(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)
	srv := &replayProcessServer{ctx: ctx, reqs: outcomeRequest("/v1/completions", outcomeTestBody)}
	server := NewStreamingServer(nil, &cancellingDirector{cancel: cancel}, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)

	// Process returns whichever of the stream error or the context error it sees first.
	_ = server.Process(srv)

	require.Equal(t, map[[2]string]float64{{"cancelled", ""}: 1}, requestOutcomes(t))
}

// A rejected request is attributed to the tenant and priority admission saw, so per-tenant
// rejections are visible.
func TestProcessLabelsOutcomeWithFairnessAndPriority(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	srv := &replayProcessServer{ctx: logutil.NewTestLoggerIntoContext(context.Background()), reqs: outcomeRequest("/v1/completions", outcomeTestBody)}
	director := &attributingDirector{fairnessID: "team-a", priority: -1, err: droppedError(errcommon.ResourceExhausted, errcommon.RequestDroppedReasonSaturated)}
	server := NewStreamingServer(nil, director, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)

	require.NoError(t, server.Process(srv))

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	var labels map[string]string
	for _, family := range families {
		if family.GetName() == "llm_d_epp_request_outcome_total" {
			require.Len(t, family.GetMetric(), 1)
			labels = map[string]string{}
			for _, label := range family.GetMetric()[0].GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
		}
	}
	require.Equal(t, "team-a", labels["fairness_id"])
	require.Equal(t, "-1", labels["priority"])
	require.Equal(t, "rejected", labels["outcome"])
}

// Requests whose ending the EPP does not observe are left out.
func TestProcessSkipsOutcomeWhenEndingIsNotObserved(t *testing.T) {
	tests := []struct {
		name string
		reqs []*extProcPb.ProcessingRequest
	}{
		{name: "route skips response processing", reqs: outcomeRequest("/v1/completions/render", outcomeTestBody)},
		{name: "stream ends before the body", reqs: outcomeRequest("/v1/completions", outcomeTestBody)[:1]},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)

			srv := &replayProcessServer{ctx: logutil.NewTestLoggerIntoContext(context.Background()), reqs: tt.reqs}
			server := NewStreamingServer(nil, &mockDirector{}, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)

			require.NoError(t, server.Process(srv))

			require.Empty(t, requestOutcomes(t))
		})
	}
}

// cancellingDirector schedules the request, then the client goes away.
type cancellingDirector struct {
	mockDirector
	cancel context.CancelFunc
}

func (d *cancellingDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	d.cancel()
	return reqCtx, nil
}

// attributingDirector resolves the tenant and priority, then admission rejects the request.
type attributingDirector struct {
	mockDirector
	fairnessID string
	priority   int
	err        error
}

func (d *attributingDirector) HandleRequest(_ context.Context, reqCtx *RequestContext, _ *fwkrh.InferenceRequestBody) (*RequestContext, error) {
	reqCtx.SchedulingRequest = &fwksched.InferenceRequest{FairnessID: d.fairnessID}
	reqCtx.Priority = d.priority
	return reqCtx, d.err
}

// requestOutcomes returns llm_d_epp_request_outcome_total keyed by {outcome, reason}.
func requestOutcomes(t *testing.T) map[[2]string]float64 {
	t.Helper()

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	got := map[[2]string]float64{}
	for _, family := range families {
		if family.GetName() != "llm_d_epp_request_outcome_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			var key [2]string
			for _, label := range metric.GetLabel() {
				switch label.GetName() {
				case "outcome":
					key[0] = label.GetValue()
				case "reason":
					key[1] = label.GetValue()
				}
			}
			got[key] += metric.GetCounter().GetValue()
		}
	}
	return got
}
