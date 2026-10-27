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
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

func TestProcessCountsModelServerErrorFromResponseStatus(t *testing.T) {
	tests := []struct {
		name      string
		key       string
		value     string
		wantCount float64
	}{
		{name: "Envoy pseudo-header with an error status", key: ":status", value: "502", wantCount: 1},
		{name: "Envoy pseudo-header with 200", key: ":status", value: "200", wantCount: 0},
		{name: "plain status header with an error status", key: "status", value: "502", wantCount: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)

			srv := &replayProcessServer{
				ctx: logutil.NewTestLoggerIntoContext(context.Background()),
				reqs: []*extProcPb.ProcessingRequest{{
					Request: &extProcPb.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extProcPb.HttpHeaders{
							Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
								{Key: tt.key, RawValue: []byte(tt.value)},
							}},
						},
					},
				}},
			}
			registry := NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard())

			require.NoError(t, NewStreamingServer(nil, &mockDirector{}, registry, 0).Process(srv))

			require.Equal(t, tt.wantCount, requestErrorCount(t, errcommon.ModelServerError))
		})
	}
}

// A request evicted after dispatch is answered with 429 and must be counted
// like any other failed request.
func TestProcessCountsEvictionAsRequestError(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()
	t.Cleanup(eppmetrics.Reset)

	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)
	srv := &blockingReplayProcessServer{replayProcessServer: replayProcessServer{
		ctx: ctx,
		reqs: []*extProcPb.ProcessingRequest{
			newRequestHeaders(map[string]string{":path": "/v1/completions", "x-request-id": "req-evicted"}),
			{Request: &extProcPb.ProcessingRequest_RequestBody{
				RequestBody: &extProcPb.HttpBody{Body: []byte(`{"model":"m","prompt":"hi"}`), EndOfStream: true},
			}},
		},
	}}
	server := NewStreamingServer(nil, &mockDirector{}, NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser()}, logr.Discard()), 0)
	server.SetEvictChannelLookup(closedEvictLookup{})

	require.NoError(t, server.Process(srv))

	last := srv.sentResponses[len(srv.sentResponses)-1]
	require.Equal(t, envoyTypePb.StatusCode_TooManyRequests, last.GetImmediateResponse().GetStatus().GetCode())
	require.Equal(t, 1.0, requestErrorCount(t, errcommon.ResourceExhausted))
}

// blockingReplayProcessServer replays its requests, then blocks like an open stream.
type blockingReplayProcessServer struct {
	replayProcessServer
}

func (m *blockingReplayProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if len(m.reqs) == 0 {
		<-m.ctx.Done()
		return nil, m.ctx.Err()
	}
	return m.replayProcessServer.Recv()
}

// closedEvictLookup evicts every request as soon as it is dispatched.
type closedEvictLookup struct{}

func (closedEvictLookup) Get(string) chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

func (closedEvictLookup) GetReason(string) errcommon.RequestDroppedReason {
	return errcommon.RequestDroppedReasonEvicted
}

func (closedEvictLookup) Deregister(string) {}

// requestErrorCount sums llm_d_epp_request_error_total over every series with the given error code.
func requestErrorCount(t *testing.T, code string) float64 {
	t.Helper()

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	var total float64
	for _, family := range families {
		if family.GetName() != "llm_d_epp_request_error_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			if metricHasLabels(metric, map[string]string{"error_code": code}) {
				total += metric.GetCounter().GetValue()
			}
		}
	}
	return total
}
