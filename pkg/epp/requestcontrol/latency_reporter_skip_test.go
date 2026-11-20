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

package requestcontrol

import (
	"context"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/responsereceived/latencyreporter"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// TestLatencyReporterIgnoresRequestsThatSkipResponseProcessing drives the real
// handlers and director. A route whose parser sets SkipResponseProcessing
// (render, count_tokens) closes the ext_proc stream right after routing, so the
// EPP never sees its response. llm_d_epp_request_outcome_total leaves such a
// request out; latency-reporter, whose outcome label claims the same values and
// precedence, must too, instead of counting a request it never saw end as failed.
func TestLatencyReporterIgnoresRequestsThatSkipResponseProcessing(t *testing.T) {
	const chatBody = `{"model":"model","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`
	tests := []struct {
		name string
		reqs []*extProcPb.ProcessingRequest
		want map[string]float64 // latency_requests_total by outcome
	}{
		{
			name: "chat completion answered 200 (control)",
			reqs: append(latencyRequest("/v1/chat/completions", chatBody), latencyResponse("200")...),
			want: map[string]float64{"completed": 1},
		},
		{
			name: "chat completions render skips response processing",
			reqs: latencyRequest("/v1/chat/completions/render", chatBody),
			want: map[string]float64{},
		},
		{
			name: "messages count_tokens skips response processing",
			reqs: latencyRequest("/v1/messages/count_tokens", chatBody),
			want: map[string]float64{},
		},
	}

	reg := prometheus.NewRegistry()
	handle := fwkplugin.NewEppHandle(context.Background(), nil, fwkplugin.WithMetricsRecorder(reg))
	raw, err := latencyreporter.PluginFactory("latency", nil, handle)
	require.NoError(t, err)
	reporter := raw.(interface {
		fwkrh.ResponseHeaderProcessor
		fwkrh.ResponseBodyProcessor
	})

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)

			endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
				Address: "192.168.1.100",
				Port:    "8000",
				ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
			}, nil, fwkdl.NewAttributes())
			director, ctx := newSinglePodDirector(t, &fwksched.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
				},
			})
			director.requestControlPlugins = *NewConfig().
				WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
				WithResponseReceivedPlugins(reporter).
				WithResponseStreamingPlugins(reporter)

			before := latencyRequestsByOutcome(t, reg)
			stream := &schedulingMetricsStream{ctx: ctx, requests: tt.reqs}
			parsers := handlers.NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser(), anthropic.NewAnthropicParser()}, logr.Discard())
			server := handlers.NewStreamingServer(nil, director, parsers, 0)
			require.NoError(t, server.Process(stream))

			got := map[string]float64{}
			for outcome, v := range latencyRequestsByOutcome(t, reg) {
				if d := v - before[outcome]; d != 0 {
					got[outcome] = d
				}
			}
			require.Equal(t, tt.want, got)
		})
	}
}

func latencyRequest(path, body string) []*extProcPb.ProcessingRequest {
	return []*extProcPb.ProcessingRequest{
		{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
			{Key: ":path", RawValue: []byte(path)},
			{Key: "x-request-id", RawValue: []byte("req-latency")},
		}}}}},
		{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{Body: []byte(body), EndOfStream: true}}},
	}
}

func latencyResponse(status string) []*extProcPb.ProcessingRequest {
	return []*extProcPb.ProcessingRequest{
		{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
			{Key: ":status", RawValue: []byte(status)},
			{Key: "content-type", RawValue: []byte("application/json")},
		}}}}},
		{Request: &extProcPb.ProcessingRequest_ResponseBody{ResponseBody: &extProcPb.HttpBody{Body: []byte(`{}`), EndOfStream: true}}},
	}
}

// latencyRequestsByOutcome sums llm_d_epp_latency_requests_total by its outcome label.
func latencyRequestsByOutcome(t *testing.T, reg *prometheus.Registry) map[string]float64 {
	t.Helper()
	families, err := reg.Gather()
	require.NoError(t, err)
	got := map[string]float64{}
	for _, family := range families {
		if family.GetName() != eppmetrics.LLMDRouterEndpointPickerSubsystem+"_latency_requests_total" {
			continue
		}
		for _, m := range family.GetMetric() {
			for _, l := range m.GetLabel() {
				if l.GetName() == "outcome" {
					got[l.GetValue()] += m.GetCounter().GetValue()
				}
			}
		}
	}
	return got
}
