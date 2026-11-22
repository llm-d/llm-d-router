/*
Copyright 2025 The Kubernetes Authors.
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
	"errors"
	"io"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

type schedulingMetricsStream struct {
	grpc.ServerStream
	ctx      context.Context
	requests []*extProcPb.ProcessingRequest
}

func (s *schedulingMetricsStream) Context() context.Context                 { return s.ctx }
func (s *schedulingMetricsStream) Send(*extProcPb.ProcessingResponse) error { return nil }
func (s *schedulingMetricsStream) Recv() (*extProcPb.ProcessingRequest, error) {
	if len(s.requests) == 0 {
		return nil, io.EOF
	}
	request := s.requests[0]
	s.requests = s.requests[1:]
	return request, nil
}

type apiSurfaceProfile struct {
	path string
	fail bool
}

func (p *apiSurfaceProfile) Run(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	p.path = request.Headers[":path"]
	request.Headers[":path"] = reqcommon.PathResponses
	if p.fail {
		return nil, errors.New("scheduling failed")
	}
	return &fwksched.ProfileRunResult{TargetEndpoints: endpoints}, nil
}

func TestRequestMetricsRetainIncomingAPISurface(t *testing.T) {
	for _, apiType := range []reqcommon.APIType{reqcommon.APITypeMessages, reqcommon.APITypeChatCompletions} {
		for _, fail := range []bool{false, true} {
			status := eppmetrics.SchedulerStatusSuccess
			if fail {
				status = eppmetrics.SchedulerStatusFailure
			}
			t.Run(apiType.String()+"/"+status, func(t *testing.T) {
				eppmetrics.Register()
				eppmetrics.Reset()
				t.Cleanup(eppmetrics.Reset)
				director, ctx := newSinglePodDirector(t, nil)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()
				profile := &apiSurfaceProfile{fail: fail}
				director.scheduler = scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(single.NewSingleProfileHandler(), map[string]fwksched.SchedulerProfile{"default": profile}))
				director.requestControlPlugins = *NewConfig().WithRequestHeaderPlugins(&mockRequestHeaderPlugin{
					name:     "rewrite-path",
					modifyFn: func(request *fwksched.InferenceRequest) { request.Headers[":path"] = reqcommon.PathChatCompletions },
				})
				stream := &schedulingMetricsStream{ctx: ctx, requests: []*extProcPb.ProcessingRequest{
					{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{{Key: ":path", RawValue: []byte(apiType.Path())}}}}}},
					{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{Body: []byte(`{"model":"model","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`), EndOfStream: true}}},
				}}
				server := handlers.NewStreamingServer(nil, director, handlers.NewParserRegistry([]fwkrh.Parser{openai.NewOpenAIParser(), anthropic.NewAnthropicParser()}, logr.Discard()), 0)
				require.NoError(t, server.Process(stream))
				require.Equal(t, reqcommon.PathChatCompletions, profile.path, "the header plugin must run before scheduling")
				counts := map[string]float64{
					"llm_d_epp_scheduler_e2e_duration_seconds":      1,
					"llm_d_epp_scheduler_attempts_total":            1,
					"llm_d_epp_request_processing_duration_seconds": 1,
				}
				if fail {
					counts["llm_d_epp_request_error_total"] = 1
				} else {
					counts["llm_d_epp_request_total"] = 1
				}
				families, err := ctrlmetrics.Registry.Gather()
				require.NoError(t, err)
				for _, family := range families {
					want, ok := counts[family.GetName()]
					if !ok {
						continue
					}
					require.Len(t, family.GetMetric(), 1, family.GetName())
					metric := family.GetMetric()[0]
					labels := map[string]string{}
					for _, label := range metric.GetLabel() {
						labels[label.GetName()] = label.GetValue()
					}
					switch family.GetName() {
					case "llm_d_epp_scheduler_e2e_duration_seconds", "llm_d_epp_scheduler_attempts_total":
						require.NotContains(t, labels, "api_surface", family.GetName())
					default:
						require.Equal(t, apiType.String(), labels["api_surface"], family.GetName())
					}
					if metric.GetCounter() != nil {
						if family.GetName() == "llm_d_epp_scheduler_attempts_total" {
							require.Equal(t, status, labels["status"])
						}
						require.Equal(t, want, metric.GetCounter().GetValue())
					} else {
						require.Equal(t, uint64(want), metric.GetHistogram().GetSampleCount())
					}
					delete(counts, family.GetName())
				}
				require.Empty(t, counts, "all request and scheduler metrics must be emitted")
			})
		}
	}
}
