/*
Copyright 2026 The Kubernetes Authors.

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

package inflightload

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics/collectors"
)

func makeInflightRequest(requestID, incomingModel, fairnessID string, priority int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:     requestID,
		IncomingModel: incomingModel,
		TargetModel:   "t1",
		FairnessID:    fairnessID,
		Objectives:    fwksched.RequestObjectives{Priority: priority},
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, 4)}},
		},
	}
}

// gaugeFor reads the gauge value for a request's labels via the production label logic.
func gaugeFor(req *fwksched.InferenceRequest) float64 {
	l := newRequestInflightLabels(req)
	return promtestutil.ToFloat64(requestInflight.WithLabelValues(l.modelName, l.targetModelName, l.fairnessID, l.priority))
}

// The producer increments the per-model gauge in PreRequest and exposes it under the documented
// name and label set.
func TestRequestInflight_Golden(t *testing.T) {
	requestInflight.Reset()
	producer := newTestProducer(t)

	req := makeInflightRequest("req1", "m1", "tenant-x", 10)
	producer.PreRequest(context.Background(), req, makeSchedulingResult("ep1"))

	expected := `
		# HELP llm_d_epp_request_inflight [ALPHA] Current number of in-flight requests in the endpoint picker (admitted, not yet completed).
		# TYPE llm_d_epp_request_inflight gauge
		llm_d_epp_request_inflight{fairness_id="tenant-x",model_name="m1",priority="10",target_model_name="t1"} 1
	`
	require.NoError(t, promtestutil.CollectAndCompare(requestInflight, strings.NewReader(expected), "llm_d_epp_request_inflight"))
}

// Every terminal path decrements the gauge exactly once. Unlike the upstream defer-Dec pattern, the
// producer splits admission (PreRequest) from completion (ResponseBody), so the decrement rides on
// PluginState eviction: end-of-stream Delete and the janitor reaper must each fire it once, with no
// leak on abort/disconnect and no double-decrement when both paths run.
func TestRequestInflight_CleanupPaths(t *testing.T) {
	endOfStream := func(p *InFlightLoadProducer, req *fwksched.InferenceRequest, res *fwksched.SchedulingResult) {
		req.SchedulingResult = res
		p.ResponseBody(context.Background(), req, &requestcontrol.Response{EndOfStream: true}, nil)
	}
	reap := func(p *InFlightLoadProducer, req *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) {
		p.PluginState.Delete(req.RequestID)
	}

	tests := []struct {
		name    string
		trigger func(*InFlightLoadProducer, *fwksched.InferenceRequest, *fwksched.SchedulingResult)
		want    float64
	}{
		{"end of stream decrements", endOfStream, 0},
		{"reaper decrements (no end of stream)", reap, 0},
		{
			name: "end of stream then reaper does not go negative",
			trigger: func(p *InFlightLoadProducer, req *fwksched.InferenceRequest, res *fwksched.SchedulingResult) {
				endOfStream(p, req, res)
				reap(p, req, res)
			},
			want: 0,
		},
		{
			name: "reaper then end of stream does not double-decrement",
			trigger: func(p *InFlightLoadProducer, req *fwksched.InferenceRequest, res *fwksched.SchedulingResult) {
				reap(p, req, res)
				endOfStream(p, req, res)
			},
			want: 0,
		},
		{
			name: "start of stream holds the request in flight",
			trigger: func(p *InFlightLoadProducer, req *fwksched.InferenceRequest, res *fwksched.SchedulingResult) {
				req.SchedulingResult = res
				p.ResponseBody(context.Background(), req, &requestcontrol.Response{StartOfStream: true}, nil)
			},
			want: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestInflight.Reset()
			producer := newTestProducer(t)
			req := makeInflightRequest("req-"+tt.name, "m1", "tenant-x", 10)
			res := makeSchedulingResult("ep1")

			producer.PreRequest(context.Background(), req, res)
			require.Equal(t, float64(1), gaugeFor(req), "PreRequest should increment once")

			tt.trigger(producer, req, res)
			require.Equal(t, tt.want, gaugeFor(req))
		})
	}
}

// Requests that are not admitted to an endpoint, or carry no model name, are not counted and write no
// gauge state entry.
func TestRequestInflight_NotCounted(t *testing.T) {
	tests := []struct {
		name   string
		req    *fwksched.InferenceRequest
		result *fwksched.SchedulingResult
	}{
		{"empty incoming model", makeInflightRequest("req-empty-model", "", "tenant-x", 10), makeSchedulingResult("ep1")},
		{"nil scheduling result", makeInflightRequest("req-nil-result", "m1", "tenant-x", 10), nil},
		{"empty profile results", makeInflightRequest("req-empty-result", "m1", "tenant-x", 10), &fwksched.SchedulingResult{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestInflight.Reset()
			producer := newTestProducer(t)

			producer.PreRequest(context.Background(), tt.req, tt.result)

			require.Equal(t, float64(0), gaugeFor(tt.req))
			_, err := producer.PluginState.Read(tt.req.RequestID, requestInflightStateKey)
			require.ErrorIs(t, err, fwkplugin.ErrNotFound, "no gauge state entry should be written")
		})
	}
}

// A request fanned out across multiple profiles/endpoints is one in-flight request: the gauge moves by
// one, not once per endpoint.
func TestRequestInflight_IncrementsOncePerRequest(t *testing.T) {
	requestInflight.Reset()
	producer := newTestProducer(t)
	ctx := context.Background()

	req := makeInflightRequest("req-multi", "m1", "tenant-x", 10)
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "prefill",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint("pod-a")}},
			"decode":  {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint("pod-b")}},
		},
	}

	producer.PreRequest(ctx, req, res)
	require.Equal(t, float64(1), gaugeFor(req))

	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Equal(t, float64(0), gaugeFor(req))
}

// Concurrent admit/complete cycles balance to zero (run under -race for the OnEvicted/atomic guard).
func TestRequestInflight_ConcurrentBalanced(t *testing.T) {
	requestInflight.Reset()
	producer := newTestProducer(t)
	ctx := context.Background()

	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			req := makeInflightRequest(fmt.Sprintf("req-%d", i), "m1", "tenant-x", 10)
			res := makeSchedulingResult("ep1")
			producer.PreRequest(ctx, req, res)
			req.SchedulingResult = res
			producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
		}(i)
	}
	wg.Wait()

	require.Equal(t, float64(0), promtestutil.ToFloat64(requestInflight.WithLabelValues("m1", "t1", "tenant-x", "10")))
}

// newRequestInflightLabels maps a request onto the gauge label set, defaulting an empty fairness id and
// stringifying the priority.
func TestNewRequestInflightLabels(t *testing.T) {
	tests := []struct {
		name string
		req  *fwksched.InferenceRequest
		want requestInflightLabels
	}{
		{
			name: "all fields set",
			req:  makeInflightRequest("r", "m1", "tenant-x", 10),
			want: requestInflightLabels{modelName: "m1", targetModelName: "t1", fairnessID: "tenant-x", priority: "10"},
		},
		{
			name: "empty fairness defaults",
			req:  makeInflightRequest("r", "m1", "", 0),
			want: requestInflightLabels{modelName: "m1", targetModelName: "t1", fairnessID: metadata.DefaultFairnessID, priority: "0"},
		},
		{
			name: "negative priority stringified",
			req:  makeInflightRequest("r", "m1", "tenant-x", -1),
			want: requestInflightLabels{modelName: "m1", targetModelName: "t1", fairnessID: "tenant-x", priority: "-1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, newRequestInflightLabels(tt.req))
		})
	}
}

// registerMetrics is idempotent for the shared gauge and registers a distinct per-producer collector,
// so two producers register into one registry and both emit per-endpoint series.
func TestRegisterMetrics_MultiProducer(t *testing.T) {
	requestInflight.Reset()
	reg := prometheus.NewRegistry()

	a := &fakeSnapshotter{requests: map[string]int64{"ns1/ep1": 1}}
	b := &fakeSnapshotter{requests: map[string]int64{"ns2/ep2": 2}}
	require.NoError(t, registerMetrics(reg, collectors.NewInFlightLoadCollector("a", a)))
	require.NoError(t, registerMetrics(reg, collectors.NewInFlightLoadCollector("b", b)))

	expected := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="ep1",namespace="ns1",producer_name="a"} 1
llm_d_epp_inflight_requests{endpoint_name="ep2",namespace="ns2",producer_name="b"} 2
`
	require.NoError(t, promtestutil.GatherAndCompare(reg, strings.NewReader(expected), "llm_d_epp_inflight_requests"))
}

func TestRegisterMetrics_NilRegisterer(t *testing.T) {
	require.Error(t, registerMetrics(nil, nil))
}

type fakeSnapshotter struct {
	requests map[string]int64
	tokens   map[string]int64
}

func (f *fakeSnapshotter) InFlightRequestsSnapshot() map[string]int64 { return f.requests }
func (f *fakeSnapshotter) InFlightTokensSnapshot() map[string]int64   { return f.tokens }
