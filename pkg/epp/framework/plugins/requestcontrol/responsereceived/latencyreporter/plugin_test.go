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

package latencyreporter

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const testConfig = `{"promptBuckets": [1024, 8192]}`

func newTestPlugin(t *testing.T, raw string) *Plugin {
	t.Helper()
	got, err := PluginFactory("latency", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	require.NoError(t, err)
	return got.(*Plugin)
}

type reqSpec struct {
	priority   int
	fairness   string
	prompt     int
	completion int
	ttft       time.Duration
	decode     time.Duration
	streaming  bool
	status     string
	cause      fwkrc.TerminationCause
	scheduling *fwksched.SchedulingResult
}

func run(p *Plugin, id string, s reqSpec) {
	req := &fwksched.InferenceRequest{
		RequestID:        id,
		TargetModel:      "m",
		FairnessID:       s.fairness,
		Headers:          map[string]string{},
		Objectives:       fwksched.RequestObjectives{Priority: s.priority},
		SchedulingResult: s.scheduling,
	}
	status := s.status
	if status == "" {
		status = "200"
	}
	p.ResponseHeader(context.Background(), req, &fwkrc.Response{Headers: map[string]string{":status": status}}, nil)

	received := time.Unix(1000, 0)
	cause := s.cause
	if cause == "" {
		cause = fwkrc.TerminationCauseNatural
	}
	resp := &fwkrc.Response{
		EndOfStream:         true,
		TerminationCause:    cause,
		ReceivedTimestamp:   received,
		FirstTokenTimestamp: received.Add(s.ttft),
		CompleteTimestamp:   received.Add(s.ttft + s.decode),
		Streaming:           s.streaming,
	}
	resp.Usage.PromptTokens = s.prompt
	resp.Usage.CompletionTokens = s.completion
	p.ResponseBody(context.Background(), req, resp, nil)
}

func count(fairness, prio, bucket, path, outcome string) float64 {
	return promtestutil.ToFloat64(latencyRequests.WithLabelValues("m", fairness, prio, bucket, path, outcome))
}

// histogram returns the sample count and sum of one histogram series.
func histogram(t *testing.T, vec *prometheus.HistogramVec, labels ...string) (uint64, float64) {
	t.Helper()
	m := &dto.Metric{}
	require.NoError(t, vec.WithLabelValues(labels...).(prometheus.Histogram).Write(m))
	return m.GetHistogram().GetSampleCount(), m.GetHistogram().GetSampleSum()
}

func TestFactory_InvalidConfig(t *testing.T) {
	cases := map[string]string{
		"unsorted buckets":  `{"promptBuckets":[8192,1024]}`,
		"dup buckets":       `{"promptBuckets":[1024,1024]}`,
		"nonpositive bound": `{"promptBuckets":[0,1024]}`,
		"unknown field":     `{"targets":[]}`,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := PluginFactory("latency", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
			require.Error(t, err)
		})
	}
}

func TestFactory_NoParameters(t *testing.T) {
	got, err := PluginFactory("latency", nil, nil)
	require.NoError(t, err)
	p := got.(*Plugin)
	assert.Equal(t, "all", p.bucket(50000))
	assert.Equal(t, defaultPrefillProfile, p.prefillProfile)
}

func TestBucketLabels(t *testing.T) {
	p := newTestPlugin(t, testConfig)
	assert.Equal(t, "le1024", p.bucket(0))
	assert.Equal(t, "le1024", p.bucket(1024))
	assert.Equal(t, "le8192", p.bucket(1025))
	assert.Equal(t, "gt8192", p.bucket(8193))
}

func TestOutcomeAndPath(t *testing.T) {
	p := newTestPlugin(t, testConfig)
	tests := []struct {
		name        string
		spec        reqSpec
		bucket      string
		path        string
		wantOutcome string
	}{
		{
			name:        "completed stream is success",
			spec:        reqSpec{prompt: 100, completion: 101, ttft: 200 * time.Millisecond, decode: 2 * time.Second, streaming: true},
			bucket:      "le1024",
			path:        pathAggregated,
			wantOutcome: outcomeSuccess,
		},
		{
			name:        "long prompt lands in the top bucket",
			spec:        reqSpec{prompt: 20000, completion: 2, ttft: 3 * time.Second, streaming: true},
			bucket:      "gt8192",
			path:        pathAggregated,
			wantOutcome: outcomeSuccess,
		},
		{
			name:        "server error is failed",
			spec:        reqSpec{prompt: 100, status: "503"},
			bucket:      "le1024",
			path:        pathAggregated,
			wantOutcome: outcomeFailed,
		},
		{
			name:        "client disconnect is failed",
			spec:        reqSpec{prompt: 100, cause: fwkrc.TerminationCauseClientDisconnect},
			bucket:      "le1024",
			path:        pathAggregated,
			wantOutcome: outcomeFailed,
		},
		{
			name: "remote prefill path",
			spec: reqSpec{prompt: 100, ttft: 100 * time.Millisecond, scheduling: &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode":  {TargetEndpoints: []fwksched.Endpoint{nil}},
				"prefill": {TargetEndpoints: []fwksched.Endpoint{nil}},
			}}},
			bucket:      "le1024",
			path:        pathRemote,
			wantOutcome: outcomeSuccess,
		},
		{
			name: "declined prefill is the local path",
			spec: reqSpec{prompt: 100, ttft: 100 * time.Millisecond, scheduling: &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode":  {TargetEndpoints: []fwksched.Endpoint{nil}},
				"prefill": nil,
			}}},
			bucket:      "le1024",
			path:        pathLocal,
			wantOutcome: outcomeSuccess,
		},
	}
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fairness := "o" + string(rune('a'+i))
			tc.spec.priority = 100
			tc.spec.fairness = fairness
			before := count(fairness, "100", tc.bucket, tc.path, tc.wantOutcome)
			run(p, "req-"+fairness, tc.spec)
			assert.Equal(t, before+1, count(fairness, "100", tc.bucket, tc.path, tc.wantOutcome))
		})
	}
}

// TestLatencyObservations checks the recorded TTFT and TPOT values. A distinct priority keeps the
// histogram series isolated from other tests.
func TestLatencyObservations(t *testing.T) {
	p := newTestPlugin(t, testConfig)

	t.Run("streamed request records TTFT and mean TPOT", func(t *testing.T) {
		n0, s0 := histogram(t, latencyTTFT, "71", "le1024", pathAggregated)
		m0, u0 := histogram(t, latencyTPOT, "71", pathAggregated)
		// 101 output tokens over 2s after the first token: TPOT = 2s / 100 = 20ms.
		run(p, "obs-stream", reqSpec{priority: 71, prompt: 100, completion: 101, ttft: 250 * time.Millisecond, decode: 2 * time.Second, streaming: true})
		n1, s1 := histogram(t, latencyTTFT, "71", "le1024", pathAggregated)
		m1, u1 := histogram(t, latencyTPOT, "71", pathAggregated)
		assert.Equal(t, n0+1, n1)
		assert.InDelta(t, 0.25, s1-s0, 1e-9)
		assert.Equal(t, m0+1, m1)
		assert.InDelta(t, 0.02, u1-u0, 1e-9)
	})

	t.Run("non-streaming request records TTFT only", func(t *testing.T) {
		m0, _ := histogram(t, latencyTPOT, "72", pathAggregated)
		n0, _ := histogram(t, latencyTTFT, "72", "le1024", pathAggregated)
		run(p, "obs-unary", reqSpec{priority: 72, prompt: 100, completion: 500, ttft: 300 * time.Millisecond, decode: 30 * time.Second})
		m1, _ := histogram(t, latencyTPOT, "72", pathAggregated)
		n1, _ := histogram(t, latencyTTFT, "72", "le1024", pathAggregated)
		assert.Equal(t, n0+1, n1)
		assert.Equal(t, m0, m1)
	})

	t.Run("failed request records no latency", func(t *testing.T) {
		n0, _ := histogram(t, latencyTTFT, "73", "le1024", pathAggregated)
		run(p, "obs-failed", reqSpec{priority: 73, prompt: 100, ttft: 100 * time.Millisecond, status: "500"})
		n1, _ := histogram(t, latencyTTFT, "73", "le1024", pathAggregated)
		assert.Equal(t, n0, n1)
	})
}

func TestPriorityFilter(t *testing.T) {
	p := newTestPlugin(t, `{"priorities": [100]}`)

	before := count("filtered", "10", "all", pathAggregated, outcomeSuccess)
	run(p, "req-filtered", reqSpec{priority: 10, fairness: "filtered", prompt: 100, ttft: time.Millisecond})
	assert.Equal(t, before, count("filtered", "10", "all", pathAggregated, outcomeSuccess), "an unlisted priority is not recorded")
	_, held := p.statuses.Load("req-filtered")
	assert.False(t, held, "per-request status is released at end of stream")

	before = count("listed", "100", "all", pathAggregated, outcomeSuccess)
	run(p, "req-listed", reqSpec{priority: 100, fairness: "listed", prompt: 100, ttft: time.Millisecond})
	assert.Equal(t, before+1, count("listed", "100", "all", pathAggregated, outcomeSuccess))
}

func TestNoPriorityFilterRecordsEveryPriority(t *testing.T) {
	p := newTestPlugin(t, testConfig)
	before := count("anyprio", "-5", "le1024", pathAggregated, outcomeSuccess)
	run(p, "req-anyprio", reqSpec{priority: -5, fairness: "anyprio", prompt: 100, ttft: time.Millisecond})
	assert.Equal(t, before+1, count("anyprio", "-5", "le1024", pathAggregated, outcomeSuccess))
}
