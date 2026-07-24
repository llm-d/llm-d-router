/*
Copyright 2025 The Kubernetes Authors.

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

package metrics

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/component-base/metrics/testutil"
	"sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	requestTotalMetric                 = inferenceObjectiveComponent + "_request_total"
	requestErrorTotalMetric            = inferenceObjectiveComponent + "_request_error_total"
	requestLatenciesMetric             = inferenceObjectiveComponent + "_request_duration_seconds"
	requestSizesMetric                 = inferenceObjectiveComponent + "_request_sizes"
	responseSizesMetric                = inferenceObjectiveComponent + "_response_sizes"
	inputTokensMetric                  = inferenceObjectiveComponent + "_input_tokens"
	outputTokensMetric                 = inferenceObjectiveComponent + "_output_tokens"
	normalizedTimePerOutputTokenMetric = inferenceObjectiveComponent + "_normalized_time_per_output_token_seconds"
	runningRequestsMetric              = inferenceObjectiveComponent + "_running_requests"
	kvCacheAvgUsageMetric              = inferencePoolComponent + "_average_kv_cache_utilization"
	queueAvgSizeMetric                 = inferencePoolComponent + "_average_queue_size"
	runningRequestsAvgMetric           = inferencePoolComponent + "_average_running_requests"
)

func TestMain(m *testing.M) {
	// Register all metrics once for the entire test suite.
	Register()
	os.Exit(m.Run())
}

func TestRecordRequestCounterandSizes(t *testing.T) {
	Reset()
	type requests struct {
		modelName       string
		targetModelName string
		fairnessID      string
		reqSize         int
	}
	scenarios := []struct {
		name string
		reqs []requests
	}{{
		name: "multiple requests",
		reqs: []requests{
			{
				modelName:       "m10",
				targetModelName: "t10",
				fairnessID:      "tenant-a",
				reqSize:         1200,
			},
			{
				modelName:       "m10",
				targetModelName: "t10",
				fairnessID:      "tenant-a",
				reqSize:         500,
			},
			{
				modelName:       "m10",
				targetModelName: "t11",
				fairnessID:      "tenant-b",
				reqSize:         2480,
			},
			{
				modelName:       "m20",
				targetModelName: "t20",
				fairnessID:      "tenant-c",
				reqSize:         80,
			},
		},
	}}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				RecordRequestCounter(req.modelName, req.targetModelName, req.fairnessID, 0)
				RecordRequestSizes(req.modelName, req.targetModelName, req.fairnessID, "0", req.reqSize)
			}

			// Verify deprecated metrics
			wantRequestTotal, err := os.Open("testdata/request_total_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRequestTotal.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantRequestTotal, requestTotalMetric); err != nil {
				t.Error(err)
			}

			wantRequestSizes, err := os.Open("testdata/request_sizes_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRequestSizes.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantRequestSizes, requestSizesMetric); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metrics
			wantRequestTotalNew, err := os.Open("testdata/llm_d_request_total_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRequestTotalNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantRequestTotalNew, "llm_d_epp_request_total"); err != nil {
				t.Error(err)
			}

			wantRequestSizesNew, err := os.Open("testdata/llm_d_request_size_bytes_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRequestSizesNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantRequestSizesNew, "llm_d_epp_request_size_bytes"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRecordRequestErrorCounter(t *testing.T) {
	Reset()
	type requests struct {
		modelName       string
		targetModelName string
		error           string
	}
	scenarios := []struct {
		name    string
		reqs    []requests
		invalid bool
	}{
		{
			name: "multiple requests",
			reqs: []requests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					error:           errcommon.Internal,
				},
				{
					modelName:       "m10",
					targetModelName: "t10",
					error:           errcommon.Internal,
				},
				{
					modelName:       "m10",
					targetModelName: "t11",
					error:           errcommon.ModelServerError,
				},
				{
					modelName:       "m20",
					targetModelName: "t20",
					error:           errcommon.ResourceExhausted,
				},
			},
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				RecordRequestErrCounter(req.modelName, req.targetModelName, "", "0", req.error)
			}

			// Verify deprecated metric
			wantRequestErrorCounter, err := os.Open("testdata/request_error_total_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRequestErrorCounter.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantRequestErrorCounter, requestErrorTotalMetric); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metric
			wantRequestErrorCounterNew, err := os.Open("testdata/llm_d_request_error_total_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRequestErrorCounterNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantRequestErrorCounterNew, "llm_d_epp_request_error_total"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRecordRequestLatencies(t *testing.T) {
	Reset()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	timeBaseline := time.Now()
	type requests struct {
		modelName       string
		targetModelName string
		receivedTime    time.Time
		completeTime    time.Time
	}
	scenarios := []struct {
		name    string
		reqs    []requests
		invalid bool
	}{
		{
			name: "multiple requests",
			reqs: []requests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 10),
				},
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 1600),
				},
				{
					modelName:       "m10",
					targetModelName: "t11",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 60),
				},
				{
					modelName:       "m20",
					targetModelName: "t20",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 120),
				},
			},
		},
		{
			name: "invalid elapsed time",
			reqs: []requests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline.Add(time.Millisecond * 10),
					completeTime:    timeBaseline,
				},
			},
			invalid: true,
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				success := RecordRequestLatencies(ctx, req.modelName, req.targetModelName, "", "0", req.receivedTime, req.completeTime)
				if success == scenario.invalid {
					t.Errorf("got record success(%v), but the request expects invalid(%v)", success, scenario.invalid)
				}
			}

			// Verify deprecated metric
			wantRequestLatencies, err := os.Open("testdata/request_duration_seconds_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRequestLatencies.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantRequestLatencies, requestLatenciesMetric); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metric
			wantRequestLatenciesNew, err := os.Open("testdata/llm_d_request_duration_seconds_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRequestLatenciesNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantRequestLatenciesNew, "llm_d_epp_request_duration_seconds"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRecordNormalizedTimePerOutputToken(t *testing.T) {
	Reset()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	timeBaseline := time.Now()
	type tokenRequests struct {
		modelName       string
		targetModelName string
		receivedTime    time.Time
		completeTime    time.Time
		outputTokens    int
	}
	scenarios := []struct {
		name    string
		reqs    []tokenRequests
		invalid bool
	}{
		{
			name: "multiple requests",
			reqs: []tokenRequests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 1000),
					outputTokens:    100,
				},
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 1600),
					outputTokens:    80,
				},
				{
					modelName:       "m10",
					targetModelName: "t11",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 6000),
					outputTokens:    300,
				},
				{
					modelName:       "m20",
					targetModelName: "t20",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 2400),
					outputTokens:    400,
				},
			},
		},
		{
			name: "invalid elapsed time",
			reqs: []tokenRequests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline.Add(time.Millisecond * 10),
					completeTime:    timeBaseline,
					outputTokens:    100,
				},
			},
			invalid: true,
		},
		{
			name: "invalid token count",
			reqs: []tokenRequests{
				{
					modelName:       "m10",
					targetModelName: "t10",
					receivedTime:    timeBaseline,
					completeTime:    timeBaseline.Add(time.Millisecond * 1000),
					outputTokens:    0,
				},
			},
			invalid: true,
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.reqs {
				success := RecordNormalizedTimePerOutputToken(ctx, req.modelName, req.targetModelName, "tenant-a", "3", req.receivedTime, req.completeTime, req.outputTokens)
				if success == scenario.invalid {
					t.Errorf("got record success(%v), but the request expects invalid(%v)", success, scenario.invalid)
				}
			}

			// Verify deprecated metric
			wantLatencyPerToken, err := os.Open("testdata/normalized_time_per_output_token_seconds_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantLatencyPerToken.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantLatencyPerToken, normalizedTimePerOutputTokenMetric); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metric
			wantLatencyPerTokenNew, err := os.Open("testdata/llm_d_normalized_time_per_output_token_seconds_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantLatencyPerTokenNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantLatencyPerTokenNew, "llm_d_epp_request_ntpot_seconds"); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metric labels directly.
			if !scenario.invalid {
				observed, err := getHistogramVecLabelValues(t, llmdNormalizedTimePerOutputToken, "m10", "t10", "tenant-a", "3")
				require.NoError(t, err)
				require.Equal(t, uint64(2), observed.GetSampleCount())
				require.InEpsilon(t, 0.03, observed.GetSampleSum(), 0.000001)
			}
		})
	}
}

func TestRecordResponseMetrics(t *testing.T) {
	Reset()
	type responses struct {
		modelName       string
		targetModelName string
		inputToken      int
		outputToken     int
		respSize        int
		cachedToken     int
	}
	scenarios := []struct {
		name string
		resp []responses
	}{{
		name: "multiple requests",
		resp: []responses{
			{
				modelName:       "m10",
				targetModelName: "t10",
				respSize:        1200,
				inputToken:      10,
				outputToken:     100,
				cachedToken:     5,
			},
			{
				modelName:       "m10",
				targetModelName: "t10",
				respSize:        500,
				inputToken:      20,
				outputToken:     200,
				cachedToken:     10,
			},
			{
				modelName:       "m10",
				targetModelName: "t11",
				respSize:        2480,
				inputToken:      30,
				outputToken:     300,
				cachedToken:     15,
			},
			{
				modelName:       "m20",
				targetModelName: "t20",
				respSize:        80,
				inputToken:      40,
				outputToken:     400,
				cachedToken:     20,
			},
		},
	}}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, resp := range scenario.resp {
				RecordInputTokens(resp.modelName, resp.targetModelName, "", "0", resp.inputToken)
				RecordOutputTokens(resp.modelName, resp.targetModelName, "", "0", resp.outputToken)
				RecordResponseSizes(resp.modelName, resp.targetModelName, "", "0", resp.respSize)
				RecordPromptCachedTokens(resp.modelName, resp.targetModelName, "", "0", resp.cachedToken)
			}

			// Verify deprecated metrics
			wantResponseSize, err := os.Open("testdata/response_sizes_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantResponseSize.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantResponseSize, responseSizesMetric); err != nil {
				t.Error(err)
			}

			wantInputToken, err := os.Open("testdata/input_tokens_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantInputToken.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantInputToken, inputTokensMetric); err != nil {
				t.Error(err)
			}

			wantOutputToken, err := os.Open("testdata/output_tokens_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantOutputToken.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantOutputToken, outputTokensMetric); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metrics
			wantResponseSizeNew, err := os.Open("testdata/llm_d_response_size_bytes_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantResponseSizeNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantResponseSizeNew, "llm_d_epp_response_size_bytes"); err != nil {
				t.Error(err)
			}

			wantInputTokenNew, err := os.Open("testdata/llm_d_input_tokens_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantInputTokenNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantInputTokenNew, "llm_d_epp_request_input_tokens"); err != nil {
				t.Error(err)
			}

			wantOutputTokenNew, err := os.Open("testdata/llm_d_output_tokens_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantOutputTokenNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantOutputTokenNew, "llm_d_epp_request_output_tokens"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRunningRequestsMetrics(t *testing.T) {
	Reset()
	type request struct {
		modelName   string
		targetModel string
		fairnessID  string
		priority    string
		complete    bool
	}

	scenarios := []struct {
		name     string
		requests []request
	}{
		{
			name: "basic test",
			requests: []request{
				{
					modelName:   "m1",
					targetModel: "t1",
					fairnessID:  "tenant-x",
					priority:    "10",
					complete:    false,
				},
				{
					modelName:   "m1",
					targetModel: "t1",
					fairnessID:  "tenant-x",
					priority:    "10",
					complete:    false,
				},
				{
					modelName:   "m1",
					targetModel: "t1",
					fairnessID:  "tenant-x",
					priority:    "10",
					complete:    true,
				},
				{
					modelName:   "m2",
					targetModel: "t2",
					fairnessID:  "tenant-y",
					priority:    "20",
					complete:    false,
				},
			},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, req := range scenario.requests {
				if req.complete {
					DecRunningRequests(req.modelName, req.targetModel, req.fairnessID, req.priority)
				} else {
					IncRunningRequests(req.modelName, req.targetModel, req.fairnessID, req.priority)
				}
			}

			// Verify deprecated metric
			wantRunningRequests, err := os.Open("testdata/running_requests_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRunningRequests.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantRunningRequests, runningRequestsMetric); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metric
			wantRunningRequestsNew, err := os.Open("testdata/llm_d_running_requests_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRunningRequestsNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantRunningRequestsNew, "llm_d_epp_request_running"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestInferencePoolMetrics(t *testing.T) {
	Reset()
	scenarios := []struct {
		name                  string
		poolName              string
		kvCacheAvg            float64
		queueSizeAvg          float64
		runningRequestsAvg    float64
		kvCacheStdDev         float64
		queueSizeStdDev       float64
		runningRequestsStdDev float64
	}{
		{
			name:                  "basic test",
			poolName:              "p1",
			kvCacheAvg:            0.3,
			queueSizeAvg:          0.4,
			runningRequestsAvg:    0.5,
			kvCacheStdDev:         0.1,
			queueSizeStdDev:       0.2,
			runningRequestsStdDev: 0.3,
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			RecordInferencePoolAvgKVCache(scenario.poolName, scenario.kvCacheAvg)
			RecordInferencePoolAvgQueueSize(scenario.poolName, scenario.queueSizeAvg)
			RecordInferencePoolAvgRunningRequests(scenario.poolName, scenario.runningRequestsAvg)
			RecordInferencePoolStdDevKVCache(scenario.poolName, scenario.kvCacheStdDev)
			RecordInferencePoolStdDevQueueSize(scenario.poolName, scenario.queueSizeStdDev)
			RecordInferencePoolStdDevRunningRequests(scenario.poolName, scenario.runningRequestsStdDev)

			// Verify deprecated metrics
			wantKVCache, err := os.Open("testdata/kv_cache_avg_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantKVCache.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantKVCache, kvCacheAvgUsageMetric); err != nil {
				t.Error(err)
			}

			wantQueueSize, err := os.Open("testdata/queue_avg_size_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantQueueSize.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantQueueSize, queueAvgSizeMetric); err != nil {
				t.Error(err)
			}

			wantRunningRequests, err := os.Open("testdata/running_requests_avg_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRunningRequests.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantRunningRequests, runningRequestsAvgMetric); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metrics
			wantKVCacheNew, err := os.Open("testdata/llm_d_kv_cache_avg_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantKVCacheNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantKVCacheNew, "llm_d_epp_average_kv_cache_utilization"); err != nil {
				t.Error(err)
			}

			wantQueueSizeNew, err := os.Open("testdata/llm_d_queue_avg_size_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantQueueSizeNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantQueueSizeNew, "llm_d_epp_average_queue_size"); err != nil {
				t.Error(err)
			}

			wantRunningRequestsNew, err := os.Open("testdata/llm_d_running_requests_avg_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRunningRequestsNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantRunningRequestsNew, "llm_d_epp_average_running_requests"); err != nil {
				t.Error(err)
			}

			wantKVCacheStdDev, err := os.Open("testdata/llm_d_kv_cache_std_dev_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantKVCacheStdDev.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantKVCacheStdDev, "llm_d_epp_std_dev_kv_cache_utilization"); err != nil {
				t.Error(err)
			}

			wantQueueSizeStdDev, err := os.Open("testdata/llm_d_queue_std_dev_size_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantQueueSizeStdDev.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantQueueSizeStdDev, "llm_d_epp_std_dev_queue_size"); err != nil {
				t.Error(err)
			}

			wantRunningRequestsStdDev, err := os.Open("testdata/llm_d_running_requests_std_dev_metrics")
			if err != nil {
				t.Fatal(err)
			}
			defer wantRunningRequestsStdDev.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantRunningRequestsStdDev, "llm_d_epp_std_dev_running_requests"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestPluginProcessingLatencies(t *testing.T) {
	Reset()
	type pluginLatency struct {
		extensionPoint string
		pluginType     string
		pluginName     string
		duration       time.Duration
	}
	scenarios := []struct {
		name      string
		latencies []pluginLatency
	}{
		{
			name: "multiple plugins",
			latencies: []pluginLatency{
				{
					extensionPoint: "ProfilePicker",
					pluginType:     "ProfileHandler",
					pluginName:     "PluginB",
					duration:       200 * time.Millisecond,
				},
				{
					extensionPoint: "Filter",
					pluginType:     "TestFilter",
					pluginName:     "PluginC",
					duration:       50 * time.Millisecond,
				},
				{
					extensionPoint: "Scorer",
					pluginType:     "TestScorer",
					pluginName:     "PluginD",
					duration:       10 * time.Millisecond,
				},
				{
					extensionPoint: "Picker",
					pluginType:     "TestPicker",
					pluginName:     "PluginE",
					duration:       10 * time.Microsecond,
				},
				{
					extensionPoint: "Admission",
					pluginType:     "TestAdmitter",
					pluginName:     "PluginF",
					duration:       5 * time.Millisecond,
				},
				{
					extensionPoint: "DataProducer",
					pluginType:     "TestDataProducer",
					pluginName:     "PluginG",
					duration:       1 * time.Millisecond,
				},
				{
					extensionPoint: "RequestParsing",
					pluginType:     "TestParser",
					pluginName:     "PluginH",
					duration:       2 * time.Millisecond,
				},
				{
					extensionPoint: "ResponseParsing",
					pluginType:     "TestParser",
					pluginName:     "PluginH",
					duration:       8 * time.Millisecond,
				},
			},
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, latency := range scenario.latencies {
				RecordPluginProcessingLatency(latency.extensionPoint, latency.pluginType, latency.pluginName, latency.duration)
			}

			// Verify deprecated metric
			wantPluginLatencies, err := os.Open("testdata/plugin_processing_latencies_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantPluginLatencies.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantPluginLatencies, "inference_extension_plugin_duration_seconds"); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metric
			wantPluginLatenciesNew, err := os.Open("testdata/llm_d_plugin_processing_latencies_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantPluginLatenciesNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantPluginLatenciesNew, "llm_d_epp_plugin_duration_seconds"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestSchedulerE2ELatency(t *testing.T) {
	Reset()
	scenarios := []struct {
		name      string
		durations []time.Duration
	}{
		{
			name: "multiple scheduling latencies",
			durations: []time.Duration{
				200 * time.Microsecond,
				800 * time.Microsecond,
				1500 * time.Microsecond,
				3 * time.Millisecond,
				8 * time.Millisecond,
				15 * time.Millisecond,
				30 * time.Millisecond,
				75 * time.Millisecond,
				150 * time.Millisecond,
			},
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, duration := range scenario.durations {
				RecordSchedulerE2ELatency(duration)
			}

			// Verify deprecated metric
			wantE2ELatency, err := os.Open("testdata/scheduler_e2e_duration_seconds_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantE2ELatency.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantE2ELatency, "inference_extension_scheduler_e2e_duration_seconds"); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metric
			wantE2ELatencyNew, err := os.Open("testdata/llm_d_scheduler_e2e_duration_seconds_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantE2ELatencyNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantE2ELatencyNew, "llm_d_epp_scheduler_e2e_duration_seconds"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestRequestProcessingLatency(t *testing.T) {
	Reset()
	durations := []time.Duration{
		200 * time.Microsecond,
		800 * time.Microsecond,
		1500 * time.Microsecond,
		3 * time.Millisecond,
		8 * time.Millisecond,
		15 * time.Millisecond,
		30 * time.Millisecond,
		75 * time.Millisecond,
		150 * time.Millisecond,
	}
	for _, duration := range durations {
		RecordRequestProcessingLatency(duration)
	}

	want, err := os.Open("testdata/llm_d_request_processing_duration_seconds_metric")
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()
	if err := promtestutil.GatherAndCompare(metrics.Registry, want, "llm_d_epp_request_processing_duration_seconds"); err != nil {
		t.Error(err)
	}
}

func TestResponseProcessingLatency(t *testing.T) {
	Reset()
	durations := []time.Duration{
		200 * time.Microsecond,
		800 * time.Microsecond,
		1500 * time.Microsecond,
		3 * time.Millisecond,
		8 * time.Millisecond,
		15 * time.Millisecond,
		30 * time.Millisecond,
		75 * time.Millisecond,
		150 * time.Millisecond,
	}
	for _, duration := range durations {
		RecordResponseProcessingLatency(duration)
	}

	want, err := os.Open("testdata/llm_d_response_processing_duration_seconds_metric")
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()
	if err := promtestutil.GatherAndCompare(metrics.Registry, want, "llm_d_epp_response_processing_duration_seconds"); err != nil {
		t.Error(err)
	}
}

func TestFlowControlDispatchCycleLengthMetric(t *testing.T) {
	Reset()
	scenarios := []struct {
		name      string
		durations []time.Duration
	}{
		{
			name: "multiple scheduling latencies",
			durations: []time.Duration{
				50 * time.Microsecond,
				150 * time.Microsecond,
				300 * time.Microsecond,
				800 * time.Microsecond,
				1500 * time.Microsecond,
				4 * time.Millisecond,
				8 * time.Millisecond,
				15 * time.Millisecond,
				30 * time.Millisecond,
				80 * time.Millisecond,
				200 * time.Millisecond,
			},
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, duration := range scenario.durations {
				RecordFlowControlDispatchCycleDuration(duration)
			}

			// Verify deprecated metric
			wantDispatchCycleLatency, err := os.Open("testdata/flow_control_dispatch_cycle_duration_seconds_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantDispatchCycleLatency.Close()
			if err := testutil.GatherAndCompare(metrics.Registry, wantDispatchCycleLatency, "inference_extension_flow_control_dispatch_cycle_duration_seconds"); err != nil {
				t.Error(err)
			}

			// Verify llm_d_epp metric
			wantDispatchCycleLatencyNew, err := os.Open("testdata/llm_d_flow_control_dispatch_cycle_duration_seconds_metric")
			if err != nil {
				t.Fatal(err)
			}
			defer wantDispatchCycleLatencyNew.Close()
			if err := promtestutil.GatherAndCompare(metrics.Registry, wantDispatchCycleLatencyNew, "llm_d_epp_flow_control_dispatch_cycle_duration_seconds"); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestFlowControlEnqueueDurationMetric(t *testing.T) {
	Reset()

	scenarios := []struct {
		name       string
		priorities []string
		outcomes   []string
		durations  []time.Duration
	}{
		{
			name: "multiple enqueue latencies",
			priorities: []string{
				"1", "1", "1", "1", "1", "1", "1",
				"2", "2", "2", "2", "2", "2", "2",
			},
			outcomes: []string{
				"Dispatched", "NotYetFinalized", "RejectedCapacity", "RejectedOther", "EvictedTTL", "EvictedContextCancelled", "EvictedOther",
				"Dispatched", "NotYetFinalized", "RejectedCapacity", "RejectedOther", "EvictedTTL", "EvictedContextCancelled", "EvictedOther",
			},
			durations: []time.Duration{
				50 * time.Microsecond,
				200 * time.Millisecond,
				400 * time.Microsecond,
				15 * time.Millisecond,
				1500 * time.Microsecond,
				80 * time.Millisecond,
				100 * time.Nanosecond,
				800 * time.Microsecond,
				1 * time.Second,
				4 * time.Millisecond,
				40 * time.Millisecond,
				8 * time.Millisecond,
				500 * time.Millisecond,
				150 * time.Microsecond,
			},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for i := range scenario.priorities {
				RecordFlowControlRequestEnqueueDuration(
					"default-fairness",
					scenario.priorities[i],
					scenario.outcomes[i],
					scenario.durations[i],
				)
			}

			// Verify deprecated metric
			func() {
				wantEnqueueLatency, err := os.Open("testdata/flow_control_enqueue_duration_seconds_metric")
				if err != nil {
					t.Fatal(err)
				}
				defer wantEnqueueLatency.Close()
				if err := testutil.GatherAndCompare(metrics.Registry, wantEnqueueLatency, "inference_extension_flow_control_request_enqueue_duration_seconds"); err != nil {
					t.Error(err)
				}
			}()

			// Verify llm_d_epp metric
			func() {
				wantEnqueueLatencyNew, err := os.Open("testdata/llm_d_flow_control_enqueue_duration_seconds_metric")
				if err != nil {
					t.Fatal(err)
				}
				defer wantEnqueueLatencyNew.Close()
				if err := promtestutil.GatherAndCompare(metrics.Registry, wantEnqueueLatencyNew, "llm_d_epp_flow_control_request_enqueue_duration_seconds"); err != nil {
					t.Error(err)
				}
			}()
		})
	}
}

func TestSchedulerAttemptsTotal(t *testing.T) {
	compareMetrics := func(t *testing.T, goldenFile string) {
		t.Helper()
		wantMetrics, err := os.Open(goldenFile)
		if err != nil {
			t.Fatal(err)
		}
		defer wantMetrics.Close()
		if err := testutil.GatherAndCompare(metrics.Registry, wantMetrics, "inference_extension_scheduler_attempts_total"); err != nil {
			t.Errorf("metric comparison failed: %v", err)
		}
	}

	compareMetricsNew := func(t *testing.T, goldenFile string) {
		t.Helper()
		wantMetrics, err := os.Open(goldenFile)
		if err != nil {
			t.Fatal(err)
		}
		defer wantMetrics.Close()
		if err := promtestutil.GatherAndCompare(metrics.Registry, wantMetrics, "llm_d_epp_scheduler_attempts_total"); err != nil {
			t.Errorf("metric comparison failed: %v", err)
		}
	}

	t.Run("success with endpoint metadata", func(t *testing.T) {
		Reset()
		result := &fwksched.SchedulingResult{
			PrimaryProfileName: "primary",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"primary": {
					TargetEndpoints: []fwksched.Endpoint{
						fwksched.NewEndpoint(
							&fwkdl.EndpointMetadata{
								NamespacedName: k8stypes.NamespacedName{Name: "pod-1", Namespace: "ns-1"},
								PodName:        "pod-1",
								Port:           "8080",
							},
							nil, nil,
						),
					},
				},
			},
		}
		RecordSchedulerAttempt(nil, "modelA", result)
		RecordSchedulerAttempt(nil, "modelA", result)
		compareMetrics(t, "testdata/scheduler_attempts_with_result_metrics")
		compareMetricsNew(t, "testdata/llm_d_scheduler_attempts_with_result_metrics")
	})

	t.Run("success with multiple endpoints uses first", func(t *testing.T) {
		Reset()
		result := &fwksched.SchedulingResult{
			PrimaryProfileName: "primary",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"primary": {
					TargetEndpoints: []fwksched.Endpoint{
						fwksched.NewEndpoint(
							&fwkdl.EndpointMetadata{
								NamespacedName: k8stypes.NamespacedName{Name: "pod-1", Namespace: "ns-1"},
								PodName:        "pod-1",
								Port:           "8080",
							},
							nil, nil,
						),
						fwksched.NewEndpoint(
							&fwkdl.EndpointMetadata{
								NamespacedName: k8stypes.NamespacedName{Name: "pod-2", Namespace: "ns-2"},
								PodName:        "pod-2",
								Port:           "9090",
							},
							nil, nil,
						),
					},
				},
			},
		}
		RecordSchedulerAttempt(nil, "modelA", result)
		RecordSchedulerAttempt(nil, "modelB", result)
		compareMetrics(t, "testdata/scheduler_attempts_multiple_endpoints_metrics")
		compareMetricsNew(t, "testdata/llm_d_scheduler_attempts_multiple_endpoints_metrics")
	})

	t.Run("success with different models and endpoints", func(t *testing.T) {
		Reset()
		resultA := &fwksched.SchedulingResult{
			PrimaryProfileName: "primary",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"primary": {
					TargetEndpoints: []fwksched.Endpoint{
						fwksched.NewEndpoint(
							&fwkdl.EndpointMetadata{
								NamespacedName: k8stypes.NamespacedName{Name: "pod-1", Namespace: "ns-1"},
								PodName:        "pod-1",
								Port:           "8080",
							},
							nil, nil,
						),
					},
				},
			},
		}
		resultB := &fwksched.SchedulingResult{
			PrimaryProfileName: "primary",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"primary": {
					TargetEndpoints: []fwksched.Endpoint{
						fwksched.NewEndpoint(
							&fwkdl.EndpointMetadata{
								NamespacedName: k8stypes.NamespacedName{Name: "pod-2", Namespace: "ns-2"},
								PodName:        "pod-2",
								Port:           "9090",
							},
							nil, nil,
						),
					},
				},
			},
		}
		RecordSchedulerAttempt(nil, "modelA", resultA)
		RecordSchedulerAttempt(nil, "modelA", resultA)
		RecordSchedulerAttempt(nil, "modelB", resultB)
		compareMetrics(t, "testdata/scheduler_attempts_different_models_metrics")
		compareMetricsNew(t, "testdata/llm_d_scheduler_attempts_different_models_metrics")
	})

	t.Run("mixed success and failure attempts", func(t *testing.T) {
		Reset()
		for range 10 {
			RecordSchedulerAttempt(nil, "modelA", nil)
		}
		for range 5 {
			RecordSchedulerAttempt(errors.New("simulated scheduling failure"), "modelA", nil)
		}
		compareMetrics(t, "testdata/scheduler_attempts_total_metrics")
		compareMetricsNew(t, "testdata/llm_d_scheduler_attempts_total_metrics")
	})
}

func getHistogramVecLabelValues(t *testing.T, h *prometheus.HistogramVec, labelValues ...string) (*dto.Histogram, error) {
	t.Helper()
	m, err := h.GetMetricWithLabelValues(labelValues...)
	if err != nil {
		return nil, err
	}
	metricDto := &dto.Metric{}
	if err := m.(prometheus.Histogram).Write(metricDto); err != nil {
		return nil, err
	}
	return metricDto.GetHistogram(), nil
}

func TestFlowControlQueueDurationMetric(t *testing.T) {
	Reset()

	const (
		pool   = "pool-1"
		model  = "qwen-3"
		target = "qwen-3-base"
	)

	records := []struct {
		fairnessID string
		priority   string
		outcome    string
		duration   time.Duration
	}{
		{fairnessID: "user-a", priority: "100", outcome: "Dispatched", duration: 10 * time.Millisecond},
		{fairnessID: "user-a", priority: "100", outcome: "Dispatched", duration: 20 * time.Millisecond},
		{fairnessID: "user-b", priority: "100", outcome: "RejectedCapacity", duration: 5 * time.Millisecond},
		{fairnessID: "user-a", priority: "50", outcome: "Dispatched", duration: 100 * time.Millisecond},
	}

	for _, rec := range records {
		RecordFlowControlRequestQueueDuration(rec.fairnessID, rec.priority, rec.outcome, pool, model, target, rec.duration)
	}

	testCases := []struct {
		name        string
		labels      prometheus.Labels
		expectCount uint64
		expectSum   float64
	}{
		{
			name: "user-a, prio 100, dispatched",
			labels: prometheus.Labels{
				"fairness_id":       "user-a",
				"priority":          "100",
				"outcome":           "Dispatched",
				"inference_pool":    pool,
				"model_name":        model,
				"target_model_name": target,
			},
			expectCount: 2,
			expectSum:   0.03,
		},
		{
			name: "user-b, prio 100, rejected",
			labels: prometheus.Labels{
				"fairness_id":       "user-b",
				"priority":          "100",
				"outcome":           "RejectedCapacity",
				"inference_pool":    pool,
				"model_name":        model,
				"target_model_name": target,
			},
			expectCount: 1,
			expectSum:   0.005,
		},
		{
			name: "user-a, prio 50, dispatched",
			labels: prometheus.Labels{
				"fairness_id":       "user-a",
				"priority":          "50",
				"outcome":           "Dispatched",
				"inference_pool":    pool,
				"model_name":        model,
				"target_model_name": target,
			},
			expectCount: 1,
			expectSum:   0.1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			labels := []string{
				tc.labels["fairness_id"],
				tc.labels["priority"],
				tc.labels["outcome"],
				tc.labels["inference_pool"],
				tc.labels["model_name"],
				tc.labels["target_model_name"],
			}

			// Deprecated metric
			hist, err := getHistogramVecLabelValues(t, flowControlRequestQueueDuration, labels...)
			require.NoError(t, err, "Failed to get histogram for labels %v", tc.labels)
			require.Equal(t, tc.expectCount, hist.GetSampleCount(), "Sample count mismatch for labels %v", tc.labels)
			require.InDelta(t, tc.expectSum, hist.GetSampleSum(), 0.00001, "Sample sum mismatch for labels %v", tc.labels)

			// llm_d_epp metric
			histNew, err := getHistogramVecLabelValues(t, llmdFlowControlRequestQueueDuration, labels...)
			require.NoError(t, err, "Failed to get llm_d_epp histogram for labels %v", tc.labels)
			require.Equal(t, tc.expectCount, histNew.GetSampleCount(), "llm_d_epp sample count mismatch for labels %v", tc.labels)
			require.InDelta(t, tc.expectSum, histNew.GetSampleSum(), 0.00001, "llm_d_epp sample sum mismatch for labels %v", tc.labels)
		})
	}
}

func TestFlowControlQueueSizeMetric(t *testing.T) {
	Reset()

	const (
		pool   = "pool-1"
		model  = "qwen-3"
		target = "qwen-3-base"
	)

	// Basic Inc/Dec
	IncFlowControlQueueSize("user-a", "100", pool, model, target)
	val, err := testutil.GetGaugeMetricValue(flowControlQueueSize.WithLabelValues("user-a", "100", pool, model, target))
	require.NoError(t, err)
	require.Equal(t, 1.0, val)

	valNew, err := testutil.GetGaugeMetricValue(llmdFlowControlQueueSize.WithLabelValues("user-a", "100", pool, model, target))
	require.NoError(t, err)
	require.Equal(t, 1.0, valNew)

	DecFlowControlQueueSize("user-a", "100", pool, model, target)
	val, err = testutil.GetGaugeMetricValue(flowControlQueueSize.WithLabelValues("user-a", "100", pool, model, target))
	require.NoError(t, err)
	require.Equal(t, 0.0, val)

	valNew, err = testutil.GetGaugeMetricValue(llmdFlowControlQueueSize.WithLabelValues("user-a", "100", pool, model, target))
	require.NoError(t, err)
	require.Equal(t, 0.0, valNew)
}

func TestFlowControlQueueBytesMetric(t *testing.T) {
	Reset()

	const (
		pool   = "pool-1"
		model  = "qwen-3"
		target = "qwen-3-base"
	)

	AddFlowControlQueueBytes("user-a", "100", pool, model, target, 32)
	val, err := testutil.GetGaugeMetricValue(flowControlQueueBytes.WithLabelValues("user-a", "100", pool, model, target))
	require.NoError(t, err)
	require.Equal(t, 32.0, val)

	valNew, err := testutil.GetGaugeMetricValue(llmdFlowControlQueueBytes.WithLabelValues("user-a", "100", pool, model, target))
	require.NoError(t, err)
	require.Equal(t, 32.0, valNew)

	SubFlowControlQueueBytes("user-a", "100", pool, model, target, 32)
	val, err = testutil.GetGaugeMetricValue(flowControlQueueBytes.WithLabelValues("user-a", "100", pool, model, target))
	require.NoError(t, err)
	require.Equal(t, 0.0, val)

	valNew, err = testutil.GetGaugeMetricValue(llmdFlowControlQueueBytes.WithLabelValues("user-a", "100", pool, model, target))
	require.NoError(t, err)
	require.Equal(t, 0.0, valNew)
}

func TestFlowControlPoolSaturationMetric(t *testing.T) {
	Reset()

	const pool = "test-pool"

	RecordFlowControlPoolSaturation(pool, 0.5)
	val, err := testutil.GetGaugeMetricValue(flowControlPoolSaturation.WithLabelValues(pool))
	require.NoError(t, err)
	require.Equal(t, 0.5, val)

	valNew, err := testutil.GetGaugeMetricValue(llmdFlowControlPoolSaturation.WithLabelValues(pool))
	require.NoError(t, err)
	require.Equal(t, 0.5, valNew)
}

func TestFlowControlRequestsTotalMetric(t *testing.T) {
	Reset()

	const pool = "test-pool"

	IncFlowControlRequestsTotal("Dispatched", "0", pool)
	IncFlowControlRequestsTotal("Dispatched", "0", pool)
	IncFlowControlRequestsTotal("RejectedCapacity", "10", pool)

	val, err := testutil.GetCounterMetricValue(llmdFlowControlRequestsTotal.WithLabelValues("Dispatched", "0", pool))
	require.NoError(t, err)
	require.Equal(t, float64(2), val)

	val, err = testutil.GetCounterMetricValue(llmdFlowControlRequestsTotal.WithLabelValues("RejectedCapacity", "10", pool))
	require.NoError(t, err)
	require.Equal(t, float64(1), val)
}

func TestRecordRequestTTFT(t *testing.T) {
	Reset()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	timeBaseline := time.Now()

	t.Run("valid streaming requests", func(t *testing.T) {
		require.True(t, RecordRequestTTFT(ctx, "m10", "t10", "tenant-a", "3", true, timeBaseline, timeBaseline.Add(10*time.Millisecond)))
		require.True(t, RecordRequestTTFT(ctx, "m10", "t10", "tenant-a", "3", true, timeBaseline, timeBaseline.Add(1600*time.Millisecond)))
		require.True(t, RecordRequestTTFT(ctx, "m20", "t20", "tenant-b", "5", false, timeBaseline, timeBaseline.Add(120*time.Millisecond)))

		h, err := getHistogramVecLabelValues(t, llmdRequestTTFT, "m10", "t10", "tenant-a", "3", "true")
		require.NoError(t, err)
		require.Equal(t, uint64(2), h.GetSampleCount())
		require.InDelta(t, 1.61, h.GetSampleSum(), 0.001)

		h, err = getHistogramVecLabelValues(t, llmdRequestTTFT, "m20", "t20", "tenant-b", "5", "false")
		require.NoError(t, err)
		require.Equal(t, uint64(1), h.GetSampleCount())
		require.InDelta(t, 0.12, h.GetSampleSum(), 0.001)
	})

	t.Run("zero first token timestamp", func(t *testing.T) {
		require.False(t, RecordRequestTTFT(ctx, "m10", "t10", "tenant-a", "3", true, timeBaseline, time.Time{}))
	})

	t.Run("first token before received", func(t *testing.T) {
		require.False(t, RecordRequestTTFT(ctx, "m10", "t10", "tenant-a", "3", true, timeBaseline.Add(10*time.Millisecond), timeBaseline))
	})
}

func TestRecordRequestTPOT(t *testing.T) {
	Reset()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	timeBaseline := time.Now()

	t.Run("valid multi-token request", func(t *testing.T) {
		received := timeBaseline
		firstToken := timeBaseline.Add(500 * time.Millisecond)
		complete := timeBaseline.Add(2000 * time.Millisecond)
		require.True(t, RecordRequestTPOT(ctx, "m10", "t10", "tenant-a", "3", received, firstToken, complete, 11))

		h, err := getHistogramVecLabelValues(t, llmdRequestTPOT, "m10", "t10", "tenant-a", "3")
		require.NoError(t, err)
		require.Equal(t, uint64(1), h.GetSampleCount())
		require.InDelta(t, 0.15, h.GetSampleSum(), 0.001)
	})

	t.Run("single token skipped", func(t *testing.T) {
		require.False(t, RecordRequestTPOT(ctx, "m10", "t10", "tenant-a", "3", timeBaseline, timeBaseline.Add(100*time.Millisecond), timeBaseline.Add(200*time.Millisecond), 1))
	})

	t.Run("zero tokens skipped", func(t *testing.T) {
		require.False(t, RecordRequestTPOT(ctx, "m10", "t10", "tenant-a", "3", timeBaseline, timeBaseline.Add(100*time.Millisecond), timeBaseline.Add(200*time.Millisecond), 0))
	})

	t.Run("zero first token timestamp", func(t *testing.T) {
		require.False(t, RecordRequestTPOT(ctx, "m10", "t10", "tenant-a", "3", timeBaseline, time.Time{}, timeBaseline.Add(200*time.Millisecond), 10))
	})

	t.Run("first token before received", func(t *testing.T) {
		require.False(t, RecordRequestTPOT(ctx, "m10", "t10", "tenant-a", "3", timeBaseline.Add(100*time.Millisecond), timeBaseline, timeBaseline.Add(200*time.Millisecond), 10))
	})

	t.Run("complete before first token", func(t *testing.T) {
		require.False(t, RecordRequestTPOT(ctx, "m10", "t10", "tenant-a", "3", timeBaseline, timeBaseline.Add(200*time.Millisecond), timeBaseline.Add(100*time.Millisecond), 10))
	})
}

func TestRecordInterTokenLatency(t *testing.T) {
	Reset()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	t.Run("valid observations", func(t *testing.T) {
		require.True(t, RecordInterTokenLatency(ctx, "m10", "t10", "tenant-a", "3", 0.05))
		require.True(t, RecordInterTokenLatency(ctx, "m10", "t10", "tenant-a", "3", 0.08))
		require.True(t, RecordInterTokenLatency(ctx, "m10", "t10", "tenant-a", "3", 0.12))
		require.True(t, RecordInterTokenLatency(ctx, "m20", "t20", "tenant-b", "5", 0.03))

		h, err := getHistogramVecLabelValues(t, llmdInterTokenLatency, "m10", "t10", "tenant-a", "3")
		require.NoError(t, err)
		require.Equal(t, uint64(3), h.GetSampleCount())
		require.InDelta(t, 0.25, h.GetSampleSum(), 0.001)

		h, err = getHistogramVecLabelValues(t, llmdInterTokenLatency, "m20", "t20", "tenant-b", "5")
		require.NoError(t, err)
		require.Equal(t, uint64(1), h.GetSampleCount())
		require.InDelta(t, 0.03, h.GetSampleSum(), 0.001)
	})

	t.Run("zero latency accepted", func(t *testing.T) {
		require.True(t, RecordInterTokenLatency(ctx, "m10", "t10", "tenant-a", "3", 0))
	})

	t.Run("negative latency rejected", func(t *testing.T) {
		require.False(t, RecordInterTokenLatency(ctx, "m10", "t10", "tenant-a", "3", -0.01))
	})
}

func TestInferenceModelRewriteDecisionsTotalMetric(t *testing.T) {
	Reset()

	RecordInferenceModelRewriteDecision("rewrite-rule-1", "model-a", "model-b")

	val, err := testutil.GetCounterMetricValue(inferenceModelRewriteDecisionsTotal.WithLabelValues("rewrite-rule-1", "model-a", "model-b"))
	require.NoError(t, err)
	require.Equal(t, 1.0, val)

	valNew, err := testutil.GetCounterMetricValue(llmdInferenceModelRewriteDecisionsTotal.WithLabelValues("rewrite-rule-1", "model-a", "model-b"))
	require.NoError(t, err)
	require.Equal(t, 1.0, valNew)
}
