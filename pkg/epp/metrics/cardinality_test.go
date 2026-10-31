/*
Copyright 2025 The llm-d Authors.

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
	"fmt"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestRequestMetricsBoundAPISurface(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	received := time.Now()
	record := func(surface reqcommon.APISurface) {
		RecordRequestCounter("model", "target", surface, "", 0)
		RecordRequestErrCounter(context.Background(), "model", "target", surface, "", "0", "502")
		require.True(t, RecordRequestLatencies(context.Background(), "model", "target", surface, "", "0", received, received.Add(time.Second)))
		RecordRequestSizes("model", "target", surface, "", "0", 100)
		RecordResponseSizes("model", "target", surface, "", "0", 200)
		RecordInputTokens("model", "target", surface, "", "0", 10)
		RecordOutputTokens("model", "target", surface, "", "0", 10)
		RecordPromptCachedTokens("model", "target", surface, "", "0", 5)
		require.True(t, RecordNormalizedTimePerOutputToken(context.Background(), "model", "target", surface, "", "0", received, received.Add(time.Second), 10))
		require.True(t, RecordRequestTTFT(context.Background(), "model", "target", surface, "", "0", true, received, received.Add(time.Millisecond)))
		require.True(t, RecordRequestTPOT(context.Background(), "model", "target", surface, "", "0", true, received, received.Add(time.Millisecond), received.Add(time.Second), 10))
		require.True(t, RecordInterTokenLatency(context.Background(), "model", "target", surface, "", "0", 0.01))
		IncRunningRequests("model", "target", surface, "", "0")
		DecRunningRequests("model", "target", surface, "", "0")
		RecordRequestProcessingLatency(surface, time.Millisecond)
		RecordResponseProcessingLatency(surface, time.Millisecond)
	}
	for _, apiType := range []reqcommon.APIType{
		reqcommon.APITypeChatCompletions, reqcommon.APITypeCompletions, reqcommon.APITypeResponses,
		reqcommon.APITypeMessages, reqcommon.APITypeVLLMGenerate, reqcommon.APITypeSGLangGenerate,
	} {
		record(apiType.Surface())
	}
	for i := 0; i < 1000; i++ {
		record(reqcommon.DetectAPISurface(fmt.Sprintf("/unrecognized/%d", i)))
		record(reqcommon.APISurface(1000 + i))
	}
	for _, collector := range []prometheus.Collector{
		llmdRequestCounter, llmdRequestErrCounter, llmdRequestLatencies, llmdRequestSizes, llmdResponseSizes,
		llmdInputTokens, llmdOutputTokens, llmdPromptCachedTokens, llmdNormalizedTimePerOutputToken,
		llmdRequestTTFT, llmdRequestTPOT, llmdInterTokenLatency, llmdRunningRequests,
		llmdRequestProcessingLatency, llmdResponseProcessingLatency,
	} {
		require.Equal(t, 7, promtestutil.CollectAndCount(collector), "six supported surfaces plus unknown must bound the series count")
	}
	require.Equal(t, float64(2000), promtestutil.ToFloat64(llmdRequestCounter.WithLabelValues(
		boundModel("model"), boundModel("target"), "", "0", reqcommon.APISurfaceUnknown.String(),
	)))
	for _, surface := range []string{reqcommon.APITypeMessages.String(), reqcommon.APISurfaceUnknown.String()} {
		require.Zero(t, promtestutil.ToFloat64(llmdRunningRequests.WithLabelValues(
			boundModel("model"), boundModel("target"), "", "0", surface,
		)))
	}
}

// resetFairnessLabelLimit restores the shared fairness cap to its default after a test.
func resetFairnessLabelLimit(t *testing.T) {
	t.Helper()
	t.Cleanup(func() { metricsutil.SetFairnessIDLabelLimit(metricsutil.DefaultFairnessIDLabelLimit) })
}

// PreAdmitModelLabels feeds the package-level limiter used by the record
// functions, so a flood of junk names must not displace a configured model.
func TestPreAdmitModelLabelsSurvivesFlood(t *testing.T) {
	const testCap = 5
	old := modelLabelLimiter
	modelLabelLimiter = metricsutil.NewBoundedLabel(testCap)
	llmdRequestCounter.Reset()
	t.Cleanup(func() {
		modelLabelLimiter = old
		llmdRequestCounter.Reset()
	})

	PreAdmitModelLabels("llama-70b", "llama-70b-canary")
	for i := 0; i < 1000; i++ {
		RecordRequestCounter(fmt.Sprintf("junk-%d", i), "target", reqcommon.APISurfaceUnknown, "fairness", 0)
	}
	RecordRequestCounter("llama-70b", "llama-70b-canary", reqcommon.APISurfaceUnknown, "fairness", 0)

	series := promtestutil.CollectAndCount(llmdRequestCounter)
	require.LessOrEqualf(t, series, testCap+2, "cardinality must stay bounded, got %d series", series)
	require.Equal(t, "llama-70b", boundModel("llama-70b"),
		"configured model emits its real label after the flood")
	require.Equal(t, "llama-70b-canary", boundModel("llama-70b-canary"),
		"configured target emits its real label after the flood")
}

// RecordRequestCounter draws its model_name label from the request body, so a
// flood of distinct names must not grow the time series set without bound.
func TestRecordRequestCounterBoundsModelCardinality(t *testing.T) {
	const testCap = 5
	old := modelLabelLimiter
	modelLabelLimiter = metricsutil.NewBoundedLabel(testCap)
	llmdRequestCounter.Reset()
	t.Cleanup(func() {
		modelLabelLimiter = old
		llmdRequestCounter.Reset()
	})

	for i := 0; i < 1000; i++ {
		RecordRequestCounter(fmt.Sprintf("model-%d", i), "target", reqcommon.APISurfaceUnknown, "fairness", 0)
	}

	count := promtestutil.CollectAndCount(llmdRequestCounter)
	require.LessOrEqualf(t, count, testCap+1,
		"model_name cardinality must stay bounded by the cap, got %d series", count)
}

// The fairness_id label is populated from a client request header, so the shared limiter must
// collapse an unbounded flood of distinct IDs into the overflow bucket instead of minting a series
// per ID.
func TestFairnessLabelFloodCollapsesToOverflow(t *testing.T) {
	const testCap = 5
	resetFairnessLabelLimit(t)
	metricsutil.SetFairnessIDLabelLimit(testCap)
	llmdFlowControlRequestEnqueueDuration.Reset()
	t.Cleanup(func() {
		llmdFlowControlRequestEnqueueDuration.Reset()
	})

	for i := 0; i < 1000; i++ {
		RecordFlowControlRequestEnqueueDuration(fmt.Sprintf("tenant-%d", i), "0", "Dispatched", time.Millisecond)
	}

	// testCap admitted IDs + 1 overflow series.
	require.Equal(t, testCap+1, promtestutil.CollectAndCount(llmdFlowControlRequestEnqueueDuration),
		"1000 distinct fairness IDs must collapse to cap+overflow series, not one series each")
}

// DeleteFlowControlFlowSeries backs the flow registry's GC hook: once a flow is collected, its
// series must not linger for the lifetime of the process.
func TestDeleteFlowControlFlowSeries(t *testing.T) {
	resetFairnessLabelLimit(t)
	metricsutil.SetFairnessIDLabelLimit(10)
	llmdFlowControlRequestEnqueueDuration.Reset()
	t.Cleanup(func() {
		llmdFlowControlRequestEnqueueDuration.Reset()
	})

	RecordFlowControlRequestEnqueueDuration("tenant-a", "0", "Dispatched", time.Millisecond)
	RecordFlowControlRequestEnqueueDuration("tenant-a", "0", "Rejected", time.Millisecond)
	RecordFlowControlRequestEnqueueDuration("tenant-b", "0", "Dispatched", time.Millisecond)
	require.Equal(t, 3, promtestutil.CollectAndCount(llmdFlowControlRequestEnqueueDuration),
		"setup: expected one series per (fairness_id, outcome) pair")

	DeleteFlowControlFlowSeries("tenant-a", "0")

	require.Equal(t, 1, promtestutil.CollectAndCount(llmdFlowControlRequestEnqueueDuration),
		"all of tenant-a's series (every outcome) must be pruned; tenant-b's must survive")
}

// The bound is applied mechanically in every record function that takes a fairness ID; this guards
// the request-metric family (distinct label ordering from the flow control family) against the
// pattern regressing there.
func TestFairnessLabelBoundOnRequestMetrics(t *testing.T) {
	const testCap = 3
	resetFairnessLabelLimit(t)
	metricsutil.SetFairnessIDLabelLimit(testCap)
	oldModels := modelLabelLimiter
	modelLabelLimiter = metricsutil.NewBoundedLabel(10)
	llmdRequestCounter.Reset()
	t.Cleanup(func() {
		modelLabelLimiter = oldModels
		llmdRequestCounter.Reset()
	})

	for i := 0; i < 100; i++ {
		RecordRequestCounter("model-a", "model-a", reqcommon.APISurfaceUnknown, fmt.Sprintf("tenant-%d", i), 0)
	}

	require.Equal(t, testCap+1, promtestutil.CollectAndCount(llmdRequestCounter),
		"100 distinct fairness IDs must collapse to cap+overflow series on the request family")
}

// RecordFlowControlRequestQueueDuration takes request-body model names; they must flow through the
// model limiter like every sibling record function.
func TestQueueDurationBoundsModelLabels(t *testing.T) {
	const testCap = 3
	oldModels := modelLabelLimiter
	modelLabelLimiter = metricsutil.NewBoundedLabel(testCap)
	resetFairnessLabelLimit(t)
	metricsutil.SetFairnessIDLabelLimit(10)
	llmdFlowControlRequestQueueDuration.Reset()
	t.Cleanup(func() {
		modelLabelLimiter = oldModels
		llmdFlowControlRequestQueueDuration.Reset()
	})

	for i := 0; i < 100; i++ {
		m := fmt.Sprintf("model-%d", i)
		RecordFlowControlRequestQueueDuration(context.Background(), "tenant", "0", "Dispatched", "pool", m, m, time.Millisecond)
	}

	require.Equal(t, testCap+1, promtestutil.CollectAndCount(llmdFlowControlRequestQueueDuration),
		"100 distinct model names must collapse to cap+overflow series, not one series each")
}

// A client can choose the overflow value itself as its fairness ID; GC of that flow must not
// delete the shared overflow series that aggregates every capped-out tenant.
func TestDeleteFlowControlFlowSeriesPreservesOverflowSeries(t *testing.T) {
	resetFairnessLabelLimit(t)
	metricsutil.SetFairnessIDLabelLimit(1)
	llmdFlowControlRequestEnqueueDuration.Reset()
	t.Cleanup(func() {
		llmdFlowControlRequestEnqueueDuration.Reset()
	})

	// tenant-a fills the single cap slot; tenant-b folds to the overflow series.
	RecordFlowControlRequestEnqueueDuration("tenant-a", "0", "Dispatched", time.Millisecond)
	RecordFlowControlRequestEnqueueDuration("tenant-b", "0", "Dispatched", time.Millisecond)
	require.Equal(t, 2, promtestutil.CollectAndCount(llmdFlowControlRequestEnqueueDuration),
		"setup: expected the admitted series plus the overflow series")

	DeleteFlowControlFlowSeries(metricsutil.OverflowValue, "0")

	require.Equal(t, 2, promtestutil.CollectAndCount(llmdFlowControlRequestEnqueueDuration),
		"deleting the overflow value must be a no-op; the shared overflow series must survive")
}

// The configured cap propagates through the record functions to the emitted series.
func TestFairnessIDLabelLimitAppliesToRecordedMetrics(t *testing.T) {
	resetFairnessLabelLimit(t)
	t.Cleanup(func() {
		llmdFlowControlRequestEnqueueDuration.Reset()
	})

	tests := []struct {
		name       string
		limit      int
		wantSeries int
	}{
		// cap distinct IDs to the limit plus one overflow series.
		{name: "low cap", limit: 3, wantSeries: 4},
		// a zero cap folds every ID onto the single overflow series.
		{name: "zero disables per-id recording", limit: 0, wantSeries: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metricsutil.SetFairnessIDLabelLimit(test.limit)
			llmdFlowControlRequestEnqueueDuration.Reset()

			for i := 0; i < 100; i++ {
				RecordFlowControlRequestEnqueueDuration(fmt.Sprintf("tenant-%d", i), "0", "Dispatched", time.Millisecond)
			}

			require.Equal(t, test.wantSeries, promtestutil.CollectAndCount(llmdFlowControlRequestEnqueueDuration))
		})
	}
}

// Each resolution is counted under its bounded ID, and folded IDs also count as overflow
// for their source.
func TestRecordFairnessIDResolution(t *testing.T) {
	resetFairnessLabelLimit(t)
	metricsutil.SetFairnessIDLabelLimit(1)
	llmdFairnessIDResolvedCounter.Reset()
	llmdFairnessIDOverflowCounter.Reset()
	t.Cleanup(func() {
		llmdFairnessIDResolvedCounter.Reset()
		llmdFairnessIDOverflowCounter.Reset()
	})

	InitFairnessIDResolutionMetrics("pool-a")
	RecordFairnessIDResolution("pool-a", "team-a", "header")
	RecordFairnessIDResolution("pool-a", "team-b", "header")
	RecordFairnessIDResolution("pool-a", "session-1", "agent_identity")
	RecordFairnessIDResolution("pool-a", "default-flow", "default")

	resolved := func(id, source string) float64 {
		return promtestutil.ToFloat64(llmdFairnessIDResolvedCounter.WithLabelValues(id, source, "pool-a"))
	}
	overflow := func(source string) float64 {
		return promtestutil.ToFloat64(llmdFairnessIDOverflowCounter.WithLabelValues(source, "pool-a"))
	}
	require.Equal(t, 1.0, resolved("team-a", "header"))
	require.Equal(t, 1.0, resolved(metricsutil.OverflowValue, "header"))
	require.Equal(t, 1.0, resolved(metricsutil.OverflowValue, "agent_identity"))
	require.Equal(t, 1.0, resolved("default-flow", "default"))
	require.Equal(t, 1.0, overflow("header"))
	require.Equal(t, 1.0, overflow("agent_identity"))
	require.Equal(t, 0.0, overflow("default"))
}

// The overflow series exist at 0 before any request, so increase() sees the first overflow.
func TestInitFairnessIDResolutionMetricsCreatesOverflowSeries(t *testing.T) {
	llmdFairnessIDOverflowCounter.Reset()
	t.Cleanup(llmdFairnessIDOverflowCounter.Reset)

	InitFairnessIDResolutionMetrics("pool-a")

	require.Equal(t, 3, promtestutil.CollectAndCount(llmdFairnessIDOverflowCounter))
	require.Equal(t, 0.0, promtestutil.ToFloat64(llmdFairnessIDOverflowCounter.WithLabelValues("header", "pool-a")))
}
