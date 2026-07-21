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
	"fmt"
	"testing"
	"time"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

func TestBoundedLabel(t *testing.T) {
	b := newBoundedLabel(2)

	require.Equal(t, "a", b.bound("a"), "first value admitted")
	require.Equal(t, "b", b.bound("b"), "second value admitted")
	require.Equal(t, overflowValue, b.bound("c"), "value beyond cap collapses to overflow")
	require.Equal(t, "a", b.bound("a"), "already-admitted value still returns itself after cap")
	require.Equal(t, overflowValue, b.bound("d"), "further unseen values keep collapsing")
	require.Equal(t, "", b.bound(""), "empty value passes through without consuming a slot")
}

// Pinned names model operator-configured models (InferenceModelRewrite sources
// and targets): they must emit their real label even when the cap is exhausted
// by unconfigured names, and must not consume cap slots themselves.
func TestBoundedLabelPin(t *testing.T) {
	b := newBoundedLabel(2)

	b.pin("configured")
	b.pin("configured") // idempotent
	require.Equal(t, "a", b.bound("a"), "pin does not consume a cap slot")
	require.Equal(t, "b", b.bound("b"), "cap still has room for a second unconfigured name")
	require.Equal(t, overflowValue, b.bound("c"), "cap full for unconfigured names")
	require.Equal(t, "configured", b.bound("configured"), "pinned name survives a full cap")

	b.pin("late")
	require.Equal(t, "late", b.bound("late"), "name pinned after the cap fills still emits its real label")
}

// PreAdmitModelLabels feeds the package-level limiter used by the record
// functions, so a flood of junk names must not displace a configured model.
func TestPreAdmitModelLabelsSurvivesFlood(t *testing.T) {
	const testCap = 5
	old := modelLabelLimiter
	modelLabelLimiter = newBoundedLabel(testCap)
	requestCounter.Reset()
	t.Cleanup(func() {
		modelLabelLimiter = old
		requestCounter.Reset()
	})

	PreAdmitModelLabels("llama-70b", "llama-70b-canary")
	for i := 0; i < 1000; i++ {
		RecordRequestCounter(fmt.Sprintf("junk-%d", i), "target", "fairness", 0)
	}
	RecordRequestCounter("llama-70b", "llama-70b-canary", "fairness", 0)

	series := promtestutil.CollectAndCount(requestCounter)
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
	modelLabelLimiter = newBoundedLabel(testCap)
	requestCounter.Reset()
	t.Cleanup(func() {
		modelLabelLimiter = old
		requestCounter.Reset()
	})

	for i := 0; i < 1000; i++ {
		RecordRequestCounter(fmt.Sprintf("model-%d", i), "target", "fairness", 0)
	}

	count := promtestutil.CollectAndCount(requestCounter)
	require.LessOrEqualf(t, count, testCap+1,
		"model_name cardinality must stay bounded by the cap, got %d series", count)
}

// The fairness_id label is populated from a client request header, so the package-level limiter
// must collapse an unbounded flood of distinct IDs into the overflow bucket instead of minting a
// series per ID.
func TestFairnessLabelFloodCollapsesToOverflow(t *testing.T) {
	const testCap = 5
	old := fairnessLabelLimiter
	fairnessLabelLimiter = newBoundedLabel(testCap)
	flowControlRequestEnqueueDuration.Reset()
	llmdFlowControlRequestEnqueueDuration.Reset()
	t.Cleanup(func() {
		fairnessLabelLimiter = old
		flowControlRequestEnqueueDuration.Reset()
		llmdFlowControlRequestEnqueueDuration.Reset()
	})

	for i := 0; i < 1000; i++ {
		RecordFlowControlRequestEnqueueDuration(fmt.Sprintf("tenant-%d", i), "0", "Dispatched", time.Millisecond)
	}

	// testCap admitted IDs + 1 overflow series, per family.
	require.Equal(t, testCap+1, promtestutil.CollectAndCount(flowControlRequestEnqueueDuration),
		"1000 distinct fairness IDs must collapse to cap+overflow series, not one series each")
	require.Equal(t, testCap+1, promtestutil.CollectAndCount(llmdFlowControlRequestEnqueueDuration),
		"the llm_d_epp family must be bounded identically")
}

// DeleteFlowControlFlowSeries backs the flow registry's GC hook: once a flow is collected, its
// series must not linger for the lifetime of the process.
func TestDeleteFlowControlFlowSeries(t *testing.T) {
	old := fairnessLabelLimiter
	fairnessLabelLimiter = newBoundedLabel(10)
	flowControlRequestEnqueueDuration.Reset()
	llmdFlowControlRequestEnqueueDuration.Reset()
	t.Cleanup(func() {
		fairnessLabelLimiter = old
		flowControlRequestEnqueueDuration.Reset()
		llmdFlowControlRequestEnqueueDuration.Reset()
	})

	RecordFlowControlRequestEnqueueDuration("tenant-a", "0", "Dispatched", time.Millisecond)
	RecordFlowControlRequestEnqueueDuration("tenant-a", "0", "Rejected", time.Millisecond)
	RecordFlowControlRequestEnqueueDuration("tenant-b", "0", "Dispatched", time.Millisecond)
	require.Equal(t, 3, promtestutil.CollectAndCount(flowControlRequestEnqueueDuration),
		"setup: expected one series per (fairness_id, outcome) pair")

	DeleteFlowControlFlowSeries("tenant-a", "0")

	require.Equal(t, 1, promtestutil.CollectAndCount(flowControlRequestEnqueueDuration),
		"all of tenant-a's series (every outcome) must be pruned; tenant-b's must survive")
	require.Equal(t, 1, promtestutil.CollectAndCount(llmdFlowControlRequestEnqueueDuration),
		"the llm_d_epp family must be pruned identically")
}

// The bound is applied mechanically in every record function that takes a fairness ID; this guards
// the request-metric family (distinct label ordering from the flow control family) against the
// pattern regressing there.
func TestFairnessLabelBoundOnRequestMetrics(t *testing.T) {
	const testCap = 3
	oldFairness := fairnessLabelLimiter
	fairnessLabelLimiter = newBoundedLabel(testCap)
	oldModels := modelLabelLimiter
	modelLabelLimiter = newBoundedLabel(10)
	requestCounter.Reset()
	llmdRequestCounter.Reset()
	t.Cleanup(func() {
		fairnessLabelLimiter = oldFairness
		modelLabelLimiter = oldModels
		requestCounter.Reset()
		llmdRequestCounter.Reset()
	})

	for i := 0; i < 100; i++ {
		RecordRequestCounter("model-a", "model-a", fmt.Sprintf("tenant-%d", i), 0)
	}

	require.Equal(t, testCap+1, promtestutil.CollectAndCount(llmdRequestCounter),
		"100 distinct fairness IDs must collapse to cap+overflow series on the request family")
	require.Equal(t, 1, promtestutil.CollectAndCount(requestCounter),
		"the deprecated family has no fairness_id label and must stay a single series")
}

// RecordFlowControlRequestQueueDuration takes request-body model names; they must flow through the
// model limiter like every sibling record function.
func TestQueueDurationBoundsModelLabels(t *testing.T) {
	const testCap = 3
	oldModels := modelLabelLimiter
	modelLabelLimiter = newBoundedLabel(testCap)
	oldFairness := fairnessLabelLimiter
	fairnessLabelLimiter = newBoundedLabel(10)
	flowControlRequestQueueDuration.Reset()
	llmdFlowControlRequestQueueDuration.Reset()
	t.Cleanup(func() {
		modelLabelLimiter = oldModels
		fairnessLabelLimiter = oldFairness
		flowControlRequestQueueDuration.Reset()
		llmdFlowControlRequestQueueDuration.Reset()
	})

	for i := 0; i < 100; i++ {
		m := fmt.Sprintf("model-%d", i)
		RecordFlowControlRequestQueueDuration("tenant", "0", "Dispatched", "pool", m, m, time.Millisecond)
	}

	require.Equal(t, testCap+1, promtestutil.CollectAndCount(flowControlRequestQueueDuration),
		"100 distinct model names must collapse to cap+overflow series, not one series each")
	require.Equal(t, testCap+1, promtestutil.CollectAndCount(llmdFlowControlRequestQueueDuration),
		"the llm_d_epp family must be bounded identically")
}

// A client can choose the overflow value itself as its fairness ID; GC of that flow must not
// delete the shared overflow series that aggregates every capped-out tenant.
func TestDeleteFlowControlFlowSeriesPreservesOverflowSeries(t *testing.T) {
	old := fairnessLabelLimiter
	fairnessLabelLimiter = newBoundedLabel(1)
	flowControlRequestEnqueueDuration.Reset()
	llmdFlowControlRequestEnqueueDuration.Reset()
	t.Cleanup(func() {
		fairnessLabelLimiter = old
		flowControlRequestEnqueueDuration.Reset()
		llmdFlowControlRequestEnqueueDuration.Reset()
	})

	// tenant-a fills the single cap slot; tenant-b folds to the overflow series.
	RecordFlowControlRequestEnqueueDuration("tenant-a", "0", "Dispatched", time.Millisecond)
	RecordFlowControlRequestEnqueueDuration("tenant-b", "0", "Dispatched", time.Millisecond)
	require.Equal(t, 2, promtestutil.CollectAndCount(flowControlRequestEnqueueDuration),
		"setup: expected the admitted series plus the overflow series")

	DeleteFlowControlFlowSeries(overflowValue, "0")

	require.Equal(t, 2, promtestutil.CollectAndCount(flowControlRequestEnqueueDuration),
		"deleting the overflow value must be a no-op; the shared overflow series must survive")
	require.Equal(t, 2, promtestutil.CollectAndCount(llmdFlowControlRequestEnqueueDuration),
		"the llm_d_epp family must be preserved identically")
}
