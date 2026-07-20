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
