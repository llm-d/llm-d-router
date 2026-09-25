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

package pipeline_test

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
)

// newMetricsRegistry registers every coordinator metric onto a fresh
// registry -- the same Register call the /metrics endpoint wiring uses --
// and clears any state left by earlier tests.
func newMetricsRegistry(t *testing.T) *prometheus.Registry {
	t.Helper()
	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	return reg
}

// histogramCount returns the sample count of the histogram series matching
// name and labels. A missing series fails the test so an absent series can
// never pass for a legitimate zero.
func histogramCount(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) uint64 {
	t.Helper()
	for _, m := range histogramSeries(t, reg, name, labels) {
		return m.GetHistogram().GetSampleCount()
	}
	return 0
}

// histogramSum returns the sample sum of the histogram series matching name
// and labels. A missing series fails the test.
func histogramSum(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) float64 {
	t.Helper()
	for _, m := range histogramSeries(t, reg, name, labels) {
		return m.GetHistogram().GetSampleSum()
	}
	return 0
}

// histogramSeries finds the metric in the family name whose labels match
// every entry of labels (nil matches any series). Exactly one series must
// match, otherwise the test fails.
func histogramSeries(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) []*dto.Metric {
	t.Helper()
	matched := histogramSeriesAll(t, reg, name, labels)
	require.Len(t, matched, 1, "expected exactly one %s series for %v", name, labels)
	return matched
}

// requireHistogramAbsent asserts that no series in the family name carries
// the given labels. Prometheus never exposes an unobserved label set, so
// "zero samples" and "absent" are the same thing on the wire: a never-taken
// path must leave no series at all.
func requireHistogramAbsent(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) {
	t.Helper()
	require.Empty(t, histogramSeriesAll(t, reg, name, labels),
		"expected no %s series for %v: the label set was never observed", name, labels)
}

func histogramSeriesAll(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) []*dto.Metric {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	var matched []*dto.Metric
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			if labelsMatch(m.GetLabel(), labels) {
				matched = append(matched, m)
			}
		}
	}
	return matched
}

func labelsMatch(actual []*dto.LabelPair, want map[string]string) bool {
	if len(want) == 0 {
		return true
	}
	got := map[string]string{}
	for _, l := range actual {
		got[l.GetName()] = l.GetValue()
	}
	for k, v := range want {
		if got[k] != v {
			return false
		}
	}
	return true
}
