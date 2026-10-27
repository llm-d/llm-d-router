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

// Package metricstest provides shared helpers for asserting coordinator
// metric observations in tests. The functions operate on a
// *prometheus.Registry (not the package-level default) so tests stay
// isolated.
package metricstest

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
)

// NewRegistry registers every coordinator metric onto a fresh registry and
// clears state left by earlier tests. Callers in different packages share
// this instead of re-defining their own newMetricsRegistry / newStepMetricsRegistry.
func NewRegistry(t *testing.T, regReg func(prometheus.Registerer) error, reset func()) *prometheus.Registry {
	t.Helper()
	reg := prometheus.NewRegistry()
	require.NoError(t, regReg(reg))
	reset()
	return reg
}

// HistogramCount returns the sample count of the histogram series matching
// name and labels. A missing series returns 0: Prometheus never exposes an
// unobserved label set, so "zero samples" and "absent" are the same on the
// wire. Use RequireHistogramAbsent when a test must assert the series was
// never observed at all.
func HistogramCount(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) uint64 {
	t.Helper()
	for _, m := range histogramSeriesAll(t, reg, name, labels) {
		return m.GetHistogram().GetSampleCount()
	}
	return 0
}

// HistogramSum returns the sample sum of the histogram series matching
// name and labels. A missing series returns 0.
func HistogramSum(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) float64 {
	t.Helper()
	for _, m := range histogramSeriesAll(t, reg, name, labels) {
		return m.GetHistogram().GetSampleSum()
	}
	return 0
}

// RequireHistogramAbsent asserts that no series in the family name carries
// the given labels. Prometheus never exposes an unobserved label set, so
// "zero samples" and "absent" are the same on the wire: a never-taken path
// must leave no series at all.
func RequireHistogramAbsent(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) {
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
