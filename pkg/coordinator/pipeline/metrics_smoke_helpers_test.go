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

	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/metrics/metricstest"
)

// newMetricsRegistry registers every coordinator metric onto a fresh
// registry -- the same Register call the /metrics endpoint wiring uses --
// and clears any state left by earlier tests. The histogram query helpers
// below are thin wrappers over the shared metricstest package; only the
// presence/absence semantics that the smoke tests rely on live here.
func newMetricsRegistry(t *testing.T) *prometheus.Registry {
	return metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
}

// histogramCount returns the sample count of the histogram series matching
// name and labels. A missing series returns 0: Prometheus never exposes an
// unobserved label set, so the callers' equality assertions treat zero as
// "never observed" and fail.
func histogramCount(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) uint64 {
	return metricstest.HistogramCount(t, reg, name, labels)
}

// histogramSum returns the sample sum of the histogram series matching
// name and labels. A missing series returns 0.
func histogramSum(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) float64 {
	return metricstest.HistogramSum(t, reg, name, labels)
}

// requireHistogramAbsent asserts that no series in the family name carries
// the given labels. Prometheus never exposes an unobserved label set, so
// "zero samples" and "absent" are the same thing on the wire: a never-taken
// path must leave no series at all.
func requireHistogramAbsent(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) {
	metricstest.RequireHistogramAbsent(t, reg, name, labels)
}
