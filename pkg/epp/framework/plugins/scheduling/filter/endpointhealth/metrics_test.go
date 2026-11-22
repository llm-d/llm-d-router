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

package endpointhealth

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func counterValue(t *testing.T, vec *prometheus.CounterVec, labels ...string) float64 {
	t.Helper()
	return testutil.ToFloat64(vec.WithLabelValues(labels...))
}

func gaugeValue(t *testing.T, vec *prometheus.GaugeVec, labels ...string) float64 {
	t.Helper()
	return testutil.ToFloat64(vec.WithLabelValues(labels...))
}

func durationPtr(d time.Duration) *metav1.Duration { return &metav1.Duration{Duration: d} }

func TestRegisterMetrics(t *testing.T) {
	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry), "re-registering the same collectors is tolerated")
	require.NoError(t, registerMetrics(nil), "a nil registerer is a no-op")
}
