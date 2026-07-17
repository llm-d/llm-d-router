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
package preciseprefixcache

import (
	"testing"

	approximateprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, approximateprefix.RegisterMetrics(registry))
	// second call must be idempotent
	require.NoError(t, approximateprefix.RegisterMetrics(registry))
}

func TestRegisterMetrics_NilRegisterer(t *testing.T) {
	require.Error(t, approximateprefix.RegisterMetrics(nil))
}

func TestRecordPrefixCacheMatch_FullHit(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	recordPrefixCacheHitRatioStats("p", "t", 0.5, 0.4, 0.2)

	maxH, err := getHistogram(approximateprefix.LlmdPrefixCacheMaxHitRatio, "p", "t")
	require.NoError(t, err)
	require.Equal(t, uint64(1), maxH.GetSampleCount())
	require.InDelta(t, 0.5, maxH.GetSampleSum(), 1e-9)

	avgH, err := getHistogram(approximateprefix.LlmdPrefixCacheAvgHitRatio, "p", "t")
	require.NoError(t, err)
	require.Equal(t, uint64(1), avgH.GetSampleCount())
	require.InDelta(t, 0.4, avgH.GetSampleSum(), 1e-9)

	stdDevH, err := getHistogram(approximateprefix.LlmdPrefixCacheStdDevHitRatio, "p", "t")
	require.NoError(t, err)
	require.Equal(t, uint64(1), stdDevH.GetSampleCount())
	require.InDelta(t, 0.2, stdDevH.GetSampleSum(), 1e-9)
}

func getHistogram(histogram *prometheus.HistogramVec, labelValues ...string) (*dto.Histogram, error) {
	metric, err := histogram.GetMetricWithLabelValues(labelValues...)
	if err != nil {
		return nil, err
	}
	dtoMetric := &dto.Metric{}
	if err := metric.(prometheus.Histogram).Write(dtoMetric); err != nil {
		return nil, err
	}
	return dtoMetric.GetHistogram(), nil
}

func resetMetrics() {
	approximateprefix.LlmdPrefixCacheMaxHitRatio.Reset()
	approximateprefix.LlmdPrefixCacheAvgHitRatio.Reset()
	approximateprefix.LlmdPrefixCacheStdDevHitRatio.Reset()
}
