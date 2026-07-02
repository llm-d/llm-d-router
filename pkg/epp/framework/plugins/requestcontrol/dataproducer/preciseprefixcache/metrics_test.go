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

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	// second call must be idempotent
	require.NoError(t, registerMetrics(registry))
}

func TestRegisterMetrics_NilRegisterer(t *testing.T) {
	require.Error(t, registerMetrics(nil))
}

func TestRecordPrefixCacheMaxMatch(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	recordPrefixCacheMaxMatch("test-plugin", "test-type", 5, 10)

	h, err := getHistogram(llmdPrefixCacheMaxHitRatio, "test-plugin", "test-type")
	require.NoError(t, err)
	require.Equal(t, uint64(1), h.GetSampleCount())
	require.InDelta(t, 0.5, h.GetSampleSum(), 1e-9)
}

func TestRecordPrefixCacheAvgMatch(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	recordPrefixCacheAvgMatch("test-plugin", "test-type", 4.0, 10)

	h, err := getHistogram(llmdPrefixCacheAvgHitRatio, "test-plugin", "test-type")
	require.NoError(t, err)
	require.Equal(t, uint64(1), h.GetSampleCount())
	require.InDelta(t, 0.4, h.GetSampleSum(), 1e-9)
}

func TestRecordPrefixCacheStdDevMatch(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	recordPrefixCacheStdDevMatch("test-plugin", "test-type", 2.0, 10)

	h, err := getHistogram(llmdPrefixCacheStdDevHitRatio, "test-plugin", "test-type")
	require.NoError(t, err)
	require.Equal(t, uint64(1), h.GetSampleCount())
	require.InDelta(t, 0.2, h.GetSampleSum(), 1e-9)
}

func TestRecordPrefixCacheMatch_FullHit(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	recordPrefixCacheMaxMatch("p", "t", 8, 8)
	recordPrefixCacheAvgMatch("p", "t", 8.0, 8)
	recordPrefixCacheStdDevMatch("p", "t", 0.0, 8)

	maxH, err := getHistogram(llmdPrefixCacheMaxHitRatio, "p", "t")
	require.NoError(t, err)
	require.Equal(t, uint64(1), maxH.GetSampleCount())
	require.InDelta(t, 1.0, maxH.GetSampleSum(), 1e-9)

	avgH, err := getHistogram(llmdPrefixCacheAvgHitRatio, "p", "t")
	require.NoError(t, err)
	require.Equal(t, uint64(1), avgH.GetSampleCount())
	require.InDelta(t, 1.0, avgH.GetSampleSum(), 1e-9)

	stdDevH, err := getHistogram(llmdPrefixCacheStdDevHitRatio, "p", "t")
	require.NoError(t, err)
	require.Equal(t, uint64(1), stdDevH.GetSampleCount())
	require.InDelta(t, 0.0, stdDevH.GetSampleSum(), 1e-9)
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
	llmdPrefixCacheMaxHitRatio.Reset()
	llmdPrefixCacheAvgHitRatio.Reset()
	llmdPrefixCacheStdDevHitRatio.Reset()
}
