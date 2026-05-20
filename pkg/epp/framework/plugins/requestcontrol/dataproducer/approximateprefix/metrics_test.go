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

package approximateprefix

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry))
}

func TestRecordPrefixCacheMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	recordPrefixCacheSize(4096)
	recordPrefixCacheMatch(10, 20)
	recordPrefixCacheMatch(0, 0)

	require.Equal(t, float64(4096), testutil.ToFloat64(prefixCacheSize.WithLabelValues()))

	hitRatio, err := getHistogram(prefixCacheHitRatio)
	require.NoError(t, err)
	require.Equal(t, uint64(1), hitRatio.GetSampleCount())
	require.Equal(t, 0.5, hitRatio.GetSampleSum())

	hitLength, err := getHistogram(prefixCacheHitLength)
	require.NoError(t, err)
	require.Equal(t, uint64(2), hitLength.GetSampleCount())
	require.Equal(t, float64(10), hitLength.GetSampleSum())
}

func getHistogram(histogram *prometheus.HistogramVec) (*dto.Histogram, error) {
	metric, err := histogram.GetMetricWithLabelValues()
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
	prefixCacheSize.Reset()
	prefixCacheHitRatio.Reset()
	prefixCacheHitLength.Reset()
}
