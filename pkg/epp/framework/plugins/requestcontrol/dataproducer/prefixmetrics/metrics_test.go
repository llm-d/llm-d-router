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

package prefixmetrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Every producer instance calls Register, so repeated calls must not panic.
func TestRegisterIsIdempotent(t *testing.T) {
	assert.NotPanics(t, func() {
		Register()
		Register()
	})
}

// A zero prediction is a real observation: the router expected no cache hit,
// and the request still contributes its prompt tokens to the denominator.
// Every field lands on its own histogram, and all four carry a sample per
// call so their sums stay divisible by one another.
func TestRecordPrediction(t *testing.T) {
	resetPredictionMetrics()
	t.Cleanup(resetPredictionMetrics)

	RecordPrediction("test-plugin", "test-type", Prediction{
		Selected: 512, BestPredicted: 768, BestAvailable: 896, PromptTokens: 1024,
	})
	RecordPrediction("test-plugin", "test-type", Prediction{
		Selected: 0, BestPredicted: 0, BestAvailable: 0, PromptTokens: 256,
	})

	for _, tc := range []struct {
		name string
		vec  *prometheus.HistogramVec
		sum  float64
	}{
		{"selected", predictedCachedTokens, 512},
		{"best predicted", bestPredictedCachedTokens, 768},
		{"best available", bestAvailableCachedTokens, 896},
		{"prompt", promptTokens, 1280},
	} {
		t.Run(tc.name, func(t *testing.T) {
			histogram, err := histogramFor(tc.vec, "test-plugin", "test-type")
			require.NoError(t, err)
			assert.Equal(t, uint64(2), histogram.GetSampleCount())
			assert.Equal(t, tc.sum, histogram.GetSampleSum())
		})
	}
}

func resetPredictionMetrics() {
	predictedCachedTokens.Reset()
	bestPredictedCachedTokens.Reset()
	bestAvailableCachedTokens.Reset()
	promptTokens.Reset()
}

func histogramFor(vec *prometheus.HistogramVec, labelValues ...string) (*dto.Histogram, error) {
	observer, err := vec.GetMetricWithLabelValues(labelValues...)
	if err != nil {
		return nil, err
	}
	metric := &dto.Metric{}
	if err := observer.(prometheus.Histogram).Write(metric); err != nil {
		return nil, err
	}
	return metric.GetHistogram(), nil
}
