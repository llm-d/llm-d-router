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
func TestRecordPrediction(t *testing.T) {
	predictedCachedTokens.Reset()
	promptTokens.Reset()
	t.Cleanup(func() {
		predictedCachedTokens.Reset()
		promptTokens.Reset()
	})

	RecordPrediction("test-plugin", "test-type", 512, 1024)
	RecordPrediction("test-plugin", "test-type", 0, 256)

	predicted, err := histogramFor(predictedCachedTokens, "test-plugin", "test-type")
	require.NoError(t, err)
	assert.Equal(t, uint64(2), predicted.GetSampleCount())
	assert.Equal(t, float64(512), predicted.GetSampleSum())

	prompt, err := histogramFor(promptTokens, "test-plugin", "test-type")
	require.NoError(t, err)
	assert.Equal(t, uint64(2), prompt.GetSampleCount())
	assert.Equal(t, float64(1280), prompt.GetSampleSum())
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
