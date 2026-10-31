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
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/observability/multimodal"
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
// call under the call's role and modality so their sums stay divisible by
// one another.
func TestRecordPrediction(t *testing.T) {
	resetPredictionMetrics()
	t.Cleanup(resetPredictionMetrics)

	RecordPrediction("test-plugin", "test-type", RoleDecode, mmobs.ModalityNone, Prediction{
		Selected: 512, BestPredicted: 768, BestAvailable: 896, PromptTokens: 1024,
	})
	RecordPrediction("test-plugin", "test-type", RoleDecode, mmobs.ModalityNone, Prediction{
		Selected: 0, BestPredicted: 0, BestAvailable: 0, PromptTokens: 256,
	})
	RecordPrediction("test-plugin", "test-type", RolePrefill, "audio,image", Prediction{
		Selected: 64, BestPredicted: 96, BestAvailable: 112, PromptTokens: 128,
	})

	for _, tc := range []struct {
		name   string
		vec    *prometheus.HistogramVec
		labels []string
		count  uint64
		sum    float64
	}{
		{"decode selected", predictedCachedTokens, []string{RoleDecode, mmobs.ModalityNone}, 2, 512},
		{"decode best predicted", bestPredictedCachedTokens, []string{RoleDecode, mmobs.ModalityNone}, 2, 768},
		{"decode best available", bestAvailableCachedTokens, []string{RoleDecode, mmobs.ModalityNone}, 2, 896},
		{"decode prompt", promptTokens, []string{RoleDecode, mmobs.ModalityNone}, 2, 1280},
		{"prefill selected", predictedCachedTokens, []string{RolePrefill, "audio,image"}, 1, 64},
		{"prefill best predicted", bestPredictedCachedTokens, []string{RolePrefill, "audio,image"}, 1, 96},
		{"prefill best available", bestAvailableCachedTokens, []string{RolePrefill, "audio,image"}, 1, 112},
		{"prefill prompt", promptTokens, []string{RolePrefill, "audio,image"}, 1, 128},
	} {
		t.Run(tc.name, func(t *testing.T) {
			histogram, err := histogramFor(tc.vec, append([]string{"test-plugin", "test-type"}, tc.labels...)...)
			require.NoError(t, err)
			assert.Equal(t, tc.count, histogram.GetSampleCount())
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

// A zero multimodal prediction is a real observation: the router expected no
// multimodal cache hit, and the request still contributes its multimodal
// tokens to the denominator. Both histograms observe the same requests so the
// ratio divides counts taken over the same observations, per role series.
func TestRecordMMPrediction(t *testing.T) {
	mmPredictedCachedTokens.Reset()
	mmPromptTokens.Reset()
	t.Cleanup(func() {
		mmPredictedCachedTokens.Reset()
		mmPromptTokens.Reset()
	})

	RecordMMPrediction("test-plugin", "test-type", RoleDecode, 512, 1024)
	RecordMMPrediction("test-plugin", "test-type", RoleDecode, 0, 256)
	RecordMMPrediction("test-plugin", "test-type", RolePrefill, 64, 128)

	predicted, err := histogramFor(mmPredictedCachedTokens, "test-plugin", "test-type", RoleDecode)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), predicted.GetSampleCount())
	assert.Equal(t, float64(512), predicted.GetSampleSum())

	prompt, err := histogramFor(mmPromptTokens, "test-plugin", "test-type", RoleDecode)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), prompt.GetSampleCount())
	assert.Equal(t, float64(1280), prompt.GetSampleSum())

	predicted, err = histogramFor(mmPredictedCachedTokens, "test-plugin", "test-type", RolePrefill)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), predicted.GetSampleCount())
	assert.Equal(t, float64(64), predicted.GetSampleSum())

	prompt, err = histogramFor(mmPromptTokens, "test-plugin", "test-type", RolePrefill)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), prompt.GetSampleCount())
	assert.Equal(t, float64(128), prompt.GetSampleSum())
}

// A request is observed once, under the comma-joined sorted list of the
// modalities it carries, so summing over modality keeps the all-requests
// ratio exact.
func TestRecordPrediction_ModalitySeries(t *testing.T) {
	predictedCachedTokens.Reset()
	promptTokens.Reset()
	t.Cleanup(func() {
		predictedCachedTokens.Reset()
		promptTokens.Reset()
	})

	RecordPrediction("test-plugin", "test-type", RoleDecode, mmobs.ModalityNone, Prediction{
		Selected: 512, PromptTokens: 1024,
	})
	RecordPrediction("test-plugin", "test-type", RoleDecode, "audio,image", Prediction{
		Selected: 100, PromptTokens: 200,
	})

	predicted, err := histogramFor(predictedCachedTokens, "test-plugin", "test-type", RoleDecode, mmobs.ModalityNone)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), predicted.GetSampleCount())
	assert.Equal(t, float64(512), predicted.GetSampleSum())

	predicted, err = histogramFor(predictedCachedTokens, "test-plugin", "test-type", RoleDecode, "audio,image")
	require.NoError(t, err)
	assert.Equal(t, uint64(1), predicted.GetSampleCount())
	assert.Equal(t, float64(100), predicted.GetSampleSum())

	prompt, err := histogramFor(promptTokens, "test-plugin", "test-type", RoleDecode, "audio,image")
	require.NoError(t, err)
	assert.Equal(t, uint64(1), prompt.GetSampleCount())
	assert.Equal(t, float64(200), prompt.GetSampleSum())
}

// Under P/D the sidecar reports the prefill stage's cached tokens, so the
// prediction follows the prefill profile when the request was disaggregated and
// the primary profile otherwise.
func TestPredictionTarget(t *testing.T) {
	decodeProfile := &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{endpointNamed("decode-pod")}}
	prefillProfile := &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{endpointNamed("prefill-pod")}}

	tests := []struct {
		name        string
		result      *fwksched.SchedulingResult
		wantProfile *fwksched.ProfileRunResult
		wantRole    string
	}{
		{
			name:     "nil result",
			result:   nil,
			wantRole: "",
		},
		{
			name: "primary only",
			result: &fwksched.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode": decodeProfile,
				},
			},
			wantProfile: decodeProfile,
			wantRole:    RoleDecode,
		},
		{
			name: "disaggregated",
			result: &fwksched.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode":  decodeProfile,
					"prefill": prefillProfile,
				},
			},
			wantProfile: prefillProfile,
			wantRole:    RolePrefill,
		},
		{
			name: "prefill profile ran without a target",
			result: &fwksched.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode":  decodeProfile,
					"prefill": nil,
				},
			},
			wantProfile: decodeProfile,
			wantRole:    RoleDecode,
		},
		{
			name: "no primary target",
			result: &fwksched.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults:     map[string]*fwksched.ProfileRunResult{"decode": {}},
			},
			wantRole: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profile, role := PredictionTarget(tt.result, "prefill")
			assert.Same(t, tt.wantProfile, profile)
			assert.Equal(t, tt.wantRole, role)
		})
	}
}

func endpointNamed(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes())
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
