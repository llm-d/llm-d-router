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
// call under the call's role so their sums stay divisible by one another.
// The two maxima are also split by the call's modality.
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
		{"decode selected", predictedCachedTokens, []string{RoleDecode}, 2, 512},
		{"decode best predicted", bestPredictedCachedTokens, []string{RoleDecode, mmobs.ModalityNone}, 2, 768},
		{"decode best available", bestAvailableCachedTokens, []string{RoleDecode, mmobs.ModalityNone}, 2, 896},
		{"decode prompt", promptTokens, []string{RoleDecode}, 2, 1280},
		{"prefill selected", predictedCachedTokens, []string{RolePrefill}, 1, 64},
		{"prefill best predicted", bestPredictedCachedTokens, []string{RolePrefill, "audio,image"}, 1, 96},
		{"prefill best available", bestAvailableCachedTokens, []string{RolePrefill, "audio,image"}, 1, 112},
		{"prefill prompt", promptTokens, []string{RolePrefill}, 1, 128},
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
