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
	"fmt"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/common/observability/multimodal"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
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
	gpuMissTokens.Reset()
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

// A token-in /generate request keys its multimodal features by whatever the
// client sends, and Summary joins those keys into the modality label. Each new
// key must not mint a new series on every prediction histogram.
func TestRecordPrediction_BoundsClientChosenModality(t *testing.T) {
	resetPredictionMetrics()
	t.Cleanup(resetPredictionMetrics)

	const requests = 200
	for i := range requests {
		request := &fwksched.InferenceRequest{Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{
				MultiModalFeatures: []fwkrequest.MultiModalFeature{
					{Modality: reqcommon.ModalityImage, Hash: "image-hash"},
					{Modality: reqcommon.Modality(fmt.Sprintf("client-modality-%d", i)), Hash: fmt.Sprintf("hash-%d", i)},
				},
			}}},
		}}
		modality, _ := mmobs.Summary(request)
		RecordPrediction("test-plugin", "test-type", RoleDecode, modality, Prediction{Selected: 1, PromptTokens: 2})
	}

	for name, vec := range map[string]*prometheus.HistogramVec{
		"predicted":      predictedCachedTokens,
		"best predicted": bestPredictedCachedTokens,
		"best available": bestAvailableCachedTokens,
		"prompt":         promptTokens,
	} {
		assert.Equal(t, 1, testutil.CollectAndCount(vec), "%s: client-chosen modalities must share one series", name)
	}
	predicted, err := histogramFor(predictedCachedTokens, "test-plugin", "test-type", RoleDecode, "image,other")
	require.NoError(t, err)
	assert.Equal(t, uint64(requests), predicted.GetSampleCount())
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
				PrefillProfileName: "prefill",
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
				PrefillProfileName: "prefill",
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
			profile, role := PredictionTarget(tt.result)
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

// Every prompt token the chosen endpoint lacks in GPU memory lands on exactly
// one reason, so the reasons sum to the prompt minus the GPU hit.
func TestRecordGPUMiss(t *testing.T) {
	resetPredictionMetrics()
	t.Cleanup(resetPredictionMetrics)

	RecordGPUMiss("test-plugin", "test-type", RolePrefill, GPUMiss{
		Selected: 100, BestPredicted: 150, BestAvailable: 180,
		SelectedConfirmed: 300, BestAvailableConfirmed: 450, PromptTokens: 500,
	})

	total := 0.0
	for reason, want := range map[string]float64{
		GPUMissFiltered:      30,
		GPUMissNotPicked:     50,
		GPUMissLowerTier:     120,
		GPUMissPeerLowerTier: 150,
		GPUMissUncached:      50,
	} {
		got := testutil.ToFloat64(gpuMissTokens.WithLabelValues("test-plugin", "test-type", RolePrefill, reason))
		assert.Equal(t, want, got, reason)
		total += got
	}
	assert.Equal(t, float64(500-100), total)
}

// A prefix held in GPU memory elsewhere is attributed to placement even when
// the chosen endpoint holds it in a lower tier, so the lower-tier reasons never
// double count it.
func TestRecordGPUMiss_PlacementBeforeLowerTier(t *testing.T) {
	resetPredictionMetrics()
	t.Cleanup(resetPredictionMetrics)

	RecordGPUMiss("test-plugin", "test-type", RolePrefill, GPUMiss{
		Selected: 100, BestPredicted: 100, BestAvailable: 400,
		SelectedConfirmed: 300, BestAvailableConfirmed: 400, PromptTokens: 500,
	})

	assert.Equal(t, 300.0, testutil.ToFloat64(gpuMissTokens.WithLabelValues("test-plugin", "test-type", RolePrefill, GPUMissFiltered)))
	assert.Equal(t, 100.0, testutil.ToFloat64(gpuMissTokens.WithLabelValues("test-plugin", "test-type", RolePrefill, GPUMissUncached)))
	assert.Equal(t, 2, testutil.CollectAndCount(gpuMissTokens), "reasons with no tokens create no series")
}

func TestRecordGPUMiss_FullHitRecordsNothing(t *testing.T) {
	resetPredictionMetrics()
	t.Cleanup(resetPredictionMetrics)

	RecordGPUMiss("test-plugin", "test-type", RolePrefill, GPUMiss{
		Selected: 512, BestPredicted: 512, BestAvailable: 512,
		SelectedConfirmed: 512, BestAvailableConfirmed: 512, PromptTokens: 512,
	})

	assert.Equal(t, 0, testutil.CollectAndCount(gpuMissTokens))
}
