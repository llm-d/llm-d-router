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

	RecordPrediction("test-plugin", "test-type", RoleDecode, 512, 1024)
	RecordPrediction("test-plugin", "test-type", RoleDecode, 0, 256)
	RecordPrediction("test-plugin", "test-type", RolePrefill, 64, 128)

	predicted, err := histogramFor(predictedCachedTokens, "test-plugin", "test-type", RoleDecode)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), predicted.GetSampleCount())
	assert.Equal(t, float64(512), predicted.GetSampleSum())

	prompt, err := histogramFor(promptTokens, "test-plugin", "test-type", RoleDecode)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), prompt.GetSampleCount())
	assert.Equal(t, float64(1280), prompt.GetSampleSum())

	predicted, err = histogramFor(predictedCachedTokens, "test-plugin", "test-type", RolePrefill)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), predicted.GetSampleCount())
	assert.Equal(t, float64(64), predicted.GetSampleSum())

	prompt, err = histogramFor(promptTokens, "test-plugin", "test-type", RolePrefill)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), prompt.GetSampleCount())
	assert.Equal(t, float64(128), prompt.GetSampleSum())
}

// Under P/D the sidecar reports the prefill stage's cached tokens, so the
// prediction follows the prefill target when the request was disaggregated and
// the primary target otherwise.
func TestPredictionTarget(t *testing.T) {
	decode := endpointNamed("decode-pod")
	prefill := endpointNamed("prefill-pod")

	tests := []struct {
		name         string
		result       *fwksched.SchedulingResult
		wantEndpoint fwksched.Endpoint
		wantRole     string
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
					"decode": {TargetEndpoints: []fwksched.Endpoint{decode}},
				},
			},
			wantEndpoint: decode,
			wantRole:     RoleDecode,
		},
		{
			name: "disaggregated",
			result: &fwksched.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode":  {TargetEndpoints: []fwksched.Endpoint{decode}},
					"prefill": {TargetEndpoints: []fwksched.Endpoint{prefill}},
				},
			},
			wantEndpoint: prefill,
			wantRole:     RolePrefill,
		},
		{
			name: "prefill profile ran without a target",
			result: &fwksched.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode":  {TargetEndpoints: []fwksched.Endpoint{decode}},
					"prefill": nil,
				},
			},
			wantEndpoint: decode,
			wantRole:     RoleDecode,
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
			endpoint, role := PredictionTarget(tt.result, "prefill")
			assert.Equal(t, tt.wantEndpoint, endpoint)
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
