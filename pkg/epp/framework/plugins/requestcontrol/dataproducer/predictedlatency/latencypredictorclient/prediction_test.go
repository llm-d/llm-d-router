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

package latencypredictorclient

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func validEncoderPredictionRequest() PredictionRequest {
	return PredictionRequest{
		KVCachePercentage:  0.5,
		InputTokenLength:   100,
		NumRequestWaiting:  1,
		NumRequestRunning:  1,
		NumTokensGenerated: 10,
	}
}

func TestValidatePredictionRequest_EncoderSizes(t *testing.T) {
	predictor := &Predictor{}

	tests := []struct {
		name        string
		inputSize   int
		matchedSize int
		wantErr     bool
	}{
		{name: "both zero", inputSize: 0, matchedSize: 0, wantErr: false},
		{name: "partial match", inputSize: 5, matchedSize: 3, wantErr: false},
		{name: "full match", inputSize: 5, matchedSize: 5, wantErr: false},
		{name: "negative input size", inputSize: -1, matchedSize: 0, wantErr: true},
		{name: "negative matched size", inputSize: 5, matchedSize: -1, wantErr: true},
		{name: "matched exceeds input", inputSize: 5, matchedSize: 6, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := validEncoderPredictionRequest()
			req.EncoderInputSize = tt.inputSize
			req.EncoderMatchedSize = tt.matchedSize
			err := predictor.ValidatePredictionRequest(req)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateTrainingEntry_EncoderSizes(t *testing.T) {
	predictor := &Predictor{}
	entry := TrainingEntry{
		KVCachePercentage:  0.5,
		InputTokenLength:   100,
		NumRequestWaiting:  1,
		NumRequestRunning:  1,
		NumTokensGenerated: 10,
		ActualTTFT:         50.0,
		ActualTPOT:         5.0,
	}

	entry.EncoderInputSize, entry.EncoderMatchedSize = 4, 2
	assert.NoError(t, predictor.ValidateTrainingEntry(entry))

	entry.EncoderInputSize, entry.EncoderMatchedSize = -1, 0
	assert.Error(t, predictor.ValidateTrainingEntry(entry))

	entry.EncoderInputSize, entry.EncoderMatchedSize = 2, 3
	assert.Error(t, predictor.ValidateTrainingEntry(entry))
}

// TestPredictBayesianRidge_EncoderCoefficients verifies the TTFT linear model
// applies encoder-cache coefficients when present and degrades to a zero
// contribution when the model was trained without them.
func TestPredictBayesianRidge_EncoderCoefficients(t *testing.T) {
	predictor := &Predictor{}
	req := validEncoderPredictionRequest()
	req.EncoderInputSize = 10
	req.EncoderMatchedSize = 4

	withoutEncoder := &MetricsResponse{Coefficients: &ModelCoefficients{
		TTFTIntercept: 100,
		TTFTCoeffs:    map[string]float64{},
		TPOTCoeffs:    map[string]float64{},
	}}
	resp, err := predictor.predictBayesianRidge(req, withoutEncoder, 0.9, ObjectiveMean)
	require.NoError(t, err)
	assert.Equal(t, 100.0, resp.TTFT)

	withEncoder := &MetricsResponse{Coefficients: &ModelCoefficients{
		TTFTIntercept: 100,
		TTFTCoeffs: map[string]float64{
			"encoder_input_size":   2,
			"encoder_matched_size": -1,
		},
		TPOTCoeffs: map[string]float64{},
	}}
	resp, err = predictor.predictBayesianRidge(req, withEncoder, 0.9, ObjectiveMean)
	require.NoError(t, err)
	assert.Equal(t, 100.0+2*10-1*4, resp.TTFT)
}
