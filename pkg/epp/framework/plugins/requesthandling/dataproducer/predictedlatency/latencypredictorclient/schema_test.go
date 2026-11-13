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
	"encoding/json"
	"math/rand/v2"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

func TestFeatureOptionsAndLegacyFieldMutation(t *testing.T) {
	features := &protocol.Features{EncoderInputSize: proto.Int64(8), EncoderMatchedSize: proto.Int64(3)}
	request, err := NewPredictionRequest(0.5, 100, 2, 3, 1, 0.25, WithFeatures(features), WithFeature("prefill_tokens_in_flight", int64(1<<53)+1))
	require.NoError(t, err)
	features.EncoderInputSize = proto.Int64(999)
	assert.Equal(t, 8, request.EncoderInputSize)
	request.InputTokenLength = 200
	data, err := json.Marshal(request)
	require.NoError(t, err)
	parsed := &protocol.PredictionRequest{}
	require.NoError(t, protocol.Unmarshal(data, parsed))
	assert.Equal(t, int64(200), parsed.GetFeatures().GetInputTokenLength())
	assert.Equal(t, int64(8), parsed.GetFeatures().GetEncoderInputSize())
	assert.Equal(t, int64(1<<53)+1, parsed.GetFeatures().GetPrefillTokensInFlight())
	_, err = NewPredictionRequest(0.5, 100, 2, 3, 1, 0.25, WithFeature("missing_feature", 1))
	require.Error(t, err)
	_, err = NewPredictionRequest(0.5, 100, 2, 3, 1, 0.25, WithFeature("encoder_input_size", -1))
	require.Error(t, err)
}

func TestLegacyAccessorsOverrideOverlappingSchemaFields(t *testing.T) {
	features := &protocol.Features{InputTokenLength: proto.Int64(123)}
	for _, row := range []any{
		PredictionRequest{Features: features},
		TrainingEntry{Features: features},
	} {
		data, err := json.Marshal(row)
		require.NoError(t, err)
		var values map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(data, &values))
		assert.JSONEq(t, `0`, string(values["input_token_length"]))
	}

	request := PredictionFromFeatures(features)
	entry := TrainingFromFeatures(features, TrainingEntry{})
	assert.Equal(t, 123, request.InputTokenLength)
	assert.Equal(t, 123, entry.InputTokenLength)
}

func TestTrainingTimestampKeepsSampleSplitInput(t *testing.T) {
	timestamp := time.Date(2026, 11, 10, 10, 20, 30, 123456000, time.FixedZone("CST", 8*60*60))
	entry := TrainingFromFeatures(&protocol.Features{InputTokenLength: proto.Int64(12)}, TrainingEntry{Timestamp: timestamp, ActualTTFT: 10})
	encoded, err := json.Marshal(entry)
	require.NoError(t, err)
	var object map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(encoded, &object))
	expected, err := timestamp.MarshalJSON()
	require.NoError(t, err)
	assert.JSONEq(t, string(expected), string(object["timestamp"]))
	assert.NotContains(t, string(encoded), "features")
}

func TestModelArtifactCompatibility(t *testing.T) {
	coefficients := ModelCoefficients{TTFTIntercept: 1, TTFTCoeffs: map[string]float64{"extra_raw_field": 2}, TPOTIntercept: 3, TPOTCoeffs: map[string]float64{}}
	encoded, err := json.Marshal(coefficients)
	require.NoError(t, err)
	generated := &protocol.ModelCoefficients{}
	require.NoError(t, protocol.Unmarshal(encoded, generated))
	assert.Equal(t, float64(2), generated.GetTtftCoefficients()["extra_raw_field"])
	var decoded ModelCoefficients
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	assert.Equal(t, coefficients, decoded)
	var info ModelInfo
	require.NoError(t, json.Unmarshal([]byte(`{"model_type":"xgboost","quantile":0.9,"model_status":{"ttft_model_ready":true},"future_field":123}`), &info))
	assert.Equal(t, "xgboost", info.ModelType)
	assert.True(t, info.ModelStatus["ttft_model_ready"])
}

func BenchmarkPredictionWire(b *testing.B) {
	request := PredictionRequest{KVCachePercentage: 0.5, InputTokenLength: 100, NumRequestWaiting: 2, NumRequestRunning: 3, NumTokensGenerated: 1, PrefixCacheScore: 0.25}
	type legacyPrediction PredictionRequest
	b.Run("legacy", func(b *testing.B) {
		for b.Loop() {
			_, err := json.Marshal(legacyPrediction(request))
			if err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("schema", func(b *testing.B) {
		for b.Loop() {
			_, err := json.Marshal(request)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkBulkPredictionWire(b *testing.B) {
	type legacyPrediction PredictionRequest
	for _, count := range []int{1, 20, 100} {
		b.Run(strconv.Itoa(count), func(b *testing.B) {
			requests := make([]PredictionRequest, count)
			legacy := make([]legacyPrediction, count)
			for index := range requests {
				requests[index] = PredictionRequest{KVCachePercentage: 0.5, InputTokenLength: 100 + index, NumRequestWaiting: 2, NumRequestRunning: 3, NumTokensGenerated: 1, PrefixCacheScore: 0.25}
				legacy[index] = legacyPrediction(requests[index])
			}
			b.Run("legacy", func(b *testing.B) {
				batch := struct {
					Requests []legacyPrediction `json:"requests"`
				}{Requests: legacy}
				for b.Loop() {
					if _, err := json.Marshal(batch); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("schema", func(b *testing.B) {
				batch := BulkPredictionRequest{Requests: requests}
				for b.Loop() {
					if _, err := json.Marshal(batch); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

func TestRidgeRegistryPreservesLegacyFormula(t *testing.T) {
	random := rand.New(rand.NewPCG(1, 2))
	for range 100 {
		request := PredictionRequest{KVCachePercentage: random.Float64(), InputTokenLength: random.IntN(1000), NumRequestWaiting: random.IntN(10), NumRequestRunning: random.IntN(10), NumTokensGenerated: random.IntN(20), PrefixCacheScore: random.Float64(), EncoderInputSize: 7, EncoderMatchedSize: 2}
		coefficients := map[string]float64{}
		for _, field := range ridgeTTFTFields {
			coefficients[string(field.Name())] = random.Float64()
		}
		for _, field := range ridgeTPOTFields {
			coefficients[string(field.Name())] = random.Float64()
		}
		expectedTTFT := 1 + coefficients["kv_cache_percentage"]*request.KVCachePercentage + coefficients["input_token_length"]*float64(request.InputTokenLength) + coefficients["num_request_waiting"]*float64(request.NumRequestWaiting) + coefficients["num_request_running"]*float64(request.NumRequestRunning) + coefficients["prefix_cache_score"]*request.PrefixCacheScore + coefficients["encoder_input_size"]*float64(request.EncoderInputSize) + coefficients["encoder_matched_size"]*float64(request.EncoderMatchedSize)
		expectedTPOT := 2 + coefficients["kv_cache_percentage"]*request.KVCachePercentage + coefficients["input_token_length"]*float64(request.InputTokenLength) + coefficients["num_request_waiting"]*float64(request.NumRequestWaiting) + coefficients["num_request_running"]*float64(request.NumRequestRunning) + coefficients["num_tokens_generated"]*float64(request.NumTokensGenerated)
		features := facadeFeatures(request, nil)
		actualTTFT, err := ridgePrediction(1, coefficients, features, ridgeTTFTFields)
		require.NoError(t, err)
		actualTPOT, err := ridgePrediction(2, coefficients, features, ridgeTPOTFields)
		require.NoError(t, err)
		assert.Equal(t, expectedTTFT, actualTTFT)
		assert.Equal(t, expectedTPOT, actualTPOT)
	}
}

func TestTreeJSONCompatibility(t *testing.T) {
	for _, data := range []string{`null`, `[]`, `[{"nodeid":0,"leaf":1.25}]`} {
		trees, err := readTrees(strings.NewReader(data))
		require.NoError(t, err)
		var expected []any
		require.NoError(t, json.Unmarshal([]byte(data), &expected))
		assert.Equal(t, expected, trees)
	}
}
