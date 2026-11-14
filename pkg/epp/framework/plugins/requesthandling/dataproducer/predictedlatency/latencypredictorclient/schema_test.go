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
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

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

func TestTreeJSONCompatibility(t *testing.T) {
	for _, data := range []string{`null`, `[]`, `[{"nodeid":0,"leaf":1.25}]`} {
		trees, err := readTrees(strings.NewReader(data))
		require.NoError(t, err)
		var expected []any
		require.NoError(t, json.Unmarshal([]byte(data), &expected))
		assert.Equal(t, expected, trees)
	}
}
