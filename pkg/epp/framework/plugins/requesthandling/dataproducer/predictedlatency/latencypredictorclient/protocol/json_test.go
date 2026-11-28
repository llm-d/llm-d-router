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

package protocol_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
)

func wireFixtures(t *testing.T) map[string]json.RawMessage {
	t.Helper()
	data, err := os.ReadFile("testdata/wire.json")
	require.NoError(t, err)
	var fixtures map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &fixtures))
	return fixtures
}

func TestWireRoundTrip(t *testing.T) {
	fixtures := wireFixtures(t)
	tests := map[string]proto.Message{
		"prediction_zero":     &protocol.PredictionRequest{},
		"prediction_extended": &protocol.PredictionRequest{},
		"prediction_legacy":   &protocol.PredictionRequest{},
		"training_ttft":       &protocol.TrainingEntry{},
		"training_tpot":       &protocol.TrainingEntry{},
		"training_legacy":     &protocol.TrainingEntry{},
		"coefficients":        &protocol.ModelCoefficients{},
		"trees":               &protocol.XGBoostTrees{},
		"buckets":             &protocol.BucketCounts{},
		"model_info":          &protocol.ModelInfo{},
		"export_metadata":     &protocol.ModelExportMetadata{},
	}
	for name, message := range tests {
		t.Run(name, func(t *testing.T) {
			require.NoError(t, protocol.Unmarshal(fixtures[name], message))
			data, err := protocol.Marshal(message)
			require.NoError(t, err)
			require.JSONEq(t, string(fixtures[name]), string(data))
		})
	}
}

func TestExistingGoTypesAcceptProtocolJSON(t *testing.T) {
	fixtures := wireFixtures(t)
	tests := []struct {
		name    string
		message proto.Message
		legacy  any
	}{
		{"prediction_zero", &protocol.PredictionRequest{}, &latencypredictorclient.PredictionRequest{}},
		{"prediction_extended", &protocol.PredictionRequest{}, &latencypredictorclient.PredictionRequest{}},
		{"training_ttft", &protocol.TrainingEntry{}, &latencypredictorclient.TrainingEntry{}},
		{"training_tpot", &protocol.TrainingEntry{}, &latencypredictorclient.TrainingEntry{}},
		{"coefficients", &protocol.ModelCoefficients{}, &latencypredictorclient.ModelCoefficients{}},
		{"trees", &protocol.XGBoostTrees{}, &latencypredictorclient.XGBoostTrees{}},
		{"model_info", &protocol.ModelInfo{}, &latencypredictorclient.ModelInfo{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.NoError(t, protocol.Unmarshal(fixtures[tt.name], tt.message))
			data, err := protocol.Marshal(tt.message)
			require.NoError(t, err)
			require.NoError(t, json.Unmarshal(data, tt.legacy))
			legacyData, err := json.Marshal(tt.legacy)
			require.NoError(t, err)
			require.JSONEq(t, string(fixtures[tt.name]), string(legacyData))
		})
	}
}

func TestOptionalZeroAndMissing(t *testing.T) {
	fixtures := wireFixtures(t)
	var message protocol.PredictionRequest
	require.NoError(t, protocol.Unmarshal(fixtures["prediction_zero"], &message))
	require.NotNil(t, message.GetFeatures().EncoderInputSize)
	require.Zero(t, message.GetFeatures().GetEncoderInputSize())
	require.NoError(t, protocol.Unmarshal(fixtures["prediction_legacy"], &message))
	require.Nil(t, message.GetFeatures().EncoderInputSize)
	data, err := protocol.Marshal(&message)
	require.NoError(t, err)
	require.JSONEq(t, string(fixtures["prediction_legacy"]), string(data))
}

func TestUnknownFields(t *testing.T) {
	var message protocol.PredictionRequest
	require.NoError(t, protocol.Unmarshal([]byte(`{"input_token_length":100,"future_feature":123,"features":{"input_token_length":999}}`), &message))
	require.Equal(t, int64(100), message.GetFeatures().GetInputTokenLength())
	data, err := protocol.Marshal(&message)
	require.NoError(t, err)
	require.JSONEq(t, `{"input_token_length":100}`, string(data))
}

func TestLargeIntegers(t *testing.T) {
	fixtures := wireFixtures(t)
	var message protocol.PredictionRequest
	require.NoError(t, protocol.Unmarshal(fixtures["prediction_extended"], &message))
	require.Equal(t, int64(9007199254740993), message.GetFeatures().GetPrefillTokensInFlight())
	require.Equal(t, int64(9223372036854775807), message.GetFeatures().GetDecodeTokensInFlight())
	data, err := protocol.Marshal(&message)
	require.NoError(t, err)
	var legacy latencypredictorclient.PredictionRequest
	require.NoError(t, json.Unmarshal(data, &legacy))
	require.Equal(t, message.GetFeatures().GetPrefillTokensInFlight(), legacy.PrefillTokensInFlight)
	require.Equal(t, message.GetFeatures().GetDecodeTokensInFlight(), legacy.DecodeTokensInFlight)
}

func TestBulkRequestsKeepFlatRows(t *testing.T) {
	fixtures := wireFixtures(t)
	tests := []struct {
		field   string
		fixture string
		message proto.Message
	}{
		{"requests", "prediction_extended", &protocol.BulkPredictionRequest{}},
		{"entries", "training_ttft", &protocol.BulkTrainingRequest{}},
	}
	for _, tt := range tests {
		t.Run(tt.field, func(t *testing.T) {
			data, err := json.Marshal(map[string][]json.RawMessage{tt.field: {fixtures[tt.fixture], fixtures[tt.fixture]}})
			require.NoError(t, err)
			require.NoError(t, protocol.Unmarshal(data, tt.message))
			encoded, err := protocol.Marshal(tt.message)
			require.NoError(t, err)
			require.JSONEq(t, string(data), string(encoded))
		})
	}
}

func TestInvalidInputDoesNotReplaceMessage(t *testing.T) {
	message := &protocol.PredictionRequest{Features: &protocol.Features{InputTokenLength: proto.Int64(100)}}
	for _, data := range []string{`{"input_token_length":1.5}`, `{"input_token_length":9223372036854775808}`, `[]`, `null`} {
		require.Error(t, protocol.Unmarshal([]byte(data), message))
		require.Equal(t, int64(100), message.GetFeatures().GetInputTokenLength())
	}
}

func TestNonfiniteNumbersAreNotSentAsJSONStrings(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		message := &protocol.PredictionRequest{Features: &protocol.Features{KvCachePercentage: proto.Float64(value)}}
		_, err := protocol.Marshal(message)
		require.Error(t, err)
	}
}
