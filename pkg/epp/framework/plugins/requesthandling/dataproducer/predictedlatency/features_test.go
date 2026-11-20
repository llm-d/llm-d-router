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

package predictedlatency

import (
	"context"
	"testing"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

func TestFeatureProviderCoverage(t *testing.T) {
	require.NoError(t, validateFeatureProviders(featureProviders))
	for _, provider := range featureProviders {
		require.NotNil(t, provider.read, "%s", provider.name)
		assert.NotZero(t, provider.uses, "%s", provider.name)
		assert.True(t, provider.defaultValue.IsValid(), "%s", provider.name)
	}
}

func TestFeatureSnapshotsAndTrainingTiming(t *testing.T) {
	config := DefaultConfig
	config.UseEncoderCacheFeatures = true
	predictor := &mockPredictor{predictions: map[string]*latencypredictor.PredictionResponse{
		"0.3": {TTFT: 10, TPOT: 2},
	}}
	producer := NewPredictedLatency("test", config, predictor)
	request := createTestInferenceRequest("snapshot", 100, 20)
	endpoint := createTestEndpoint("worker", 0.3, 2, 99)
	endpoint.Put(producer.inFlightLoadDataKey, &attrconcurrency.InFlightLoad{Tokens: 123, Requests: 7})
	items := []attrmm.MatchItem{{Hash: "image", Size: 5}}
	endpoint.Put(producer.encoderCacheDataKey, attrmm.NewEncoderCacheMatchInfo(items, items))
	require.NoError(t, producer.Produce(context.Background(), request, []fwksched.Endpoint{endpoint}))
	state, err := producer.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	require.Len(t, predictor.capturedBulkStrictRequests, 1)
	prediction := predictor.capturedBulkStrictRequests[0]
	assert.Equal(t, int64(123), prediction.PrefillTokensInFlight)
	assert.Equal(t, 7, prediction.NumRequestRunning)
	assert.Equal(t, 5, prediction.EncoderInputSize)
	assert.Equal(t, 5, prediction.EncoderMatchedSize)
	assert.Equal(t, 1, prediction.NumTokensGenerated)

	endpoint.Put(producer.inFlightLoadDataKey, &attrconcurrency.InFlightLoad{Tokens: 999, Requests: 44})
	latest := &fwkdl.Metrics{KVCacheUsagePercent: 0.8, WaitingQueueSize: 9, RunningRequestsSize: 100}
	state.ttft = 50
	recordTTFTTrainingData(context.Background(), predictor, state, latest, endpoint.GetMetadata(), time.Now())
	require.Len(t, predictor.capturedTrainingEntries, 1)
	training := predictor.capturedTrainingEntries[0]
	assert.Equal(t, prediction.PrefillTokensInFlight, training.PrefillTokensInFlight)
	assert.Equal(t, prediction.NumRequestRunning, training.NumRequestRunning)
	assert.Equal(t, prediction.InputTokenLength, training.InputTokenLength)
	assert.Equal(t, 0.8, training.KVCachePercentage)
	assert.Equal(t, 9, training.NumRequestWaiting)
	assert.Equal(t, 0, training.NumTokensGenerated)
	assert.Equal(t, 5, training.EncoderMatchedSize)

	snapshot := endpointFeaturesFor(state, endpoint.GetMetadata())
	before := proto.Clone(snapshot)
	tpot := composeFeatures(state, latest, snapshot, tpotFeatures)
	assert.Zero(t, tpot.GetEncoderInputSize())
	assert.Zero(t, tpot.GetEncoderMatchedSize())
	assert.Zero(t, tpot.GetPrefixCacheScore())
	assert.Equal(t, int64(123), tpot.GetPrefillTokensInFlight())
	assert.True(t, proto.Equal(before, snapshot), "composing training features must not mutate dispatch snapshots")
}

func TestFeatureProviderRejectsIncompleteRegistry(t *testing.T) {
	require.Error(t, validateFeatureProviders(featureProviders[1:]))
	duplicate := append([]featureProvider{featureProviders[0]}, featureProviders...)
	require.Error(t, validateFeatureProviders(duplicate))
	unknown := append([]featureProvider(nil), featureProviders...)
	unknown[0].name = "unknown_field"
	require.Error(t, validateFeatureProviders(unknown))
}
