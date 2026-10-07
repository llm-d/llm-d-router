/*
Copyright 2025 The Kubernetes Authors.
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
	"errors"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
)

type fixedBulkPredictor struct {
	*mockPredictor
	response *latencypredictorclient.BulkPredictionResponse
	err      error
}

func (p *fixedBulkPredictor) PredictBulkStrict(_ context.Context, _ []latencypredictorclient.PredictionRequest) (*latencypredictorclient.BulkPredictionResponse, error) {
	return p.response, p.err
}

func TestProducePredictionFailureObservability(t *testing.T) {
	tests := []struct {
		name       string
		predictor  *fixedBulkPredictor
		wantReason string
	}{
		{
			name:       "predictor error",
			predictor:  &fixedBulkPredictor{mockPredictor: &mockPredictor{}, err: errors.New("predictor unavailable")},
			wantReason: predictionFailureReasonPredictorError,
		},
		{
			name:       "nil response",
			predictor:  &fixedBulkPredictor{mockPredictor: &mockPredictor{}},
			wantReason: predictionFailureReasonNilResponse,
		},
		{
			name: "short response",
			predictor: &fixedBulkPredictor{mockPredictor: &mockPredictor{}, response: &latencypredictorclient.BulkPredictionResponse{
				Predictions: []latencypredictorclient.PredictionResponse{},
			}},
			wantReason: predictionFailureReasonLengthMismatch,
		},
		{
			name: "long response",
			predictor: &fixedBulkPredictor{mockPredictor: &mockPredictor{}, response: &latencypredictorclient.BulkPredictionResponse{
				Predictions: []latencypredictorclient.PredictionResponse{{}, {}},
			}},
			wantReason: predictionFailureReasonLengthMismatch,
		},
		{
			name: "reported failed prediction",
			predictor: &fixedBulkPredictor{mockPredictor: &mockPredictor{}, response: &latencypredictorclient.BulkPredictionResponse{
				Predictions:       []latencypredictorclient.PredictionResponse{{}},
				FailedPredictions: 1,
			}},
			wantReason: predictionFailureReasonPredictorError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetMetrics()
			t.Cleanup(resetMetrics)
			pl := NewPredictedLatency("test-plugin", DefaultConfig, tt.predictor)
			request := createTestInferenceRequest(tt.name, 0, 0)
			endpoint := createTestEndpoint("pod-a", 0.5, 0, 0)

			require.NoError(t, pl.Produce(t.Context(), request, []fwksched.Endpoint{endpoint}))
			_, hasPrediction := endpoint.Get(pl.latencyPredictionInfoDataKey)
			assert.False(t, hasPrediction)
			assert.Equal(t, float64(1), testutil.ToFloat64(llmdRequestPredictionFailures.WithLabelValues(
				"test-plugin", LatencyDataProviderPluginType, tt.wantReason)))
		})
	}
}

func TestProducePredictionSuccessAndDisabledDoNotCountFailure(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)
	endpoint := createTestEndpoint("pod-a", 0.5, 0, 0)
	predictor := &fixedBulkPredictor{mockPredictor: &mockPredictor{}, response: &latencypredictorclient.BulkPredictionResponse{
		Predictions: []latencypredictorclient.PredictionResponse{{TTFT: 1, TPOT: 0.1}},
	}}
	pl := NewPredictedLatency("test-plugin", DefaultConfig, predictor)
	require.NoError(t, pl.Produce(t.Context(), createTestInferenceRequest("success", 0, 0), []fwksched.Endpoint{endpoint}))
	_, hasPrediction := endpoint.Get(pl.latencyPredictionInfoDataKey)
	assert.True(t, hasPrediction)

	disabledConfig := DefaultConfig
	disabledConfig.PredictInProduce = false
	disabled := NewPredictedLatency("test-plugin", disabledConfig, &fixedBulkPredictor{
		mockPredictor: &mockPredictor{}, err: errors.New("must not be called"),
	})
	require.NoError(t, disabled.Produce(t.Context(), createTestInferenceRequest("disabled", 0, 0), []fwksched.Endpoint{endpoint}))
	for _, reason := range []string{
		predictionFailureReasonRequestError,
		predictionFailureReasonPredictorError,
		predictionFailureReasonNilResponse,
		predictionFailureReasonLengthMismatch,
	} {
		assert.Equal(t, float64(0), testutil.ToFloat64(llmdRequestPredictionFailures.WithLabelValues(
			"test-plugin", LatencyDataProviderPluginType, reason)))
	}
}

func TestProduceCancelledRequestDoesNotCountPredictionFailure(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pl := NewPredictedLatency("test-plugin", DefaultConfig, &fixedBulkPredictor{
		mockPredictor: &mockPredictor{}, err: context.Canceled,
	})
	err := pl.Produce(ctx, createTestInferenceRequest("cancelled", 0, 0), []fwksched.Endpoint{
		createTestEndpoint("pod-a", 0.5, 0, 0),
	})
	assert.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, float64(0), testutil.ToFloat64(llmdRequestPredictionFailures.WithLabelValues(
		"test-plugin", LatencyDataProviderPluginType, predictionFailureReasonPredictorError)))
}

func TestProducesConsumes(t *testing.T) {
	pl := NewPredictedLatency(LatencyDataProviderPluginType, DefaultConfig, nil)

	produces := pl.Produces()
	expectedProduceKey := attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(pl.TypedName().Name)
	assert.Contains(t, produces, expectedProduceKey)

	consumes := pl.Consumes()
	assert.Contains(t, consumes.Required, attrprefix.PrefixCacheMatchInfoDataKey)
	assert.NotContains(t, consumes.Required, attrmm.EncoderCacheMatchInfoKey,
		"encoder-cache match data must not be consumed when the feature is disabled")
}

func TestConsumes_EncoderCacheFeatureEnabled(t *testing.T) {
	cfg := DefaultConfig
	cfg.UseEncoderCacheFeatures = true
	pl := NewPredictedLatency(LatencyDataProviderPluginType, cfg, nil)

	consumes := pl.Consumes()
	assert.Contains(t, consumes.Required, attrmm.EncoderCacheMatchInfoKey)
}

// TestProduce_CapturesEncoderCacheSizes verifies that Produce reads the
// multimodal encoder-cache match data attached to endpoints and captures the
// request's input size plus the per-endpoint matched size, leaving endpoints
// without match data (text-only requests) at 0.
func TestProduce_CapturesEncoderCacheSizes(t *testing.T) {
	cfg := DefaultConfig
	cfg.PredictInProduce = false
	cfg.UseEncoderCacheFeatures = true
	pl := NewPredictedLatency(LatencyDataProviderPluginType, cfg, nil)

	request := createTestInferenceRequest("encoder-test", 0, 0)
	matched := createTestEndpoint("pod-matched", 0.1, 0, 0)
	unmatched := createTestEndpoint("pod-unmatched", 0.1, 0, 0)

	items := []attrmm.MatchItem{{Hash: "img-a", Size: 1}, {Hash: "img-b", Size: 1}}
	matched.Put(pl.encoderCacheDataKey, attrmm.NewEncoderCacheMatchInfo(items[:1], items))
	unmatched.Put(pl.encoderCacheDataKey, attrmm.NewEncoderCacheMatchInfo(nil, items))

	require.NoError(t, pl.Produce(context.Background(), request, []fwksched.Endpoint{matched, unmatched}))

	plCtx, err := pl.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	assert.Equal(t, 2, plCtx.encoderInputSize)
	assert.Equal(t, 1, plCtx.encoderMatchedSizeForEndpoints["pod-matched"])
	assert.Equal(t, 0, plCtx.encoderMatchedSizeForEndpoints["pod-unmatched"])
}

// TestProduce_ClampsInconsistentEncoderMatchData verifies that match data
// whose matched size exceeds the input size is clamped so downstream
// predictor validation (matched <= input) cannot reject the request.
func TestProduce_ClampsInconsistentEncoderMatchData(t *testing.T) {
	cfg := DefaultConfig
	cfg.PredictInProduce = false
	cfg.UseEncoderCacheFeatures = true
	pl := NewPredictedLatency(LatencyDataProviderPluginType, cfg, nil)

	request := createTestInferenceRequest("encoder-clamp-test", 0, 0)
	endpoint := createTestEndpoint("pod-a", 0.1, 0, 0)

	matched := []attrmm.MatchItem{{Hash: "img-a", Size: 3}}
	requestItems := []attrmm.MatchItem{{Hash: "img-b", Size: 1}}
	endpoint.Put(pl.encoderCacheDataKey, attrmm.NewEncoderCacheMatchInfo(matched, requestItems))

	require.NoError(t, pl.Produce(context.Background(), request, []fwksched.Endpoint{endpoint}))

	plCtx, err := pl.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	assert.Equal(t, 1, plCtx.encoderInputSize)
	assert.Equal(t, 1, plCtx.encoderMatchedSizeForEndpoints["pod-a"])
}

// TestProduce_EncoderCacheFeatureDisabledIgnoresMatchData is the negative
// control: with the feature off, attached match data is not read.
func TestProduce_EncoderCacheFeatureDisabledIgnoresMatchData(t *testing.T) {
	cfg := DefaultConfig
	cfg.PredictInProduce = false
	pl := NewPredictedLatency(LatencyDataProviderPluginType, cfg, nil)

	request := createTestInferenceRequest("encoder-disabled-test", 0, 0)
	endpoint := createTestEndpoint("pod-a", 0.1, 0, 0)
	items := []attrmm.MatchItem{{Hash: "img-a", Size: 1}}
	endpoint.Put(pl.encoderCacheDataKey, attrmm.NewEncoderCacheMatchInfo(items, items))

	require.NoError(t, pl.Produce(context.Background(), request, []fwksched.Endpoint{endpoint}))

	plCtx, err := pl.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	assert.Equal(t, 0, plCtx.encoderInputSize)
	assert.Empty(t, plCtx.encoderMatchedSizeForEndpoints)
}

// TestProduce_CancelledContextDoesNotPublish verifies that when the
// director's Produce window has already closed (ctx cancelled), the plugin
// does not publish the SLO context into the ttlcache. If it did, ResponseBody
// would later find the context and issue an orphan decrement against counters
// PreRequest never incremented — draining prefillTokensInFlight negative.
func TestProduce_CancelledContextDoesNotPublish(t *testing.T) {
	cfg := DefaultConfig
	cfg.PredictInProduce = false // skip the prediction sidecar path
	pl := NewPredictedLatency(LatencyDataProviderPluginType, cfg, nil)

	request := createTestInferenceRequest("cancel-test", 0, 0)
	endpoint := createTestEndpoint("pod-a", 0.1, 0, 0)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before the plugin runs

	err := pl.Produce(ctx, request, []fwksched.Endpoint{endpoint})
	assert.ErrorIs(t, err, context.Canceled, "should propagate ctx.Err() on cancelled context")

	_, getErr := pl.getPredictedLatencyContextForRequest(request)
	assert.Error(t, getErr, "SLO context should NOT be stored when ctx is cancelled")
}

// TestProduce_LivesContextPublishes is the positive control for the
// cancellation test above: with a live context, the fast-path store still fires.
func TestProduce_LiveContextPublishes(t *testing.T) {
	cfg := DefaultConfig
	cfg.PredictInProduce = false
	pl := NewPredictedLatency(LatencyDataProviderPluginType, cfg, nil)

	request := createTestInferenceRequest("live-test", 0, 0)
	endpoint := createTestEndpoint("pod-a", 0.1, 0, 0)

	err := pl.Produce(context.Background(), request, []fwksched.Endpoint{endpoint})
	assert.NoError(t, err)

	_, getErr := pl.getPredictedLatencyContextForRequest(request)
	assert.NoError(t, getErr, "SLO context should be stored on the happy path")
}
