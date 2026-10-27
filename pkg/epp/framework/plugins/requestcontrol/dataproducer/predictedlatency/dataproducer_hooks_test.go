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
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
)

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

func TestProduce_PredictionFailureObservability(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	tests := []struct {
		name       string
		predictor  latencypredictor.PredictorInterface
		wantReason string
	}{
		{
			name:       "predictor unavailable",
			predictor:  nil,
			wantReason: predictionFailureReasonPredictorUnavailable,
		},
		{
			name:       "request error",
			predictor:  &mockPredictor{err: errors.New("connection refused")},
			wantReason: predictionFailureReasonRequestError,
		},
		{
			name:       "nil response",
			predictor:  &mockPredictor{nilBulkResponse: true},
			wantReason: predictionFailureReasonNilResponse,
		},
		{
			name: "length mismatch",
			predictor: &mockPredictor{
				bulkPredictionsOverride: []latencypredictor.PredictionResponse{},
			},
			wantReason: predictionFailureReasonLengthMismatch,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pluginName := "test-" + tc.wantReason
			pl := NewPredictedLatency(pluginName, DefaultConfig, tc.predictor)

			var errorLogs []string
			logger := funcr.New(func(prefix, args string) {
				errorLogs = append(errorLogs, prefix+args)
			}, funcr.Options{Verbosity: 0})
			ctx := log.IntoContext(context.Background(), logger)

			before := promtestutil.ToFloat64(llmdRequestPredictionFailures.WithLabelValues(pluginName, LatencyDataProviderPluginType, tc.wantReason))

			endpoint := createTestEndpoint("pod-a", 0.1, 0, 0)
			req1 := createTestInferenceRequest("req-1", 0, 0)
			req2 := createTestInferenceRequest("req-2", 0, 0)

			require.NoError(t, pl.Produce(ctx, req1, []fwksched.Endpoint{endpoint}))
			require.NoError(t, pl.Produce(ctx, req2, []fwksched.Endpoint{endpoint}))

			after := promtestutil.ToFloat64(llmdRequestPredictionFailures.WithLabelValues(pluginName, LatencyDataProviderPluginType, tc.wantReason))
			assert.InDelta(t, 2.0, after-before, 1e-9, "every failed prediction must increment request_prediction_failures_total")
			assert.Len(t, errorLogs, 1, "error log must fire on first failure and rate-limit rapid follow-up failures")
		})
	}

	t.Run("context canceled ignored", func(t *testing.T) {
		resetMetrics()
		pl := NewPredictedLatency("test-canceled", DefaultConfig, &mockPredictor{err: context.Canceled})

		var errorLogs []string
		logger := funcr.New(func(prefix, args string) {
			errorLogs = append(errorLogs, prefix+args)
		}, funcr.Options{Verbosity: 0})
		ctx := log.IntoContext(context.Background(), logger)

		endpoint := createTestEndpoint("pod-a", 0.1, 0, 0)
		req := createTestInferenceRequest("req-canceled", 0, 0)

		require.NoError(t, pl.Produce(ctx, req, []fwksched.Endpoint{endpoint}))
		assert.Equal(t, 0, promtestutil.CollectAndCount(llmdRequestPredictionFailures), "context cancellation must not increment request_prediction_failures_total")
		assert.Empty(t, errorLogs, "context cancellation must not emit error log")
	})

	t.Run("coalesced HTTP length mismatch", func(t *testing.T) {
		resetMetrics()
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(latencypredictor.BulkPredictionResponse{
				Predictions: []latencypredictor.PredictionResponse{},
			})
		}))
		t.Cleanup(server.Close)

		cfg := latencypredictor.DefaultConfig()
		cfg.PredictionURLs = []string{server.URL}
		cfg.TrainingURL = server.URL
		cfg.CoalesceWindow = time.Millisecond

		var errorLogs []string
		logger := funcr.New(func(prefix, args string) {
			errorLogs = append(errorLogs, prefix+args)
		}, funcr.Options{Verbosity: 0})
		ctx := log.IntoContext(context.Background(), logger)

		predictor := latencypredictor.New(cfg, logger)
		t.Cleanup(func() { predictor.Stop(context.Background()) })

		pluginName := "test-coalesced-length-mismatch"
		pl := NewPredictedLatency(pluginName, DefaultConfig, predictor)
		endpoint := createTestEndpoint("pod-a", 0.1, 0, 0)
		req := createTestInferenceRequest("req-coalesced", 0, 0)

		require.NoError(t, pl.Produce(ctx, req, []fwksched.Endpoint{endpoint}))
		after := promtestutil.ToFloat64(llmdRequestPredictionFailures.WithLabelValues(pluginName, LatencyDataProviderPluginType, predictionFailureReasonLengthMismatch))
		assert.InDelta(t, 1.0, after, 1e-9, "coalesced length mismatch must increment length_mismatch counter")
		assert.Len(t, errorLogs, 1)
	})
}
