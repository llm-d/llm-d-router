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
	"errors"
	"testing"
	"time"

	"github.com/go-logr/zapr"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
)

type blockingBulkPredictor struct {
	*mockPredictor
}

func (p *blockingBulkPredictor) PredictBulkStrict(ctx context.Context, _ []latencypredictorclient.PredictionRequest) (*latencypredictorclient.BulkPredictionResponse, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestProduceDeadlinePredictionFailureIsObservable(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Millisecond)
	defer cancel()
	pl := NewPredictedLatency("test-plugin", DefaultConfig, &blockingBulkPredictor{mockPredictor: &mockPredictor{}})

	err := pl.Produce(ctx, createTestInferenceRequest("deadline", 0, 0), []fwksched.Endpoint{
		createTestEndpoint("pod-a", 0.5, 0, 0),
	})

	assert.ErrorIs(t, err, context.DeadlineExceeded)
	assert.Equal(t, float64(1), testutil.ToFloat64(llmdRequestPredictionFailures.WithLabelValues(
		"test-plugin", LatencyDataProviderPluginType, predictionFailureReasonPredictorError)))
}

func TestProducePredictionFailureLogsAreRateLimited(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	core, observed := observer.New(zapcore.InfoLevel)
	ctx := log.IntoContext(t.Context(), zapr.NewLogger(zap.New(core)))
	pl := NewPredictedLatency("test-plugin", DefaultConfig, &fixedBulkPredictor{
		mockPredictor: &mockPredictor{},
		err:           errors.New("predictor unavailable"),
	})
	endpoint := createTestEndpoint("pod-a", 0.5, 0, 0)
	for i := 0; i < 3; i++ {
		require.NoError(t, pl.Produce(ctx, createTestInferenceRequest("failure", 0, 0), []fwksched.Endpoint{endpoint}))
	}

	assert.Equal(t, 1, observed.FilterMessage("Latency prediction failed").Len())
	assert.Equal(t, 0, observed.FilterMessage("Bulk prediction failed").Len())
	assert.Equal(t, 0, observed.FilterMessage("bulk prediction failed").Len())
}
