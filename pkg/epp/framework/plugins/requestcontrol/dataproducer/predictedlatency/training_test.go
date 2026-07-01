/*
Copyright 2025 The Kubernetes Authors.

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
	"strings"
	"testing"

	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestBulkPredictWithMetrics(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
			"0.6": {TTFT: 0.6, TPOT: 0.04},
		},
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
		{KVCacheUsagePercent: 0.6},
	}
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod2"},
		},
	}
	inputTokenLengths := []int{1, 1}
	generatedTokenCounts := []int{1, 1}
	prefixCacheScores := []float64{0.0, 0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil)

	assert.NoError(t, err)
	assert.Len(t, results, 2)
	assert.Equal(t, 0.5, results[0].TTFT)
	assert.Equal(t, 0.03, results[0].TPOT)
	assert.Equal(t, 0.6, results[1].TTFT)
	assert.Equal(t, 0.04, results[1].TPOT)
}

func TestBulkPredictWithMetrics_Error(t *testing.T) {
	mockPredictor := &mockPredictor{
		err: errors.New("prediction failed"),
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
	}
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	inputTokenLengths := []int{1}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil)

	assert.Error(t, err)
	assert.Nil(t, results)
}

func TestBulkPredictWithMetrics_InputMismatch(t *testing.T) {
	mockPredictor := &mockPredictor{}
	metricsStates := []*fwkdl.Metrics{{}}
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	inputTokenLengths := []int{1, 1} // Mismatch length
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil)

	assert.Error(t, err)
	assert.Nil(t, results)
	assert.True(t, strings.Contains(err.Error(), "input slice lengths must match"))
}

func TestBulkPredictWithMetrics_WithPredictedLatencyCtx(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
		},
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
	}
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	inputTokenLengths := []int{1}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	plCtx := &predictedLatencyCtx{
		schedulingRequest: fwksched.InferenceRequest{
			TargetModel: "test-model",
		},
		incomingModelName: "incoming-model",
	}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", plCtx, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil)

	assert.NoError(t, err)
	assert.Len(t, results, 1)
	assert.Equal(t, 0.5, results[0].TTFT)
	assert.Equal(t, 0.03, results[0].TPOT)
}

func TestBulkPredictWithMetrics_ChatCompletionsInputTokenLength(t *testing.T) {
	mp := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
		},
	}

	metricsStates := []*fwkdl.Metrics{{KVCacheUsagePercent: 0.5}}
	pods := []*fwkdl.EndpointMetadata{
		{NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"}},
	}

	inputTokenLengths := []int{2}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mp, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, []int64{0}, nil)

	assert.NoError(t, err)
	assert.Len(t, results, 1)
	assert.Equal(t, 0.5, results[0].TTFT)
}

func TestBulkPredictWithMetrics_NilMetricsState(t *testing.T) {
	mockPredictor := &mockPredictor{}
	metricsStates := []*fwkdl.Metrics{nil} // Nil metrics state
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	inputTokenLengths := []int{1}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), "test-plugin", "test-type", nil, mockPredictor, metricsStates, "", pods, inputTokenLengths, generatedTokenCounts, prefixCacheScores, nil, nil)

	assert.Error(t, err)
	assert.Nil(t, results)
	assert.True(t, strings.Contains(err.Error(), "metrics state at index 0 cannot be nil"))
}
