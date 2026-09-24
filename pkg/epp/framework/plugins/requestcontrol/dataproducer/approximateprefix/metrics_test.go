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

package approximateprefix

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry))
}

func TestRecordPrefixCacheMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	recordPrefixCacheSize("test-plugin", "test-type", 4096)
	recordPrefixCacheMatch("test-plugin", "test-type", 10, 20)
	recordPrefixCacheMatch("test-plugin", "test-type", 0, 0)

	require.Equal(t, float64(4096), testutil.ToFloat64(llmdPrefixCacheSize.WithLabelValues("test-plugin", "test-type")))

	hitRatio, err := getHistogram(llmdPrefixCacheHitRatio, "test-plugin", "test-type")
	require.NoError(t, err)
	require.Equal(t, uint64(1), hitRatio.GetSampleCount())
	require.Equal(t, 0.5, hitRatio.GetSampleSum())

	hitLength, err := getHistogram(llmdPrefixCacheHitLength, "test-plugin", "test-type")
	require.NoError(t, err)
	require.Equal(t, uint64(2), hitLength.GetSampleCount())
	require.Equal(t, float64(10), hitLength.GetSampleSum())
}

func getHistogram(histogram *prometheus.HistogramVec, labelValues ...string) (*dto.Histogram, error) {
	metric, err := histogram.GetMetricWithLabelValues(labelValues...)
	if err != nil {
		return nil, err
	}
	dtoMetric := &dto.Metric{}
	if err := metric.(prometheus.Histogram).Write(dtoMetric); err != nil {
		return nil, err
	}
	return dtoMetric.GetHistogram(), nil
}

func resetMetrics() {
	llmdPrefixCacheSize.Reset()
	llmdPrefixCacheHitRatio.Reset()
	llmdPrefixCacheHitLength.Reset()
}

// PreRequest reports the prefix hit for the chosen endpoint in tokens, together
// with the prompt tokens it was measured against.
func TestPreRequestRecordsPrediction(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-records"
	p := producerForPrediction(t, name, 2)
	endpoints, result := endpointAndResult()

	// Seed the indexer: nothing is cached yet, so the prediction is zero while
	// the prompt still lands in the denominator.
	tokens := []uint32{1, 2, 3, 4}
	runPrediction(t, p, "seed", tokens, endpoints, result)
	require.Equal(t, float64(0), metricSum(t, predictedCachedTokensMetric, name))
	require.Equal(t, float64(len(tokens)), metricSum(t, promptTokensMetric, name))

	// The same prompt now matches every block on the endpoint that was chosen.
	runPrediction(t, p, "repeat", tokens, endpoints, result)
	assert.Equal(t, float64(len(tokens)), metricSum(t, predictedCachedTokensMetric, name))
	assert.Equal(t, float64(2*len(tokens)), metricSum(t, promptTokensMetric, name))
}

// A prompt whose length is not a multiple of the block size still hashes its
// trailing partial block, so the block-to-token conversion has to be bounded by
// the prompt length or a full match reports more tokens than the prompt holds.
func TestPreRequestPredictionBoundedByPromptLength(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-partial-block"
	p := producerForPrediction(t, name, 4)
	endpoints, result := endpointAndResult()

	// 5 tokens at block size 4 hash to 2 blocks, the second covering 1 token.
	tokens := []uint32{1, 2, 3, 4, 5}
	runPrediction(t, p, "seed", tokens, endpoints, result)
	require.Equal(t, float64(0), metricSum(t, predictedCachedTokensMetric, name))

	runPrediction(t, p, "repeat", tokens, endpoints, result)
	assert.Equal(t, float64(len(tokens)), metricSum(t, predictedCachedTokensMetric, name),
		"a full match must report the prompt's 5 tokens, not 2 blocks * 4 tokens")
}

// Each prompt is bounded on its own: clamping the aggregate would let a short
// prompt's overshoot hide under a long prompt's length.
func TestPreRequestPredictionBoundsEachPromptSeparately(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-multi-prompt"
	p := producerForPrediction(t, name, 4)
	endpoints, result := endpointAndResult()

	// 5 tokens (2 blocks, 1 partial) alongside 8 tokens (2 full blocks).
	body := &fwkrh.InferenceRequestBody{
		TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{
			{TokenIDs: []uint32{1, 2, 3, 4, 5}},
			{TokenIDs: []uint32{6, 7, 8, 9, 10, 11, 12, 13}},
		}},
	}
	runPredictionWithBody(t, p, "seed", body, endpoints, result)
	runPredictionWithBody(t, p, "repeat", body, endpoints, result)

	assert.Equal(t, float64(13), metricSum(t, predictedCachedTokensMetric, name))
	assert.Equal(t, float64(26), metricSum(t, promptTokensMetric, name))
}

// A token cap below the block size hashes nothing, so no endpoint can be
// predicted to hold any of the prompt. The prompt still has to reach the
// denominator, or the misconfiguration leaves the metric empty instead of
// reporting a zero hit rate.
func TestPreRequestPredictionCountsUnhashedPrompts(t *testing.T) {
	disableMinBlockSizeClamp(t)

	const name = "approx-predicted-no-hashes"
	p, err := newDataProducer(context.Background(), name, config{
		BlockSizeTokens:        4,
		MaxPrefixTokensToMatch: 2,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}, testHandle())
	require.NoError(t, err)
	endpoints, result := endpointAndResult()

	tokens := []uint32{1, 2, 3, 4, 5}
	runPrediction(t, p, "unhashed", tokens, endpoints, result)

	assert.Equal(t, float64(0), metricSum(t, predictedCachedTokensMetric, name))
	assert.Equal(t, float64(len(tokens)), metricSum(t, promptTokensMetric, name))
}

func producerForPrediction(t *testing.T, name string, blockSize int) *dataProducer {
	t.Helper()
	p, err := newDataProducer(context.Background(), name, config{
		BlockSizeTokens:        blockSize,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}, testHandle())
	require.NoError(t, err)
	return p
}

func endpointAndResult() ([]fwksched.Endpoint, *fwksched.SchedulingResult) {
	endpoints := []fwksched.Endpoint{fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes())}
	return endpoints, &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: endpoints},
		},
	}
}

func runPrediction(t *testing.T, p *dataProducer, id string, tokens []uint32,
	endpoints []fwksched.Endpoint, result *fwksched.SchedulingResult,
) {
	t.Helper()
	runPredictionWithBody(t, p, id, tokenizedBody(tokens), endpoints, result)
}

func runPredictionWithBody(t *testing.T, p *dataProducer, id string, body *fwkrh.InferenceRequestBody,
	endpoints []fwksched.Endpoint, result *fwksched.SchedulingResult,
) {
	t.Helper()
	req := &fwksched.InferenceRequest{RequestID: id, TargetModel: "m", Body: body}
	require.NoError(t, p.Produce(context.Background(), req, endpoints))
	require.NoError(t, p.PreRequest(context.Background(), req, result))
	p.wg.Wait()
}

const (
	predictedCachedTokensMetric = "llm_d_epp_prefix_predicted_cached_tokens"
	promptTokensMetric          = "llm_d_epp_prefix_prompt_tokens"
)

// metricSum reads a shared prefix metric out of the registry it is registered
// against, since those metrics live in another package.
func metricSum(t *testing.T, metricName, pluginName string) float64 {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != metricName {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				if label.GetName() == "plugin_name" && label.GetValue() == pluginName {
					return metric.GetHistogram().GetSampleSum()
				}
			}
		}
	}
	return 0
}
