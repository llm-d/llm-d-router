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

package lengthadaptiveload

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
)

func newEndpointWithLoad(name string, requests, tokens int64) fwksched.Endpoint {
	return newEndpointWithLoadAndLabels(name, requests, tokens, nil)
}

func newEndpointWithLoadAndLabels(name string, requests, tokens int64, labels map[string]string) fwksched.Endpoint {
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:     types.NamespacedName{Namespace: "default", Name: name},
			Labels: labels,
		},
		&fwkdl.Metrics{},
		nil,
	)
	ep.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{
		Requests: requests,
		Tokens:   tokens,
	})
	return ep
}

func newTokenizedRequest(numTokens int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: "test-req",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: make([]uint32, numTokens)}},
			},
		},
	}
}

func TestFactory(t *testing.T) {
	tests := []struct {
		name       string
		pluginName string
		jsonParams string
		expectErr  bool
		check      func(t *testing.T, s *LengthAdaptiveLoadScorer)
	}{
		{
			name:       "default parameters",
			pluginName: "default-scorer",
			jsonParams: `{}`,
			expectErr:  false,
			check: func(t *testing.T, s *LengthAdaptiveLoadScorer) {
				assert.Equal(t, "default-scorer", s.TypedName().Name)
				assert.Equal(t, LengthAdaptiveLoadScorerType, s.TypedName().Type)
				assert.Equal(t, defaultShortThresholdTokens, s.params.ShortThresholdTokens)
				assert.Equal(t, defaultLongThresholdTokens, s.params.LongThresholdTokens)
				assert.Equal(t, defaultMaxConcurrency, s.params.MaxConcurrency)
				assert.Equal(t, defaultMaxTokenCapacity, s.params.MaxTokenCapacity)
				assert.Equal(t, DefaultMaxConcurrencyLabel, s.params.MaxConcurrencyLabel)
				assert.Equal(t, DefaultMaxTokenCapacityLabel, s.params.MaxTokenCapacityLabel)
			},
		},
		{
			name:       "custom parameters",
			pluginName: "custom-scorer",
			jsonParams: `{"shortThresholdTokens": 256, "longThresholdTokens": 8192, "maxConcurrency": 128, "maxTokenCapacity": 100000, "maxConcurrencyLabel": "custom/max-seqs", "maxTokenCapacityLabel": "custom/max-tokens", "inFlightLoadProducerName": "custom-producer"}`,
			expectErr:  false,
			check: func(t *testing.T, s *LengthAdaptiveLoadScorer) {
				assert.Equal(t, 256, s.params.ShortThresholdTokens)
				assert.Equal(t, 8192, s.params.LongThresholdTokens)
				assert.Equal(t, 128, s.params.MaxConcurrency)
				assert.Equal(t, 100000, s.params.MaxTokenCapacity)
				assert.Equal(t, "custom/max-seqs", s.params.MaxConcurrencyLabel)
				assert.Equal(t, "custom/max-tokens", s.params.MaxTokenCapacityLabel)
				assert.Equal(t, "custom-producer", s.params.InFlightLoadProducerName)
			},
		},
		{
			name:       "longThresholdTokens less than or equal to shortThresholdTokens is adjusted",
			pluginName: "clamped-thresholds",
			jsonParams: `{"shortThresholdTokens": 1024, "longThresholdTokens": 512}`,
			expectErr:  false,
			check: func(t *testing.T, s *LengthAdaptiveLoadScorer) {
				assert.Equal(t, 1024, s.params.ShortThresholdTokens)
				assert.Equal(t, 1025, s.params.LongThresholdTokens)
			},
		},
		{
			name:       "malformed JSON returns error",
			pluginName: "bad-json",
			jsonParams: `{"shortThresholdTokens": "invalid"}`,
			expectErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := json.RawMessage(tt.jsonParams)
			p, err := Factory(tt.pluginName, fwkplugin.StrictDecoder(raw), nil)
			if tt.expectErr {
				require.Error(t, err)
				assert.Nil(t, p)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, p)
			scorer, ok := p.(*LengthAdaptiveLoadScorer)
			require.True(t, ok)
			if tt.check != nil {
				tt.check(t, scorer)
			}
		})
	}
}

func TestLengthAdaptiveLoadScorer_MetadataAndConsumes(t *testing.T) {
	scorer := NewLengthAdaptiveLoadScorer(&Parameters{
		InFlightLoadProducerName: "custom-inflight",
	}).WithName("decode-lb")

	assert.Equal(t, LengthAdaptiveLoadScorerType, scorer.TypedName().Type)
	assert.Equal(t, "decode-lb", scorer.TypedName().Name)
	assert.Equal(t, fwksched.Distribution, scorer.Category())

	deps := scorer.Consumes()
	require.Len(t, deps.Required, 2)
	assert.Contains(t, deps.Required, tokenizer.TokenizedPromptDataKey)
	expectedLoadKey := attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName("custom-inflight")
	assert.Contains(t, deps.Required, expectedLoadKey)
}

func TestLengthAdaptiveLoadScorer_2PodDeepInferScenario(t *testing.T) {
	ctx := context.Background()
	scorer := NewLengthAdaptiveLoadScorer(&Parameters{
		ShortThresholdTokens: 512,
		LongThresholdTokens:  16384,
		MaxConcurrency:       256,
		MaxTokenCapacity:     150000,
	})

	// Pod 1: 1 giant 100K-token request -> queueScore = 255/256 = 0.99609375, tokenScore = 50K/150K = 0.33333333
	// Pod 2: 200 tiny 100-token requests (20K total) -> queueScore = 56/256 = 0.21875, tokenScore = 130K/150K = 0.86666667
	pod1 := newEndpointWithLoad("pod1", 1, 100000)
	pod2 := newEndpointWithLoad("pod2", 200, 20000)
	endpoints := []fwksched.Endpoint{pod1, pod2}

	t.Run("short request (100 tokens, alpha=0.0) routes to Pod 1 with idle batch slots", func(t *testing.T) {
		req := newTokenizedRequest(100)
		scores := scorer.Score(ctx, req, endpoints)

		assert.InDelta(t, 255.0/256.0, scores[pod1], 0.0001)
		assert.InDelta(t, 56.0/256.0, scores[pod2], 0.0001)
		assert.Greater(t, scores[pod1], scores[pod2], "Pod 1 should win for short request")
	})

	t.Run("long request (32K tokens, alpha=1.0) routes to Pod 2 with low KV token load", func(t *testing.T) {
		req := newTokenizedRequest(32000)
		scores := scorer.Score(ctx, req, endpoints)

		assert.InDelta(t, 50000.0/150000.0, scores[pod1], 0.0001)
		assert.InDelta(t, 130000.0/150000.0, scores[pod2], 0.0001)
		assert.Greater(t, scores[pod2], scores[pod1], "Pod 2 should win for long request")
	})

	t.Run("giant request (100K tokens, alpha=1.0) triggers KV capacity guardrail on Pod 1", func(t *testing.T) {
		req := newTokenizedRequest(100000)
		scores := scorer.Score(ctx, req, endpoints)

		// Pod 1: 100K + 100K = 200K > 150K MaxTokenCapacity -> guarded to 0.0
		// Pod 2: 20K + 100K = 120K <= 150K MaxTokenCapacity -> 130K/150K = 0.8667
		assert.Equal(t, 0.0, scores[pod1], "Pod 1 exceeds KV token capacity and must be guarded to 0.0")
		assert.InDelta(t, 130000.0/150000.0, scores[pod2], 0.0001)
		assert.Greater(t, scores[pod2], scores[pod1])
	})

	t.Run("medium request (8448 tokens, alpha=0.5) blends queueScore and tokenScore equally", func(t *testing.T) {
		// Midpoint between 512 and 16384 is (512 + 16384) / 2 = 8448 -> alpha = 0.5
		req := newTokenizedRequest(8448)
		scores := scorer.Score(ctx, req, endpoints)

		expectedPod1 := 0.5*(255.0/256.0) + 0.5*(50000.0/150000.0)
		expectedPod2 := 0.5*(56.0/256.0) + 0.5*(130000.0/150000.0)
		assert.InDelta(t, expectedPod1, scores[pod1], 0.0001)
		assert.InDelta(t, expectedPod2, scores[pod2], 0.0001)
	})
}

func TestLengthAdaptiveLoadScorer_EdgeCases(t *testing.T) {
	ctx := context.Background()
	scorer := NewLengthAdaptiveLoadScorer(nil)

	t.Run("empty endpoints returns empty map", func(t *testing.T) {
		scores := scorer.Score(ctx, newTokenizedRequest(100), nil)
		assert.Empty(t, scores)
	})

	t.Run("idle pod with 0 requests and 0 tokens scores 1.0 without division by zero", func(t *testing.T) {
		idlePod := newEndpointWithLoad("idle-pod", 0, 0)
		missingAttrPod := fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "no-attr-pod"}},
			&fwkdl.Metrics{},
			nil,
		)
		typedNilPod := fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: "typed-nil-pod"}},
			&fwkdl.Metrics{},
			nil,
		)
		var nilLoad *attrconcurrency.InFlightLoad
		typedNilPod.Put(attrconcurrency.InFlightLoadDataKey, nilLoad)

		endpoints := []fwksched.Endpoint{idlePod, missingAttrPod, typedNilPod}
		for _, tokCount := range []int{0, 100, 8448, 32000} {
			scores := scorer.Score(ctx, newTokenizedRequest(tokCount), endpoints)
			assert.Equal(t, 1.0, scores[idlePod])
			assert.Equal(t, 1.0, scores[missingAttrPod])
			assert.Equal(t, 1.0, scores[typedNilPod])
		}
	})

	t.Run("fallback to RequestSizeBytes when TokenizedRequest is absent", func(t *testing.T) {
		pod1 := newEndpointWithLoad("pod1", 1, 100000)
		pod2 := newEndpointWithLoad("pod2", 200, 20000)
		endpoints := []fwksched.Endpoint{pod1, pod2}

		// 32000 tokens * 4 BytesPerToken = 128000 bytes -> alpha = 1.0
		longRawReq := &fwksched.InferenceRequest{
			RequestID:        "raw-long",
			RequestSizeBytes: 32000 * fwkrequest.BytesPerToken,
		}
		scores := scorer.Score(ctx, longRawReq, endpoints)
		assert.Greater(t, scores[pod2], scores[pod1], "Byte-size fallback should classify 128KB request as long")
	})

	t.Run("all pods exceeding guardrail falls back to rawScores", func(t *testing.T) {
		// Both pods are at MaxConcurrency (256), so (load.Requests + 1) <= 256 is false for both.
		podLessLoaded := newEndpointWithLoad("pod-less", 256, 30000)
		podMoreLoaded := newEndpointWithLoad("pod-more", 256, 120000)
		endpoints := []fwksched.Endpoint{podLessLoaded, podMoreLoaded}

		req := newTokenizedRequest(4096)
		scores := scorer.Score(ctx, req, endpoints)

		assert.Greater(t, scores[podLessLoaded], 0.0, "Fallback rawScores should be non-zero")
		assert.Greater(t, scores[podLessLoaded], scores[podMoreLoaded], "Less-loaded pod must still score higher in fallback")
	})
}

func TestLengthAdaptiveLoadScorer_ScopeInvocation(t *testing.T) {
	ctx := context.Background()
	scorer := NewLengthAdaptiveLoadScorer(nil)
	datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{scorer})

	req := newTokenizedRequest(1024)
	pod1 := newEndpointWithLoad("pod1", 10, 5000)
	pod2 := newEndpointWithLoad("pod2", 20, 10000)

	scopedReq, scopedEndpoints, violations := datalayer.ScopeInvocation(
		logr.Discard(),
		"Score",
		scorer,
		req,
		[]fwksched.Endpoint{pod1, pod2},
	)
	scores := scorer.Score(ctx, scopedReq, scopedEndpoints)
	require.Len(t, scores, 2)
	require.NoError(t, violations.Write())
}

func TestLengthAdaptiveLoadScorer_DecodeLifecycleRetainsTokensUntilEndOfStream(t *testing.T) {
	ctx := context.Background()
	scorer := NewLengthAdaptiveLoadScorer(nil)

	// Simulate inflight-load-producer releasing tokens at StartOfStream (load.Tokens = 0)
	// while req1 (32K tokens) is still actively decoding on pod1.
	pod1 := newEndpointWithLoad("pod1", 1, 0)
	pod2 := newEndpointWithLoad("pod2", 0, 0)

	req1 := newTokenizedRequest(32000)
	req1.RequestID = "req-1"
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{pod1}},
		},
	}

	require.NoError(t, scorer.PreRequest(ctx, req1, result))

	// Even after StartOfStream zeroes InFlightLoad.Tokens on pod1, scorer still tracks 32K KV tokens on pod1.
	scorer.ResponseBody(ctx, req1, &fwkrh.Response{StartOfStream: true}, nil)
	scoresDuringDecode := scorer.Score(ctx, newTokenizedRequest(32000), []fwksched.Endpoint{pod1, pod2})
	assert.InDelta(t, (150000.0-32000.0)/150000.0, scoresDuringDecode[pod1], 0.0001)
	assert.Equal(t, 1.0, scoresDuringDecode[pod2])

	// Once EndOfStream arrives, the 32K KV tokens on pod1 are released.
	scorer.ResponseBody(ctx, req1, &fwkrh.Response{EndOfStream: true}, nil)
	scoresAfterEOS := scorer.Score(ctx, newTokenizedRequest(32000), []fwksched.Endpoint{pod1, pod2})
	assert.Equal(t, 1.0, scoresAfterEOS[pod1])
}

func TestLengthAdaptiveLoadScorer_HeterogeneousPerPodCapacityLabels(t *testing.T) {
	ctx := context.Background()
	scorer := NewLengthAdaptiveLoadScorer(&Parameters{
		ShortThresholdTokens: 512,
		LongThresholdTokens:  16384,
		MaxConcurrency:       256,
		MaxTokenCapacity:     150000,
	})

	// Heterogeneous pool:
	// - podT4 (16 GB): 64 max seqs, 30,000 max KV tokens; currently holding 15,000 tokens (50% KV used)
	// - podA10G (24 GB): 128 max seqs, 60,000 max KV tokens; currently holding 40,000 tokens (66.7% KV used)
	podT4 := newEndpointWithLoadAndLabels("pod-t4", 8, 15000, map[string]string{
		DefaultMaxConcurrencyLabel:   "64",
		DefaultMaxTokenCapacityLabel: "30000",
	})
	podA10G := newEndpointWithLoadAndLabels("pod-a10g", 16, 40000, map[string]string{
		DefaultMaxConcurrencyLabel:   "128",
		DefaultMaxTokenCapacityLabel: "60000",
	})
	endpoints := []fwksched.Endpoint{podT4, podA10G}

	t.Run("long request that fits on both pods prefers pod with higher relative KV headroom", func(t *testing.T) {
		// On podT4: 15K + 12K = 27K <= 30K (fits); on podA10G: 40K + 12K = 52K <= 60K (fits).
		// Both have identical queueScore: podT4 = 1 - 8/64 = 0.875, podA10G = 1 - 16/128 = 0.875.
		// podT4 tokenScore = 1 - 15000/30000 = 0.50; podA10G tokenScore = 1 - 40000/60000 = 0.3333.
		req := newTokenizedRequest(12000)
		scores := scorer.Score(ctx, req, endpoints)
		assert.Greater(t, scores[podT4], scores[podA10G], "podT4 has 50%% free KV vs podA10G's 33.3%% free KV")
	})

	t.Run("18K-token request exceeds podT4's 30K label guardrail and routes to podA10G", func(t *testing.T) {
		// 18K tokens (alpha = 1.0):
		// - podT4: 15K + 18K = 33K > 30K label limit -> guarded to 0.0!
		// - podA10G: 40K + 18K = 58K <= 60K label limit -> tokenScore = (60K - 40K)/60K = 0.3333
		req := newTokenizedRequest(18000)
		scores := scorer.Score(ctx, req, endpoints)
		assert.Equal(t, 0.0, scores[podT4], "podT4 must be guarded to 0.0 because 15K + 18K > 30K")
		assert.InDelta(t, 20000.0/60000.0, scores[podA10G], 0.0001)
	})

	t.Run("invalid or non-positive label values fall back to plugin parameters", func(t *testing.T) {
		badLabelPod := newEndpointWithLoadAndLabels("pod-bad-labels", 1, 75000, map[string]string{
			DefaultMaxConcurrencyLabel:   "not-a-number",
			DefaultMaxTokenCapacityLabel: "-500",
		})
		req := newTokenizedRequest(32000)
		scores := scorer.Score(ctx, req, []fwksched.Endpoint{badLabelPod})
		// Falls back to MaxTokenCapacity = 150000 -> tokenScore = (150000 - 75000)/150000 = 0.5
		assert.InDelta(t, 0.5, scores[badLabelPod], 0.0001)
	})
}

func BenchmarkLengthAdaptiveLoadScorer_Score(b *testing.B) {
	numPods := 8
	endpoints := make([]fwksched.Endpoint, numPods)
	scorer := NewLengthAdaptiveLoadScorer(nil)

	for i := 0; i < numPods; i++ {
		endpoints[i] = newEndpointWithLoad(
			fmt.Sprintf("pod%d", i),
			int64(i*20),
			int64(i*15000),
		)
	}

	req := newTokenizedRequest(4096)
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = scorer.Score(ctx, req, endpoints)
	}
}
