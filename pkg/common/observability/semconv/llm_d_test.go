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

package semconv

import (
	"testing"

	"go.opentelemetry.io/otel/attribute"
)

func TestLLMDSemanticConventions(t *testing.T) {
	tests := []struct {
		name     string
		got      attribute.KeyValue
		wantKey  string
		wantType attribute.Type
	}{
		// EPP Scheduling & Scoring
		{
			name:     "LLMDEPPProfileName",
			got:      LLMDEPPProfileName("default"),
			wantKey:  "llm_d.epp.scheduling.profile.name",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPFilterCandidateEndpoints",
			got:      LLMDEPPFilterCandidateEndpoints(5),
			wantKey:  "llm_d.epp.filter.candidate_endpoints",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPPickerTopScores",
			got:      LLMDEPPPickerTopScores([]float64{1.0, 0.8}),
			wantKey:  "llm_d.epp.picker.top_scores",
			wantType: attribute.FLOAT64SLICE,
		},
		{
			name:     "LLMDEPPScorerType",
			got:      LLMDEPPScorerType("precise_prefix_cache"),
			wantKey:  "llm_d.epp.scorer.type",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPScorerScoreMax",
			got:      LLMDEPPScorerScoreMax(95.5),
			wantKey:  "llm_d.epp.scorer.score.max",
			wantType: attribute.FLOAT64,
		},
		// EPP Profile Handler & Disaggregation
		{
			name:     "LLMDEPPProfileHandlerDecision",
			got:      LLMDEPPProfileHandlerDecision("run_decode"),
			wantKey:  "llm_d.epp.profile_handler.decision",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPPDDisaggregationUsed",
			got:      LLMDEPPPDDisaggregationUsed(true),
			wantKey:  "llm_d.epp.pd.disaggregation_used",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDEPPDisaggReason",
			got:      LLMDEPPDisaggReason("prefix_cache"),
			wantKey:  "llm_d.epp.disagg.reason",
			wantType: attribute.STRING,
		},
		// EPP Producer
		{
			name:     "LLMDEPPProducerMaxMatchBlocks",
			got:      LLMDEPPProducerMaxMatchBlocks(8),
			wantKey:  "llm_d.epp.producer.max_match_blocks",
			wantType: attribute.INT64,
		},
		// KV Cache
		{
			name:     "LLMDKVCacheBlockKeysCount",
			got:      LLMDKVCacheBlockKeysCount(16),
			wantKey:  "llm_d.kv_cache.block_keys.count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheBlockHitRatio",
			got:      LLMDKVCacheBlockHitRatio(0.75),
			wantKey:  "llm_d.kv_cache.block_hit_ratio",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDKVCacheLookupCacheHit",
			got:      LLMDKVCacheLookupCacheHit(true),
			wantKey:  "llm_d.kv_cache.lookup.cache_hit",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDKVCacheScorerAlgorithm",
			got:      LLMDKVCacheScorerAlgorithm("lru"),
			wantKey:  "llm_d.kv_cache.scorer.algorithm",
			wantType: attribute.STRING,
		},
		// Sidecar / PD Proxy
		{
			name:     "LLMDPDProxyConnector",
			got:      LLMDPDProxyConnector("nixlv2"),
			wantKey:  "llm_d.pd_proxy.connector",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyPrefillTarget",
			got:      LLMDPDProxyPrefillTarget("10.0.0.1:8080"),
			wantKey:  "llm_d.pd_proxy.prefill_target",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyTotalDurationMs",
			got:      LLMDPDProxyTotalDurationMs(123.45),
			wantKey:  "llm_d.pd_proxy.total_duration_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDPDProxyDecodeStreaming",
			got:      LLMDPDProxyDecodeStreaming(true),
			wantKey:  "llm_d.pd_proxy.decode.streaming",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDPDProxyChunkedDecodeChunks",
			got:      LLMDPDProxyChunkedDecodeChunks(4),
			wantKey:  "llm_d.pd_proxy.chunked_decode.chunks",
			wantType: attribute.INT64,
		},
		// EC Proxy
		{
			name:     "LLMDECProxyEncodeDisaggregationUsed",
			got:      LLMDECProxyEncodeDisaggregationUsed(true),
			wantKey:  "llm_d.ec_proxy.encode_disaggregation_used",
			wantType: attribute.BOOL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.got.Key) != tt.wantKey {
				t.Errorf("got key %q, want %q", tt.got.Key, tt.wantKey)
			}
			if tt.got.Value.Type() != tt.wantType {
				t.Errorf("got type %v, want %v", tt.got.Value.Type(), tt.wantType)
			}
		})
	}
}
