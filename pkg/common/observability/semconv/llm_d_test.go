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
		// EPP Scheduling
		{
			name:     "LLMDEPPProfileName",
			got:      LLMDEPPProfileName("default"),
			wantKey:  "llm_d.epp.scheduling.profile.name",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPFilterDecision",
			got:      LLMDEPPFilterDecision("sticky"),
			wantKey:  "llm_d.epp.filter.decision",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPFilterCandidateEndpoints",
			got:      LLMDEPPFilterCandidateEndpoints(8),
			wantKey:  "llm_d.epp.filter.candidate_endpoints",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPFilterFilteredEndpoints",
			got:      LLMDEPPFilterFilteredEndpoints(5),
			wantKey:  "llm_d.epp.filter.filtered_endpoints",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPFilterStickyEndpoints",
			got:      LLMDEPPFilterStickyEndpoints(3),
			wantKey:  "llm_d.epp.filter.sticky_endpoints",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPFilterAffinityThreshold",
			got:      LLMDEPPFilterAffinityThreshold(0.8),
			wantKey:  "llm_d.epp.filter.affinity_threshold",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDEPPFilterTTFTPenaltyMs",
			got:      LLMDEPPFilterTTFTPenaltyMs(1500),
			wantKey:  "llm_d.epp.filter.ttft_penalty_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDEPPScorerCount",
			got:      LLMDEPPScorerCount(4),
			wantKey:  "llm_d.epp.scorer.count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPScoringCandidateEndpoints",
			got:      LLMDEPPScoringCandidateEndpoints(10),
			wantKey:  "llm_d.epp.scoring.candidate_endpoints",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPPickerCandidateEndpoints",
			got:      LLMDEPPPickerCandidateEndpoints(3),
			wantKey:  "llm_d.epp.picker.candidate_endpoints",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPPickerTopEndpoints",
			got:      LLMDEPPPickerTopEndpoints([]string{"pod-1", "pod-2"}),
			wantKey:  "llm_d.epp.picker.top_endpoints",
			wantType: attribute.STRINGSLICE,
		},
		{
			name:     "LLMDEPPPickerTopScores",
			got:      LLMDEPPPickerTopScores([]float64{1.0, 0.5}),
			wantKey:  "llm_d.epp.picker.top_scores",
			wantType: attribute.FLOAT64SLICE,
		},

		// EPP Scorer
		{
			name:     "LLMDEPPScorerType",
			got:      LLMDEPPScorerType("precise_prefix_cache"),
			wantKey:  "llm_d.epp.scorer.type",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPScorerName",
			got:      LLMDEPPScorerName("scorer-1"),
			wantKey:  "llm_d.epp.scorer.name",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPScorerWeight",
			got:      LLMDEPPScorerWeight(0.5),
			wantKey:  "llm_d.epp.scorer.weight",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDEPPScorerCandidateEndpoints",
			got:      LLMDEPPScorerCandidateEndpoints(5),
			wantKey:  "llm_d.epp.scorer.candidate_endpoints",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPScorerScoreMax",
			got:      LLMDEPPScorerScoreMax(95.5),
			wantKey:  "llm_d.epp.scorer.score.max",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDEPPScorerScoreAvg",
			got:      LLMDEPPScorerScoreAvg(70.2),
			wantKey:  "llm_d.epp.scorer.score.avg",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDEPPScorerEndpointsScored",
			got:      LLMDEPPScorerEndpointsScored(5),
			wantKey:  "llm_d.epp.scorer.endpoints_scored",
			wantType: attribute.INT64,
		},

		// EPP Profile Handler
		{
			name:     "LLMDEPPProfileHandlerDecision",
			got:      LLMDEPPProfileHandlerDecision("run_decode"),
			wantKey:  "llm_d.epp.profile_handler.decision",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPProfileHandlerSelectedProfile",
			got:      LLMDEPPProfileHandlerSelectedProfile("prefill"),
			wantKey:  "llm_d.epp.profile_handler.selected_profile",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPProfileHandlerTotalProfiles",
			got:      LLMDEPPProfileHandlerTotalProfiles(3),
			wantKey:  "llm_d.epp.profile_handler.total_profiles",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPProfileHandlerExecutedProfiles",
			got:      LLMDEPPProfileHandlerExecutedProfiles(2),
			wantKey:  "llm_d.epp.profile_handler.executed_profiles",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPProfileHandlerDecodeFailed",
			got:      LLMDEPPProfileHandlerDecodeFailed(true),
			wantKey:  "llm_d.epp.profile_handler.decode_failed",
			wantType: attribute.BOOL,
		},

		// EPP Disagg
		{
			name:     "LLMDEPPDisaggReason",
			got:      LLMDEPPDisaggReason("prefix_cache"),
			wantKey:  "llm_d.epp.disagg.reason",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPPDReason",
			got:      LLMDEPPPDReason("no_prefill_profile_result"),
			wantKey:  "llm_d.epp.pd.reason",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPPDDisaggregationUsed",
			got:      LLMDEPPPDDisaggregationUsed(true),
			wantKey:  "llm_d.epp.pd.disaggregation_used",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDEPPPDPrefillPodAddress",
			got:      LLMDEPPPDPrefillPodAddress("10.0.0.1"),
			wantKey:  "llm_d.epp.pd.prefill_pod_address",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPPDPrefillPodPort",
			got:      LLMDEPPPDPrefillPodPort("8080"),
			wantKey:  "llm_d.epp.pd.prefill_pod_port",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPEncodeDisaggregationUsed",
			got:      LLMDEPPEncodeDisaggregationUsed(true),
			wantKey:  "llm_d.epp.encode.disaggregation_used",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDEPPEncodeReason",
			got:      LLMDEPPEncodeReason("no_encode_profile_result"),
			wantKey:  "llm_d.epp.encode.reason",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPEncodeEndpoints",
			got:      LLMDEPPEncodeEndpoints("10.0.0.2:8000"),
			wantKey:  "llm_d.epp.encode.endpoints",
			wantType: attribute.STRING,
		},

		// EPP Producer
		{
			name:     "LLMDEPPProducerCandidateEndpoints",
			got:      LLMDEPPProducerCandidateEndpoints(4),
			wantKey:  "llm_d.epp.producer.candidate_endpoints",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPProducerResult",
			got:      LLMDEPPProducerResult("success"),
			wantKey:  "llm_d.epp.producer.result",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPProducerMaxMatchBlocks",
			got:      LLMDEPPProducerMaxMatchBlocks(8),
			wantKey:  "llm_d.epp.producer.max_match_blocks",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDEPPProducerTotalBlocks",
			got:      LLMDEPPProducerTotalBlocks(16),
			wantKey:  "llm_d.epp.producer.total_blocks",
			wantType: attribute.INT64,
		},

		// EPP Token Producer
		{
			name:     "LLMDEPPTokenProducerBackend",
			got:      LLMDEPPTokenProducerBackend("huggingface"),
			wantKey:  "llm_d.epp.token_producer.backend",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPTokenProducerResult",
			got:      LLMDEPPTokenProducerResult("success"),
			wantKey:  "llm_d.epp.token_producer.result",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDEPPTokenProducerTokenCount",
			got:      LLMDEPPTokenProducerTokenCount(128),
			wantKey:  "llm_d.epp.token_producer.token_count",
			wantType: attribute.INT64,
		},

		// KV Cache
		{
			name:     "LLMDKVCachePodCount",
			got:      LLMDKVCachePodCount(5),
			wantKey:  "llm_d.kv_cache.pod_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheTokenCount",
			got:      LLMDKVCacheTokenCount(512),
			wantKey:  "llm_d.kv_cache.token_count",
			wantType: attribute.INT64,
		},
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
			name:     "LLMDKVCacheBlocksFound",
			got:      LLMDKVCacheBlocksFound(12),
			wantKey:  "llm_d.kv_cache.blocks_found",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheIndexWalkKeyCount",
			got:      LLMDKVCacheIndexWalkKeyCount(8),
			wantKey:  "llm_d.kv_cache.index.walk.key_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheIndexWalkKeysPresent",
			got:      LLMDKVCacheIndexWalkKeysPresent(6),
			wantKey:  "llm_d.kv_cache.index.walk.keys_present",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheIndexAddEngineKeyCount",
			got:      LLMDKVCacheIndexAddEngineKeyCount(4),
			wantKey:  "llm_d.kv_cache.index.add.engine_key_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheIndexAddRequestKeyCount",
			got:      LLMDKVCacheIndexAddRequestKeyCount(4),
			wantKey:  "llm_d.kv_cache.index.add.request_key_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheIndexAddPodEntryCount",
			got:      LLMDKVCacheIndexAddPodEntryCount(2),
			wantKey:  "llm_d.kv_cache.index.add.pod_entry_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheIndexAddDeviceTierCount",
			got:      LLMDKVCacheIndexAddDeviceTierCount(1),
			wantKey:  "llm_d.kv_cache.index.add.device_tier_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheIndexEvictKeyType",
			got:      LLMDKVCacheIndexEvictKeyType("engine"),
			wantKey:  "llm_d.kv_cache.index.evict.key_type",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDKVCacheIndexEvictPodEntryCount",
			got:      LLMDKVCacheIndexEvictPodEntryCount(1),
			wantKey:  "llm_d.kv_cache.index.evict.pod_entry_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheIndexEvictDeviceTierCount",
			got:      LLMDKVCacheIndexEvictDeviceTierCount(1),
			wantKey:  "llm_d.kv_cache.index.evict.device_tier_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheIndexLookupBlockCount",
			got:      LLMDKVCacheIndexLookupBlockCount(16),
			wantKey:  "llm_d.kv_cache.index.lookup.block_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheLookupPodFilterCount",
			got:      LLMDKVCacheLookupPodFilterCount(2),
			wantKey:  "llm_d.kv_cache.lookup.pod_filter_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheLookupCacheHit",
			got:      LLMDKVCacheLookupCacheHit(true),
			wantKey:  "llm_d.kv_cache.lookup.cache_hit",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDKVCacheLookupBlocksFound",
			got:      LLMDKVCacheLookupBlocksFound(8),
			wantKey:  "llm_d.kv_cache.lookup.blocks_found",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCachePrefixMatchKeyCount",
			got:      LLMDKVCachePrefixMatchKeyCount(10),
			wantKey:  "llm_d.kv_cache.prefix_match.key_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCachePrefixMatchPodFilterCount",
			got:      LLMDKVCachePrefixMatchPodFilterCount(3),
			wantKey:  "llm_d.kv_cache.prefix_match.pod_filter_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCachePrefixMatchWalked",
			got:      LLMDKVCachePrefixMatchWalked(true),
			wantKey:  "llm_d.kv_cache.prefix_match.walked",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDKVCachePrefixMatchPodsMatched",
			got:      LLMDKVCachePrefixMatchPodsMatched(2),
			wantKey:  "llm_d.kv_cache.prefix_match.pods_matched",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCachePrefixMatchLongestChain",
			got:      LLMDKVCachePrefixMatchLongestChain(5),
			wantKey:  "llm_d.kv_cache.prefix_match.longest_chain",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheScorerAlgorithm",
			got:      LLMDKVCacheScorerAlgorithm("prefix_match"),
			wantKey:  "llm_d.kv_cache.scorer.algorithm",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDKVCacheScorerKeyCount",
			got:      LLMDKVCacheScorerKeyCount(10),
			wantKey:  "llm_d.kv_cache.scorer.key_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheScoreMax",
			got:      LLMDKVCacheScoreMax(1.0),
			wantKey:  "llm_d.kv_cache.score.max",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDKVCacheScoreAvg",
			got:      LLMDKVCacheScoreAvg(0.5),
			wantKey:  "llm_d.kv_cache.score.avg",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDKVCacheScorerPodsScored",
			got:      LLMDKVCacheScorerPodsScored(4),
			wantKey:  "llm_d.kv_cache.scorer.pods_scored",
			wantType: attribute.INT64,
		},

		// KV Cache Events
		{
			name:     "LLMDKVCacheEventsTopic",
			got:      LLMDKVCacheEventsTopic("events/v1"),
			wantKey:  "llm_d.kv_cache.events.topic",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDKVCacheEventsSequence",
			got:      LLMDKVCacheEventsSequence(12345),
			wantKey:  "llm_d.kv_cache.events.sequence",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheEventsPayloadSizeBytes",
			got:      LLMDKVCacheEventsPayloadSizeBytes(1024),
			wantKey:  "llm_d.kv_cache.events.payload_size_bytes",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDKVCacheEventsSourceEndpoint",
			got:      LLMDKVCacheEventsSourceEndpoint("tcp://10.0.0.1:5555"),
			wantKey:  "llm_d.kv_cache.events.source_endpoint",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDKVCacheEventsPodID",
			got:      LLMDKVCacheEventsPodID("pod-abc"),
			wantKey:  "llm_d.kv_cache.events.pod_id",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDKVCacheEventsEventCount",
			got:      LLMDKVCacheEventsEventCount(10),
			wantKey:  "llm_d.kv_cache.events.event_count",
			wantType: attribute.INT64,
		},

		// Sidecar / Proxy
		{
			name:     "LLMDPDProxyConnector",
			got:      LLMDPDProxyConnector("nixlv2"),
			wantKey:  "llm_d.pd_proxy.connector",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyKVConnector",
			got:      LLMDPDProxyKVConnector("nixlv2"),
			wantKey:  "llm_d.pd_proxy.kv_connector",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyECConnector",
			got:      LLMDPDProxyECConnector("ec"),
			wantKey:  "llm_d.pd_proxy.ec_connector",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyRequestID",
			got:      LLMDPDProxyRequestID("req-123"),
			wantKey:  "llm_d.pd_proxy.request_id",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyRequestPath",
			got:      LLMDPDProxyRequestPath("/v1/chat/completions"),
			wantKey:  "llm_d.pd_proxy.request_path",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyPrefillTarget",
			got:      LLMDPDProxyPrefillTarget("10.0.0.1:8000"),
			wantKey:  "llm_d.pd_proxy.prefill_target",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyPrefillCandidates",
			got:      LLMDPDProxyPrefillCandidates(3),
			wantKey:  "llm_d.pd_proxy.prefill_candidates",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDPDProxyDecodeTarget",
			got:      LLMDPDProxyDecodeTarget("10.0.0.2:8000"),
			wantKey:  "llm_d.pd_proxy.decode.target",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyReason",
			got:      LLMDPDProxyReason("no_prefill_header"),
			wantKey:  "llm_d.pd_proxy.reason",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyError",
			got:      LLMDPDProxyError("ssrf_protection_denied"),
			wantKey:  "llm_d.pd_proxy.error",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyDeniedTarget",
			got:      LLMDPDProxyDeniedTarget("10.0.0.3:8000"),
			wantKey:  "llm_d.pd_proxy.denied_target",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyKVCacheSource",
			got:      LLMDPDProxyKVCacheSource("remote"),
			wantKey:  "llm_d.pd_proxy.kv_cache_source",
			wantType: attribute.STRING,
		},
		{
			name:     "LLMDPDProxyDisaggregationUsed",
			got:      LLMDPDProxyDisaggregationUsed(true),
			wantKey:  "llm_d.pd_proxy.disaggregation_used",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDPDProxyConcurrentPD",
			got:      LLMDPDProxyConcurrentPD(true),
			wantKey:  "llm_d.pd_proxy.concurrent_pd",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDPDProxyParallelDispatch",
			got:      LLMDPDProxyParallelDispatch(true),
			wantKey:  "llm_d.pd_proxy.parallel_dispatch",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDPDProxyParallelWindowMs",
			got:      LLMDPDProxyParallelWindowMs(15.5),
			wantKey:  "llm_d.pd_proxy.parallel_window_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDPDProxyTotalDurationMs",
			got:      LLMDPDProxyTotalDurationMs(120.5),
			wantKey:  "llm_d.pd_proxy.total_duration_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDPDProxyTrueTTFTMs",
			got:      LLMDPDProxyTrueTTFTMs(45.2),
			wantKey:  "llm_d.pd_proxy.true_ttft_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDPDProxyPrefillDurationMsSummary",
			got:      LLMDPDProxyPrefillDurationMsSummary(30.1),
			wantKey:  "llm_d.pd_proxy.prefill_duration_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDPDProxyDecodeDurationMsSummary",
			got:      LLMDPDProxyDecodeDurationMsSummary(85.4),
			wantKey:  "llm_d.pd_proxy.decode_duration_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDPDProxyCoordinatorOverheadMs",
			got:      LLMDPDProxyCoordinatorOverheadMs(5.0),
			wantKey:  "llm_d.pd_proxy.coordinator_overhead_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDPDProxyPrefillAsync",
			got:      LLMDPDProxyPrefillAsync(true),
			wantKey:  "llm_d.pd_proxy.prefill.async",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDPDProxyPrefillStatusCode",
			got:      LLMDPDProxyPrefillStatusCode(200),
			wantKey:  "llm_d.pd_proxy.prefill.status_code",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDPDProxyPrefillDurationMs",
			got:      LLMDPDProxyPrefillDurationMs(28.3),
			wantKey:  "llm_d.pd_proxy.prefill.duration_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDPDProxyDecodeConcurrentWithPrefill",
			got:      LLMDPDProxyDecodeConcurrentWithPrefill(true),
			wantKey:  "llm_d.pd_proxy.decode.concurrent_with_prefill",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDPDProxyDecodeDataParallel",
			got:      LLMDPDProxyDecodeDataParallel(false),
			wantKey:  "llm_d.pd_proxy.decode.data_parallel",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDPDProxyDecodeStreaming",
			got:      LLMDPDProxyDecodeStreaming(true),
			wantKey:  "llm_d.pd_proxy.decode.streaming",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDPDProxyDecodeDurationMs",
			got:      LLMDPDProxyDecodeDurationMs(80.0),
			wantKey:  "llm_d.pd_proxy.decode.duration_ms",
			wantType: attribute.FLOAT64,
		},
		{
			name:     "LLMDPDProxyChunkedDecodeChunkSize",
			got:      LLMDPDProxyChunkedDecodeChunkSize(64),
			wantKey:  "llm_d.pd_proxy.chunked_decode.chunk_size",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDPDProxyChunkedDecodeStreaming",
			got:      LLMDPDProxyChunkedDecodeStreaming(true),
			wantKey:  "llm_d.pd_proxy.chunked_decode.streaming",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDPDProxyChunkedDecodeChunks",
			got:      LLMDPDProxyChunkedDecodeChunks(4),
			wantKey:  "llm_d.pd_proxy.chunked_decode.chunks",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDPDProxyChunkedDecodeTotalTokens",
			got:      LLMDPDProxyChunkedDecodeTotalTokens(256),
			wantKey:  "llm_d.pd_proxy.chunked_decode.total_tokens",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDPDProxyChunkedDecodeDurationMs",
			got:      LLMDPDProxyChunkedDecodeDurationMs(35.0),
			wantKey:  "llm_d.pd_proxy.chunked_decode.duration_ms",
			wantType: attribute.FLOAT64,
		},

		// EC Proxy
		{
			name:     "LLMDECProxyEncodeDisaggregationUsed",
			got:      LLMDECProxyEncodeDisaggregationUsed(true),
			wantKey:  "llm_d.ec_proxy.encode_disaggregation_used",
			wantType: attribute.BOOL,
		},
		{
			name:     "LLMDECProxyEncoderCount",
			got:      LLMDECProxyEncoderCount(3),
			wantKey:  "llm_d.ec_proxy.encoder_count",
			wantType: attribute.INT64,
		},
		{
			name:     "LLMDECProxyEncoderCandidates",
			got:      LLMDECProxyEncoderCandidates(4),
			wantKey:  "llm_d.ec_proxy.encoder_candidates",
			wantType: attribute.INT64,
		},

		// OpenAI API
		{
			name:     "LLMDOpenAIAPI",
			got:      LLMDOpenAIAPI("chat_completions"),
			wantKey:  "llm_d.openai.api",
			wantType: attribute.STRING,
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
