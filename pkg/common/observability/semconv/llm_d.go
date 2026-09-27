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
	"go.opentelemetry.io/otel/attribute"
)

// Internal llm-d specific attribute keys.
// All custom router, scheduler, scorer, and sidecar attributes are namespaced under "llm_d.*".
const (
	// EPP Scheduling attributes
	LLMDEPPProfileNameKey               = attribute.Key("llm_d.epp.scheduling.profile.name")
	LLMDEPPFilterDecisionKey            = attribute.Key("llm_d.epp.filter.decision")
	LLMDEPPFilterCandidateEndpointsKey  = attribute.Key("llm_d.epp.filter.candidate_endpoints")
	LLMDEPPFilterFilteredEndpointsKey   = attribute.Key("llm_d.epp.filter.filtered_endpoints")
	LLMDEPPFilterStickyEndpointsKey     = attribute.Key("llm_d.epp.filter.sticky_endpoints")
	LLMDEPPFilterAffinityThresholdKey   = attribute.Key("llm_d.epp.filter.affinity_threshold")
	LLMDEPPFilterTTFTPenaltyMsKey       = attribute.Key("llm_d.epp.filter.ttft_penalty_ms")
	LLMDEPPScorerCountKey               = attribute.Key("llm_d.epp.scorer.count")
	LLMDEPPScoringCandidateEndpointsKey = attribute.Key("llm_d.epp.scoring.candidate_endpoints")
	LLMDEPPPickerCandidateEndpointsKey  = attribute.Key("llm_d.epp.picker.candidate_endpoints")
	LLMDEPPPickerTopEndpointsKey        = attribute.Key("llm_d.epp.picker.top_endpoints")
	LLMDEPPPickerTopScoresKey           = attribute.Key("llm_d.epp.picker.top_scores")

	// EPP Scorer attributes
	LLMDEPPScorerTypeKey               = attribute.Key("llm_d.epp.scorer.type")
	LLMDEPPScorerNameKey               = attribute.Key("llm_d.epp.scorer.name")
	LLMDEPPScorerWeightKey             = attribute.Key("llm_d.epp.scorer.weight")
	LLMDEPPScorerCandidateEndpointsKey = attribute.Key("llm_d.epp.scorer.candidate_endpoints")
	LLMDEPPScorerScoreMaxKey           = attribute.Key("llm_d.epp.scorer.score.max")
	LLMDEPPScorerScoreAvgKey           = attribute.Key("llm_d.epp.scorer.score.avg")
	LLMDEPPScorerEndpointsScoredKey    = attribute.Key("llm_d.epp.scorer.endpoints_scored")

	// EPP Profile Handler & Disaggregation attributes
	LLMDEPPDisaggReasonKey                   = attribute.Key("llm_d.epp.disagg.reason")
	LLMDEPPPDReasonKey                       = attribute.Key("llm_d.epp.pd.reason")
	LLMDEPPPDDisaggregationUsedKey           = attribute.Key("llm_d.epp.pd.disaggregation_used")
	LLMDEPPPDPrefillPodAddressKey            = attribute.Key("llm_d.epp.pd.prefill_pod_address")
	LLMDEPPPDPrefillPodPortKey               = attribute.Key("llm_d.epp.pd.prefill_pod_port")
	LLMDEPPEncodeDisaggregationUsedKey       = attribute.Key("llm_d.epp.encode.disaggregation_used")
	LLMDEPPEncodeReasonKey                   = attribute.Key("llm_d.epp.encode.reason")
	LLMDEPPEncodeEndpointsKey                = attribute.Key("llm_d.epp.encode.endpoints")
	LLMDEPPProfileHandlerDecisionKey         = attribute.Key("llm_d.epp.profile_handler.decision")
	LLMDEPPProfileHandlerSelectedProfileKey  = attribute.Key("llm_d.epp.profile_handler.selected_profile")
	LLMDEPPProfileHandlerTotalProfilesKey    = attribute.Key("llm_d.epp.profile_handler.total_profiles")
	LLMDEPPProfileHandlerExecutedProfilesKey = attribute.Key("llm_d.epp.profile_handler.executed_profiles")
	LLMDEPPProfileHandlerDecodeFailedKey     = attribute.Key("llm_d.epp.profile_handler.decode_failed")

	// EPP Producer attributes
	LLMDEPPProducerCandidateEndpointsKey = attribute.Key("llm_d.epp.producer.candidate_endpoints")
	LLMDEPPProducerResultKey             = attribute.Key("llm_d.epp.producer.result")
	LLMDEPPProducerMaxMatchBlocksKey     = attribute.Key("llm_d.epp.producer.max_match_blocks")
	LLMDEPPProducerTotalBlocksKey        = attribute.Key("llm_d.epp.producer.total_blocks")

	// EPP Token Producer attributes
	LLMDEPPTokenProducerBackendKey    = attribute.Key("llm_d.epp.token_producer.backend")
	LLMDEPPTokenProducerResultKey     = attribute.Key("llm_d.epp.token_producer.result")
	LLMDEPPTokenProducerTokenCountKey = attribute.Key("llm_d.epp.token_producer.token_count")

	// KV Cache attributes
	LLMDKVCachePodCountKey                  = attribute.Key("llm_d.kv_cache.pod_count")
	LLMDKVCacheTokenCountKey                = attribute.Key("llm_d.kv_cache.token_count")
	LLMDKVCacheBlockKeysCountKey            = attribute.Key("llm_d.kv_cache.block_keys.count")
	LLMDKVCacheBlockHitRatioKey             = attribute.Key("llm_d.kv_cache.block_hit_ratio")
	LLMDKVCacheBlocksFoundKey               = attribute.Key("llm_d.kv_cache.blocks_found")
	LLMDKVCacheIndexWalkKeyCountKey         = attribute.Key("llm_d.kv_cache.index.walk.key_count")
	LLMDKVCacheIndexWalkKeysPresentKey      = attribute.Key("llm_d.kv_cache.index.walk.keys_present")
	LLMDKVCacheIndexAddEngineKeyCountKey    = attribute.Key("llm_d.kv_cache.index.add.engine_key_count")
	LLMDKVCacheIndexAddRequestKeyCountKey   = attribute.Key("llm_d.kv_cache.index.add.request_key_count")
	LLMDKVCacheIndexAddPodEntryCountKey     = attribute.Key("llm_d.kv_cache.index.add.pod_entry_count")
	LLMDKVCacheIndexAddDeviceTierCountKey   = attribute.Key("llm_d.kv_cache.index.add.device_tier_count")
	LLMDKVCacheIndexEvictKeyTypeKey         = attribute.Key("llm_d.kv_cache.index.evict.key_type")
	LLMDKVCacheIndexEvictPodEntryCountKey   = attribute.Key("llm_d.kv_cache.index.evict.pod_entry_count")
	LLMDKVCacheIndexEvictDeviceTierCountKey = attribute.Key("llm_d.kv_cache.index.evict.device_tier_count")
	LLMDKVCacheIndexLookupBlockCountKey     = attribute.Key("llm_d.kv_cache.index.lookup.block_count")
	LLMDKVCacheLookupPodFilterCountKey      = attribute.Key("llm_d.kv_cache.lookup.pod_filter_count")
	LLMDKVCacheLookupCacheHitKey            = attribute.Key("llm_d.kv_cache.lookup.cache_hit")
	LLMDKVCacheLookupBlocksFoundKey         = attribute.Key("llm_d.kv_cache.lookup.blocks_found")
	LLMDKVCachePrefixMatchKeyCountKey       = attribute.Key("llm_d.kv_cache.prefix_match.key_count")
	LLMDKVCachePrefixMatchPodFilterCountKey = attribute.Key("llm_d.kv_cache.prefix_match.pod_filter_count")
	LLMDKVCachePrefixMatchWalkedKey         = attribute.Key("llm_d.kv_cache.prefix_match.walked")
	LLMDKVCachePrefixMatchPodsMatchedKey    = attribute.Key("llm_d.kv_cache.prefix_match.pods_matched")
	LLMDKVCachePrefixMatchLongestChainKey   = attribute.Key("llm_d.kv_cache.prefix_match.longest_chain")
	LLMDKVCacheScorerAlgorithmKey           = attribute.Key("llm_d.kv_cache.scorer.algorithm")
	LLMDKVCacheScorerKeyCountKey            = attribute.Key("llm_d.kv_cache.scorer.key_count")
	LLMDKVCacheScoreMaxKey                  = attribute.Key("llm_d.kv_cache.score.max")
	LLMDKVCacheScoreAvgKey                  = attribute.Key("llm_d.kv_cache.score.avg")
	LLMDKVCacheScorerPodsScoredKey          = attribute.Key("llm_d.kv_cache.scorer.pods_scored")

	// KV Cache Event attributes
	LLMDKVCacheEventsTopicKey            = attribute.Key("llm_d.kv_cache.events.topic")
	LLMDKVCacheEventsSequenceKey         = attribute.Key("llm_d.kv_cache.events.sequence")
	LLMDKVCacheEventsPayloadSizeBytesKey = attribute.Key("llm_d.kv_cache.events.payload_size_bytes")
	LLMDKVCacheEventsSourceEndpointKey   = attribute.Key("llm_d.kv_cache.events.source_endpoint")
	LLMDKVCacheEventsPodIDKey            = attribute.Key("llm_d.kv_cache.events.pod_id")
	LLMDKVCacheEventsEventCountKey       = attribute.Key("llm_d.kv_cache.events.event_count")

	// Sidecar / Proxy attributes
	LLMDPDProxyConnectorKey                   = attribute.Key("llm_d.pd_proxy.connector")
	LLMDPDProxyKVConnectorKey                 = attribute.Key("llm_d.pd_proxy.kv_connector")
	LLMDPDProxyECConnectorKey                 = attribute.Key("llm_d.pd_proxy.ec_connector")
	LLMDPDProxyRequestIDKey                   = attribute.Key("llm_d.pd_proxy.request_id")
	LLMDPDProxyRequestPathKey                 = attribute.Key("llm_d.pd_proxy.request_path")
	LLMDPDProxyPrefillTargetKey               = attribute.Key("llm_d.pd_proxy.prefill_target")
	LLMDPDProxyPrefillCandidatesKey           = attribute.Key("llm_d.pd_proxy.prefill_candidates")
	LLMDPDProxyDecodeTargetKey                = attribute.Key("llm_d.pd_proxy.decode.target")
	LLMDPDProxyReasonKey                      = attribute.Key("llm_d.pd_proxy.reason")
	LLMDPDProxyErrorKey                       = attribute.Key("llm_d.pd_proxy.error")
	LLMDPDProxyDeniedTargetKey                = attribute.Key("llm_d.pd_proxy.denied_target")
	LLMDPDProxyKVCacheSourceKey               = attribute.Key("llm_d.pd_proxy.kv_cache_source")
	LLMDPDProxyDisaggregationUsedKey          = attribute.Key("llm_d.pd_proxy.disaggregation_used")
	LLMDPDProxyConcurrentPDKey                = attribute.Key("llm_d.pd_proxy.concurrent_pd")
	LLMDPDProxyParallelDispatchKey            = attribute.Key("llm_d.pd_proxy.parallel_dispatch")
	LLMDPDProxyParallelWindowMsKey            = attribute.Key("llm_d.pd_proxy.parallel_window_ms")
	LLMDPDProxyTotalDurationMsKey             = attribute.Key("llm_d.pd_proxy.total_duration_ms")
	LLMDPDProxyTrueTTFTMsKey                  = attribute.Key("llm_d.pd_proxy.true_ttft_ms")
	LLMDPDProxyPrefillDurationMsSummaryKey    = attribute.Key("llm_d.pd_proxy.prefill_duration_ms")
	LLMDPDProxyDecodeDurationMsSummaryKey     = attribute.Key("llm_d.pd_proxy.decode_duration_ms")
	LLMDPDProxyCoordinatorOverheadMsKey       = attribute.Key("llm_d.pd_proxy.coordinator_overhead_ms")
	LLMDPDProxyPrefillAsyncKey                = attribute.Key("llm_d.pd_proxy.prefill.async")
	LLMDPDProxyPrefillStatusCodeKey           = attribute.Key("llm_d.pd_proxy.prefill.status_code")
	LLMDPDProxyPrefillDurationMsKey           = attribute.Key("llm_d.pd_proxy.prefill.duration_ms")
	LLMDPDProxyDecodeConcurrentWithPrefillKey = attribute.Key("llm_d.pd_proxy.decode.concurrent_with_prefill")
	LLMDPDProxyDecodeDataParallelKey          = attribute.Key("llm_d.pd_proxy.decode.data_parallel")
	LLMDPDProxyDecodeStreamingKey             = attribute.Key("llm_d.pd_proxy.decode.streaming")
	LLMDPDProxyDecodeDurationMsKey            = attribute.Key("llm_d.pd_proxy.decode.duration_ms")
	LLMDPDProxyChunkedDecodeChunkSizeKey      = attribute.Key("llm_d.pd_proxy.chunked_decode.chunk_size")
	LLMDPDProxyChunkedDecodeStreamingKey      = attribute.Key("llm_d.pd_proxy.chunked_decode.streaming")
	LLMDPDProxyChunkedDecodeChunksKey         = attribute.Key("llm_d.pd_proxy.chunked_decode.chunks")
	LLMDPDProxyChunkedDecodeTotalTokensKey    = attribute.Key("llm_d.pd_proxy.chunked_decode.total_tokens")
	LLMDPDProxyChunkedDecodeDurationMsKey     = attribute.Key("llm_d.pd_proxy.chunked_decode.duration_ms")

	// EC Proxy attributes
	LLMDECProxyEncodeDisaggregationUsedKey = attribute.Key("llm_d.ec_proxy.encode_disaggregation_used")
	LLMDECProxyEncoderCountKey             = attribute.Key("llm_d.ec_proxy.encoder_count")
	LLMDECProxyEncoderCandidatesKey        = attribute.Key("llm_d.ec_proxy.encoder_candidates")

	// OpenAI API attributes
	LLMDOpenAIAPIKey = attribute.Key("llm_d.openai.api")
)

// Typed helper functions for llm-d internal attributes.

// EPP Scheduling helpers

// LLMDEPPProfileName returns an attribute for the scheduled profile name.
func LLMDEPPProfileName(name string) attribute.KeyValue {
	return LLMDEPPProfileNameKey.String(name)
}

// LLMDEPPFilterDecision returns an attribute for the outcome a filter decided on.
func LLMDEPPFilterDecision(decision string) attribute.KeyValue {
	return LLMDEPPFilterDecisionKey.String(decision)
}

// LLMDEPPFilterCandidateEndpoints returns an attribute for the number of endpoints a filter received.
func LLMDEPPFilterCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPFilterCandidateEndpointsKey.Int(count)
}

// LLMDEPPFilterFilteredEndpoints returns an attribute for the number of endpoints a filter returned.
func LLMDEPPFilterFilteredEndpoints(count int) attribute.KeyValue {
	return LLMDEPPFilterFilteredEndpointsKey.Int(count)
}

// LLMDEPPFilterStickyEndpoints returns an attribute for the number of endpoints meeting the affinity threshold.
func LLMDEPPFilterStickyEndpoints(count int) attribute.KeyValue {
	return LLMDEPPFilterStickyEndpointsKey.Int(count)
}

// LLMDEPPFilterAffinityThreshold returns an attribute for the configured prefix cache affinity threshold.
func LLMDEPPFilterAffinityThreshold(threshold float64) attribute.KeyValue {
	return LLMDEPPFilterAffinityThresholdKey.Float64(threshold)
}

// LLMDEPPFilterTTFTPenaltyMs returns an attribute for the TTFT penalty the load gate measured, in milliseconds.
func LLMDEPPFilterTTFTPenaltyMs(penalty float64) attribute.KeyValue {
	return LLMDEPPFilterTTFTPenaltyMsKey.Float64(penalty)
}

// LLMDEPPScorerCount returns an attribute for the number of scorers configured in a profile.
func LLMDEPPScorerCount(count int) attribute.KeyValue {
	return LLMDEPPScorerCountKey.Int(count)
}

// LLMDEPPScoringCandidateEndpoints returns an attribute for candidate endpoints entering the scorer chain.
func LLMDEPPScoringCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPScoringCandidateEndpointsKey.Int(count)
}

// LLMDEPPPickerCandidateEndpoints returns an attribute for candidate endpoints entering the picker.
func LLMDEPPPickerCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPPickerCandidateEndpointsKey.Int(count)
}

// LLMDEPPPickerTopEndpoints returns an attribute for the top scored endpoint names.
func LLMDEPPPickerTopEndpoints(endpoints []string) attribute.KeyValue {
	return LLMDEPPPickerTopEndpointsKey.StringSlice(endpoints)
}

// LLMDEPPPickerTopScores returns an attribute for the top scored endpoint values.
func LLMDEPPPickerTopScores(scores []float64) attribute.KeyValue {
	return LLMDEPPPickerTopScoresKey.Float64Slice(scores)
}

// EPP Scorer helpers

// LLMDEPPScorerType returns an attribute for the EPP scorer type.
func LLMDEPPScorerType(val string) attribute.KeyValue {
	return LLMDEPPScorerTypeKey.String(val)
}

// LLMDEPPScorerName returns an attribute for the EPP scorer instance name.
func LLMDEPPScorerName(val string) attribute.KeyValue {
	return LLMDEPPScorerNameKey.String(val)
}

// LLMDEPPScorerWeight returns an attribute for the EPP scorer weight.
func LLMDEPPScorerWeight(val float64) attribute.KeyValue {
	return LLMDEPPScorerWeightKey.Float64(val)
}

// LLMDEPPScorerCandidateEndpoints returns an attribute for the number of candidate endpoints for a scorer.
func LLMDEPPScorerCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPScorerCandidateEndpointsKey.Int(count)
}

// LLMDEPPScorerScoreMax returns an attribute for the maximum score generated by a scorer.
func LLMDEPPScorerScoreMax(score float64) attribute.KeyValue {
	return LLMDEPPScorerScoreMaxKey.Float64(score)
}

// LLMDEPPScorerScoreAvg returns an attribute for the average score generated by a scorer.
func LLMDEPPScorerScoreAvg(score float64) attribute.KeyValue {
	return LLMDEPPScorerScoreAvgKey.Float64(score)
}

// LLMDEPPScorerEndpointsScored returns an attribute for the number of endpoints scored.
func LLMDEPPScorerEndpointsScored(count int) attribute.KeyValue {
	return LLMDEPPScorerEndpointsScoredKey.Int(count)
}

// EPP Profile Handler helpers

// LLMDEPPProfileHandlerDecision returns an attribute for the profile handler decision.
func LLMDEPPProfileHandlerDecision(decision string) attribute.KeyValue {
	return LLMDEPPProfileHandlerDecisionKey.String(decision)
}

// LLMDEPPProfileHandlerSelectedProfile returns an attribute for the selected profile name.
func LLMDEPPProfileHandlerSelectedProfile(profile string) attribute.KeyValue {
	return LLMDEPPProfileHandlerSelectedProfileKey.String(profile)
}

// LLMDEPPProfileHandlerTotalProfiles returns an attribute for total profiles evaluated.
func LLMDEPPProfileHandlerTotalProfiles(total int) attribute.KeyValue {
	return LLMDEPPProfileHandlerTotalProfilesKey.Int(total)
}

// LLMDEPPProfileHandlerExecutedProfiles returns an attribute for executed profiles count.
func LLMDEPPProfileHandlerExecutedProfiles(executed int) attribute.KeyValue {
	return LLMDEPPProfileHandlerExecutedProfilesKey.Int(executed)
}

// LLMDEPPProfileHandlerDecodeFailed returns an attribute indicating whether decode execution failed.
func LLMDEPPProfileHandlerDecodeFailed(failed bool) attribute.KeyValue {
	return LLMDEPPProfileHandlerDecodeFailedKey.Bool(failed)
}

// EPP Disagg helpers

// LLMDEPPDisaggReason returns an attribute for disaggregation reason.
func LLMDEPPDisaggReason(reason string) attribute.KeyValue {
	return LLMDEPPDisaggReasonKey.String(reason)
}

// LLMDEPPPDReason returns an attribute for prefill/decode disaggregation reason.
func LLMDEPPPDReason(reason string) attribute.KeyValue {
	return LLMDEPPPDReasonKey.String(reason)
}

// LLMDEPPPDDisaggregationUsed returns an attribute indicating whether PD disaggregation was used.
func LLMDEPPPDDisaggregationUsed(used bool) attribute.KeyValue {
	return LLMDEPPPDDisaggregationUsedKey.Bool(used)
}

// LLMDEPPPDPrefillPodAddress returns an attribute for selected prefill pod address.
func LLMDEPPPDPrefillPodAddress(addr string) attribute.KeyValue {
	return LLMDEPPPDPrefillPodAddressKey.String(addr)
}

// LLMDEPPPDPrefillPodPort returns an attribute for selected prefill pod port.
func LLMDEPPPDPrefillPodPort(port string) attribute.KeyValue {
	return LLMDEPPPDPrefillPodPortKey.String(port)
}

// LLMDEPPEncodeDisaggregationUsed returns an attribute indicating whether encode disaggregation was used.
func LLMDEPPEncodeDisaggregationUsed(used bool) attribute.KeyValue {
	return LLMDEPPEncodeDisaggregationUsedKey.Bool(used)
}

// LLMDEPPEncodeReason returns an attribute for encode disaggregation reason.
func LLMDEPPEncodeReason(reason string) attribute.KeyValue {
	return LLMDEPPEncodeReasonKey.String(reason)
}

// LLMDEPPEncodeEndpoints returns an attribute for encode endpoints.
func LLMDEPPEncodeEndpoints(endpoints string) attribute.KeyValue {
	return LLMDEPPEncodeEndpointsKey.String(endpoints)
}

// EPP Producer helpers

// LLMDEPPProducerCandidateEndpoints returns an attribute for producer candidate endpoints count.
func LLMDEPPProducerCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPProducerCandidateEndpointsKey.Int(count)
}

// LLMDEPPProducerResult returns an attribute for producer result.
func LLMDEPPProducerResult(result string) attribute.KeyValue {
	return LLMDEPPProducerResultKey.String(result)
}

// LLMDEPPProducerMaxMatchBlocks returns an attribute for max matched blocks in producer.
func LLMDEPPProducerMaxMatchBlocks(blocks int) attribute.KeyValue {
	return LLMDEPPProducerMaxMatchBlocksKey.Int(blocks)
}

// LLMDEPPProducerTotalBlocks returns an attribute for total blocks in producer.
func LLMDEPPProducerTotalBlocks(blocks int) attribute.KeyValue {
	return LLMDEPPProducerTotalBlocksKey.Int(blocks)
}

// EPP Token Producer helpers

// LLMDEPPTokenProducerBackend returns an attribute for token producer backend.
func LLMDEPPTokenProducerBackend(backend string) attribute.KeyValue {
	return LLMDEPPTokenProducerBackendKey.String(backend)
}

// LLMDEPPTokenProducerResult returns an attribute for token producer result.
func LLMDEPPTokenProducerResult(result string) attribute.KeyValue {
	return LLMDEPPTokenProducerResultKey.String(result)
}

// LLMDEPPTokenProducerTokenCount returns an attribute for token producer token count.
func LLMDEPPTokenProducerTokenCount(count int) attribute.KeyValue {
	return LLMDEPPTokenProducerTokenCountKey.Int(count)
}

// KV Cache helpers

// LLMDKVCachePodCount returns an attribute for KV cache pod count.
func LLMDKVCachePodCount(count int) attribute.KeyValue {
	return LLMDKVCachePodCountKey.Int(count)
}

// LLMDKVCacheTokenCount returns an attribute for KV cache token count.
func LLMDKVCacheTokenCount(count int) attribute.KeyValue {
	return LLMDKVCacheTokenCountKey.Int(count)
}

// LLMDKVCacheBlockKeysCount returns an attribute for KV cache block keys count.
func LLMDKVCacheBlockKeysCount(count int) attribute.KeyValue {
	return LLMDKVCacheBlockKeysCountKey.Int(count)
}

// LLMDKVCacheBlockHitRatio returns an attribute for KV cache block hit ratio.
func LLMDKVCacheBlockHitRatio(ratio float64) attribute.KeyValue {
	return LLMDKVCacheBlockHitRatioKey.Float64(ratio)
}

// LLMDKVCacheBlocksFound returns an attribute for KV cache blocks found.
func LLMDKVCacheBlocksFound(blocks int) attribute.KeyValue {
	return LLMDKVCacheBlocksFoundKey.Int(blocks)
}

// LLMDKVCacheIndexWalkKeyCount returns an attribute for KV cache index walk key count.
func LLMDKVCacheIndexWalkKeyCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexWalkKeyCountKey.Int(count)
}

// LLMDKVCacheIndexWalkKeysPresent returns an attribute for KV cache index walk keys present.
func LLMDKVCacheIndexWalkKeysPresent(present int) attribute.KeyValue {
	return LLMDKVCacheIndexWalkKeysPresentKey.Int(present)
}

// LLMDKVCacheIndexAddEngineKeyCount returns an attribute for KV cache index add engine keys count.
func LLMDKVCacheIndexAddEngineKeyCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexAddEngineKeyCountKey.Int(count)
}

// LLMDKVCacheIndexAddRequestKeyCount returns an attribute for KV cache index add request keys count.
func LLMDKVCacheIndexAddRequestKeyCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexAddRequestKeyCountKey.Int(count)
}

// LLMDKVCacheIndexAddPodEntryCount returns an attribute for KV cache index add pod entries count.
func LLMDKVCacheIndexAddPodEntryCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexAddPodEntryCountKey.Int(count)
}

// LLMDKVCacheIndexAddDeviceTierCount returns an attribute for KV cache index add device tiers count.
func LLMDKVCacheIndexAddDeviceTierCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexAddDeviceTierCountKey.Int(count)
}

// LLMDKVCacheIndexEvictKeyType returns an attribute for KV cache index evicted key type.
func LLMDKVCacheIndexEvictKeyType(keyType string) attribute.KeyValue {
	return LLMDKVCacheIndexEvictKeyTypeKey.String(keyType)
}

// LLMDKVCacheIndexEvictPodEntryCount returns an attribute for KV cache index evicted pod entries count.
func LLMDKVCacheIndexEvictPodEntryCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexEvictPodEntryCountKey.Int(count)
}

// LLMDKVCacheIndexEvictDeviceTierCount returns an attribute for KV cache index evicted device tiers count.
func LLMDKVCacheIndexEvictDeviceTierCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexEvictDeviceTierCountKey.Int(count)
}

// LLMDKVCacheIndexLookupBlockCount returns an attribute for KV cache index lookup block count.
func LLMDKVCacheIndexLookupBlockCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexLookupBlockCountKey.Int(count)
}

// LLMDKVCacheLookupPodFilterCount returns an attribute for KV cache lookup pod filter count.
func LLMDKVCacheLookupPodFilterCount(count int) attribute.KeyValue {
	return LLMDKVCacheLookupPodFilterCountKey.Int(count)
}

// LLMDKVCacheLookupCacheHit returns an attribute for KV cache lookup cache hit.
func LLMDKVCacheLookupCacheHit(hit bool) attribute.KeyValue {
	return LLMDKVCacheLookupCacheHitKey.Bool(hit)
}

// LLMDKVCacheLookupBlocksFound returns an attribute for KV cache lookup blocks found.
func LLMDKVCacheLookupBlocksFound(blocks int) attribute.KeyValue {
	return LLMDKVCacheLookupBlocksFoundKey.Int(blocks)
}

// LLMDKVCachePrefixMatchKeyCount returns an attribute for prefix match key count.
func LLMDKVCachePrefixMatchKeyCount(count int) attribute.KeyValue {
	return LLMDKVCachePrefixMatchKeyCountKey.Int(count)
}

// LLMDKVCachePrefixMatchPodFilterCount returns an attribute for prefix match pod filter count.
func LLMDKVCachePrefixMatchPodFilterCount(count int) attribute.KeyValue {
	return LLMDKVCachePrefixMatchPodFilterCountKey.Int(count)
}

// LLMDKVCachePrefixMatchWalked returns an attribute indicating whether key walker was used.
func LLMDKVCachePrefixMatchWalked(walked bool) attribute.KeyValue {
	return LLMDKVCachePrefixMatchWalkedKey.Bool(walked)
}

// LLMDKVCachePrefixMatchPodsMatched returns an attribute for prefix match matched pods count.
func LLMDKVCachePrefixMatchPodsMatched(count int) attribute.KeyValue {
	return LLMDKVCachePrefixMatchPodsMatchedKey.Int(count)
}

// LLMDKVCachePrefixMatchLongestChain returns an attribute for prefix match longest block chain.
func LLMDKVCachePrefixMatchLongestChain(chain int) attribute.KeyValue {
	return LLMDKVCachePrefixMatchLongestChainKey.Int(chain)
}

// LLMDKVCacheScorerAlgorithm returns an attribute for KV cache scorer algorithm strategy.
func LLMDKVCacheScorerAlgorithm(algo string) attribute.KeyValue {
	return LLMDKVCacheScorerAlgorithmKey.String(algo)
}

// LLMDKVCacheScorerKeyCount returns an attribute for KV cache scorer key count.
func LLMDKVCacheScorerKeyCount(count int) attribute.KeyValue {
	return LLMDKVCacheScorerKeyCountKey.Int(count)
}

// LLMDKVCacheScoreMax returns an attribute for KV cache max score.
func LLMDKVCacheScoreMax(score float64) attribute.KeyValue {
	return LLMDKVCacheScoreMaxKey.Float64(score)
}

// LLMDKVCacheScoreAvg returns an attribute for KV cache average score.
func LLMDKVCacheScoreAvg(score float64) attribute.KeyValue {
	return LLMDKVCacheScoreAvgKey.Float64(score)
}

// LLMDKVCacheScorerPodsScored returns an attribute for KV cache scored pods count.
func LLMDKVCacheScorerPodsScored(count int) attribute.KeyValue {
	return LLMDKVCacheScorerPodsScoredKey.Int(count)
}

// KV Cache Event helpers

// LLMDKVCacheEventsTopic returns an attribute for KV cache event topic.
func LLMDKVCacheEventsTopic(topic string) attribute.KeyValue {
	return LLMDKVCacheEventsTopicKey.String(topic)
}

// LLMDKVCacheEventsSequence returns an attribute for KV cache event sequence.
func LLMDKVCacheEventsSequence(seq int64) attribute.KeyValue {
	return LLMDKVCacheEventsSequenceKey.Int64(seq)
}

// LLMDKVCacheEventsPayloadSizeBytes returns an attribute for KV cache event payload size.
func LLMDKVCacheEventsPayloadSizeBytes(size int) attribute.KeyValue {
	return LLMDKVCacheEventsPayloadSizeBytesKey.Int(size)
}

// LLMDKVCacheEventsSourceEndpoint returns an attribute for KV cache event source endpoint.
func LLMDKVCacheEventsSourceEndpoint(ep string) attribute.KeyValue {
	return LLMDKVCacheEventsSourceEndpointKey.String(ep)
}

// LLMDKVCacheEventsPodID returns an attribute for KV cache event pod ID.
func LLMDKVCacheEventsPodID(podID string) attribute.KeyValue {
	return LLMDKVCacheEventsPodIDKey.String(podID)
}

// LLMDKVCacheEventsEventCount returns an attribute for KV cache event batch count.
func LLMDKVCacheEventsEventCount(count int) attribute.KeyValue {
	return LLMDKVCacheEventsEventCountKey.Int(count)
}

// Sidecar / Proxy helpers

// LLMDPDProxyConnector returns an attribute for PD proxy connector name.
func LLMDPDProxyConnector(conn string) attribute.KeyValue {
	return LLMDPDProxyConnectorKey.String(conn)
}

// LLMDPDProxyKVConnector returns an attribute for PD proxy KV connector name.
func LLMDPDProxyKVConnector(conn string) attribute.KeyValue {
	return LLMDPDProxyKVConnectorKey.String(conn)
}

// LLMDPDProxyECConnector returns an attribute for PD proxy EC connector name.
func LLMDPDProxyECConnector(conn string) attribute.KeyValue {
	return LLMDPDProxyECConnectorKey.String(conn)
}

// LLMDPDProxyRequestID returns an attribute for PD proxy request ID.
func LLMDPDProxyRequestID(id string) attribute.KeyValue {
	return LLMDPDProxyRequestIDKey.String(id)
}

// LLMDPDProxyRequestPath returns an attribute for PD proxy request path.
func LLMDPDProxyRequestPath(path string) attribute.KeyValue {
	return LLMDPDProxyRequestPathKey.String(path)
}

// LLMDPDProxyPrefillTarget returns an attribute for PD proxy prefill target host/port.
func LLMDPDProxyPrefillTarget(target string) attribute.KeyValue {
	return LLMDPDProxyPrefillTargetKey.String(target)
}

// LLMDPDProxyPrefillCandidates returns an attribute for PD proxy prefill candidate count.
func LLMDPDProxyPrefillCandidates(candidates int) attribute.KeyValue {
	return LLMDPDProxyPrefillCandidatesKey.Int(candidates)
}

// LLMDPDProxyDecodeTarget returns an attribute for PD proxy decode target host/port.
func LLMDPDProxyDecodeTarget(target string) attribute.KeyValue {
	return LLMDPDProxyDecodeTargetKey.String(target)
}

// LLMDPDProxyReason returns an attribute for PD proxy decision reason.
func LLMDPDProxyReason(reason string) attribute.KeyValue {
	return LLMDPDProxyReasonKey.String(reason)
}

// LLMDPDProxyError returns an attribute for PD proxy error classification.
func LLMDPDProxyError(err string) attribute.KeyValue {
	return LLMDPDProxyErrorKey.String(err)
}

// LLMDPDProxyDeniedTarget returns an attribute for PD proxy denied target address.
func LLMDPDProxyDeniedTarget(target string) attribute.KeyValue {
	return LLMDPDProxyDeniedTargetKey.String(target)
}

// LLMDPDProxyKVCacheSource returns an attribute for PD proxy KV cache source header value.
func LLMDPDProxyKVCacheSource(source string) attribute.KeyValue {
	return LLMDPDProxyKVCacheSourceKey.String(source)
}

// LLMDPDProxyDisaggregationUsed returns an attribute indicating whether PD proxy disaggregation was used.
func LLMDPDProxyDisaggregationUsed(used bool) attribute.KeyValue {
	return LLMDPDProxyDisaggregationUsedKey.Bool(used)
}

// LLMDPDProxyConcurrentPD returns an attribute indicating whether concurrent PD was used.
func LLMDPDProxyConcurrentPD(concurrent bool) attribute.KeyValue {
	return LLMDPDProxyConcurrentPDKey.Bool(concurrent)
}

// LLMDPDProxyParallelDispatch returns an attribute indicating whether parallel dispatch was used.
func LLMDPDProxyParallelDispatch(parallel bool) attribute.KeyValue {
	return LLMDPDProxyParallelDispatchKey.Bool(parallel)
}

// LLMDPDProxyParallelWindowMs returns an attribute for PD proxy parallel window in milliseconds.
func LLMDPDProxyParallelWindowMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyParallelWindowMsKey.Float64(ms)
}

// LLMDPDProxyTotalDurationMs returns an attribute for PD proxy total duration in milliseconds.
func LLMDPDProxyTotalDurationMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyTotalDurationMsKey.Float64(ms)
}

// LLMDPDProxyTrueTTFTMs returns an attribute for PD proxy true TTFT in milliseconds.
func LLMDPDProxyTrueTTFTMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyTrueTTFTMsKey.Float64(ms)
}

// LLMDPDProxyPrefillDurationMsSummary returns an attribute for summary prefill duration in milliseconds.
func LLMDPDProxyPrefillDurationMsSummary(ms float64) attribute.KeyValue {
	return LLMDPDProxyPrefillDurationMsSummaryKey.Float64(ms)
}

// LLMDPDProxyDecodeDurationMsSummary returns an attribute for summary decode duration in milliseconds.
func LLMDPDProxyDecodeDurationMsSummary(ms float64) attribute.KeyValue {
	return LLMDPDProxyDecodeDurationMsSummaryKey.Float64(ms)
}

// LLMDPDProxyCoordinatorOverheadMs returns an attribute for coordinator overhead in milliseconds.
func LLMDPDProxyCoordinatorOverheadMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyCoordinatorOverheadMsKey.Float64(ms)
}

// LLMDPDProxyPrefillAsync returns an attribute indicating whether prefill was asynchronous.
func LLMDPDProxyPrefillAsync(async bool) attribute.KeyValue {
	return LLMDPDProxyPrefillAsyncKey.Bool(async)
}

// LLMDPDProxyPrefillStatusCode returns an attribute for prefill HTTP response status code.
func LLMDPDProxyPrefillStatusCode(code int) attribute.KeyValue {
	return LLMDPDProxyPrefillStatusCodeKey.Int(code)
}

// LLMDPDProxyPrefillDurationMs returns an attribute for prefill span duration in milliseconds.
func LLMDPDProxyPrefillDurationMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyPrefillDurationMsKey.Float64(ms)
}

// LLMDPDProxyDecodeConcurrentWithPrefill returns an attribute indicating whether decode ran concurrently with prefill.
func LLMDPDProxyDecodeConcurrentWithPrefill(concurrent bool) attribute.KeyValue {
	return LLMDPDProxyDecodeConcurrentWithPrefillKey.Bool(concurrent)
}

// LLMDPDProxyDecodeDataParallel returns an attribute indicating whether decode used data parallel.
func LLMDPDProxyDecodeDataParallel(dp bool) attribute.KeyValue {
	return LLMDPDProxyDecodeDataParallelKey.Bool(dp)
}

// LLMDPDProxyDecodeStreaming returns an attribute indicating whether decode used streaming.
func LLMDPDProxyDecodeStreaming(streaming bool) attribute.KeyValue {
	return LLMDPDProxyDecodeStreamingKey.Bool(streaming)
}

// LLMDPDProxyDecodeDurationMs returns an attribute for decode span duration in milliseconds.
func LLMDPDProxyDecodeDurationMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyDecodeDurationMsKey.Float64(ms)
}

// LLMDPDProxyChunkedDecodeChunkSize returns an attribute for chunked decode chunk size.
func LLMDPDProxyChunkedDecodeChunkSize(size int) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeChunkSizeKey.Int(size)
}

// LLMDPDProxyChunkedDecodeStreaming returns an attribute indicating whether chunked decode was streaming.
func LLMDPDProxyChunkedDecodeStreaming(streaming bool) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeStreamingKey.Bool(streaming)
}

// LLMDPDProxyChunkedDecodeChunks returns an attribute for number of chunks in chunked decode.
func LLMDPDProxyChunkedDecodeChunks(chunks int) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeChunksKey.Int(chunks)
}

// LLMDPDProxyChunkedDecodeTotalTokens returns an attribute for total tokens in chunked decode.
func LLMDPDProxyChunkedDecodeTotalTokens(tokens int) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeTotalTokensKey.Int(tokens)
}

// LLMDPDProxyChunkedDecodeDurationMs returns an attribute for chunked decode duration in milliseconds.
func LLMDPDProxyChunkedDecodeDurationMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeDurationMsKey.Float64(ms)
}

// EC Proxy helpers

// LLMDECProxyEncodeDisaggregationUsed returns an attribute indicating whether encode disaggregation was used.
func LLMDECProxyEncodeDisaggregationUsed(used bool) attribute.KeyValue {
	return LLMDECProxyEncodeDisaggregationUsedKey.Bool(used)
}

// LLMDECProxyEncoderCount returns an attribute for allowed encoder count in EC proxy.
func LLMDECProxyEncoderCount(count int) attribute.KeyValue {
	return LLMDECProxyEncoderCountKey.Int(count)
}

// LLMDECProxyEncoderCandidates returns an attribute for candidate encoder count in EC proxy.
func LLMDECProxyEncoderCandidates(candidates int) attribute.KeyValue {
	return LLMDECProxyEncoderCandidatesKey.Int(candidates)
}

// OpenAI API helpers

// LLMDOpenAIAPI returns an attribute for OpenAI API endpoint type.
func LLMDOpenAIAPI(api string) attribute.KeyValue {
	return LLMDOpenAIAPIKey.String(api)
}
