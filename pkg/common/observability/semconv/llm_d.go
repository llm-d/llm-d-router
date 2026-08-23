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
// All custom router, scheduler, scorer, cache, and sidecar attributes are namespaced under "llm_d.*".
const (
	// EPP Scheduling, Filtering & Picking attributes
	LLMDEPPProfileNameKey               = attribute.Key("llm_d.epp.scheduling.profile.name")
	LLMDEPPFilterCandidateEndpointsKey  = attribute.Key("llm_d.epp.filter.candidate_endpoints")
	LLMDEPPFilterFilteredEndpointsKey   = attribute.Key("llm_d.epp.filter.filtered_endpoints")
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

	// EPP Data Producer attributes
	LLMDEPPProducerCandidateEndpointsKey = attribute.Key("llm_d.epp.producer.candidate_endpoints")
	LLMDEPPProducerResultKey             = attribute.Key("llm_d.epp.producer.result")
	LLMDEPPProducerMaxMatchBlocksKey     = attribute.Key("llm_d.epp.producer.max_match_blocks")
	LLMDEPPProducerTotalBlocksKey        = attribute.Key("llm_d.epp.producer.total_blocks")

	// KV Cache & Indexer attributes
	LLMDKVCachePodCountKey                  = attribute.Key("llm_d.kv_cache.pod_count")
	LLMDKVCacheTokenCountKey                = attribute.Key("llm_d.kv_cache.token_count")
	LLMDKVCacheBlockKeysCountKey            = attribute.Key("llm_d.kv_cache.block_keys.count")
	LLMDKVCacheBlockHitRatioKey             = attribute.Key("llm_d.kv_cache.block_hit_ratio")
	LLMDKVCacheBlocksFoundKey               = attribute.Key("llm_d.kv_cache.blocks_found")
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
	LLMDKVCacheScorerAlgorithmKey           = attribute.Key("llm_d.kv_cache.scorer.algorithm")
	LLMDKVCacheScorerKeyCountKey            = attribute.Key("llm_d.kv_cache.scorer.key_count")
	LLMDKVCacheScoreMaxKey                  = attribute.Key("llm_d.kv_cache.score.max")
	LLMDKVCacheScoreAvgKey                  = attribute.Key("llm_d.kv_cache.score.avg")
	LLMDKVCacheScorerPodsScoredKey          = attribute.Key("llm_d.kv_cache.scorer.pods_scored")

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
	LLMDECProxyEncoderAllowedKey           = attribute.Key("llm_d.ec_proxy.encoder_allowed")
	LLMDECProxyEncoderCandidatesKey        = attribute.Key("llm_d.ec_proxy.encoder_candidates")

	// OpenAI API attributes
	LLMDOpenAIAPIKey = attribute.Key("llm_d.openai.api")
)

// Typed helper functions for llm-d internal attributes.

// EPP Scheduling helpers
func LLMDEPPProfileName(name string) attribute.KeyValue {
	return LLMDEPPProfileNameKey.String(name)
}

func LLMDEPPFilterCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPFilterCandidateEndpointsKey.Int(count)
}

func LLMDEPPFilterFilteredEndpoints(count int) attribute.KeyValue {
	return LLMDEPPFilterFilteredEndpointsKey.Int(count)
}

func LLMDEPPScorerCount(count int) attribute.KeyValue {
	return LLMDEPPScorerCountKey.Int(count)
}

func LLMDEPPScoringCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPScoringCandidateEndpointsKey.Int(count)
}

func LLMDEPPPickerCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPPickerCandidateEndpointsKey.Int(count)
}

func LLMDEPPPickerTopEndpoints(endpoints []string) attribute.KeyValue {
	return LLMDEPPPickerTopEndpointsKey.StringSlice(endpoints)
}

func LLMDEPPPickerTopScores(scores []float64) attribute.KeyValue {
	return LLMDEPPPickerTopScoresKey.Float64Slice(scores)
}

// EPP Scorer helpers
func LLMDEPPScorerType(val string) attribute.KeyValue {
	return LLMDEPPScorerTypeKey.String(val)
}

func LLMDEPPScorerName(val string) attribute.KeyValue {
	return LLMDEPPScorerNameKey.String(val)
}

func LLMDEPPScorerWeight(val float64) attribute.KeyValue {
	return LLMDEPPScorerWeightKey.Float64(val)
}

func LLMDEPPScorerCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPScorerCandidateEndpointsKey.Int(count)
}

func LLMDEPPScorerScoreMax(score float64) attribute.KeyValue {
	return LLMDEPPScorerScoreMaxKey.Float64(score)
}

func LLMDEPPScorerScoreAvg(score float64) attribute.KeyValue {
	return LLMDEPPScorerScoreAvgKey.Float64(score)
}

func LLMDEPPScorerEndpointsScored(count int) attribute.KeyValue {
	return LLMDEPPScorerEndpointsScoredKey.Int(count)
}

// EPP Profile Handler helpers
func LLMDEPPProfileHandlerDecision(decision string) attribute.KeyValue {
	return LLMDEPPProfileHandlerDecisionKey.String(decision)
}

func LLMDEPPProfileHandlerSelectedProfile(profile string) attribute.KeyValue {
	return LLMDEPPProfileHandlerSelectedProfileKey.String(profile)
}

func LLMDEPPProfileHandlerTotalProfiles(total int) attribute.KeyValue {
	return LLMDEPPProfileHandlerTotalProfilesKey.Int(total)
}

func LLMDEPPProfileHandlerExecutedProfiles(executed int) attribute.KeyValue {
	return LLMDEPPProfileHandlerExecutedProfilesKey.Int(executed)
}

func LLMDEPPProfileHandlerDecodeFailed(failed bool) attribute.KeyValue {
	return LLMDEPPProfileHandlerDecodeFailedKey.Bool(failed)
}

// EPP Disagg helpers
func LLMDEPPDisaggReason(reason string) attribute.KeyValue {
	return LLMDEPPDisaggReasonKey.String(reason)
}

func LLMDEPPPDReason(reason string) attribute.KeyValue {
	return LLMDEPPPDReasonKey.String(reason)
}

func LLMDEPPPDDisaggregationUsed(used bool) attribute.KeyValue {
	return LLMDEPPPDDisaggregationUsedKey.Bool(used)
}

func LLMDEPPPDPrefillPodAddress(addr string) attribute.KeyValue {
	return LLMDEPPPDPrefillPodAddressKey.String(addr)
}

func LLMDEPPPDPrefillPodPort(port string) attribute.KeyValue {
	return LLMDEPPPDPrefillPodPortKey.String(port)
}

func LLMDEPPEncodeDisaggregationUsed(used bool) attribute.KeyValue {
	return LLMDEPPEncodeDisaggregationUsedKey.Bool(used)
}

func LLMDEPPEncodeReason(reason string) attribute.KeyValue {
	return LLMDEPPEncodeReasonKey.String(reason)
}

func LLMDEPPEncodeEndpoints(endpoints string) attribute.KeyValue {
	return LLMDEPPEncodeEndpointsKey.String(endpoints)
}

// EPP Producer helpers
func LLMDEPPProducerCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPProducerCandidateEndpointsKey.Int(count)
}

func LLMDEPPProducerResult(result string) attribute.KeyValue {
	return LLMDEPPProducerResultKey.String(result)
}

func LLMDEPPProducerMaxMatchBlocks(blocks int) attribute.KeyValue {
	return LLMDEPPProducerMaxMatchBlocksKey.Int(blocks)
}

func LLMDEPPProducerTotalBlocks(blocks int) attribute.KeyValue {
	return LLMDEPPProducerTotalBlocksKey.Int(blocks)
}

// KV Cache helpers
func LLMDKVCachePodCount(count int) attribute.KeyValue {
	return LLMDKVCachePodCountKey.Int(count)
}

func LLMDKVCacheTokenCount(count int) attribute.KeyValue {
	return LLMDKVCacheTokenCountKey.Int(count)
}

func LLMDKVCacheBlockKeysCount(count int) attribute.KeyValue {
	return LLMDKVCacheBlockKeysCountKey.Int(count)
}

func LLMDKVCacheBlockHitRatio(ratio float64) attribute.KeyValue {
	return LLMDKVCacheBlockHitRatioKey.Float64(ratio)
}

func LLMDKVCacheBlocksFound(blocks int) attribute.KeyValue {
	return LLMDKVCacheBlocksFoundKey.Int(blocks)
}

func LLMDKVCacheIndexAddEngineKeyCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexAddEngineKeyCountKey.Int(count)
}

func LLMDKVCacheIndexAddRequestKeyCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexAddRequestKeyCountKey.Int(count)
}

func LLMDKVCacheIndexAddPodEntryCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexAddPodEntryCountKey.Int(count)
}

func LLMDKVCacheIndexAddDeviceTierCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexAddDeviceTierCountKey.Int(count)
}

func LLMDKVCacheIndexEvictKeyType(keyType string) attribute.KeyValue {
	return LLMDKVCacheIndexEvictKeyTypeKey.String(keyType)
}

func LLMDKVCacheIndexEvictPodEntryCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexEvictPodEntryCountKey.Int(count)
}

func LLMDKVCacheIndexEvictDeviceTierCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexEvictDeviceTierCountKey.Int(count)
}

func LLMDKVCacheIndexLookupBlockCount(count int) attribute.KeyValue {
	return LLMDKVCacheIndexLookupBlockCountKey.Int(count)
}

func LLMDKVCacheLookupPodFilterCount(count int) attribute.KeyValue {
	return LLMDKVCacheLookupPodFilterCountKey.Int(count)
}

func LLMDKVCacheLookupCacheHit(hit bool) attribute.KeyValue {
	return LLMDKVCacheLookupCacheHitKey.Bool(hit)
}

func LLMDKVCacheLookupBlocksFound(blocks int) attribute.KeyValue {
	return LLMDKVCacheLookupBlocksFoundKey.Int(blocks)
}

func LLMDKVCacheScorerAlgorithm(algo string) attribute.KeyValue {
	return LLMDKVCacheScorerAlgorithmKey.String(algo)
}

func LLMDKVCacheScorerKeyCount(count int) attribute.KeyValue {
	return LLMDKVCacheScorerKeyCountKey.Int(count)
}

func LLMDKVCacheScoreMax(score float64) attribute.KeyValue {
	return LLMDKVCacheScoreMaxKey.Float64(score)
}

func LLMDKVCacheScoreAvg(score float64) attribute.KeyValue {
	return LLMDKVCacheScoreAvgKey.Float64(score)
}

func LLMDKVCacheScorerPodsScored(count int) attribute.KeyValue {
	return LLMDKVCacheScorerPodsScoredKey.Int(count)
}

// Sidecar / Proxy helpers
func LLMDPDProxyConnector(conn string) attribute.KeyValue {
	return LLMDPDProxyConnectorKey.String(conn)
}

func LLMDPDProxyKVConnector(conn string) attribute.KeyValue {
	return LLMDPDProxyKVConnectorKey.String(conn)
}

func LLMDPDProxyECConnector(conn string) attribute.KeyValue {
	return LLMDPDProxyECConnectorKey.String(conn)
}

func LLMDPDProxyRequestID(id string) attribute.KeyValue {
	return LLMDPDProxyRequestIDKey.String(id)
}

func LLMDPDProxyRequestPath(path string) attribute.KeyValue {
	return LLMDPDProxyRequestPathKey.String(path)
}

func LLMDPDProxyPrefillTarget(target string) attribute.KeyValue {
	return LLMDPDProxyPrefillTargetKey.String(target)
}

func LLMDPDProxyPrefillCandidates(candidates int) attribute.KeyValue {
	return LLMDPDProxyPrefillCandidatesKey.Int(candidates)
}

func LLMDPDProxyDecodeTarget(target string) attribute.KeyValue {
	return LLMDPDProxyDecodeTargetKey.String(target)
}

func LLMDPDProxyReason(reason string) attribute.KeyValue {
	return LLMDPDProxyReasonKey.String(reason)
}

func LLMDPDProxyError(err string) attribute.KeyValue {
	return LLMDPDProxyErrorKey.String(err)
}

func LLMDPDProxyDeniedTarget(target string) attribute.KeyValue {
	return LLMDPDProxyDeniedTargetKey.String(target)
}

func LLMDPDProxyKVCacheSource(source string) attribute.KeyValue {
	return LLMDPDProxyKVCacheSourceKey.String(source)
}

func LLMDPDProxyDisaggregationUsed(used bool) attribute.KeyValue {
	return LLMDPDProxyDisaggregationUsedKey.Bool(used)
}

func LLMDPDProxyConcurrentPD(concurrent bool) attribute.KeyValue {
	return LLMDPDProxyConcurrentPDKey.Bool(concurrent)
}

func LLMDPDProxyParallelDispatch(parallel bool) attribute.KeyValue {
	return LLMDPDProxyParallelDispatchKey.Bool(parallel)
}

func LLMDPDProxyParallelWindowMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyParallelWindowMsKey.Float64(ms)
}

func LLMDPDProxyTotalDurationMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyTotalDurationMsKey.Float64(ms)
}

func LLMDPDProxyTrueTTFTMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyTrueTTFTMsKey.Float64(ms)
}

func LLMDPDProxyPrefillDurationMsSummary(ms float64) attribute.KeyValue {
	return LLMDPDProxyPrefillDurationMsSummaryKey.Float64(ms)
}

func LLMDPDProxyDecodeDurationMsSummary(ms float64) attribute.KeyValue {
	return LLMDPDProxyDecodeDurationMsSummaryKey.Float64(ms)
}

func LLMDPDProxyCoordinatorOverheadMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyCoordinatorOverheadMsKey.Float64(ms)
}

func LLMDPDProxyPrefillAsync(async bool) attribute.KeyValue {
	return LLMDPDProxyPrefillAsyncKey.Bool(async)
}

func LLMDPDProxyPrefillStatusCode(code int) attribute.KeyValue {
	return LLMDPDProxyPrefillStatusCodeKey.Int(code)
}

func LLMDPDProxyPrefillDurationMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyPrefillDurationMsKey.Float64(ms)
}

func LLMDPDProxyDecodeConcurrentWithPrefill(concurrent bool) attribute.KeyValue {
	return LLMDPDProxyDecodeConcurrentWithPrefillKey.Bool(concurrent)
}

func LLMDPDProxyDecodeDataParallel(dp bool) attribute.KeyValue {
	return LLMDPDProxyDecodeDataParallelKey.Bool(dp)
}

func LLMDPDProxyDecodeStreaming(streaming bool) attribute.KeyValue {
	return LLMDPDProxyDecodeStreamingKey.Bool(streaming)
}

func LLMDPDProxyDecodeDurationMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyDecodeDurationMsKey.Float64(ms)
}

func LLMDPDProxyChunkedDecodeChunkSize(size int) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeChunkSizeKey.Int(size)
}

func LLMDPDProxyChunkedDecodeStreaming(streaming bool) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeStreamingKey.Bool(streaming)
}

func LLMDPDProxyChunkedDecodeChunks(chunks int) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeChunksKey.Int(chunks)
}

func LLMDPDProxyChunkedDecodeTotalTokens(tokens int) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeTotalTokensKey.Int(tokens)
}

func LLMDPDProxyChunkedDecodeDurationMs(ms float64) attribute.KeyValue {
	return LLMDPDProxyChunkedDecodeDurationMsKey.Float64(ms)
}

// EC Proxy helpers
func LLMDECProxyEncodeDisaggregationUsed(used bool) attribute.KeyValue {
	return LLMDECProxyEncodeDisaggregationUsedKey.Bool(used)
}

func LLMDECProxyEncoderCount(count int) attribute.KeyValue {
	return LLMDECProxyEncoderCountKey.Int(count)
}

func LLMDECProxyEncoderAllowed(count int) attribute.KeyValue {
	return LLMDECProxyEncoderAllowedKey.Int(count)
}

func LLMDECProxyEncoderCandidates(candidates int) attribute.KeyValue {
	return LLMDECProxyEncoderCandidatesKey.Int(candidates)
}

// OpenAI API helpers
func LLMDOpenAIAPI(api string) attribute.KeyValue {
	return LLMDOpenAIAPIKey.String(api)
}
