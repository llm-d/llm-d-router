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

package semconv

import (
	"reflect"
	"testing"

	"go.opentelemetry.io/otel/attribute"
)

// These expectations are independent of the registry to protect the emitted API.
func TestLLMDAttributeCompatibility(t *testing.T) {
	stringValue := "example"
	intValue := -7
	floatValue := 1.25
	boolValue := true
	tests := []struct{ got, want attribute.KeyValue }{
		{LLMDEPPFairnessID(stringValue), attribute.Key("llm_d.epp.fairness.id").String("example")},
		{LLMDEPPFairnessSource(stringValue), attribute.Key("llm_d.epp.fairness.source").String("example")},
		{LLMDEPPProfileName(stringValue), attribute.Key("llm_d.epp.scheduling.profile.name").String("example")},
		{LLMDEPPFilterDecision(stringValue), attribute.Key("llm_d.epp.filter.decision").String("example")},
		{LLMDEPPFilterCandidateEndpoints(intValue), attribute.Key("llm_d.epp.filter.candidate_endpoints").Int(-7)},
		{LLMDEPPFilterFilteredEndpoints(intValue), attribute.Key("llm_d.epp.filter.filtered_endpoints").Int(-7)},
		{LLMDEPPFilterStickyEndpoints(intValue), attribute.Key("llm_d.epp.filter.sticky_endpoints").Int(-7)},
		{LLMDEPPFilterAffinityThreshold(floatValue), attribute.Key("llm_d.epp.filter.affinity_threshold").Float64(1.25)},
		{LLMDEPPFilterTTFTPenaltyMs(floatValue), attribute.Key("llm_d.epp.filter.ttft_penalty_ms").Float64(1.25)},
		{LLMDEPPScorerCount(intValue), attribute.Key("llm_d.epp.scorer.count").Int(-7)},
		{LLMDEPPScoringCandidateEndpoints(intValue), attribute.Key("llm_d.epp.scoring.candidate_endpoints").Int(-7)},
		{LLMDEPPPickerCandidateEndpoints(intValue), attribute.Key("llm_d.epp.picker.candidate_endpoints").Int(-7)},
		{LLMDEPPPickerTopEndpoints([]string{"a", "b"}), attribute.Key("llm_d.epp.picker.top_endpoints").StringSlice([]string{"a", "b"})},
		{LLMDEPPPickerTopScores([]float64{1.25, -2.5}), attribute.Key("llm_d.epp.picker.top_scores").Float64Slice([]float64{1.25, -2.5})},
		{LLMDEPPScorerType(stringValue), attribute.Key("llm_d.epp.scorer.type").String("example")},
		{LLMDEPPScorerName(stringValue), attribute.Key("llm_d.epp.scorer.name").String("example")},
		{LLMDEPPScorerWeight(floatValue), attribute.Key("llm_d.epp.scorer.weight").Float64(1.25)},
		{LLMDEPPScorerCandidateEndpoints(intValue), attribute.Key("llm_d.epp.scorer.candidate_endpoints").Int(-7)},
		{LLMDEPPScorerScoreMax(floatValue), attribute.Key("llm_d.epp.scorer.score.max").Float64(1.25)},
		{LLMDEPPScorerScoreAvg(floatValue), attribute.Key("llm_d.epp.scorer.score.avg").Float64(1.25)},
		{LLMDEPPScorerEndpointsScored(intValue), attribute.Key("llm_d.epp.scorer.endpoints_scored").Int(-7)},
		{LLMDEPPProfileHandlerDecision(stringValue), attribute.Key("llm_d.epp.profile_handler.decision").String("example")},
		{LLMDEPPProfileHandlerSelectedProfile(stringValue), attribute.Key("llm_d.epp.profile_handler.selected_profile").String("example")},
		{LLMDEPPProfileHandlerTotalProfiles(intValue), attribute.Key("llm_d.epp.profile_handler.total_profiles").Int(-7)},
		{LLMDEPPProfileHandlerExecutedProfiles(intValue), attribute.Key("llm_d.epp.profile_handler.executed_profiles").Int(-7)},
		{LLMDEPPProfileHandlerDecodeFailed(boolValue), attribute.Key("llm_d.epp.profile_handler.decode_failed").Bool(true)},
		{LLMDEPPProfileHandlerPrefillFailed(boolValue), attribute.Key("llm_d.epp.profile_handler.prefill_failed").Bool(true)},
		{LLMDEPPDisaggReason(stringValue), attribute.Key("llm_d.epp.disagg.reason").String("example")},
		{LLMDEPPPDReason(stringValue), attribute.Key("llm_d.epp.pd.reason").String("example")},
		{LLMDEPPPDDisaggregationUsed(boolValue), attribute.Key("llm_d.epp.pd.disaggregation_used").Bool(true)},
		{LLMDEPPPDPrefillPodAddress(stringValue), attribute.Key("llm_d.epp.pd.prefill_pod_address").String("example")},
		{LLMDEPPPDPrefillPodPort(stringValue), attribute.Key("llm_d.epp.pd.prefill_pod_port").String("example")},
		{LLMDEPPEncodeDisaggregationUsed(boolValue), attribute.Key("llm_d.epp.encode.disaggregation_used").Bool(true)},
		{LLMDEPPEncodeReason(stringValue), attribute.Key("llm_d.epp.encode.reason").String("example")},
		{LLMDEPPEncodeEndpoints(stringValue), attribute.Key("llm_d.epp.encode.endpoints").String("example")},
		{LLMDEPPProducerCandidateEndpoints(intValue), attribute.Key("llm_d.epp.producer.candidate_endpoints").Int(-7)},
		{LLMDEPPProducerResult(stringValue), attribute.Key("llm_d.epp.producer.result").String("example")},
		{LLMDEPPProducerMaxMatchBlocks(intValue), attribute.Key("llm_d.epp.producer.max_match_blocks").Int(-7)},
		{LLMDEPPProducerTotalBlocks(intValue), attribute.Key("llm_d.epp.producer.total_blocks").Int(-7)},
		{LLMDEPPTokenProducerBackend(stringValue), attribute.Key("llm_d.epp.token_producer.backend").String("example")},
		{LLMDEPPTokenProducerResult(stringValue), attribute.Key("llm_d.epp.token_producer.result").String("example")},
		{LLMDEPPTokenProducerTokenCount(intValue), attribute.Key("llm_d.epp.token_producer.token_count").Int(-7)},
		{LLMDKVCachePodCount(intValue), attribute.Key("llm_d.kv_cache.pod_count").Int(-7)},
		{LLMDKVCacheTokenCount(intValue), attribute.Key("llm_d.kv_cache.token_count").Int(-7)},
		{LLMDKVCacheBlockKeysCount(intValue), attribute.Key("llm_d.kv_cache.block_keys.count").Int(-7)},
		{LLMDKVCacheBlockHitRatio(floatValue), attribute.Key("llm_d.kv_cache.block_hit_ratio").Float64(1.25)},
		{LLMDKVCacheBlocksFound(intValue), attribute.Key("llm_d.kv_cache.blocks_found").Int(-7)},
		{LLMDKVCacheIndexWalkKeyCount(intValue), attribute.Key("llm_d.kv_cache.index.walk.key_count").Int(-7)},
		{LLMDKVCacheIndexWalkKeysPresent(intValue), attribute.Key("llm_d.kv_cache.index.walk.keys_present").Int(-7)},
		{LLMDKVCacheIndexAddEngineKeyCount(intValue), attribute.Key("llm_d.kv_cache.index.add.engine_key_count").Int(-7)},
		{LLMDKVCacheIndexAddRequestKeyCount(intValue), attribute.Key("llm_d.kv_cache.index.add.request_key_count").Int(-7)},
		{LLMDKVCacheIndexAddPodEntryCount(intValue), attribute.Key("llm_d.kv_cache.index.add.pod_entry_count").Int(-7)},
		{LLMDKVCacheIndexAddDeviceTierCount(intValue), attribute.Key("llm_d.kv_cache.index.add.device_tier_count").Int(-7)},
		{LLMDKVCacheIndexEvictKeyType(stringValue), attribute.Key("llm_d.kv_cache.index.evict.key_type").String("example")},
		{LLMDKVCacheIndexEvictKeyCount(intValue), attribute.Key("llm_d.kv_cache.index.evict.key_count").Int(-7)},
		{LLMDKVCacheIndexEvictPodEntryCount(intValue), attribute.Key("llm_d.kv_cache.index.evict.pod_entry_count").Int(-7)},
		{LLMDKVCacheIndexEvictDeviceTierCount(intValue), attribute.Key("llm_d.kv_cache.index.evict.device_tier_count").Int(-7)},
		{LLMDKVCacheIndexLookupBlockCount(intValue), attribute.Key("llm_d.kv_cache.index.lookup.block_count").Int(-7)},
		{LLMDKVCacheLookupPodFilterCount(intValue), attribute.Key("llm_d.kv_cache.lookup.pod_filter_count").Int(-7)},
		{LLMDKVCacheLookupCacheHit(boolValue), attribute.Key("llm_d.kv_cache.lookup.cache_hit").Bool(true)},
		{LLMDKVCacheLookupBlocksFound(intValue), attribute.Key("llm_d.kv_cache.lookup.blocks_found").Int(-7)},
		{LLMDKVCachePrefixMatchKeyCount(intValue), attribute.Key("llm_d.kv_cache.prefix_match.key_count").Int(-7)},
		{LLMDKVCachePrefixMatchPodFilterCount(intValue), attribute.Key("llm_d.kv_cache.prefix_match.pod_filter_count").Int(-7)},
		{LLMDKVCachePrefixMatchWalked(boolValue), attribute.Key("llm_d.kv_cache.prefix_match.walked").Bool(true)},
		{LLMDKVCachePrefixMatchPodsMatched(intValue), attribute.Key("llm_d.kv_cache.prefix_match.pods_matched").Int(-7)},
		{LLMDKVCachePrefixMatchLongestChain(intValue), attribute.Key("llm_d.kv_cache.prefix_match.longest_chain").Int(-7)},
		{LLMDKVCacheEventsTopic(stringValue), attribute.Key("llm_d.kv_cache.events.topic").String("example")},
		{LLMDKVCacheEventsSequence(int64(42)), attribute.Key("llm_d.kv_cache.events.sequence").Int64(int64(42))},
		{LLMDKVCacheEventsPayloadSizeBytes(intValue), attribute.Key("llm_d.kv_cache.events.payload_size_bytes").Int(-7)},
		{LLMDKVCacheEventsSourceEndpoint(stringValue), attribute.Key("llm_d.kv_cache.events.source_endpoint").String("example")},
		{LLMDKVCacheEventsPodID(stringValue), attribute.Key("llm_d.kv_cache.events.pod_id").String("example")},
		{LLMDKVCacheEventsEventCount(intValue), attribute.Key("llm_d.kv_cache.events.event_count").Int(-7)},
		{LLMDPDProxyConnector(stringValue), attribute.Key("llm_d.pd_proxy.connector").String("example")},
		{LLMDPDProxyKVConnector(stringValue), attribute.Key("llm_d.pd_proxy.kv_connector").String("example")},
		{LLMDPDProxyECConnector(stringValue), attribute.Key("llm_d.pd_proxy.ec_connector").String("example")},
		{LLMDPDProxyRequestID(stringValue), attribute.Key("llm_d.pd_proxy.request_id").String("example")},
		{LLMDPDProxyRequestPath(stringValue), attribute.Key("llm_d.pd_proxy.request_path").String("example")},
		{LLMDPDProxyPrefillTarget(stringValue), attribute.Key("llm_d.pd_proxy.prefill_target").String("example")},
		{LLMDPDProxyPrefillCandidates(intValue), attribute.Key("llm_d.pd_proxy.prefill_candidates").Int(-7)},
		{LLMDPDProxyBootstrapRoom(int64(42)), attribute.Key("llm_d.pd_proxy.bootstrap_room").Int64(int64(42))},
		{LLMDPDProxyDecodeTarget(stringValue), attribute.Key("llm_d.pd_proxy.decode.target").String("example")},
		{LLMDPDProxyReason(stringValue), attribute.Key("llm_d.pd_proxy.reason").String("example")},
		{LLMDPDProxyError(stringValue), attribute.Key("llm_d.pd_proxy.error").String("example")},
		{LLMDPDProxyDeniedTarget(stringValue), attribute.Key("llm_d.pd_proxy.denied_target").String("example")},
		{LLMDPDProxyKVCacheSource(stringValue), attribute.Key("llm_d.pd_proxy.kv_cache_source").String("example")},
		{LLMDPDProxyDisaggregationUsed(boolValue), attribute.Key("llm_d.pd_proxy.disaggregation_used").Bool(true)},
		{LLMDPDProxyConcurrentPD(boolValue), attribute.Key("llm_d.pd_proxy.concurrent_pd").Bool(true)},
		{LLMDPDProxyParallelDispatch(boolValue), attribute.Key("llm_d.pd_proxy.parallel_dispatch").Bool(true)},
		{LLMDPDProxyParallelWindowMs(floatValue), attribute.Key("llm_d.pd_proxy.parallel_window_ms").Float64(1.25)},
		{LLMDPDProxyTotalDurationMs(floatValue), attribute.Key("llm_d.pd_proxy.total_duration_ms").Float64(1.25)},
		{LLMDPDProxyTrueTTFTMs(floatValue), attribute.Key("llm_d.pd_proxy.true_ttft_ms").Float64(1.25)},
		{LLMDPDProxyPrefillDurationMsSummary(floatValue), attribute.Key("llm_d.pd_proxy.prefill_duration_ms").Float64(1.25)},
		{LLMDPDProxyDecodeDurationMsSummary(floatValue), attribute.Key("llm_d.pd_proxy.decode_duration_ms").Float64(1.25)},
		{LLMDPDProxyCoordinatorOverheadMs(floatValue), attribute.Key("llm_d.pd_proxy.coordinator_overhead_ms").Float64(1.25)},
		{LLMDPDProxyPrefillAsync(boolValue), attribute.Key("llm_d.pd_proxy.prefill.async").Bool(true)},
		{LLMDPDProxyPrefillStatusCode(intValue), attribute.Key("llm_d.pd_proxy.prefill.status_code").Int(-7)},
		{LLMDPDProxyPrefillDurationMs(floatValue), attribute.Key("llm_d.pd_proxy.prefill.duration_ms").Float64(1.25)},
		{LLMDPDProxyDecodeConcurrentWithPrefill(boolValue), attribute.Key("llm_d.pd_proxy.decode.concurrent_with_prefill").Bool(true)},
		{LLMDPDProxyDecodeDataParallel(boolValue), attribute.Key("llm_d.pd_proxy.decode.data_parallel").Bool(true)},
		{LLMDPDProxyDecodeStreaming(boolValue), attribute.Key("llm_d.pd_proxy.decode.streaming").Bool(true)},
		{LLMDPDProxyDecodeDurationMs(floatValue), attribute.Key("llm_d.pd_proxy.decode.duration_ms").Float64(1.25)},
		{LLMDPDProxyChunkedDecodeChunkSize(intValue), attribute.Key("llm_d.pd_proxy.chunked_decode.chunk_size").Int(-7)},
		{LLMDPDProxyChunkedDecodeStreaming(boolValue), attribute.Key("llm_d.pd_proxy.chunked_decode.streaming").Bool(true)},
		{LLMDPDProxyChunkedDecodeChunks(intValue), attribute.Key("llm_d.pd_proxy.chunked_decode.chunks").Int(-7)},
		{LLMDPDProxyChunkedDecodeTotalTokens(intValue), attribute.Key("llm_d.pd_proxy.chunked_decode.total_tokens").Int(-7)},
		{LLMDPDProxyChunkedDecodeDurationMs(floatValue), attribute.Key("llm_d.pd_proxy.chunked_decode.duration_ms").Float64(1.25)},
		{LLMDECProxyEncodeDisaggregationUsed(boolValue), attribute.Key("llm_d.ec_proxy.encode_disaggregation_used").Bool(true)},
		{LLMDECProxyEncoderCount(intValue), attribute.Key("llm_d.ec_proxy.encoder_count").Int(-7)},
		{LLMDECProxyEncoderCandidates(intValue), attribute.Key("llm_d.ec_proxy.encoder_candidates").Int(-7)},
		{LLMDCoordinatorPipelineStepCount(intValue), attribute.Key("llm_d.coordinator.pipeline.step_count").Int(-7)},
		{LLMDCoordinatorPipelineExecutionPath(stringValue), attribute.Key("llm_d.coordinator.pipeline.execution_path").String("example")},
		{LLMDOpenAIAPI(stringValue), attribute.Key("llm_d.openai.api").String("example")},
	}
	for _, tt := range tests {
		t.Run(string(tt.want.Key), func(t *testing.T) {
			if !reflect.DeepEqual(tt.got, tt.want) {
				t.Fatalf("got %v, want %v", tt.got, tt.want)
			}
		})
	}
}
