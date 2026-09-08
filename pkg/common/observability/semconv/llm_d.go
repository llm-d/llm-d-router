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
// All custom router, scheduler, scorer, and cache attributes are namespaced under "llm_d.*".
const (
	// LLMDEPPScorerTypeKey is the attribute key for the scorer type.
	LLMDEPPScorerTypeKey = attribute.Key("llm_d.epp.scorer.type")
	// LLMDEPPScorerNameKey is the attribute key for the scorer instance name.
	LLMDEPPScorerNameKey = attribute.Key("llm_d.epp.scorer.name")
	// LLMDEPPScorerWeightKey is the attribute key for the scorer weight.
	LLMDEPPScorerWeightKey = attribute.Key("llm_d.epp.scorer.weight")
	// LLMDEPPScorerCandidateEndpointsKey is the attribute key for candidate endpoints count scored by a scorer.
	LLMDEPPScorerCandidateEndpointsKey = attribute.Key("llm_d.epp.scorer.candidate_endpoints")
	// LLMDEPPScorerScoreMaxKey is the attribute key for the max score produced by a scorer.
	LLMDEPPScorerScoreMaxKey = attribute.Key("llm_d.epp.scorer.score.max")
	// LLMDEPPScorerScoreAvgKey is the attribute key for the average score produced by a scorer.
	LLMDEPPScorerScoreAvgKey = attribute.Key("llm_d.epp.scorer.score.avg")
	// LLMDEPPScorerEndpointsScoredKey is the attribute key for total endpoints scored.
	LLMDEPPScorerEndpointsScoredKey = attribute.Key("llm_d.epp.scorer.endpoints_scored")

	// LLMDEPPDisaggReasonKey is the attribute key for disaggregation reason.
	LLMDEPPDisaggReasonKey = attribute.Key("llm_d.epp.disagg.reason")
	// LLMDEPPPDReasonKey is the attribute key for prefill/decode disaggregation reason.
	LLMDEPPPDReasonKey = attribute.Key("llm_d.epp.pd.reason")
	// LLMDEPPPDDisaggregationUsedKey is the attribute key indicating whether PD disaggregation was used.
	LLMDEPPPDDisaggregationUsedKey = attribute.Key("llm_d.epp.pd.disaggregation_used")
	// LLMDEPPPDPrefillPodAddressKey is the attribute key for selected prefill pod address.
	LLMDEPPPDPrefillPodAddressKey = attribute.Key("llm_d.epp.pd.prefill_pod_address")
	// LLMDEPPPDPrefillPodPortKey is the attribute key for selected prefill pod port.
	LLMDEPPPDPrefillPodPortKey = attribute.Key("llm_d.epp.pd.prefill_pod_port")
	// LLMDEPPEncodeDisaggregationUsedKey is the attribute key indicating whether encode disaggregation was used.
	LLMDEPPEncodeDisaggregationUsedKey = attribute.Key("llm_d.epp.encode.disaggregation_used")
	// LLMDEPPEncodeReasonKey is the attribute key for encode disaggregation reason.
	LLMDEPPEncodeReasonKey = attribute.Key("llm_d.epp.encode.reason")
	// LLMDEPPEncodeEndpointsKey is the attribute key for encode endpoint details.
	LLMDEPPEncodeEndpointsKey = attribute.Key("llm_d.epp.encode.endpoints")
	// LLMDEPPProfileHandlerDecisionKey is the attribute key for profile handler decision.
	LLMDEPPProfileHandlerDecisionKey = attribute.Key("llm_d.epp.profile_handler.decision")
	// LLMDEPPProfileHandlerSelectedProfileKey is the attribute key for selected profile name.
	LLMDEPPProfileHandlerSelectedProfileKey = attribute.Key("llm_d.epp.profile_handler.selected_profile")
	// LLMDEPPProfileHandlerTotalProfilesKey is the attribute key for total profile count.
	LLMDEPPProfileHandlerTotalProfilesKey = attribute.Key("llm_d.epp.profile_handler.total_profiles")
	// LLMDEPPProfileHandlerExecutedProfilesKey is the attribute key for executed profile count.
	LLMDEPPProfileHandlerExecutedProfilesKey = attribute.Key("llm_d.epp.profile_handler.executed_profiles")

	// LLMDEPPProducerCandidateEndpointsKey is the attribute key for producer candidate endpoints count.
	LLMDEPPProducerCandidateEndpointsKey = attribute.Key("llm_d.epp.producer.candidate_endpoints")
	// LLMDEPPProducerResultKey is the attribute key for producer execution result.
	LLMDEPPProducerResultKey = attribute.Key("llm_d.epp.producer.result")

	// LLMDKVCachePodCountKey is the attribute key for KV cache pod count.
	LLMDKVCachePodCountKey = attribute.Key("llm_d.kv_cache.pod_count")
	// LLMDKVCacheTokenCountKey is the attribute key for KV cache token count.
	LLMDKVCacheTokenCountKey = attribute.Key("llm_d.kv_cache.token_count")
	// LLMDKVCacheBlockKeysCountKey is the attribute key for KV cache block keys count.
	LLMDKVCacheBlockKeysCountKey = attribute.Key("llm_d.kv_cache.block_keys.count")
)

// Typed helper functions for llm-d internal attributes.

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

// LLMDEPPProducerCandidateEndpoints returns an attribute for producer candidate endpoints count.
func LLMDEPPProducerCandidateEndpoints(count int) attribute.KeyValue {
	return LLMDEPPProducerCandidateEndpointsKey.Int(count)
}

// LLMDEPPProducerResult returns an attribute for producer result.
func LLMDEPPProducerResult(result string) attribute.KeyValue {
	return LLMDEPPProducerResultKey.String(result)
}

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
