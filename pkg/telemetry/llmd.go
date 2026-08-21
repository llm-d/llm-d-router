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

package telemetry

import (
	"go.opentelemetry.io/otel/attribute"
)

// Internal llm-d specific attribute keys.
// All custom router, scheduler, scorer, and sidecar attributes are namespaced under "llm_d.*".
const (
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

	// EPP Data Producer attributes
	LLMDEPPProducerCandidateEndpointsKey = attribute.Key("llm_d.epp.producer.candidate_endpoints")
	LLMDEPPProducerResultKey             = attribute.Key("llm_d.epp.producer.result")

	// KV Cache attributes
	LLMDKVCachePodCountKey       = attribute.Key("llm_d.kv_cache.pod_count")
	LLMDKVCacheTokenCountKey     = attribute.Key("llm_d.kv_cache.token_count")
	LLMDKVCacheBlockKeysCountKey = attribute.Key("llm_d.kv_cache.block_keys.count")
)

// Typed helper functions for llm-d internal attributes.

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
