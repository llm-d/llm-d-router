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

package request

const (
	RequestIDHeaderKey = "x-request-id"
	// PeerTopologyHeaderKey carries the prefill endpoint's encoded topology
	// from the prefill EPP's response, through the coordinator, to the decode
	// EPP's request, for topology-affinity-filter and topology-affinity-scorer
	// running in coordinator deployments.
	PeerTopologyHeaderKey = "x-peer-topology"

	FieldKVTransferParams     = "kv_transfer_params"
	FieldECTransferParams     = "ec_transfer_params"
	FieldMaxOutputTokens      = "max_output_tokens" // Used by Responses API
	FieldMinTokens            = "min_tokens"
	FieldSamplingParams       = "sampling_params"
	FieldExtraArgs            = "extra_args"
	FieldDoRemotePrefill      = "do_remote_prefill"
	FieldDoRemoteDecode       = "do_remote_decode"
	FieldRemoteBlockIDs       = "remote_block_ids"
	FieldRemoteEngineID       = "remote_engine_id"
	FieldRemoteHost           = "remote_host"
	FieldRemotePort           = "remote_port"
	FieldCacheHitThreshold    = "cache_hit_threshold"
	FieldContinueFinalMessage = "continue_final_message"
	FieldAddGenerationPrompt  = "add_generation_prompt"
)
