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
	// RevisionDecisionIDHeaderKey identifies requests that belong to the same
	// rollout decision. This is needed only for roles such as encode that create
	// several parallel subrequests from one user request (for example, one per
	// image). The coordinator must give every subrequest the same value so they
	// use the same revision.
	RevisionDecisionIDHeaderKey = "x-llm-d-revision-decision-id"

	FieldKVTransferParams     = "kv_transfer_params"
	FieldECTransferParams     = "ec_transfer_params"
	FieldMaxTokens            = "max_tokens"
	FieldMaxCompletionTokens  = "max_completion_tokens"
	FieldMaxOutputTokens      = "max_output_tokens" // Used by Responses API
	FieldMinTokens            = "min_tokens"
	FieldStream               = "stream"
	FieldStreamOptions        = "stream_options"
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
