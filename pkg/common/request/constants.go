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
	// EPPProfileHeaderKey names the scheduling profile the EPP must run for a
	// request. The coordinator sets it on every phase call, Envoy routes on it,
	// and the header-profile-handler reads it.
	EPPProfileHeaderKey = "x-llm-d-epp-profile"
	// DisaggregatedRevisionHeaderKey carries the selected rollout revision
	// between phases of a disaggregated request.
	DisaggregatedRevisionHeaderKey = "x-llm-d-disagg-revision"
	// RevisionDecisionIDHeaderKey identifies requests that belong to the same
	// rollout decision. This is needed only for roles such as encode that create
	// several parallel subrequests from one user request (for example, one per
	// image). The coordinator must give every subrequest the same value so they
	// use the same revision.
	RevisionDecisionIDHeaderKey = "x-llm-d-revision-decision-id"
	// PeerTopologyHeaderKey carries the prefill endpoint's encoded topology
	// from the prefill EPP's response, through the coordinator, to the decode
	// EPP's request, for topology-affinity-filter and topology-affinity-scorer
	// running in coordinator deployments.
	PeerTopologyHeaderKey = "x-llm-d-peer-topology"

	// DefaultFairnessID is the default fairness ID used when no ID is provided in the request.
	// This ensures that requests without explicit fairness identifiers are still grouped and managed by the Flow Control
	// system.
	DefaultFairnessID = "default-flow"

	// DefaultTenantID prefixes a composed fairness ID for a request that
	// carries a derived program root but no fairness-ID header. Such traffic
	// shares one synthetic tenant, so its programs stay addressable as a
	// group. A gateway stamping this exact tenant string shares the prefix.
	DefaultTenantID = "default-tenant"

	// HeaderContentType names the HTTP header that carries the media type of a body.
	HeaderContentType = "content-type"
	// ContentTypeJSON is the media type of JSON bodies.
	ContentTypeJSON = "application/json"
	// MediaTypeEventStream is the SSE media type without optional parameters.
	MediaTypeEventStream = "text/event-stream"
	// HeaderContentTypeOptions names the HTTP header that controls MIME sniffing of a body.
	HeaderContentTypeOptions = "x-content-type-options"
	// ContentTypeOptionsNoSniff disables MIME sniffing; http.Error sets it on every body it writes.
	ContentTypeOptionsNoSniff = "nosniff"
	// PseudoHeaderStatus names the pseudo-header Envoy reports a response's
	// status in. HeaderStatus is the bare name, which the EPP also accepts.
	PseudoHeaderStatus = ":status"
	HeaderStatus       = "status"

	FieldKVTransferParams = "kv_transfer_params"
	// FieldPromptTokenIDs is the kv_transfer_params field carrying
	// pre-tokenized prompt token IDs.
	FieldPromptTokenIDs       = "prompt_token_ids"
	FieldECTransferParams     = "ec_transfer_params"
	FieldECDescriptorMetadata = "metadata" // ec_transfer_params descriptor field: the item's placeholder metadata
	FieldMaxTokens            = "max_tokens"
	FieldMaxCompletionTokens  = "max_completion_tokens"
	FieldMaxOutputTokens      = "max_output_tokens" // Used by Responses API
	FieldMinTokens            = "min_tokens"
	FieldPrompt               = "prompt"
	FieldStream               = "stream"
	FieldStreamOptions        = "stream_options"
	FieldIncludeUsage         = "include_usage"          // stream_options field: emit a final usage-only frame
	FieldContinuousUsageStats = "continuous_usage_stats" // stream_options field: vLLM emits usage on every frame
	FieldEcho                 = "echo"
	FieldTruncatePromptTokens = "truncate_prompt_tokens" //#nosec G101 -- request field name, not a credential
	FieldTools                = "tools"
	FieldToolChoice           = "tool_choice"
	FieldResponseFormat       = "response_format"
	FieldSamplingParams       = "sampling_params"
	FieldN                    = "n"
	FieldDoRemotePrefill      = "do_remote_prefill"
	FieldDoRemoteDecode       = "do_remote_decode"
	FieldRemoteBlockIDs       = "remote_block_ids"
	FieldRemoteEngineID       = "remote_engine_id"
	FieldRemoteHost           = "remote_host"
	FieldRemotePort           = "remote_port"
	// OffloadingConnector P2P tier roles, carried inside kv_transfer_params. The
	// role is encoded by the nesting key, named for the remote party it
	// describes: remote_decoder on the prefill request, remote_prefiller on the
	// decode request, remote_kv_source for a cached-prefix pull. Each block
	// carries its own kv_request_id.
	FieldRemoteDecoder         = "remote_decoder"
	FieldRemotePrefiller       = "remote_prefiller"
	FieldRemoteKVSource        = "remote_kv_source"
	FieldKVRequestID           = "kv_request_id"
	FieldCacheHitThreshold     = "cache_hit_threshold"
	FieldContinueFinalMessage  = "continue_final_message"
	FieldAddGenerationPrompt   = "add_generation_prompt"
	FieldPreviousResponseID    = "previous_response_id"
	FieldConversation          = "conversation"
	FieldBackground            = "background"
	FieldInput                 = "input"
	FieldInstructions          = "instructions"
	FieldContent               = "content"
	FieldFileID                = "file_id"
	FieldMessages              = "messages"
	FieldModel                 = "model"
	FieldRole                  = "role"
	FieldText                  = "text"
	FieldType                  = "type"
	FieldURL                   = "url"
	FieldUUID                  = "uuid" // content part field vLLM keys a multimodal item's hash on
	FieldImageURL              = "image_url"
	FieldStore                 = "store"
	FieldMMProcessorKwargs     = "mm_processor_kwargs"
	FieldMediaIOKwargs         = "media_io_kwargs"
	FieldOutput                = "output"
	FieldFunctions             = "functions"
	FieldLogprobs              = "logprobs"
	FieldReturnTokenIDs        = "return_token_ids"
	FieldUseBeamSearch         = "use_beam_search"
	FieldReturnPromptText      = "return_prompt_text"
	FieldPromptLogprobTokenIDs = "prompt_logprob_token_ids" //#nosec G101 -- JSON field name, not a credential
	FieldPromptLogprobs        = "prompt_logprobs"
	FieldReasoningEffort       = "reasoning_effort"
	FieldChatTemplateKwargs    = "chat_template_kwargs"
	FieldEnableThinking        = "enable_thinking"
	FieldThinking              = "thinking" // Anthropic Messages thinking config

	// SGLang bootstrap coordination fields, carried inside kv_transfer_params.
	// The prefill pod echoes them back so the decode pod can open the bootstrap
	// channel to it.
	FieldBootstrapHost = "bootstrap_host"
	FieldBootstrapPort = "bootstrap_port"
	FieldBootstrapRoom = "bootstrap_room"
)

// Role values, the strings a message's FieldRole takes.
const (
	RoleAssistant = "assistant"
)

// Choice fields of a Chat Completions or Completions response, streamed or not.
// A non-streaming chat reply holds each choice's output under FieldMessage; a
// streamed chunk holds the increment under FieldDelta.
const (
	FieldChoices      = "choices"
	FieldIndex        = "index"
	FieldFinishReason = "finish_reason"
	FieldMessage      = "message"
	FieldDelta        = "delta"
)

// Usage fields in a response body. Chat Completions and Completions report
// prompt_tokens and completion_tokens with details under prompt_tokens_details;
// Responses and Conversations report input_tokens and output_tokens with details
// under input_tokens_details. total_tokens is named the same in every API.
// Messages splits the prompt into the part served from cache and the rest, so
// input_tokens alone excludes cache_read_input_tokens and
// cache_creation_input_tokens.
const (
	FieldUsage                    = "usage"
	FieldPromptTokens             = "prompt_tokens"
	FieldCompletionTokens         = "completion_tokens"
	FieldInputTokens              = "input_tokens"
	FieldOutputTokens             = "output_tokens"
	FieldTotalTokens              = "total_tokens"
	FieldPromptTokensDetails      = "prompt_tokens_details"       //#nosec G101 -- JSON field name, not a credential
	FieldInputTokensDetails       = "input_tokens_details"        //#nosec G101 -- JSON field name, not a credential
	FieldCachedTokens             = "cached_tokens"               //#nosec G101 -- JSON field name, not a credential
	FieldCacheReadInputTokens     = "cache_read_input_tokens"     //#nosec G101 -- JSON field name, not a credential
	FieldCacheCreationInputTokens = "cache_creation_input_tokens" //#nosec G101 -- JSON field name, not a credential
)

// FieldResponse wraps the response object in Responses stream events, which is
// where those events carry usage; the Messages message_start event wraps its
// object under FieldMessage the same way.
const FieldResponse = "response"

// Server-sent event framing for streamed responses. SGLang sends the bare
// marker as an event payload, the OpenAI APIs send the framed line.
const (
	SSEDataPrefix = "data: "
	SSEDoneMarker = "[DONE]"
	SSEDone       = SSEDataPrefix + SSEDoneMarker
)

// ToolChoiceNone is the tool_choice value that forbids the model from calling a tool.
const ToolChoiceNone = "none"

// Content part types, the values a content part's FieldType takes. A
// chat-completions *_url part nests its URL and options under one object keyed
// by the part type, so PartTypeImageURL and FieldImageURL hold the same string
// in different roles; a Responses input_image instead carries a bare image_url
// string with its options as siblings.
//
// The Responses input content union is input_text / input_image / input_file, so
// a Responses request carrying one of the others is refused by the model server.
// PartModality does not pre-empt that: it answers by part type alone, so such a
// part is still collected and primed. FieldFileID on an input_image, input_file
// or computer_screenshot names a Files API upload the serving engine must fetch.
const (
	PartTypeText               = "text"
	PartTypeInputText          = "input_text"
	PartTypeOutputText         = "output_text"
	PartTypeRefusal            = "refusal"
	PartTypeImageURL           = "image_url"
	PartTypeAudioURL           = "audio_url"
	PartTypeVideoURL           = "video_url"
	PartTypeInputAudio         = "input_audio"
	PartTypeInputImage         = "input_image"
	PartTypeInputFile          = "input_file"
	PartTypeComputerScreenshot = "computer_screenshot"
)

// ResponsesItemTypeMessage is the Responses input item type of a role-tagged
// message. An item that omits FieldType is a message too.
const ResponsesItemTypeMessage = "message"

// Anthropic Messages block types. An image block holds its image in a
// FieldSource object; a tool_result block holds the tool's output under
// FieldContent, as a string or as an array of blocks.
const (
	BlockTypeImage      = "image"
	BlockTypeToolResult = "tool_result"
)

// DefaultImageMediaType is vLLM's fallback when an Anthropic base64 image
// source omits media_type.
const DefaultImageMediaType = "image/jpeg"

// Fields of an Anthropic Messages image source. A source of type
// ImageSourceTypeURL names its image by FieldURL; any other source carries
// base64 FieldData and an optional FieldMediaType.
const (
	FieldSource        = "source"
	FieldData          = "data"
	FieldMediaType     = "media_type"
	ImageSourceTypeURL = "url"
)

// Message roles a content walk tells apart.
const (
	RoleUser   = "user"
	RoleSystem = "system"
)
