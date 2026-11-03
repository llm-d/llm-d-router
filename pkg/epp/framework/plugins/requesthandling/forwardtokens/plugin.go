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

// Package forwardtokens forwards the token IDs the router already rendered to the model server.
package forwardtokens

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	commonrequest "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

const (
	// PluginType is the registered type name of the token forwarding plugin.
	PluginType = "forward-prompt-token-ids"

	fieldReturnPromptText = "return_prompt_text"
	fieldSuffix           = "suffix"
)

var (
	_ fwkrh.PreRequest      = &Plugin{}
	_ plugin.ConsumerPlugin = &Plugin{}
)

// Config configures token forwarding.
type Config struct {
	// MinTokens skips prompts shorter than this; a short prompt costs the server little to
	// re-tokenize, and the forwarded list grows the body. 0 forwards every prompt.
	MinTokens int `json:"minTokens,omitempty"`
	// TrustCallerTokenIDs keeps a caller-supplied kv_transfer_params.prompt_token_ids
	// instead of replacing it with the EPP's own render. Only enable it when
	// every caller is a trusted coordinator following vLLM #48145's decode-leg
	// pattern: an end user supplying the field makes the server serve those IDs
	// instead of rendering messages, bypassing every messages-level gateway
	// policy (injected system prompts, guardrails, prompt-token limits).
	// Defaults to false: the caller's list is removed before the token-producer
	// renders, and the EPP's own render is forwarded.
	TrustCallerTokenIDs bool `json:"trustCallerTokenIDs,omitempty"`
}

// Plugin forwards the token-producer's IDs to the model server so it does not render the prompt again:
// in kv_transfer_params.prompt_token_ids for the chat-shaped APIs, and as the prompt itself for
// /v1/completions.
type Plugin struct {
	typedName plugin.TypedName
	// trustCallerTokenIDs comes from Config.TrustCallerTokenIDs.
	trustCallerTokenIDs bool
	minTokens           int
}

// PluginFactory constructs a token forwarding plugin.
func PluginFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	cfg := Config{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", PluginType, err)
		}
	}
	return New(name, cfg)
}

// New constructs a token forwarding plugin from validated configuration.
func New(name string, cfg Config) (*Plugin, error) {
	if cfg.MinTokens < 0 {
		return nil, fmt.Errorf("%s: minTokens must be >= 0, got %d", PluginType, cfg.MinTokens)
	}
	return &Plugin{
		typedName:           plugin.TypedName{Type: PluginType, Name: name},
		minTokens:           cfg.MinTokens,
		trustCallerTokenIDs: cfg.TrustCallerTokenIDs,
	}, nil
}

// TypedName returns the plugin's registered type and configured name.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Consumes requires a token-producer with the vllm backend. The estimate backend's IDs are packed
// bytes, not engine tokens, so forwarding them would serve a different prompt; startup fails instead.
func (p *Plugin) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{
			tokenproducer.TokenizedPromptDataKey: plugin.ConsumedData{
				Type:    fwkrequest.TokenizedRequest{},
				Matches: plugin.MatchKeyValues(map[string]string{"backend": tokenproducer.BackendVLLM}),
			},
		},
	}
}

// PreRequest adds the rendered token IDs to a single-prompt JSON request. For chat-shaped APIs other
// kv_transfer_params fields are preserved, and an ID list the client sent is kept.
func (p *Plugin) PreRequest(ctx context.Context, request *scheduling.InferenceRequest,
	_ *scheduling.SchedulingResult) error {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return nil
	}
	body := request.Body
	if len(body.TokenizedRequest.Prompts) != 1 {
		return nil
	}
	prompt := body.TokenizedRequest.Prompts[0]
	if len(prompt.TokenIDs) == 0 || len(prompt.TokenIDs) < p.minTokens {
		return nil
	}
	// The token-producer's legacy /v1/messages and /v1/responses translations
	// do not guarantee token parity with what vLLM renders from the client's
	// request (they drop fields like thinking, reasoning and tool_choice), so
	// their IDs must not be forwarded as authoritative.
	if body.TokenizedRequest.LegacyTranslated {
		return nil
	}
	// An untrusted caller-supplied kv_transfer_params.prompt_token_ids taints
	// the render: vLLM's render returns the caller's list instead of rendering
	// messages, so the token-producer's IDs are the caller's IDs. Strip the
	// field and skip forwarding; the engine renders from messages, which is
	// the behavior without this plugin. Trusted coordinators (vLLM #48145's
	// decode-leg pattern) set trustCallerTokenIDs to keep today's behavior.
	if !p.trustCallerTokenIDs {
		if payload, isMap := body.Payload.(fwkrequest.PayloadMap); isMap {
			if stripped := stripCallerTokenIDs(payload); stripped {
				body.MutatePayloadMap(func(m fwkrequest.PayloadMap) {
					m[commonrequest.FieldKVTransferParams] = payload[commonrequest.FieldKVTransferParams]
				})
				return nil
			}
		}
	}
	// Multimodal prompts are skipped: vLLM only reuses forwarded IDs for
	// non-text content when the processed multimodal features are forwarded
	// alongside them, which is deferred to a follow-up PR (pending
	// vllm-project/vllm#64721). Forwarding IDs alone would make the engine
	// serve a prompt without the media.
	if prompt.MultiModalFeatures != nil {
		return nil
	}
	payload, ok := body.Payload.(fwkrequest.PayloadMap)
	if !ok {
		logger := log.FromContext(ctx).WithName(p.typedName.String()).V(logging.DEBUG)
		logger.Info("skipping token forwarding: request payload is not mutable JSON",
			"requestID", request.RequestID)
		return nil
	}
	switch {
	case body.Completions != nil:
		p.forwardCompletions(body, payload, prompt.TokenIDs)
	case isChatShaped(body):
		// A stateful Responses request is rendered by the worker from history it stores, which the
		// router's render does not include.
		if body.Responses != nil && rejectStatefulResponses(body.Responses, payload) {
			return nil
		}
		p.forwardChat(ctx, request, payload, prompt.TokenIDs)
	}
	return nil
}

// forwardChat writes the IDs into kv_transfer_params.prompt_token_ids, which vLLM's chat path
// (also serving Messages and Responses) reads to skip templating and tokenization.
func (p *Plugin) forwardChat(ctx context.Context, request *scheduling.InferenceRequest,
	payload fwkrequest.PayloadMap, ids []uint32) {
	// echo and return_prompt_text need vLLM to render the messages itself:
	// with return_prompt_text the reuse branch skips templating, so the
	// response's prompt_text would silently be null; with echo vLLM discards
	// the forwarded IDs and re-renders anyway, so forwarding is pure overhead.
	if hasField(payload, commonrequest.FieldEcho) || hasField(payload, fieldReturnPromptText) {
		return
	}
	params, clientHasIDs, ok := kvTransferParams(payload)
	if !ok {
		log.FromContext(ctx).WithName(p.typedName.String()).V(logging.DEBUG).Info(
			"skipping token forwarding: kv_transfer_params is not an object", "requestID", request.RequestID)
		return
	}
	if clientHasIDs {
		return
	}
	params[commonrequest.FieldPromptTokenIDs] = ids
	request.Body.MutatePayloadMap(func(m fwkrequest.PayloadMap) {
		m[commonrequest.FieldKVTransferParams] = params
	})
}

// forwardCompletions replaces a text prompt with its IDs; vLLM's completions path takes a token
// array as the prompt without tokenizing it. It leaves alone prompts that are already token IDs,
// and echo and suffix requests, whose responses or rendering depend on the prompt text.
func (p *Plugin) forwardCompletions(body *fwkrequest.InferenceRequestBody,
	payload fwkrequest.PayloadMap, ids []uint32) {
	if len(body.Completions.Prompt.TokenIDs) > 0 || hasField(payload, commonrequest.FieldEcho) || hasField(payload, fieldSuffix) {
		return
	}
	body.MutatePayloadMap(func(m fwkrequest.PayloadMap) {
		m[commonrequest.FieldPrompt] = ids
	})
}

// hasField reports whether payload sets field to a value other than null, false or an empty string.
func hasField(payload fwkrequest.PayloadMap, field string) bool {
	v, ok := payload[field]
	if !ok {
		return false
	}
	if raw, isRaw := v.(json.RawMessage); isRaw {
		if err := parserutil.Unmarshal(raw, &v); err != nil {
			return true
		}
	}
	switch t := v.(type) {
	case nil:
		return false
	case bool:
		return t
	case string:
		return t != ""
	}
	return true
}

// rejectStatefulResponses reports whether the typed Responses request references
// server-side state the router's render does not include, or requests a
// background mode the forwarded IDs cannot carry. previous_response_id and
// conversation come from the parser-decoded typed body; background is a
// top-level parameter checked from the payload (one field lookup, no input
// decode); file references require walking the parser-decoded input array.
func rejectStatefulResponses(responses *fwkrequest.ResponsesRequest, payload fwkrequest.PayloadMap) bool {
	if responses.PreviousResponseID != "" || responses.Conversation != nil {
		return true
	}
	// Background is a top-level Responses parameter, never an input-item key.
	if bg, exists := payload[commonrequest.FieldBackground]; exists && bg != nil {
		return true
	}
	// File references live at arbitrary depth in input items (input[].content[],
	// function_call_output.output[], computer_call_output.output). Walk the
	// parser-decoded input, not the raw payload bytes.
	if inputList, ok := responses.Input.([]any); ok {
		for _, item := range inputList {
			if m, ok := item.(map[string]any); ok {
				if hasFileReference(m) {
					return true
				}
			}
		}
	}
	return false
}

// hasFileReference reports whether a decoded input item references a server-side
// file at any nesting depth, matching RejectStatefulResponsesFields' walk.
func hasFileReference(item map[string]any) bool {
	for key, val := range item {
		if key == commonrequest.FieldFileID {
			return true
		}
		if nested, ok := val.(map[string]any); ok {
			if hasFileReference(nested) {
				return true
			}
		}
		if list, ok := val.([]any); ok {
			for _, elem := range list {
				if m, ok := elem.(map[string]any); ok {
					if hasFileReference(m) {
						return true
					}
				}
			}
		}
	}
	return false
}

// isChatShaped reports whether vLLM renders the request through its chat path, the only path
// that honors kv_transfer_params.prompt_token_ids. Known limitation: vLLM's Harmony
// (GPT-OSS) renderer has no reuse branch. /v1/responses requests rendered through Harmony
// ignore the forwarded IDs and re-render server-side, so forwarding them is wasted bytes but
// not incorrect (tracked upstream as a vLLM-side gap).
func isChatShaped(body *fwkrequest.InferenceRequestBody) bool {
	return body.ChatCompletions != nil || body.Messages != nil || body.Responses != nil
}

// stripCallerTokenIDs removes kv_transfer_params.prompt_token_ids from the
// payload in both wire forms (decoded map and json.RawMessage), returning
// whether a strip happened. The raw form is what the parsers actually produce
// for chat-shaped APIs, so both must be handled.
func stripCallerTokenIDs(payload fwkrequest.PayloadMap) bool {
	value, exists := payload[commonrequest.FieldKVTransferParams]
	if !exists || value == nil {
		return false
	}
	switch v := value.(type) {
	case map[string]any:
		if ids, has := v[commonrequest.FieldPromptTokenIDs]; has && ids != nil {
			delete(v, commonrequest.FieldPromptTokenIDs)
			return true
		}
	case json.RawMessage:
		var rawParams map[string]json.RawMessage
		if err := parserutil.Unmarshal(v, &rawParams); err != nil || rawParams == nil {
			return false
		}
		if raw, has := rawParams[commonrequest.FieldPromptTokenIDs]; has &&
			!bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			delete(rawParams, commonrequest.FieldPromptTokenIDs)
			if cleaned, err := json.Marshal(rawParams); err == nil {
				payload[commonrequest.FieldKVTransferParams] = json.RawMessage(cleaned)
				return true
			}
		}
	}
	return false
}

// kvTransferParams returns the request's kv_transfer_params as a map, or a
// new one when absent. clientHasIDs reports whether the client already
// supplied prompt_token_ids (an explicit null counts as absent): when true,
// the raw bytes are left untouched and the returned map must not be written
// to or forwarded.
func kvTransferParams(payload fwkrequest.PayloadMap) (params map[string]any, clientHasIDs, ok bool) {
	value, exists := payload[commonrequest.FieldKVTransferParams]
	if !exists || value == nil {
		return map[string]any{}, false, true
	}
	switch v := value.(type) {
	case map[string]any:
		if ids, exists := v[commonrequest.FieldPromptTokenIDs]; exists && ids != nil {
			return v, true, true
		}
		return v, false, true
	case json.RawMessage:
		// Fast path: when the client already supplied prompt_token_ids, peek
		// with json.RawMessage values instead of materializing every token as
		// a boxed interface{}. forwardChat only needs to know the key exists
		// before leaving the request untouched.
		var rawParams map[string]json.RawMessage
		if err := parserutil.Unmarshal(v, &rawParams); err != nil || rawParams == nil {
			return nil, false, false
		}
		if raw, exists := rawParams[commonrequest.FieldPromptTokenIDs]; exists &&
			!bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return nil, true, true
		}
		var params map[string]any
		if err := parserutil.Unmarshal(v, &params); err != nil || params == nil {
			return nil, false, false
		}
		return params, false, true
	}
	return nil, false, false
}
