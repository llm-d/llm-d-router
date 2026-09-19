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

// Package outlenbucket provides a RequestHeaderProcessor plugin that predicts the
// output-length bin for a request from request-time signals
// (enable_thinking, thinking_budget/reasoning_budget, has_tools, tool_choice,
// response_format, continue_final_message, max_output_tokens) and
// publishes it as a request attribute. Downstream consumers -- the in-flight
// token estimator today, and flow-control queue ordering / KV-pressure gating
// in the future -- read it via scheduling.ReadRequestAttribute to make
// output-length-aware decisions.
package outlenbucket

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// AttributeKey is the request-attribute key under which this plugin publishes
// the predicted output-length bin. Downstream consumers read it via
// scheduling.ReadRequestAttribute[Bucket]. It is a DataKey rather than a plain
// string because #2190 keyed the per-request attribute store by DataKey.
var AttributeKey = plugin.NewDataKey("outlen-bucket", "")

const (
	// PluginType is the plugin type name used in the EPP config.
	PluginType = "outlen-bucket"

	// longBudgetThresholdTokens is the thinking_budget above which a request is
	// classified LONG even when enable_thinking is not explicitly set.
	longBudgetThresholdTokens = 4000
	// shortMaxOutputTokens is the max_output_tokens below which a request is
	// classified SHORT on the strength of an explicit client cap alone.
	shortMaxOutputTokens = 500
	// longFloorTokens is the lower edge of the LONG bin. A client cap
	// (max_output_tokens) strictly below this makes a >=2000-token generation
	// physically impossible, so it vetoes a tentative LONG classification
	// (the max_output_tokens "bin ceiling" — purely restrictive, high precision).
	longFloorTokens = 2000
)

// Bucket is the predicted output-length category for a request,
// derived from request-time signals before any tokens are generated.
type Bucket int8

const (
	// Unknown means no reliable signal was found; consumers apply their own
	// neutral middle estimate. It is the zero value, so a missing attribute
	// reads as Unknown.
	Unknown Bucket = iota
	// Short predicts < 500 output tokens (e.g. tool-call JSON responses).
	Short
	// Long predicts >= 2000 output tokens (e.g. reasoning chains).
	Long
)

func (b Bucket) String() string {
	switch b {
	case Short:
		return "SHORT"
	case Long:
		return "LONG"
	default:
		return "UNKNOWN"
	}
}

// EstimateOutlen predicts the output-length bin using request-time signals.
// Precedence (first match wins): LONG pushers (enable_thinking,
// DeepSeek thinking.type="enabled", thinking_budget/reasoning_budget>4000)
// are checked first; SHORT pushers (tool_choice, has_tools,
// continue_final_message, response_format, max_output_tokens<500)
// follow; everything else is UNKNOWN. A max_output_tokens cap below the LONG
// floor downgrades LONG last. Input length is intentionally excluded -- it has
// no correlation with output length.
func EstimateOutlen(body *fwkrh.InferenceRequestBody) Bucket {
	if body == nil {
		return Unknown
	}

	var enableThinking *bool
	var thinkingBudget *int64
	hasTools := false
	continueFinalMessage := false
	// has_tools, continue_final_message, enable_thinking, thinking_budget, and the
	// vendor-specific DeepSeek/Nemotron signals are only carried on the chat-completions
	// shape (vLLM populates them from the client's chat_template_kwargs / extra_body).
	if body.ChatCompletions != nil {
		hasTools = len(body.ChatCompletions.Tools) > 0
		continueFinalMessage = body.ChatCompletions.ContinueFinalMessage
		kwArgs := body.ChatCompletions.ChatTemplateKWArgs
		enableThinking = boolPtrFromAny(kwArgs["enable_thinking"])
		// DeepSeek V4 activation: extra_body={"thinking":{"type":"enabled"|"disabled"}}.
		if enableThinking == nil {
			if m, ok := kwArgs["thinking"].(map[string]any); ok {
				switch stringFromAny(m["type"]) {
				case "enabled":
					t := true
					enableThinking = &t
				case "disabled":
					f := false
					enableThinking = &f
				}
			}
		}
		thinkingBudget = int64PtrFromAny(kwArgs["thinking_budget"])
		if thinkingBudget == nil {
			// Nemotron uses reasoning_budget as the budget key.
			thinkingBudget = int64PtrFromAny(kwArgs["reasoning_budget"])
		}
	}

	// tool_choice and response_format are OpenAI top-level body fields, not typed
	// on the request — read them from the raw payload map.
	var toolChoice, responseFormat string
	if payload, ok := payloadMap(body); ok {
		toolChoice = toolChoiceKind(payload["tool_choice"])
		responseFormat = responseFormatType(payload["response_format"])
	}

	bucket := classifyOutlen(classifyInput{
		enableThinking:       enableThinking,
		thinkingBudget:       thinkingBudget,
		hasTools:             hasTools,
		continueFinalMessage: continueFinalMessage,
		toolChoice:           toolChoice,
		responseFormat:       responseFormat,
		maxOutputTokens:      body.MaxOutputTokens,
	})

	// Bin ceiling (always last): a hard client cap below the LONG floor makes a
	// LONG generation physically impossible, so downgrade. Purely restrictive.
	return applyMaxOutputCeiling(bucket, body.MaxOutputTokens)
}

// classifyInput carries the request-time signals for the output-length classifier.
type classifyInput struct {
	enableThinking       *bool
	thinkingBudget       *int64
	hasTools             bool
	continueFinalMessage bool
	toolChoice           string
	responseFormat       string
	maxOutputTokens      *int64
}

// classifyOutlen applies the precedence cascade documented on EstimateOutlen.
func classifyOutlen(in classifyInput) Bucket {
	thinking := in.enableThinking != nil && *in.enableThinking

	// --- LONG pushers (checked first; over-calling LONG is the cheap error) ---

	// Thinking mode -> always long (reasoning chains, measured p50 = 3,848-16,530 tokens).
	if thinking {
		return Long
	}
	// Large thinking/reasoning budget, only when enable_thinking is not explicitly
	// set -> treat as LONG. An explicit enable_thinking=false is the stronger signal
	// and is respected: the request falls through rather than being forced to LONG.
	if in.enableThinking == nil && in.thinkingBudget != nil && *in.thinkingBudget > longBudgetThresholdTokens {
		return Long
	}

	// --- SHORT pushers (must be high-precision) ---

	// Forced tool call -> short tool-call JSON.
	if in.toolChoice == "required" || in.toolChoice == "named" {
		return Short
	}
	// Tools without thinking -> short tool-call JSON (measured p50 = 41 tokens, 100% precision).
	// Guard: enable_thinking must be explicitly false or absent (Nemotron ARC-AGI proves
	// has_tools alone is NOT a SHORT signal under thinking); and tool_choice="none" vetoes it
	// (tools are advertised but the model is told not to call them, so the SHORT premise fails).
	if in.hasTools && !thinking && in.toolChoice != "none" {
		return Short
	}
	// Continuing/completing a partially-written assistant turn -> short by construction.
	if in.continueFinalMessage {
		return Short
	}
	// Structured output (JSON) -> bounded, tends short.
	if in.responseFormat == "json_object" || in.responseFormat == "json_schema" {
		return Short
	}
	// Explicit short cap set by the client -> treat as short.
	if in.maxOutputTokens != nil && *in.maxOutputTokens > 0 && *in.maxOutputTokens < shortMaxOutputTokens {
		return Short
	}

	return Unknown
}

// applyMaxOutputCeiling downgrades a tentative LONG bin when the client's
// max_output_tokens cap makes a LONG (>= longFloorTokens) generation impossible.
// It only ever downgrades LONG, so it cannot lower precision on SHORT/UNKNOWN.
func applyMaxOutputCeiling(bucket Bucket, maxOutputTokens *int64) Bucket {
	if bucket != Long || maxOutputTokens == nil || *maxOutputTokens <= 0 {
		return bucket
	}
	if *maxOutputTokens < shortMaxOutputTokens {
		return Short
	}
	if *maxOutputTokens < longFloorTokens {
		return Unknown
	}
	return bucket
}

// PluginFactory is the factory function for the outlen-bucket plugin.
func PluginFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
	}, nil
}

// compile-time interface assertion
var _ requestcontrol.RequestHeaderProcessor = &Plugin{}

// Plugin predicts the output-length bin for a request and stores it as a request
// attribute for output-length-aware scheduling.
type Plugin struct {
	typedName plugin.TypedName
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// RequestHeader runs after the request body is parsed and attached, but before
// admission control. It classifies the request into an output-length bin and
// publishes the result as a request attribute.
func (p *Plugin) RequestHeader(_ context.Context, request *scheduling.InferenceRequest) error {
	if request == nil || request.Body == nil {
		return nil
	}
	request.PutAttribute(AttributeKey, EstimateOutlen(request.Body))
	return nil
}

// payloadMap returns the request's raw JSON payload as a map, if it was parsed
// into one. tool_choice and response_format are not typed on the request body,
// so they are read here.
func payloadMap(body *fwkrh.InferenceRequestBody) (fwkrh.PayloadMap, bool) {
	if body == nil || body.Payload == nil {
		return nil, false
	}
	return body.Payload.AsMap()
}

// stringFromAny returns v as a string when it is one, else "" ("not set").
func stringFromAny(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// toolChoiceKind normalizes the OpenAI tool_choice field. It is either a string
// ("none" | "auto" | "required") or an object forcing a specific function
// ({"type":"function","function":{"name":...}}), which we report as "named".
// Anything else yields "" ("not set").
func toolChoiceKind(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		// A specific tool is forced -> a short tool-call JSON response.
		return "named"
	}
	return ""
}

// responseFormatType returns the response_format.type ("text" | "json_object" |
// "json_schema") from the OpenAI response_format object, or "" if absent.
func responseFormatType(v any) string {
	if m, ok := v.(map[string]any); ok {
		return stringFromAny(m["type"])
	}
	return ""
}

// toJSONNumber normalizes float64 (from json.Unmarshal without UseNumber) and
// json.Number (from a decoder with UseNumber) into a single json.Number,
// eliminating duplicate numeric-type handling across the coercion helpers.
func toJSONNumber(v any) (json.Number, bool) {
	switch t := v.(type) {
	case json.Number:
		return t, true
	case float64:
		return json.Number(strconv.FormatFloat(t, 'f', -1, 64)), true
	}
	return "", false
}

// boolPtrFromAny coerces a JSON-decoded value into a *bool. It accepts a native
// bool, the strings "true"/"false"/"1"/"0", and numeric 0/1 (float64 or
// json.Number). Any other value yields nil ("not set").
func boolPtrFromAny(v any) *bool {
	switch t := v.(type) {
	case bool:
		return &t
	case string:
		if b, err := strconv.ParseBool(t); err == nil {
			return &b
		}
	}
	if n, ok := toJSONNumber(v); ok {
		if f, err := n.Float64(); err == nil {
			b := f != 0
			return &b
		}
	}
	return nil
}

// int64PtrFromAny coerces a JSON-decoded value into a *int64. It accepts
// float64, json.Number, an integer string, and native int/int64. Any other
// value (or a non-integral / unparsable one) yields nil ("not set").
func int64PtrFromAny(v any) *int64 {
	switch t := v.(type) {
	case int:
		i := int64(t)
		return &i
	case int64:
		return &t
	case string:
		if i, err := strconv.ParseInt(t, 10, 64); err == nil {
			return &i
		}
	}
	if n, ok := toJSONNumber(v); ok {
		if i, err := n.Int64(); err == nil {
			return &i
		}
	}
	return nil
}
