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
// (enable_thinking, has_tools, thinking_budget) and publishes it as a request
// attribute. Downstream consumers -- the in-flight token estimator today, and
// flow-control queue ordering / KV-pressure gating in the future -- read it via
// scheduling.ReadRequestAttribute to make output-length-aware decisions.
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

// AttributeKey is the request-attribute key under which this plugin
// publishes the predicted output-length bin. Downstream consumers read it via
// scheduling.ReadRequestAttribute[Bucket].
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

// EstimateOutlen predicts the output-length bin using request-time signals
// (enable_thinking, thinking_budget, has_tools, max_output_tokens).
func EstimateOutlen(body *fwkrh.InferenceRequestBody) Bucket {
	if body == nil {
		return Unknown
	}

	var enableThinking *bool
	var thinkingBudget *int64
	hasTools := requestHasTools(body)

	// enable_thinking / thinking_budget are only carried in chat-completions
	// chat_template_kwargs (vLLM populates them from the client's extra_body); the
	// Claude messages and OpenAI responses shapes do not surface these signals.
	if body.ChatCompletions != nil {
		kwArgs := body.ChatCompletions.ChatTemplateKWArgs
		if v, ok := kwArgs["enable_thinking"]; ok {
			enableThinking = boolPtrFromAny(v)
		}
		if v, ok := kwArgs["thinking_budget"]; ok {
			thinkingBudget = int64PtrFromAny(v)
		}
	}

	// Thinking mode -> always long (reasoning chains).
	if enableThinking != nil && *enableThinking {
		return Long
	}

	// Large thinking budget, only when enable_thinking is not explicitly set ->
	// treat as LONG. An explicit enable_thinking=false is the stronger signal and
	// is respected: the request falls through rather than being forced to LONG.
	if enableThinking == nil && thinkingBudget != nil && *thinkingBudget > longBudgetThresholdTokens {
		return Long
	}

	// Tools without thinking -> short tool-call JSON. The enable_thinking guard
	// matters: has_tools=true alone is not a SHORT signal when thinking is also on.
	if hasTools && (enableThinking == nil || !*enableThinking) {
		return Short
	}

	// Explicit short cap set by the client -> treat as short.
	if body.MaxOutputTokens != nil && *body.MaxOutputTokens > 0 && *body.MaxOutputTokens < shortMaxOutputTokens {
		return Short
	}

	return Unknown
}

// requestHasTools reports whether the request carries tool definitions. Only the
// chat-completions shape is inspected: the thinking signals (enable_thinking,
// thinking_budget) that distinguish a SHORT tool-call from a LONG reasoning
// request are only surfaced there, so classifying tools on a shape whose thinking
// signals we cannot read would risk labeling a thinking request SHORT.
func requestHasTools(body *fwkrh.InferenceRequestBody) bool {
	return body.ChatCompletions != nil && len(body.ChatCompletions.Tools) > 0
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
