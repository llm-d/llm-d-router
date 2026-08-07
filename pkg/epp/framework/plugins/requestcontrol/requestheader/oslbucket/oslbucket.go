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

// Package oslbucket provides a RequestHeaderProcessor plugin that predicts the
// output-sequence-length (OSL) bin for a request from request-time signals
// (enable_thinking, has_tools, thinking_budget) and publishes it as a request
// attribute. Downstream consumers — the in-flight token estimator today, and
// flow-control queue ordering / KV-pressure gating in the future — read it via
// scheduling.ReadRequestAttribute to make output-length-aware decisions.
package oslbucket

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// OSLBucketKey is the request-attribute key under which this plugin
	// publishes the predicted OSL bin. Downstream consumers read it via
	// scheduling.ReadRequestAttribute[OSLBucket].
	OSLBucketKey = "osl-bucket"
	// PluginType is the plugin type name used in the EPP config.
	PluginType = "osl-bucket"

	// longBudgetThresholdTokens is the thinking_budget above which a request is
	// classified LONG even when enable_thinking is not explicitly set.
	longBudgetThresholdTokens = 4000
	// shortMaxOutputTokens is the max_output_tokens below which a request is
	// classified SHORT on the strength of an explicit client cap alone.
	shortMaxOutputTokens = 500
)

// OSLBucket is the predicted output-sequence-length category for a request,
// derived from request-time signals before any tokens are generated.
type OSLBucket int8

const (
	// OSLBucketUnknown means no reliable signal was found; consumers use their
	// own fallback (e.g. the ratio-based token estimate). It is the zero value,
	// so a missing attribute reads as UNKNOWN.
	OSLBucketUnknown OSLBucket = iota
	// OSLBucketShort predicts < 500 output tokens (e.g. tool-call JSON responses).
	OSLBucketShort
	// OSLBucketLong predicts >= 2000 output tokens (e.g. reasoning chains).
	OSLBucketLong
)

func (b OSLBucket) String() string {
	switch b {
	case OSLBucketShort:
		return "SHORT"
	case OSLBucketLong:
		return "LONG"
	default:
		return "UNKNOWN"
	}
}

// EstimateOSLBucket predicts the output-length bin using request-time signals.
//
// Validated on 22,575 samples across 5 real-world LLM datasets
// (KIMI-K2.5, rStar-Coder, xlam-function-calling-60k, WildChat-4.8M,
// Nemotron-SFT-ARC-AGI-v1):
//
//   - enable_thinking=true  -> LONG:  90.1% precision, 79.7% recall
//   - enable_thinking=false AND has_tools=true -> SHORT: 100.0% precision, 56.9% recall
//
// ISL is intentionally excluded: no correlation with OSL, adds noise.
// See research-directions/osl-aware-scheduling/README.md for full analysis.
func EstimateOSLBucket(body *fwkrh.InferenceRequestBody) OSLBucket {
	if body == nil {
		return OSLBucketUnknown
	}

	var enableThinking *bool
	var thinkingBudget *int64
	hasTools := false
	if body.ChatCompletions != nil {
		hasTools = len(body.ChatCompletions.Tools) > 0
		kwArgs := body.ChatCompletions.ChatTemplateKWArgs
		if v, ok := kwArgs["enable_thinking"]; ok {
			enableThinking = boolPtrFromAny(v)
		}
		if v, ok := kwArgs["thinking_budget"]; ok {
			thinkingBudget = int64PtrFromAny(v)
		}
	}

	// Thinking mode -> always long (reasoning chains, measured p50 = 3,848-16,530 tokens).
	if enableThinking != nil && *enableThinking {
		return OSLBucketLong
	}

	// Large thinking budget without explicit enable_thinking -> treat as LONG.
	if thinkingBudget != nil && *thinkingBudget > longBudgetThresholdTokens {
		return OSLBucketLong
	}

	// Tools without thinking -> short tool-call JSON (measured p50 = 41 tokens, 100% precision).
	// Guard: enable_thinking must be explicitly false or absent. Nemotron ARC-AGI proves that
	// has_tools=true alone is NOT a SHORT signal when enable_thinking is also true.
	if hasTools && (enableThinking == nil || !*enableThinking) {
		return OSLBucketShort
	}

	// Explicit short cap set by the client -> treat as short.
	if body.MaxOutputTokens != nil && *body.MaxOutputTokens > 0 && *body.MaxOutputTokens < shortMaxOutputTokens {
		return OSLBucketShort
	}

	return OSLBucketUnknown
}

// PluginFactory is the factory function for the OSL bucket plugin.
func PluginFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
	}, nil
}

// compile-time interface assertion
var _ requestcontrol.RequestHeaderProcessor = &Plugin{}

// Plugin predicts the OSL bin for a request and stores it as a request
// attribute for output-length-aware scheduling.
type Plugin struct {
	typedName plugin.TypedName
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// RequestHeader runs after the request body is parsed and attached, but before
// admission control. It classifies the request into an OSL bin and publishes
// the result as a request attribute.
func (p *Plugin) RequestHeader(_ context.Context, request *scheduling.InferenceRequest) error {
	if request == nil || request.Body == nil {
		return nil
	}
	request.PutAttribute(OSLBucketKey, EstimateOSLBucket(request.Body))
	return nil
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
	case float64:
		b := t != 0
		return &b
	case json.Number:
		if f, err := t.Float64(); err == nil {
			b := f != 0
			return &b
		}
	}
	return nil
}

// int64PtrFromAny coerces a JSON-decoded value into a *int64. It accepts
// float64, json.Number, an integer string, and native int/int64. Any other
// value (or a non-integral / unparseable one) yields nil ("not set").
func int64PtrFromAny(v any) *int64 {
	switch t := v.(type) {
	case float64:
		i := int64(t)
		return &i
	case json.Number:
		if i, err := t.Int64(); err == nil {
			return &i
		}
	case string:
		if i, err := strconv.ParseInt(t, 10, 64); err == nil {
			return &i
		}
	case int:
		i := int64(t)
		return &i
	case int64:
		return &t
	}
	return nil
}
