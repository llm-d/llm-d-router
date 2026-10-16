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

package outlensketch

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/cespare/xxhash/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// signalSig hashes the request-time signals into a stable 64-bit signature. Folded
// into the histogram key, it gives every distinct signal combination its own
// histogram, so the signals split the distribution without being averaged into the
// prompt geometry. The signals mirror the offline study's signature: model,
// reasoning level, tool presence/choice, continue-final-message, response format,
// output-token cap, thinking budget, turn count, and code presence (coarsened into
// buckets so the signature stays low-cardinality).
//
// The hash value need not match any other process -- the sketch builds its own
// histograms -- it need only map identical signal combinations to the same key.
func signalSig(body *fwkrh.InferenceRequestBody) uint64 {
	payload := payloadMap(body)
	s := fmt.Sprintf("%s|%d|%d|%s|%d|%s|%d|%d|%d|%d",
		body.Model,
		reasoningBucket(body, payload),
		boolToInt(hasTools(body)),
		toolChoiceKind(payload["tool_choice"]),
		boolToInt(hasContinueFinalMessage(body)),
		responseFormatKind(payload["response_format"]),
		maxOutputBucket(body.MaxOutputTokens),
		thinkingBudgetBucket(body),
		numMessagesBucket(body),
		boolToInt(hasCode(body)),
	)
	return xxhash.Sum64String(s)
}

// reasoningBucket coarsens the reasoning level: 2 high, 0 none/low, 1 unknown.
func reasoningBucket(body *fwkrh.InferenceRequestBody, payload map[string]any) int {
	switch asString(payload["reasoning_effort"]) {
	case "high":
		return 2
	case "low":
		return 0
	}
	if enabled, ok := enableThinking(body); ok {
		if enabled {
			return 2
		}
		return 0
	}
	return 1
}

// thinkingBudgetBucket coarsens the thinking/reasoning budget: 0 unset, 1 <=1024,
// 2 <=8000, 3 larger.
func thinkingBudgetBucket(body *fwkrh.InferenceRequestBody) int {
	tb, ok := thinkingBudget(body)
	if !ok || tb <= 0 {
		return 0
	}
	switch {
	case tb <= 1024:
		return 1
	case tb <= 8000:
		return 2
	default:
		return 3
	}
}

// numMessagesBucket coarsens the turn count: 0 single, 1 short dialog, 2 medium,
// 3 agentic.
func numMessagesBucket(body *fwkrh.InferenceRequestBody) int {
	n := 0
	if body.ChatCompletions != nil {
		n = len(body.ChatCompletions.Messages)
	}
	switch {
	case n <= 1:
		return 0
	case n <= 3:
		return 1
	case n <= 8:
		return 2
	default:
		return 3
	}
}

// maxOutputBucket coarsens the client output cap: 0 unset, 1 >=2000, 2 below.
func maxOutputBucket(maxOutputTokens *int64) int {
	if maxOutputTokens == nil {
		return 0
	}
	if *maxOutputTokens >= 2000 {
		return 1
	}
	return 2
}

// hasCode reports whether any chat message carries code markers, matching the
// study's has_code feature over the full prompt. The scan is a bounded substring
// search over text already in memory.
func hasCode(body *fwkrh.InferenceRequestBody) bool {
	if body.ChatCompletions == nil {
		return false
	}
	for _, m := range body.ChatCompletions.Messages {
		if containsAny(m.Content.PlainText(), "```", "def ", "{\n") {
			return true
		}
	}
	return false
}

func hasTools(body *fwkrh.InferenceRequestBody) bool {
	return body.ChatCompletions != nil && len(body.ChatCompletions.Tools) > 0
}

func hasContinueFinalMessage(body *fwkrh.InferenceRequestBody) bool {
	return body.ChatCompletions != nil && body.ChatCompletions.ContinueFinalMessage
}

// enableThinking resolves the thinking flag from chat_template_kwargs
// (enable_thinking) or the DeepSeek thinking.type object; ok is false when neither
// is present.
func enableThinking(body *fwkrh.InferenceRequestBody) (enabled, ok bool) {
	if body.ChatCompletions == nil {
		return false, false
	}
	kw := body.ChatCompletions.ChatTemplateKWArgs
	if b, present := asBool(kw["enable_thinking"]); present {
		return b, true
	}
	if m, isMap := kw["thinking"].(map[string]any); isMap {
		switch asString(m["type"]) {
		case "enabled":
			return true, true
		case "disabled":
			return false, true
		}
	}
	return false, false
}

// thinkingBudget resolves the thinking budget from thinking_budget or the Nemotron
// reasoning_budget alias.
func thinkingBudget(body *fwkrh.InferenceRequestBody) (int64, bool) {
	if body.ChatCompletions == nil {
		return 0, false
	}
	kw := body.ChatCompletions.ChatTemplateKWArgs
	if v, ok := asInt64(kw["thinking_budget"]); ok {
		return v, true
	}
	return asInt64(kw["reasoning_budget"])
}

func payloadMap(body *fwkrh.InferenceRequestBody) map[string]any {
	if body == nil || body.Payload == nil {
		return nil
	}
	m, ok := body.Payload.AsMap()
	if !ok {
		return nil
	}
	return m
}

// toolChoiceKind normalizes the OpenAI tool_choice field: a string
// (none|auto|required) or an object forcing a function ("named"); "" when unset.
func toolChoiceKind(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		return "named"
	}
	return ""
}

// responseFormatKind returns the response_format type (e.g. json_object,
// json_schema), accepting either an object with a "type" field or a bare string;
// "" when unset.
func responseFormatKind(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		return asString(t["type"])
	}
	return ""
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asBool(v any) (val, ok bool) {
	switch t := v.(type) {
	case bool:
		return t, true
	case string:
		if b, err := strconv.ParseBool(t); err == nil {
			return b, true
		}
	case float64:
		return t != 0, true
	}
	return false, false
}

func asInt64(v any) (int64, bool) {
	switch t := v.(type) {
	case float64:
		return int64(t), true
	case int64:
		return t, true
	case int:
		return int64(t), true
	case string:
		if n, err := strconv.ParseInt(t, 10, 64); err == nil {
			return n, true
		}
	}
	return 0, false
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
