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

package outlenbucket

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// chatBody builds a request body with a chat-completions payload carrying the
// given tools and chat_template_kwargs.
func chatBody(tools []any, kwArgs map[string]any, maxOut *int64) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Tools:              tools,
			ChatTemplateKWArgs: kwArgs,
		},
		MaxOutputTokens: maxOut,
	}
}

// bodyOpts configures bodyWith for the extended signal tests. The raw-payload
// fields (tool_choice, response_format) live in payload; continueFinal and
// tools/kwArgs live on the typed ChatCompletions.
type bodyOpts struct {
	tools         []any
	kwArgs        map[string]any
	payload       map[string]any
	continueFinal bool
	maxOut        *int64
}

// bodyWith builds a request body exercising both the typed chat-completions
// fields and the raw JSON payload map that tool_choice/response_format are read from.
func bodyWith(o bodyOpts) *fwkrh.InferenceRequestBody {
	b := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Tools:                o.tools,
			ChatTemplateKWArgs:   o.kwArgs,
			ContinueFinalMessage: o.continueFinal,
		},
		MaxOutputTokens: o.maxOut,
	}
	if o.payload != nil {
		b.Payload = fwkrh.PayloadMap(o.payload)
	}
	return b
}

func TestEstimateOutlen(t *testing.T) {
	oneTool := []any{map[string]any{"type": "function"}}

	tests := []struct {
		name string
		body *fwkrh.InferenceRequestBody
		want Bucket
	}{
		{name: "nil body", body: nil, want: Unknown},
		{name: "empty body", body: &fwkrh.InferenceRequestBody{}, want: Unknown},
		{
			name: "enable_thinking=true -> LONG",
			body: chatBody(nil, map[string]any{"enable_thinking": true}, nil),
			want: Long,
		},
		{
			name: "enable_thinking=false, no tools -> UNKNOWN",
			body: chatBody(nil, map[string]any{"enable_thinking": false}, nil),
			want: Unknown,
		},
		{
			name: "has_tools=true, enable_thinking absent -> SHORT",
			body: chatBody(oneTool, nil, nil),
			want: Short,
		},
		{
			name: "has_tools=true, enable_thinking=false -> SHORT",
			body: chatBody(oneTool, map[string]any{"enable_thinking": false}, nil),
			want: Short,
		},
		{
			name: "has_tools=true, enable_thinking=true -> LONG (thinking overrides)",
			body: chatBody(oneTool, map[string]any{"enable_thinking": true}, nil),
			want: Long,
		},
		{
			name: "thinking_budget>4000 without enable_thinking -> LONG",
			body: chatBody(nil, map[string]any{"thinking_budget": float64(8000)}, nil),
			want: Long,
		},
		{
			name: "thinking_budget>4000 with enable_thinking=false -> UNKNOWN (explicit false wins)",
			body: chatBody(nil, map[string]any{"enable_thinking": false, "thinking_budget": float64(8000)}, nil),
			want: Unknown,
		},
		{
			name: "thinking_budget<=4000 -> UNKNOWN",
			body: chatBody(nil, map[string]any{"thinking_budget": float64(4000)}, nil),
			want: Unknown,
		},
		{
			name: "max_output_tokens<500 -> SHORT",
			body: chatBody(nil, nil, ptr.To(int64(100))),
			want: Short,
		},
		{
			name: "max_output_tokens=499 -> SHORT",
			body: chatBody(nil, nil, ptr.To(int64(499))),
			want: Short,
		},
		{
			name: "max_output_tokens=500 -> UNKNOWN (boundary)",
			body: chatBody(nil, nil, ptr.To(int64(500))),
			want: Unknown,
		},
		{
			name: "max_output_tokens=0 -> UNKNOWN (zero ignored)",
			body: chatBody(nil, nil, ptr.To(int64(0))),
			want: Unknown,
		},
		{
			name: "enable_thinking as string \"true\" -> LONG",
			body: chatBody(nil, map[string]any{"enable_thinking": "true"}, nil),
			want: Long,
		},
		{
			name: "no chat completions, short cap -> SHORT",
			body: &fwkrh.InferenceRequestBody{MaxOutputTokens: ptr.To(int64(50))},
			want: Short,
		},
		{
			name: "tools only on messages shape -> UNKNOWN (not inspected)",
			body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{Tools: []fwkrh.AnthropicTool{{Name: "f"}}}},
			want: Unknown,
		},
		{
			name: "tools only on responses shape -> UNKNOWN (not inspected)",
			body: &fwkrh.InferenceRequestBody{Responses: &fwkrh.ResponsesRequest{Tools: oneTool}},
			want: Unknown,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := EstimateOutlen(tc.body)
			require.Equal(t, tc.want, got, "got %s want %s", got, tc.want)
		})
	}
}

func TestPlugin_RequestHeader_PublishesAttribute(t *testing.T) {
	p, err := PluginFactory("outlen", nil, nil)
	require.NoError(t, err)
	plugin := p.(*Plugin)

	t.Run("LONG bucket published", func(t *testing.T) {
		req := &scheduling.InferenceRequest{
			Body: chatBody(nil, map[string]any{"enable_thinking": true}, nil),
		}
		require.NoError(t, plugin.RequestHeader(context.Background(), req))

		got, ok := scheduling.ReadRequestAttribute[Bucket](req, AttributeKey)
		require.True(t, ok, "attribute must be set")
		require.Equal(t, Long, got)
	})

	t.Run("SHORT bucket published", func(t *testing.T) {
		req := &scheduling.InferenceRequest{
			Body: chatBody([]any{map[string]any{"type": "function"}}, nil, nil),
		}
		require.NoError(t, plugin.RequestHeader(context.Background(), req))

		got, ok := scheduling.ReadRequestAttribute[Bucket](req, AttributeKey)
		require.True(t, ok)
		require.Equal(t, Short, got)
	})

	t.Run("UNKNOWN still published", func(t *testing.T) {
		req := &scheduling.InferenceRequest{Body: &fwkrh.InferenceRequestBody{}}
		require.NoError(t, plugin.RequestHeader(context.Background(), req))

		got, ok := scheduling.ReadRequestAttribute[Bucket](req, AttributeKey)
		require.True(t, ok)
		require.Equal(t, Unknown, got)
	})

	t.Run("nil body is a no-op", func(t *testing.T) {
		req := &scheduling.InferenceRequest{}
		require.NoError(t, plugin.RequestHeader(context.Background(), req))

		_, ok := scheduling.ReadRequestAttribute[Bucket](req, AttributeKey)
		require.False(t, ok, "no attribute when body is nil")
	})
}

func TestBoolPtrFromAny(t *testing.T) {
	require.Equal(t, true, *boolPtrFromAny(true))
	require.Equal(t, false, *boolPtrFromAny(false))
	require.Equal(t, true, *boolPtrFromAny("true"))
	require.Equal(t, false, *boolPtrFromAny("false"))
	require.Equal(t, true, *boolPtrFromAny("1"))
	require.Equal(t, false, *boolPtrFromAny("0"))
	require.Equal(t, true, *boolPtrFromAny(float64(1)))
	require.Equal(t, false, *boolPtrFromAny(float64(0)))
	require.Equal(t, true, *boolPtrFromAny(json.Number("1")))
	require.Nil(t, boolPtrFromAny("maybe"))
	require.Nil(t, boolPtrFromAny(nil))
	require.Nil(t, boolPtrFromAny([]int{1}))
}

func TestInt64PtrFromAny(t *testing.T) {
	require.Equal(t, int64(8000), *int64PtrFromAny(float64(8000)))
	require.Equal(t, int64(8000), *int64PtrFromAny(json.Number("8000")))
	require.Equal(t, int64(8000), *int64PtrFromAny("8000"))
	require.Equal(t, int64(42), *int64PtrFromAny(42))
	require.Equal(t, int64(42), *int64PtrFromAny(int64(42)))
	require.Nil(t, int64PtrFromAny("not-a-number"))
	require.Nil(t, int64PtrFromAny(nil))
	require.Nil(t, int64PtrFromAny(true))
}

// namedToolChoice is an OpenAI tool_choice object forcing a specific function.
var namedToolChoice = map[string]any{"type": "function", "function": map[string]any{"name": "get_weather"}}

// TestEstimateOutlen_ExtendedSignals covers signals read from the raw payload
// map (tool_choice, response_format), the typed continue_final_message,
// vendor-specific normalizations (DeepSeek thinking.type, Nemotron reasoning_budget),
// the tool_choice="none" veto, and the max_output_tokens bin ceiling.
func TestEstimateOutlen_ExtendedSignals(t *testing.T) {
	oneTool := []any{map[string]any{"type": "function"}}

	tests := []struct {
		name string
		body *fwkrh.InferenceRequestBody
		want Bucket
	}{
		// --- LONG pushers: DeepSeek thinking.type vendor normalization ---
		{
			name: "thinking.type=enabled -> LONG (normalizes to enable_thinking=true)",
			body: bodyWith(bodyOpts{kwArgs: map[string]any{"thinking": map[string]any{"type": "enabled"}}}),
			want: Long,
		},
		{
			name: "thinking.type=disabled -> UNKNOWN (normalizes to enable_thinking=false)",
			body: bodyWith(bodyOpts{kwArgs: map[string]any{"thinking": map[string]any{"type": "disabled"}}}),
			want: Unknown,
		},
		// --- LONG pushers: Nemotron reasoning_budget alias ---
		{
			name: "reasoning_budget=5000 -> LONG (Nemotron alias for thinking_budget)",
			body: bodyWith(bodyOpts{kwArgs: map[string]any{"reasoning_budget": int64(5000)}}),
			want: Long,
		},
		{
			name: "thinking_budget=100 + reasoning_budget=5000 -> UNKNOWN (thinking_budget wins; 100 < 4000)",
			body: bodyWith(bodyOpts{kwArgs: map[string]any{"thinking_budget": int64(100), "reasoning_budget": int64(5000)}}),
			want: Unknown,
		},
		// --- SHORT pushers ---
		{
			name: "tool_choice=required -> SHORT",
			body: bodyWith(bodyOpts{payload: map[string]any{"tool_choice": "required"}}),
			want: Short,
		},
		{
			name: "tool_choice=named object -> SHORT",
			body: bodyWith(bodyOpts{payload: map[string]any{"tool_choice": namedToolChoice}}),
			want: Short,
		},
		{
			name: "continue_final_message=true -> SHORT",
			body: bodyWith(bodyOpts{continueFinal: true}),
			want: Short,
		},
		{
			name: "response_format json_object -> SHORT",
			body: bodyWith(bodyOpts{payload: map[string]any{"response_format": map[string]any{"type": "json_object"}}}),
			want: Short,
		},
		{
			name: "response_format json_schema -> SHORT",
			body: bodyWith(bodyOpts{payload: map[string]any{"response_format": map[string]any{"type": "json_schema"}}}),
			want: Short,
		},
		{
			name: "response_format text -> UNKNOWN (not a SHORT signal)",
			body: bodyWith(bodyOpts{payload: map[string]any{"response_format": map[string]any{"type": "text"}}}),
			want: Unknown,
		},
		// --- tool_choice="none" veto of the has_tools -> SHORT rule ---
		{
			name: "has_tools + tool_choice=none -> UNKNOWN (veto: tools won't be called)",
			body: bodyWith(bodyOpts{tools: oneTool, payload: map[string]any{"tool_choice": "none"}}),
			want: Unknown,
		},
		{
			name: "has_tools + tool_choice=auto -> SHORT (auto does not veto)",
			body: bodyWith(bodyOpts{tools: oneTool, payload: map[string]any{"tool_choice": "auto"}}),
			want: Short,
		},
		{
			name: "has_tools + no tool_choice -> SHORT (existing behavior preserved)",
			body: bodyWith(bodyOpts{tools: oneTool}),
			want: Short,
		},
		// --- max_output_tokens bin ceiling (downgrades a tentative LONG) ---
		{
			name: "enable_thinking=true + max_output=1500 -> UNKNOWN (LONG vetoed by cap<2000)",
			body: bodyWith(bodyOpts{kwArgs: map[string]any{"enable_thinking": true}, maxOut: ptr.To(int64(1500))}),
			want: Unknown,
		},
		{
			name: "enable_thinking=true + max_output=100 -> SHORT (LONG vetoed by cap<500)",
			body: bodyWith(bodyOpts{kwArgs: map[string]any{"enable_thinking": true}, maxOut: ptr.To(int64(100))}),
			want: Short,
		},
		{
			name: "enable_thinking=true + max_output=3000 -> LONG (cap above LONG floor)",
			body: bodyWith(bodyOpts{kwArgs: map[string]any{"enable_thinking": true}, maxOut: ptr.To(int64(3000))}),
			want: Long,
		},
		{
			name: "enable_thinking=true + max_output=0 -> LONG (zero cap ignored)",
			body: bodyWith(bodyOpts{kwArgs: map[string]any{"enable_thinking": true}, maxOut: ptr.To(int64(0))}),
			want: Long,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := EstimateOutlen(tc.body)
			require.Equal(t, tc.want, got, "got %s want %s", got, tc.want)
		})
	}
}

func TestApplyMaxOutputCeiling(t *testing.T) {
	// Only LONG is ever downgraded; SHORT/UNKNOWN pass through untouched.
	require.Equal(t, Short, applyMaxOutputCeiling(Short, ptr.To(int64(50))))
	require.Equal(t, Unknown, applyMaxOutputCeiling(Unknown, ptr.To(int64(50))))
	// LONG with a cap below the LONG floor is downgraded by cap size.
	require.Equal(t, Short, applyMaxOutputCeiling(Long, ptr.To(int64(499))))
	require.Equal(t, Unknown, applyMaxOutputCeiling(Long, ptr.To(int64(500))))
	require.Equal(t, Unknown, applyMaxOutputCeiling(Long, ptr.To(int64(1999))))
	// LONG with a cap at/above the floor, nil, or zero is left as LONG.
	require.Equal(t, Long, applyMaxOutputCeiling(Long, ptr.To(int64(2000))))
	require.Equal(t, Long, applyMaxOutputCeiling(Long, nil))
	require.Equal(t, Long, applyMaxOutputCeiling(Long, ptr.To(int64(0))))
}

func TestStringFromAny(t *testing.T) {
	require.Equal(t, "high", stringFromAny("high"))
	require.Equal(t, "", stringFromAny(nil))
	require.Equal(t, "", stringFromAny(42))
	require.Equal(t, "", stringFromAny(map[string]any{"type": "function"}))
}

func TestToolChoiceKind(t *testing.T) {
	require.Equal(t, "none", toolChoiceKind("none"))
	require.Equal(t, "auto", toolChoiceKind("auto"))
	require.Equal(t, "required", toolChoiceKind("required"))
	require.Equal(t, "named", toolChoiceKind(namedToolChoice))
	require.Equal(t, "", toolChoiceKind(nil))
	require.Equal(t, "", toolChoiceKind(42))
}

func TestResponseFormatType(t *testing.T) {
	require.Equal(t, "json_object", responseFormatType(map[string]any{"type": "json_object"}))
	require.Equal(t, "json_schema", responseFormatType(map[string]any{"type": "json_schema", "json_schema": map[string]any{}}))
	require.Equal(t, "text", responseFormatType(map[string]any{"type": "text"}))
	require.Equal(t, "", responseFormatType(map[string]any{}))
	require.Equal(t, "", responseFormatType("json_object"))
	require.Equal(t, "", responseFormatType(nil))
}

