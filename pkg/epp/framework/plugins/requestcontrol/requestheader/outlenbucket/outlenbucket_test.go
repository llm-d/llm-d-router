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
