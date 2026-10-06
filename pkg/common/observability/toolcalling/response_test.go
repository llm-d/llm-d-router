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

package toolcalling

import (
	"testing"

	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestResponseSummarySpanAttributes(t *testing.T) {
	attributes := (ResponseSummary{
		ToolCallingRequested:     true,
		UpstreamToolCallPresent:  true,
		ForwardedToolCallPresent: false,
	}).SpanAttributes()

	require.Len(t, attributes, 2)
	require.Equal(t, ResponseAttributeUpstreamToolCallPresent, string(attributes[0].Key))
	require.True(t, attributes[0].Value.AsBool())
	require.Equal(t, ResponseAttributeForwardedToolCallPresent, string(attributes[1].Key))
	require.False(t, attributes[1].Value.AsBool())
}

func TestResponseDetectorJSON(t *testing.T) {
	tests := []struct {
		name    string
		surface reqcommon.APIType
		body    string
		want    bool
	}{
		{
			name:    "chat completions tool calls",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"tool_calls":[{"type":"function","function":{"name":"sentinel_name","arguments":"sentinel_arguments"}}]}}]}`,
			want:    true,
		},
		{
			name:    "empty chat tool calls",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"tool_calls":[]}}]}`,
		},
		{
			name:    "legacy chat function call",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"function_call":{"name":"sentinel_name","arguments":"sentinel_arguments"}}}]}`,
			want:    true,
		},
		{
			name:    "messages tool use",
			surface: reqcommon.APITypeMessages,
			body:    `{"content":[{"type":"tool_use","name":"sentinel_name","input":{"secret":"sentinel_arguments"}}]}`,
			want:    true,
		},
		{
			name:    "text mentioning tool use is not a tool call",
			surface: reqcommon.APITypeMessages,
			body:    `{"content":[{"type":"text","text":"tool_use"}]}`,
		},
		{
			name:    "malformed response",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			detector, err := NewResponseDetector(tt.surface, false)
			require.NoError(t, err)
			require.Equal(t, tt.want, detector.Observe([]byte(tt.body), true))
		})
	}
}

func TestResponseDetectorSSEAcrossChunkBoundaries(t *testing.T) {
	tests := []struct {
		name    string
		surface reqcommon.APIType
		chunks  []string
	}{
		{
			name:    "chat completions delta",
			surface: reqcommon.APITypeChatCompletions,
			chunks: []string{
				`data: {"choices":[{"delta":{"tool_`,
				`calls":[{"index":0,"function":{"name":"sentinel_name",`,
				`"arguments":"sentinel_arguments"}}]}}]}` + "\r\n",
			},
		},
		{
			name:    "messages content block start",
			surface: reqcommon.APITypeMessages,
			chunks: []string{
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",",
				"\"content_block\":{\"type\":\"tool_use\",\"name\":\"sentinel_name\",\"input\":{\"secret\":\"sentinel_arguments\"}}}\n\n",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			detector, err := NewResponseDetector(tt.surface, true)
			require.NoError(t, err)
			for i, chunk := range tt.chunks {
				got := detector.Observe([]byte(chunk), i == len(tt.chunks)-1)
				if i < len(tt.chunks)-1 {
					require.False(t, got)
				}
			}
			require.True(t, detector.toolCallPresent)
			require.Empty(t, detector.line, "completed SSE payload should not remain buffered")
		})
	}
}

func TestResponseDetectorDoesNotMatchIrrelevantOrInvalidSSE(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	for _, chunk := range []string{
		`data: {"choices":[{"delta":{"content":"tool_calls"}}]}` + "\n",
		"data: [DONE]\n",
		"data: {invalid json}\n",
	} {
		require.False(t, detector.Observe([]byte(chunk), false))
	}
	require.False(t, detector.Observe(nil, true))
}

func TestResponseDetectorRejectsUnsupportedSurface(t *testing.T) {
	for _, api := range []reqcommon.APIType{reqcommon.APIType(-1), reqcommon.APITypeResponses} {
		_, err := NewResponseDetector(api, false)
		require.Error(t, err)
	}
}

func TestResponseDetectorDoesNotRetainCompletedPayload(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	payload := `data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"sentinel_name","arguments":"sentinel_arguments"}}]}}]}` + "\n"
	require.True(t, detector.Observe([]byte(payload), false))
	require.Empty(t, detector.line)
}

func TestResponseSummaryOmitsAttributesForNonToolCallingRequest(t *testing.T) {
	attributes := (ResponseSummary{}).SpanAttributes()
	require.Empty(t, attributes)
}
