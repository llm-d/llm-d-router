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
		ToolCallingRequested:         true,
		UpstreamToolCallPresent:      true,
		ForwardedToolCallPresent:     false,
		UpstreamDetectionIncomplete:  true,
		ForwardedDetectionIncomplete: false,
	}).SpanAttributes()

	require.Len(t, attributes, 4)
	require.Equal(t, ResponseAttributeUpstreamToolCallPresent, string(attributes[0].Key))
	require.True(t, attributes[0].Value.AsBool())
	require.Equal(t, ResponseAttributeForwardedToolCallPresent, string(attributes[1].Key))
	require.False(t, attributes[1].Value.AsBool())
	require.Equal(t, ResponseAttributeUpstreamDetectionIncomplete, string(attributes[2].Key))
	require.True(t, attributes[2].Value.AsBool())
	require.Equal(t, ResponseAttributeForwardedDetectionIncomplete, string(attributes[3].Key))
	require.False(t, attributes[3].Value.AsBool())
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
			name:    "chat content mentioning tool calls is not a tool call",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"content":"tool_calls"}}]}`,
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

func TestResponseDetectorFlushesFinalSSELineAtEndOfStream(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	payload := []byte(`data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}`)
	require.False(t, detector.Observe(payload, false), "an incomplete SSE line is not parsed before its boundary")
	require.True(t, detector.Observe(nil, true), "the final unterminated SSE line is parsed at end of stream")
	require.Empty(t, detector.line)
	require.Equal(t, make([]byte, cap(detector.line)), detector.line[:cap(detector.line)], "flushed event data is cleared from the detector buffer")
}

func TestResponseDetectorMessagesSSEWithoutToolUse(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeMessages, true)
	require.NoError(t, err)
	for _, chunk := range []string{
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\"}}\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n",
	} {
		require.False(t, detector.Observe([]byte(chunk), false))
	}
	require.False(t, detector.Observe(nil, true))
	require.Empty(t, detector.line)
}

func TestResponseDetectorTruncatedSSEDoesNotReportToolCall(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	require.False(t, detector.Observe([]byte(`data: {"choices":[{"delta":{"tool_calls":[`), false))
	require.False(t, detector.Observe(nil, true))
	require.Empty(t, detector.line)
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
	require.NotContains(t, string(detector.line[:cap(detector.line)]), "sentinel_", "completed payload bytes must be cleared")
	require.True(t, detector.Observe([]byte("data: more content"), false))
	require.Empty(t, detector.line, "detector must stop buffering after finding a tool call")
}

func TestResponseDetectorBoundsOversizedSSELine(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)

	oversized := make([]byte, len("data: ")+maxResponseSSELineBytes+1)
	copy(oversized, "data: ")
	for i := len("data: "); i < len(oversized); i++ {
		oversized[i] = 'x'
	}
	require.False(t, detector.Observe(oversized, false))
	require.True(t, detector.DetectionIncomplete())
	require.LessOrEqual(t, len(detector.line), maxResponseSSELineBytes)
	require.Empty(t, detector.line, "overflow clears buffered bytes while discarding the rest of the line")

	toolCallLine := []byte("\n" + `data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}` + "\n")
	require.True(t, detector.Observe(toolCallLine, false), "detection resumes at the next SSE line")
	require.True(t, detector.DetectionIncomplete(), "incomplete status remains visible after later detection")
}

func TestResponseDetectorCloseClearsIncompleteSSELine(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	payload := []byte(`data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"sentinel_arguments"}}]}}]}`)
	require.False(t, detector.Observe(payload, false))
	require.Contains(t, string(detector.line[:cap(detector.line)]), "sentinel_arguments")

	detector.Close()

	require.Empty(t, detector.line)
	require.Equal(t, make([]byte, cap(detector.line)), detector.line[:cap(detector.line)])
	require.False(t, detector.Observe([]byte("more data\n"), false), "closed detector must ignore later chunks")
}

func TestResponseSummaryOmitsAttributesForNonToolCallingRequest(t *testing.T) {
	attributes := (ResponseSummary{}).SpanAttributes()
	require.Empty(t, attributes)
}
