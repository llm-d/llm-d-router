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
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
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
	require.Equal(t, semconv.LLMDToolCallingResponseUpstreamPresentKey, attributes[0].Key)
	require.True(t, attributes[0].Value.AsBool())
	require.Equal(t, semconv.LLMDToolCallingResponseForwardedPresentKey, attributes[1].Key)
	require.False(t, attributes[1].Value.AsBool())
	require.Equal(t, semconv.LLMDToolCallingResponseUpstreamDetectionIncompleteKey, attributes[2].Key)
	require.True(t, attributes[2].Value.AsBool())
	require.Equal(t, semconv.LLMDToolCallingResponseForwardedDetectionIncompleteKey, attributes[3].Key)
	require.False(t, attributes[3].Value.AsBool())
}

func TestResponseDetectorJSON(t *testing.T) {
	tests := []struct {
		name       string
		surface    reqcommon.APIType
		body       string
		want       bool
		incomplete bool
	}{
		{
			name:    "chat completions tool calls",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"tool_calls":[{"type":"function","function":{"name":"sentinel_name","arguments":"sentinel_arguments"}}]}}]}`,
			want:    true,
		},
		{
			name:    "escaped chat tool calls",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"\u0074ool_calls":[{"index":0}]}}]}`,
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
			name:    "escaped legacy chat function call",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"\u0066unction_call":{"name":"sentinel_name"}}}]}`,
			want:    true,
		},
		{
			name:    "messages tool use",
			surface: reqcommon.APITypeMessages,
			body:    `{"content":[{"type":"tool_use","name":"sentinel_name","input":{"secret":"sentinel_arguments"}}]}`,
			want:    true,
		},
		{
			name:    "messages nested message tool use",
			surface: reqcommon.APITypeMessages,
			body:    `{"type":"message_start","message":{"content":[{"type":"tool_use","name":"sentinel_name"}]}}`,
			want:    true,
		},
		{
			name:    "unexpected tool calls object before function call",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"tool_calls":{"nested":[1]},"function_call":{"name":"sentinel_name"}}}]}`,
			want:    true,
		},
		{
			name:    "unexpected function call array before tool calls",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"function_call":[{"nested":1}],"tool_calls":[{"index":0}]}}]}`,
			want:    true,
		},
		{
			name:    "unexpected content type object before tool use",
			surface: reqcommon.APITypeMessages,
			body:    `{"content":[{"type":{"nested":["tool_use"]},"text":"hello"},{"type":"tool_use"}]}`,
			want:    true,
		},
		{
			name:    "unexpected tool calls object without tool call",
			surface: reqcommon.APITypeChatCompletions,
			body:    `{"choices":[{"message":{"tool_calls":{"nested":["tool_calls"]}}}]}`,
		},
		{
			name:    "unexpected content type object without tool use",
			surface: reqcommon.APITypeMessages,
			body:    `{"content":[{"type":{"nested":["tool_use"]},"text":"hello"}]}`,
		},
		{
			name:    "escaped messages tool use",
			surface: reqcommon.APITypeMessages,
			body:    `{"content":[{"type":"\u0074ool_use","name":"sentinel_name"}]}`,
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
			name:       "malformed response",
			surface:    reqcommon.APITypeChatCompletions,
			body:       `{"choices":[`,
			incomplete: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			detector, err := NewResponseDetector(tt.surface, false)
			require.NoError(t, err)
			require.Equal(t, tt.want, detector.Observe([]byte(tt.body), true))
			require.Equal(t, tt.incomplete, detector.DetectionIncomplete())
		})
	}
}

func TestResponseDetectorFragmentedJSON(t *testing.T) {
	for _, tt := range []struct {
		surface reqcommon.APIType
		body    string
	}{
		{reqcommon.APITypeChatCompletions, `{"choices":[{"message":{"tool_calls":[{"function":{"name":"sentinel_name"}}]}}]}`},
		{reqcommon.APITypeMessages, `{"content":[{"type":"tool_use","name":"sentinel_name"}]}`},
	} {
		for _, finalChunk := range []bool{false, true} {
			detector, err := NewResponseDetector(tt.surface, false)
			require.NoError(t, err)
			split := len(tt.body) / 2
			require.False(t, detector.Observe([]byte(tt.body[:split]), false))
			require.False(t, detector.DetectionIncomplete())
			require.Equal(t, finalChunk, detector.Observe([]byte(tt.body[split:]), finalChunk))
			if !finalChunk {
				require.True(t, detector.Observe(nil, true))
			}
			require.False(t, detector.DetectionIncomplete())
			require.Empty(t, detector.eventData)
		}
	}
}

func BenchmarkResponseDetectorObserveJSON(b *testing.B) {
	padding := strings.Repeat("x", 1024)
	largeArgument := strings.Repeat("x", 64*1024)
	for _, tt := range []struct {
		name    string
		surface reqcommon.APIType
		body    string
		want    bool
	}{
		{"chat short text event", reqcommon.APITypeChatCompletions, `{"choices":[{"delta":{"content":"hello"}}]}`, false},
		{"chat 1KB text event", reqcommon.APITypeChatCompletions, `{"choices":[{"delta":{"content":"` + padding + `"}}]}`, false},
		{"chat tool event", reqcommon.APITypeChatCompletions, `{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}`, true},
		{"chat tool event with large arguments", reqcommon.APITypeChatCompletions, `{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"` + largeArgument + `"}}]}}]}`, true},
		{"messages short text event", reqcommon.APITypeMessages, `{"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}`, false},
		{"messages 1KB text event", reqcommon.APITypeMessages, `{"type":"content_block_delta","delta":{"type":"text_delta","text":"` + padding + `"}}`, false},
		{"messages tool event", reqcommon.APITypeMessages, `{"type":"content_block_start","content_block":{"type":"tool_use"}}`, true},
		{"messages tool event with large input", reqcommon.APITypeMessages, `{"type":"content_block_start","content_block":{"type":"tool_use","input":{"value":"` + largeArgument + `"}}}`, true},
		{"escaped chat tool event", reqcommon.APITypeChatCompletions, `{"choices":[{"delta":{"\u0074ool_calls":[{"index":0}]}}]}`, true},
	} {
		b.Run(tt.name, func(b *testing.B) {
			detector := &ResponseDetector{surface: tt.surface}
			body := []byte(tt.body)
			detector.observeJSON(body)
			if detector.toolCallPresent != tt.want || detector.detectionIncomplete {
				b.Fatal("unexpected detector result")
			}
			b.ReportAllocs()
			for b.Loop() {
				detector.observeJSON(body)
			}
		})
	}
}

func TestResponseDetectorReportsUnreadablePayload(t *testing.T) {
	for _, tt := range []struct {
		name           string
		eventStream    bool
		body           string
		wantIncomplete bool
	}{
		{name: "malformed JSON", body: `{invalid json}`, wantIncomplete: true},
		{name: "truncated JSON", body: `{"choices":[`, wantIncomplete: true},
		{name: "malformed JSON after tool call", body: `{"choices":[{"message":{"tool_calls":[{"index":0}]}}],`, wantIncomplete: true},
		{name: "empty JSON body", wantIncomplete: true},
		{name: "valid JSON without tools", body: `{"choices":[{"message":{"content":"hello"}}]}`},
		{name: "malformed SSE event", eventStream: true, body: "data: {invalid json}\n\n", wantIncomplete: true},
		{name: "truncated SSE event", eventStream: true, body: `data: {"choices":[`, wantIncomplete: true},
		{name: "valid SSE without tools", eventStream: true, body: `data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n"},
		{name: "SSE control events", eventStream: true, body: strings.Join([]string{": keepalive", "data:", "data: [DONE]"}, "\n\n") + "\n\n"},
		{name: "SSE unfinished control line", eventStream: true, body: ": keepalive\nid: 7\nretry: 1000\nevent: ping"},
		{name: "SSE partial BOM at EOF", eventStream: true, body: "\xef"},
		{name: "SSE two-byte BOM prefix at EOF", eventStream: true, body: "\xef\xbb"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, tt.eventStream)
			require.NoError(t, err)
			require.False(t, detector.Observe([]byte(tt.body), true))
			require.Equal(t, tt.wantIncomplete, detector.DetectionIncomplete())
		})
	}
}

func TestResponseDetectorRetainsIncompleteStatusAfterLaterToolCall(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	require.False(t, detector.Observe([]byte("data: {invalid json}\n\n"), false))
	require.True(t, detector.DetectionIncomplete())
	require.True(t, detector.Observe([]byte(`data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}`+"\n\n"), true))
	require.True(t, detector.DetectionIncomplete())
}

func TestResponseDetectorPreservesValidJSONNumbers(t *testing.T) {
	for _, eventStream := range []bool{false, true} {
		payload := `{"usage":{"total_tokens":1e400},"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}`
		if eventStream {
			payload = "data: " + payload + "\n\n"
		}
		detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, eventStream)
		require.NoError(t, err)
		require.True(t, detector.Observe([]byte(payload), true))
		require.False(t, detector.DetectionIncomplete())
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
				`"arguments":"sentinel_arguments"}}]}}]}` + "\r\n\r\n",
			},
		},
		{
			name:    "escaped chat completions delta",
			surface: reqcommon.APITypeChatCompletions,
			chunks: []string{
				`data: {"choices":[{"delta":{"\u00`,
				`74ool_calls":[{"index":0}]}}]}` + "\n\n",
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
		{
			name:    "escaped messages content block start",
			surface: reqcommon.APITypeMessages,
			chunks: []string{
				`data: {"type":"content_block_start","content_block":{"type":"\u00`,
				`74ool_use"}}` + "\n\n",
			},
		},
		{
			name:    "messages nested message tool use",
			surface: reqcommon.APITypeMessages,
			chunks: []string{
				`data: {"type":"message_start","message":{"content":[{"type":"tool_use"}]}}` + "\n\n",
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

func TestResponseDetectorSSEEventFraming(t *testing.T) {
	for _, ending := range []struct {
		name  string
		value string
	}{
		{name: "LF", value: "\n"},
		{name: "CR", value: "\r"},
		{name: "CRLF", value: "\r\n"},
	} {
		for _, tt := range []struct {
			name    string
			surface reqcommon.APIType
			lines   []string
		}{
			{name: "chat", surface: reqcommon.APITypeChatCompletions, lines: []string{
				`data: {"choices":[{"delta":`,
				": keepalive",
				"data",
				`data:{"tool_calls":[{"index":0}]}}]}`,
			}},
			{name: "messages", surface: reqcommon.APITypeMessages, lines: []string{
				"event: content_block_start",
				`data: {"type":"content_block_start",`,
				`data: "content_block":{"type":"tool_use","name":"sentinel_name"}}`,
			}},
		} {
			t.Run(ending.name+"/"+tt.name, func(t *testing.T) {
				event := strings.Join(tt.lines, ending.value) + ending.value + ending.value
				for split := 0; split <= len(event); split++ {
					detector, err := NewResponseDetector(tt.surface, true)
					require.NoError(t, err)
					detector.Observe([]byte(event[:split]), false)
					require.True(t, detector.Observe([]byte(event[split:]), false), "split at byte %d", split)
					require.False(t, detector.DetectionIncomplete())
					require.Empty(t, detector.eventData)
				}
			})
		}
	}
}

func TestResponseDetectorSSEByteOrderMark(t *testing.T) {
	for _, api := range []struct {
		surface reqcommon.APIType
		payload string
	}{
		{reqcommon.APITypeChatCompletions, `{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}`},
		{reqcommon.APITypeMessages, `{"type":"content_block_start","content_block":{"type":"tool_use"}}`},
	} {
		for _, tt := range []struct {
			name, prefix, ending string
			want, incomplete     bool
		}{
			{name: "plain", ending: "\n\n", want: true},
			{name: "initial BOM", prefix: "\xef\xbb\xbf", ending: "\n\n", want: true},
			{name: "BOM before empty line", prefix: "\xef\xbb\xbf\n", ending: "\n\n", want: true},
			{name: "BOM after empty line", prefix: "\n\xef\xbb\xbf", ending: "\n\n"},
			{name: "BOM after short comment", prefix: ":\n\xef\xbb\xbf", ending: "\n\n"},
			{name: "BOM after comment", prefix: ": keepalive\n\xef\xbb\xbf", ending: "\n\n"},
			{name: "repeated BOM", prefix: "\xef\xbb\xbf\xef\xbb\xbf", ending: "\n\n"},
			{name: "partial BOM before field", prefix: "\xef\xbb", ending: "\n\n"},
			{name: "unfinished BOM-prefixed event", prefix: "\xef\xbb\xbf", ending: "\n", incomplete: true},
		} {
			t.Run(api.surface.String()+"/"+tt.name, func(t *testing.T) {
				body := []byte(tt.prefix + "data: " + api.payload + tt.ending)
				for _, split := range []int{0, 1, 2, 3, len(body)} {
					detector, err := NewResponseDetector(api.surface, true)
					require.NoError(t, err)
					detector.Observe(body[:split], false)
					for i := split; i < len(body); i++ {
						detector.Observe(body[i:i+1], false)
					}
					require.Equal(t, tt.want, detector.Observe(nil, true), "split at byte %d", split)
					require.Equal(t, tt.incomplete, detector.DetectionIncomplete())
					require.Empty(t, detector.line)
					require.Empty(t, detector.eventData)
					require.Equal(t, strings.Repeat("\x00", cap(detector.line)), string(detector.line[:cap(detector.line)]))
					require.Equal(t, strings.Repeat("\x00", cap(detector.eventData)), string(detector.eventData[:cap(detector.eventData)]))
				}
			})
		}
	}
}

func TestResponseDetectorWaitsForSSEEventBoundary(t *testing.T) {
	payload := `data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}` + "\n"
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	require.False(t, detector.Observe([]byte(payload), false))
	require.True(t, detector.Observe([]byte("\n"), false))

	detector, err = NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	require.False(t, detector.Observe([]byte(payload+"data: invalid\n\n"), false), "all data lines belong to the same JSON payload")
}

func TestResponseDetectorBoundsMultilineSSEEvent(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	line := "data: " + strings.Repeat("x", 1024) + "\n"
	for i := 0; i <= maxResponseSSEEventBytes/1024; i++ {
		require.False(t, detector.Observe([]byte(line), false))
		require.LessOrEqual(t, len(detector.line)+len(detector.eventData), maxResponseSSEEventBytes)
	}
	require.True(t, detector.DetectionIncomplete(), "many short lines must not bypass the buffer limit")
	require.Empty(t, detector.eventData)
	payload := `data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}` + "\n"
	require.False(t, detector.Observe([]byte(payload), false), "the remainder of an oversized event must be discarded")
	require.True(t, detector.Observe([]byte("\n"+payload+"\n"), false), "detection resumes at the next event")
	require.True(t, detector.DetectionIncomplete())
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

func TestResponseDetectorRequiresFinalSSEEventBoundary(t *testing.T) {
	for _, api := range []struct {
		surface reqcommon.APIType
		payload string
	}{
		{reqcommon.APITypeChatCompletions, `{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}`},
		{reqcommon.APITypeMessages, `{"type":"content_block_start","content_block":{"type":"tool_use"}}`},
	} {
		for _, ending := range []struct {
			name, suffix string
			complete     bool
		}{
			{name: "none"},
			{name: "LF", suffix: "\n"},
			{name: "CR", suffix: "\r"},
			{name: "CRLF", suffix: "\r\n"},
			{name: "LF-LF", suffix: "\n\n", complete: true},
			{name: "CR-CR", suffix: "\r\r", complete: true},
			{name: "CRLF-CRLF", suffix: "\r\n\r\n", complete: true},
		} {
			t.Run(api.surface.String()+"/"+ending.name, func(t *testing.T) {
				detector, err := NewResponseDetector(api.surface, true)
				require.NoError(t, err)
				require.False(t, detector.Observe([]byte("data: "+api.payload), false))
				for _, b := range []byte(ending.suffix) {
					detector.Observe([]byte{b}, false)
				}
				require.Equal(t, ending.complete, detector.Observe(nil, true))
				require.Equal(t, !ending.complete, detector.DetectionIncomplete())
				require.Empty(t, detector.line)
				require.Empty(t, detector.eventData)
				require.Equal(t, make([]byte, cap(detector.line)), detector.line[:cap(detector.line)])
				require.Equal(t, make([]byte, cap(detector.eventData)), detector.eventData[:cap(detector.eventData)])
			})
		}
	}
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
	payload := `data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"sentinel_name","arguments":"sentinel_arguments"}}]}}]}` + "\n\n"
	require.True(t, detector.Observe([]byte(payload), false))
	require.Empty(t, detector.line)
	require.NotContains(t, string(detector.line[:cap(detector.line)]), "sentinel_", "completed payload bytes must be cleared")
	require.Empty(t, detector.eventData)
	require.NotContains(t, string(detector.eventData[:cap(detector.eventData)]), "sentinel_")
	require.True(t, detector.Observe([]byte("data: more content"), false))
	require.Empty(t, detector.line, "detector must stop buffering after finding a tool call")
}

func TestResponseDetectorBoundsOversizedSSELine(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)

	oversized := make([]byte, len("data: ")+maxResponseSSEEventBytes+1)
	copy(oversized, "data: ")
	for i := len("data: "); i < len(oversized); i++ {
		oversized[i] = 'x'
	}
	require.False(t, detector.Observe(oversized, false))
	require.True(t, detector.DetectionIncomplete())
	require.LessOrEqual(t, len(detector.line), maxResponseSSEEventBytes)
	require.Empty(t, detector.line, "overflow clears buffered bytes while discarding the rest of the line")

	toolCallLine := []byte("\n\n" + `data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}` + "\n\n")
	require.True(t, detector.Observe(toolCallLine, false), "detection resumes at the next SSE event")
	require.True(t, detector.DetectionIncomplete(), "incomplete status remains visible after later detection")
}

func TestResponseDetectorCloseClearsIncompleteSSELine(t *testing.T) {
	detector, err := NewResponseDetector(reqcommon.APITypeChatCompletions, true)
	require.NoError(t, err)
	payload := []byte(`data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"sentinel_arguments"}}]}}]}`)
	require.False(t, detector.Observe(payload, false))
	require.Contains(t, string(detector.line[:cap(detector.line)]), "sentinel_arguments")
	require.False(t, detector.Observe([]byte("\n"), false))
	require.Contains(t, string(detector.eventData), "sentinel_arguments")

	detector.Close()

	require.Empty(t, detector.line)
	require.Equal(t, make([]byte, cap(detector.line)), detector.line[:cap(detector.line)])
	require.Empty(t, detector.eventData)
	require.Equal(t, make([]byte, cap(detector.eventData)), detector.eventData[:cap(detector.eventData)])
	require.False(t, detector.Observe([]byte("more data\n"), false), "closed detector must ignore later chunks")
}

func TestResponseSummaryOmitsAttributesForNonToolCallingRequest(t *testing.T) {
	attributes := (ResponseSummary{}).SpanAttributes()
	require.Empty(t, attributes)
}
