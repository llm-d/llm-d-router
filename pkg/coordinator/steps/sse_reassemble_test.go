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

package steps

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// frame unmarshals a streaming SSE data payload the way the scanner does, so the
// numeric types the reassembler sees (float64) match the runtime path.
func frame(t *testing.T, payload string) map[string]any {
	t.Helper()
	var m map[string]any
	require.NoError(t, json.Unmarshal([]byte(payload), &m))
	return m
}

func TestSSEReassemble_Chat(t *testing.T) {
	r := newSSEReassembler(sseShapeChat, 0)
	r.add(frame(t, `{"id":"cmpl-1","object":"chat.completion.chunk","created":100,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"He"},"finish_reason":null}]}`))
	r.add(frame(t, `{"choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}`))
	r.add(frame(t, `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`))
	r.add(frame(t, `{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`))

	got := r.result()
	require.Equal(t, "cmpl-1", got["id"])
	require.Equal(t, "chat.completion", got["object"])
	require.Equal(t, "m", got["model"])
	require.EqualValues(t, 100, got["created"])

	choices := got["choices"].([]any)
	require.Len(t, choices, 1)
	choice := choices[0].(map[string]any)
	require.EqualValues(t, 0, choice["index"])
	require.Equal(t, "stop", choice["finish_reason"])
	msg := choice["message"].(map[string]any)
	require.Equal(t, "assistant", msg["role"])
	require.Equal(t, "Hello", msg["content"])

	usage := got["usage"].(map[string]any)
	require.EqualValues(t, 5, usage["prompt_tokens"])
	require.EqualValues(t, 2, usage["completion_tokens"])
	require.EqualValues(t, 7, usage["total_tokens"])
}

// TestDeltaHasUnfoldableField checks the whitelist: role and content fold, so only
// another key with a non-null value is unfoldable; a null-valued optional does not
// count (SDKs serialize an absent field that way).
func TestDeltaHasUnfoldableField(t *testing.T) {
	tests := []struct {
		name  string
		delta map[string]any
		want  bool
	}{
		{"empty", map[string]any{}, false},
		{"role only", map[string]any{"role": "assistant"}, false},
		{"content only", map[string]any{"content": "hi"}, false},
		{"role and content", map[string]any{"role": "assistant", "content": "hi"}, false},
		{"null tool_calls ignored", map[string]any{"content": "hi", "tool_calls": nil}, false},
		{"reasoning", map[string]any{"reasoning": "let me think"}, true},
		{"reasoning_content (older vLLM)", map[string]any{"reasoning_content": "let me think"}, true},
		{"tool_calls", map[string]any{"content": "", "tool_calls": []any{map[string]any{}}}, true},
		{"refusal", map[string]any{"refusal": "no"}, true},
		{"annotations", map[string]any{"content": "x", "annotations": []any{}}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, deltaHasUnfoldableField(tc.delta))
		})
	}
}

// TestSSEReassemble_FlagsLossyDelta verifies the reassembler flags a chat delta
// carrying a field beyond role and content (here reasoning, a server-side
// default), so the scan can fall back rather than drop it.
func TestSSEReassemble_FlagsLossyDelta(t *testing.T) {
	r := newSSEReassembler(sseShapeChat, 0)
	r.add(frame(t, `{"choices":[{"index":0,"delta":{"role":"assistant"}}]}`))
	require.False(t, r.lossyFrame, "a role-only delta is not lossy")

	r.add(frame(t, `{"choices":[{"index":0,"delta":{"reasoning":"let me think"}}]}`))
	require.True(t, r.lossyFrame, "a reasoning delta must flag the reply lossy")
}

// TestSSEReassemble_IgnoresMalformedFrameParts verifies the reassembler skips a
// choices entry that is not an object and a frame with no choices array, folding
// only the well-formed choice, so a malformed frame cannot panic or corrupt the
// reply.
func TestSSEReassemble_IgnoresMalformedFrameParts(t *testing.T) {
	r := newSSEReassembler(sseShapeChat, 0)
	// A choices array mixing a non-object entry with a real one.
	r.add(frame(t, `{"choices":["not-an-object",{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`))
	// A frame whose choices field is not an array at all.
	r.add(frame(t, `{"choices":"nope"}`))
	r.add(frame(t, `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`))

	choices := r.result()["choices"].([]any)
	require.Len(t, choices, 1)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	require.Equal(t, "hi", msg["content"])
}

// TestSSEReassemble_UsageChargedToCeiling verifies the retained usage block counts
// toward bufferedBytes, so usage-only frames that add new fields, or grow an
// existing one, trip the per-request ceiling instead of growing the heap past the
// reservation. A field that repeats at the same size is charged once, so the
// trailing usage frame of a well-behaved stream costs only its own size.
func TestSSEReassemble_UsageChargedToCeiling(t *testing.T) {
	r := newSSEReassembler(sseShapeChat, 0)
	r.add(frame(t, `{"choices":[],"usage":{"prompt_tokens":5}}`))
	first := r.bufferedBytes()
	require.Positive(t, first, "a retained usage field is charged")
	r.add(frame(t, `{"choices":[],"usage":{"prompt_tokens":5}}`))
	require.Equal(t, first, r.bufferedBytes(), "a repeated field is not charged twice")
	r.add(frame(t, `{"choices":[],"usage":{"completion_tokens":2}}`))
	require.Greater(t, r.bufferedBytes(), first, "a new field is charged")

	var sb strings.Builder
	for i := 0; i < 1000; i++ {
		fmt.Fprintf(&sb, "data: {\"choices\":[],\"usage\":{\"field_%04d\":\"%s\"}}\n\n", i, strings.Repeat("x", 64))
	}
	sb.WriteString("data: [DONE]\n\n")
	err := scanForcedResponse(strings.NewReader(sb.String()), newSSEReassembler(sseShapeChat, 1), 4096, 0)
	require.ErrorIs(t, err, errForceStreamCeiling, "usage-only frames past the reservation trip the ceiling")

	// One key, first seen small, then resent with a value past the reservation.
	grow := "data: {\"choices\":[],\"usage\":{\"note\":\"x\"}}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"note\":\"" + strings.Repeat("y", 3500) + "\"}}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"note\":\"" + strings.Repeat("y", 4500) + "\"}}\n\n" +
		"data: [DONE]\n\n"
	err = scanForcedResponse(strings.NewReader(grow), newSSEReassembler(sseShapeChat, 1), 4096, 0)
	require.ErrorIs(t, err, errForceStreamCeiling, "a repeated field that grows is charged its growth")

	// A field that shrinks gives back the difference, so the charge tracks what is held.
	shrink := newSSEReassembler(sseShapeChat, 0)
	shrink.add(frame(t, `{"choices":[],"usage":{"note":"`+strings.Repeat("y", 100)+`"}}`))
	shrink.add(frame(t, `{"choices":[],"usage":{"note":"y"}}`))
	require.EqualValues(t, len(`note`)+len(`"y"`), shrink.bufferedBytes())
}

// TestSSEReassemble_MergesUsageAcrossFrames verifies usage folds field by field
// across frames: a server that reports halves in separate frames (or overrides an
// earlier value) ends with every field, matching mergeUsage.
func TestSSEReassemble_MergesUsageAcrossFrames(t *testing.T) {
	r := newSSEReassembler(sseShapeChat, 0)
	r.add(frame(t, `{"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`))
	r.add(frame(t, `{"choices":[],"usage":{"prompt_tokens":5}}`))
	r.add(frame(t, `{"choices":[],"usage":{"completion_tokens":2,"total_tokens":7}}`))

	usage := r.result()["usage"].(map[string]any)
	require.EqualValues(t, 5, usage["prompt_tokens"])
	require.EqualValues(t, 2, usage["completion_tokens"])
	require.EqualValues(t, 7, usage["total_tokens"])
}

func TestSSEReassemble_Text(t *testing.T) {
	r := newSSEReassembler(sseShapeText, 0)
	r.add(frame(t, `{"id":"cmpl-2","object":"text_completion","created":200,"model":"m","choices":[{"index":0,"text":"He","finish_reason":null}]}`))
	r.add(frame(t, `{"choices":[{"index":0,"text":"llo","finish_reason":"length"}]}`))
	r.add(frame(t, `{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`))

	got := r.result()
	require.Equal(t, "cmpl-2", got["id"])
	require.Equal(t, "text_completion", got["object"])

	choices := got["choices"].([]any)
	require.Len(t, choices, 1)
	choice := choices[0].(map[string]any)
	require.EqualValues(t, 0, choice["index"])
	require.Equal(t, "Hello", choice["text"])
	require.Equal(t, "length", choice["finish_reason"])

	usage := got["usage"].(map[string]any)
	require.EqualValues(t, 7, usage["total_tokens"])
}

// The text shape has no chunk-vs-final object rename, so an object the stream
// never carried defaults rather than coming through empty.
func TestSSEReassemble_TextDefaultsObject(t *testing.T) {
	r := newSSEReassembler(sseShapeText, 0)
	r.add(frame(t, `{"id":"g-1","choices":[{"index":0,"text":"hi"}]}`))
	require.Equal(t, "text_completion", r.result()["object"])
}

func TestSSEReassemble_MultipleChoices(t *testing.T) {
	r := newSSEReassembler(sseShapeChat, 0)
	r.add(frame(t, `{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"A"}},{"index":1,"delta":{"role":"assistant","content":"B"}}]}`))
	r.add(frame(t, `{"choices":[{"index":1,"delta":{"content":"b"},"finish_reason":"stop"},{"index":0,"delta":{"content":"a"},"finish_reason":"stop"}]}`))

	choices := r.result()["choices"].([]any)
	require.Len(t, choices, 2)

	byIndex := map[int]map[string]any{}
	for _, c := range choices {
		cm := c.(map[string]any)
		idx := cm["index"].(int)
		byIndex[idx] = cm
	}
	require.Equal(t, "Aa", byIndex[0]["message"].(map[string]any)["content"])
	require.Equal(t, "Bb", byIndex[1]["message"].(map[string]any)["content"])
}

// Choices must come out ordered by index regardless of the order the frames
// first mentioned them.
func TestSSEReassemble_ChoiceOrder(t *testing.T) {
	r := newSSEReassembler(sseShapeText, 0)
	r.add(frame(t, `{"choices":[{"index":2,"text":"c"},{"index":0,"text":"a"}]}`))
	r.add(frame(t, `{"choices":[{"index":1,"text":"b"}]}`))

	choices := r.result()["choices"].([]any)
	require.Len(t, choices, 3)
	for i, c := range choices {
		require.EqualValues(t, i, c.(map[string]any)["index"])
	}
}

func TestSSEReassemble_BufferedBytesTracksContent(t *testing.T) {
	r := newSSEReassembler(sseShapeText, 0)
	require.Zero(t, r.bufferedBytes())
	r.add(frame(t, `{"choices":[{"index":0,"text":"hello"}]}`))
	require.EqualValues(t, 5, r.bufferedBytes())
	r.add(frame(t, `{"choices":[{"index":0,"text":"!"}]}`))
	require.EqualValues(t, 6, r.bufferedBytes())
}

func TestShapeForAPIType(t *testing.T) {
	for _, tc := range []struct {
		apiType reqcommon.APIType
		want    sseShape
	}{
		{reqcommon.APITypeChatCompletions, sseShapeChat},
		{reqcommon.APITypeCompletions, sseShapeText},
		{reqcommon.APITypeVLLMGenerate, sseShapeGenerate},
		{reqcommon.APITypeResponses, sseShapeResponses},
		{reqcommon.APITypeMessages, sseShapeMessages},
	} {
		got, ok := shapeForAPIType(tc.apiType)
		require.True(t, ok)
		require.Equal(t, tc.want, got)
	}

	// An API the force-stream path does not handle reports ok=false, so the caller
	// declines to force-stream rather than misparsing the stream as text.
	_, ok := shapeForAPIType(reqcommon.APITypeSGLangGenerate)
	require.False(t, ok)
}

// TestSSEReassemble_DropsChoicesBeyondN verifies the reassembler ignores a choice
// whose index is at or beyond the declared n, so an upstream returning more
// choices than reserved cannot grow memory past the reservation.
func TestSSEReassemble_DropsChoicesBeyondN(t *testing.T) {
	r := newSSEReassembler(sseShapeText, 1)
	r.add(frame(t, `{"choices":[{"index":0,"text":"keep"},{"index":5,"text":"drop"}]}`))

	choices := r.result()["choices"].([]any)
	require.Len(t, choices, 1)
	require.EqualValues(t, 0, choices[0].(map[string]any)["index"])
	require.Equal(t, "keep", choices[0].(map[string]any)["text"])
	// The dropped choice's content is not charged to the ceiling.
	require.EqualValues(t, len("keep"), r.bufferedBytes())
}

// TestSSEReassemble_DropsNegativeChoiceIndex verifies a choice with a negative
// index is ignored rather than creating a map entry, so a misbehaving upstream
// cannot grow the choice map with out-of-range indices.
func TestSSEReassemble_DropsNegativeChoiceIndex(t *testing.T) {
	r := newSSEReassembler(sseShapeText, 1)
	r.add(frame(t, `{"choices":[{"index":-1,"text":"drop"},{"index":0,"text":"keep"}]}`))

	require.Len(t, r.choices, 1, "the negative-index choice must not create an entry")
	choices := r.result()["choices"].([]any)
	require.Len(t, choices, 1)
	require.EqualValues(t, 0, choices[0].(map[string]any)["index"])
	require.Equal(t, "keep", choices[0].(map[string]any)["text"])
	require.EqualValues(t, len("keep"), r.bufferedBytes())
}

// TestSSEReassemble_ChoiceIndex verifies how a choice's index selects its
// accumulator. An absent or null index folds into choice 0, so a single-choice
// stream that omits it still reassembles. An index present but not a non-negative
// integer (a malformed frame) is dropped like a negative one, rather than
// truncated or guessed into another choice.
func TestSSEReassemble_ChoiceIndex(t *testing.T) {
	tests := []struct {
		name  string
		index string
		want  bool
	}{
		{"absent", ``, true},
		{"null", `"index":null,`, true},
		{"non-numeric", `"index":"oops",`, false},
		{"non-integral", `"index":1.5,`, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := newSSEReassembler(sseShapeText, 2)
			r.add(frame(t, `{"choices":[{`+tc.index+`"text":"hi"}]}`))

			choices := r.result()["choices"].([]any)
			if !tc.want {
				require.Empty(t, choices, "an invalid index must not create a choice")
				require.Zero(t, r.bufferedBytes(), "a dropped choice is not charged to the ceiling")
				return
			}
			require.Len(t, choices, 1)
			require.EqualValues(t, 0, choices[0].(map[string]any)["index"])
			require.Equal(t, "hi", choices[0].(map[string]any)["text"])
		})
	}
}

// TestSSEReassemble_CapturesSystemFingerprint verifies the chat envelope carries
// system_fingerprint through from the first chunk that sets it to the reply.
func TestSSEReassemble_CapturesSystemFingerprint(t *testing.T) {
	r := newSSEReassembler(sseShapeChat, 0)
	r.add(frame(t, `{"id":"c","model":"m","system_fingerprint":"fp_123","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`))
	r.add(frame(t, `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`))

	require.Equal(t, "fp_123", r.result()["system_fingerprint"])
}

// The Responses stream is typed events; response.completed carries the terminal
// response object, which the reassembler emits verbatim. The deltas before it
// drive only the per-request ceiling, so the reply is exactly the object the pod
// would have returned non-streaming.
func TestSSEReassemble_Responses(t *testing.T) {
	completed := `{"id":"resp-1","object":"response","model":"llama-3","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}`

	r := newSSEReassembler(sseShapeResponses, 0)
	r.add(frame(t, `{"type":"response.created","response":{"id":"resp-1","model":"llama-3","status":"in_progress"}}`))
	r.add(frame(t, `{"type":"response.output_text.delta","delta":"Hello"}`))
	r.add(frame(t, `{"type":"response.output_text.delta","delta":" world"}`))
	r.add(frame(t, `{"type":"response.completed","response":`+completed+`}`))

	// Only the generated text is charged to the ceiling, not the envelope.
	require.EqualValues(t, len("Hello world"), r.bufferedBytes())

	var want map[string]any
	require.NoError(t, json.Unmarshal([]byte(completed), &want))
	require.Equal(t, want, r.result())
}

// TestSSEReassemble_ResponsesDeltaAfterTerminal verifies a stray output_text.delta
// arriving after the terminal event is ignored: the captured object stands and the
// ceiling estimate does not grow, so a reply the pod already completed is not
// re-charged and cannot re-trip the ceiling.
func TestSSEReassemble_ResponsesDeltaAfterTerminal(t *testing.T) {
	completed := `{"id":"r","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`
	r := newSSEReassembler(sseShapeResponses, 0)
	r.add(frame(t, `{"type":"response.output_text.delta","delta":"hi"}`))
	r.add(frame(t, `{"type":"response.completed","response":`+completed+`}`))
	before := r.bufferedBytes()

	r.add(frame(t, `{"type":"response.output_text.delta","delta":"xxxxxxxxxx"}`))
	require.Equal(t, before, r.bufferedBytes(), "a post-terminal delta must not grow the ceiling estimate")

	var want map[string]any
	require.NoError(t, json.Unmarshal([]byte(completed), &want))
	require.Equal(t, want, r.result(), "the completed object stands despite the late delta")
}

// A terminal event other than response.completed (response.incomplete on a
// truncated generation, response.failed on an error) carries its own status and,
// on failure, an error block. The reassembler emits that object verbatim rather
// than synthesizing a completed reply, so the client sees the real outcome.
func TestSSEReassemble_ResponsesTerminalStates(t *testing.T) {
	tests := []struct {
		name  string
		event string
		resp  string
	}{
		{
			name:  "incomplete",
			event: responseEventIncomplete,
			resp:  `{"id":"resp-1","object":"response","model":"m","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial"}]}],"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}`,
		},
		{
			name:  "failed",
			event: responseEventFailed,
			resp:  `{"id":"resp-1","object":"response","model":"m","status":"failed","error":{"code":"server_error","message":"boom"}}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := newSSEReassembler(sseShapeResponses, 0)
			r.add(frame(t, `{"type":"response.created","response":{"id":"resp-1","model":"m","status":"in_progress"}}`))
			r.add(frame(t, `{"type":"response.output_text.delta","delta":"partial"}`))
			r.add(frame(t, `{"type":"`+tc.event+`","response":`+tc.resp+`}`))

			var want map[string]any
			require.NoError(t, json.Unmarshal([]byte(tc.resp), &want))
			require.Equal(t, want, r.result())
		})
	}
}

// Without a terminal event, a Responses stream has no status or usage carrier, so
// it is not a servable reply: terminated reports false and scanForcedResponse
// (tested separately) treats it as truncated and falls back, rather than
// synthesizing a complete-looking object.
func TestSSEReassemble_ResponsesNoTerminalEventNotTerminated(t *testing.T) {
	r := newSSEReassembler(sseShapeResponses, 0)
	r.add(frame(t, `{"type":"response.created","response":{"id":"resp-2","model":"m","status":"in_progress"}}`))
	r.add(frame(t, `{"type":"response.output_text.delta","delta":"hi"}`))

	require.False(t, r.terminated(), "a Responses stream with no terminal event is not a complete reply")
}

// The Anthropic Messages stream is typed events; the reply is assembled from the
// envelope (message_start), the text deltas (content_block_delta/text_delta), and
// the terminal message_delta (stop_reason + authoritative usage). The reassembled
// reply must match the pod's non-streaming message object.
func TestSSEReassemble_Messages(t *testing.T) {
	r := newSSEReassembler(sseShapeMessages, 0)
	r.add(frame(t, `{"type":"message_start","message":{"id":"msg_1","model":"claude","role":"assistant","usage":{"input_tokens":25}}}`))
	r.add(frame(t, `{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}`))
	r.add(frame(t, `{"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}`))
	r.add(frame(t, `{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"</done>"},"usage":{"input_tokens":25,"output_tokens":2}}`))
	r.add(frame(t, `{"type":"message_stop"}`))

	require.True(t, r.terminated())
	// The text plus the two retained usage fields, input_tokens charged once.
	require.EqualValues(t, len("Hello world")+len(`input_tokens`+`25`)+len(`output_tokens`+`2`), r.bufferedBytes())

	got := r.result()
	require.Equal(t, "msg_1", got["id"])
	require.Equal(t, "message", got["type"])
	require.Equal(t, "assistant", got["role"])
	require.Equal(t, "claude", got["model"])
	require.Equal(t, "stop_sequence", got["stop_reason"])
	require.Equal(t, "</done>", got["stop_sequence"])

	content := got["content"].([]any)
	require.Len(t, content, 1)
	block := content[0].(map[string]any)
	require.Equal(t, "text", block["type"])
	require.Equal(t, "Hello world", block["text"])

	// Usage is the message_delta block, which carries the final output_tokens.
	usage := got["usage"].(map[string]any)
	require.EqualValues(t, 25, usage["input_tokens"])
	require.EqualValues(t, 2, usage["output_tokens"])
}

// message_delta usage is cumulative and authoritative: its input_tokens supersede
// the message_start value and its output_tokens are added.
func TestSSEReassemble_MessagesUsageMerge(t *testing.T) {
	r := newSSEReassembler(sseShapeMessages, 0)
	r.add(frame(t, `{"type":"message_start","message":{"usage":{"input_tokens":2679,"output_tokens":3}}}`))
	r.add(frame(t, `{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}`))
	r.add(frame(t, `{"type":"message_delta","usage":{"input_tokens":10682,"cache_creation_input_tokens":64,"cache_read_input_tokens":128,"output_tokens":510,"server_tool_use":{"web_search_requests":1}}}`))

	usage := r.result()["usage"].(map[string]any)
	require.EqualValues(t, 10682, usage["input_tokens"], "message_delta input_tokens supersede message_start")
	require.EqualValues(t, 510, usage["output_tokens"])
	require.EqualValues(t, 64, usage["cache_creation_input_tokens"])
	require.EqualValues(t, 128, usage["cache_read_input_tokens"])
	require.Equal(t, map[string]any{"web_search_requests": float64(1)}, usage["server_tool_use"])
}

// A non-text content block marks the reply lossy so force-streaming falls back and
// the pass-through keeps the block. It is detected both when announced in
// content_block_start and when its delta arrives.
func TestSSEReassemble_MessagesNonTextLossy(t *testing.T) {
	for _, tc := range []struct {
		name  string
		frame string
	}{
		{"tool_use content_block_start", `{"type":"content_block_start","content_block":{"type":"tool_use","id":"t1","name":"get_weather"}}`},
		{"input_json_delta", `{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"city\""}}`},
		{"thinking_delta", `{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"let me think"}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := newSSEReassembler(sseShapeMessages, 0)
			r.add(frame(t, `{"type":"message_start","message":{"id":"m","role":"assistant"}}`))
			require.False(t, r.lossyFrame)
			r.add(frame(t, tc.frame))
			require.True(t, r.lossyFrame, "a non-text block must flag the reply lossy")
		})
	}
}

// A text content_block_start and a text_delta do not make the reply lossy.
func TestSSEReassemble_MessagesTextNotLossy(t *testing.T) {
	r := newSSEReassembler(sseShapeMessages, 0)
	r.add(frame(t, `{"type":"content_block_start","content_block":{"type":"text","text":""}}`))
	r.add(frame(t, `{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}`))
	require.False(t, r.lossyFrame)
}

// Without a terminal event (message_delta/message_stop), a Messages stream is not a
// complete reply: terminated reports false and scanForcedResponse treats it as a
// truncation and falls back.
func TestSSEReassemble_MessagesNoTerminalNotTerminated(t *testing.T) {
	r := newSSEReassembler(sseShapeMessages, 0)
	r.add(frame(t, `{"type":"message_start","message":{"id":"m"}}`))
	r.add(frame(t, `{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}`))
	require.False(t, r.terminated())
}

// Frames mirror the real vLLM /inference/v1/generate stream: one token_id per
// frame under a generate-tokens-* request_id, finish_reason on the final content
// frame, then a trailing choices:[] usage frame (data: [DONE] follows upstream).
// The reassembled reply must match the endpoint's non-streaming reply: a
// request_id envelope with prompt_logprobs and kv_transfer_params null, per
// choice only {index, logprobs, finish_reason, token_ids}, and no usage block.
func TestSSEReassemble_Generate(t *testing.T) {
	r := newSSEReassembler(sseShapeGenerate, 0)
	r.add(frame(t, `{"request_id":"generate-tokens-abc","choices":[{"index":0,"logprobs":null,"finish_reason":null,"token_ids":[576]}],"usage":null}`))
	r.add(frame(t, `{"request_id":"generate-tokens-abc","choices":[{"index":0,"logprobs":null,"finish_reason":null,"token_ids":[9396]}],"usage":null}`))
	r.add(frame(t, `{"request_id":"generate-tokens-abc","choices":[{"index":0,"logprobs":null,"finish_reason":"length","token_ids":[374]}],"usage":null}`))
	r.add(frame(t, `{"request_id":"generate-tokens-abc","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":16,"total_tokens":26}}`))

	got := r.result()
	// The generate reply carries a request_id envelope with prompt_logprobs and
	// kv_transfer_params null, not the OpenAI id/object/model fields.
	require.Equal(t, "generate-tokens-abc", got["request_id"])
	require.NotContains(t, got, "object")
	require.NotContains(t, got, "id")
	require.Contains(t, got, "prompt_logprobs")
	require.Nil(t, got["prompt_logprobs"])
	require.Contains(t, got, "kv_transfer_params")
	require.Nil(t, got["kv_transfer_params"])
	// The non-streaming generate reply has no usage block, so the trailing
	// frame's usage is dropped.
	require.NotContains(t, got, "usage")

	choices := got["choices"].([]any)
	require.Len(t, choices, 1)
	choice := choices[0].(map[string]any)
	require.EqualValues(t, 0, choice["index"])
	require.Equal(t, "length", choice["finish_reason"])
	require.Nil(t, choice["logprobs"])
	// A dense model's reply carries no routed_experts field.
	require.NotContains(t, choice, "routed_experts")
	require.Equal(t, []any{float64(576), float64(9396), float64(374)}, choice["token_ids"])
}

// A choice that carried no tokens still emits token_ids as an array, matching
// the documented generate reply shape rather than a null.
func TestSSEReassemble_GenerateEmptyTokens(t *testing.T) {
	r := newSSEReassembler(sseShapeGenerate, 0)
	r.add(frame(t, `{"request_id":"g","choices":[{"index":0,"finish_reason":"stop"}]}`))
	choice := r.result()["choices"].([]any)[0].(map[string]any)
	require.Equal(t, []any{}, choice["token_ids"])
}

func TestSSEReassemble_GenerateBufferedBytesCountsTokens(t *testing.T) {
	r := newSSEReassembler(sseShapeGenerate, 0)
	require.Zero(t, r.bufferedBytes())
	r.add(frame(t, `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3]}]}`))
	require.EqualValues(t, 3*forceStreamBytesPerToken, r.bufferedBytes())
}
