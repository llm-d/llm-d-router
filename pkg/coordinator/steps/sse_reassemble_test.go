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
	r := newSSEReassembler(sseShapeChat)
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

func TestSSEReassemble_Text(t *testing.T) {
	r := newSSEReassembler(sseShapeText)
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
	r := newSSEReassembler(sseShapeText)
	r.add(frame(t, `{"id":"g-1","choices":[{"index":0,"text":"hi"}]}`))
	require.Equal(t, "text_completion", r.result()["object"])
}

func TestSSEReassemble_MultipleChoices(t *testing.T) {
	r := newSSEReassembler(sseShapeChat)
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
	r := newSSEReassembler(sseShapeText)
	r.add(frame(t, `{"choices":[{"index":2,"text":"c"},{"index":0,"text":"a"}]}`))
	r.add(frame(t, `{"choices":[{"index":1,"text":"b"}]}`))

	choices := r.result()["choices"].([]any)
	require.Len(t, choices, 3)
	for i, c := range choices {
		require.EqualValues(t, i, c.(map[string]any)["index"])
	}
}

func TestSSEReassemble_BufferedBytesTracksContent(t *testing.T) {
	r := newSSEReassembler(sseShapeText)
	require.Zero(t, r.bufferedBytes())
	r.add(frame(t, `{"choices":[{"index":0,"text":"hello"}]}`))
	require.EqualValues(t, 5, r.bufferedBytes())
	r.add(frame(t, `{"choices":[{"index":0,"text":"!"}]}`))
	require.EqualValues(t, 6, r.bufferedBytes())
}

func TestShapeForAPIType(t *testing.T) {
	require.Equal(t, sseShapeChat, shapeForAPIType(reqcommon.APITypeChatCompletions))
	require.Equal(t, sseShapeText, shapeForAPIType(reqcommon.APITypeCompletions))
	require.Equal(t, sseShapeGenerate, shapeForAPIType(reqcommon.APITypeVLLMGenerate))
}

// Frames mirror the real vLLM /inference/v1/generate stream: one token_id per
// frame under a generate-tokens-* request_id, finish_reason on the final content
// frame, then a trailing choices:[] usage frame (data: [DONE] follows upstream).
// The reassembled reply must match the endpoint's non-streaming reply: a
// request_id envelope with prompt_logprobs and kv_transfer_params null, per
// choice only {index, logprobs, finish_reason, token_ids}, and no usage block.
func TestSSEReassemble_Generate(t *testing.T) {
	r := newSSEReassembler(sseShapeGenerate)
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
	r := newSSEReassembler(sseShapeGenerate)
	r.add(frame(t, `{"request_id":"g","choices":[{"index":0,"finish_reason":"stop"}]}`))
	choice := r.result()["choices"].([]any)[0].(map[string]any)
	require.Equal(t, []any{}, choice["token_ids"])
}

func TestSSEReassemble_GenerateBufferedBytesCountsTokens(t *testing.T) {
	r := newSSEReassembler(sseShapeGenerate)
	require.Zero(t, r.bufferedBytes())
	r.add(frame(t, `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3]}]}`))
	require.EqualValues(t, 3*forceStreamBytesPerToken, r.bufferedBytes())
}
