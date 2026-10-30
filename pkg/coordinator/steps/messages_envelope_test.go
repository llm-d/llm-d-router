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

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func wrapMessagesMap(t *testing.T, generate string) map[string]any {
	t.Helper()
	wrapped, err := wrapMessages([]byte(generate), "fallback-id", "fallback-model", reqcommon.OutputModeText)
	if err != nil {
		t.Fatalf("wrapMessages: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(wrapped, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return got
}

// TestWrapMessages_TextResponse verifies the envelope rewrite: the text becomes
// a text block, finish_reason becomes stop_reason, usage is renamed with cached
// tokens split out, and generate-only and OpenAI-only fields are dropped.
func TestWrapMessages_TextResponse(t *testing.T) {
	got := wrapMessagesMap(t, `{
		"request_id": "req-1", "model": "test-model", "created": 1700000000, "object": "chat.completion",
		"output_mode": "text",
		"usage": {"prompt_tokens": 15, "completion_tokens": 9, "total_tokens": 24,
			"prompt_tokens_details": {"cached_tokens": 5}},
		"kv_transfer_params": {"remote_host": "10.0.0.1"},
		"prompt_token_ids": [1, 2, 3],
		"choices": [{"index": 0, "finish_reason": "stop", "text": "2 + 2 = 4", "token_ids": [17]}]
	}`)

	if got[reqcommon.FieldType] != reqcommon.TypeMessage || got[reqcommon.FieldRole] != reqcommon.RoleAssistant ||
		got[reqcommon.FieldID] != "msg_req-1" || got["model"] != "test-model" {
		t.Errorf("envelope = %v", got)
	}
	if got[reqcommon.FieldStopReason] != reqcommon.StopReasonEndTurn || got[reqcommon.FieldStopSequence] != nil {
		t.Errorf("stop_reason/stop_sequence = %v/%v", got[reqcommon.FieldStopReason], got[reqcommon.FieldStopSequence])
	}
	for _, field := range []string{
		reqcommon.FieldObject, reqcommon.FieldCreated, reqcommon.FieldChoices, reqcommon.FieldRequestID,
		reqcommon.FieldOutputMode, reqcommon.FieldKVTransferParams, "prompt_token_ids",
	} {
		if _, ok := got[field]; ok {
			t.Errorf("field %q reached the client", field)
		}
	}

	block, _ := got[reqcommon.FieldContent].([]any)[0].(map[string]any)
	if block[reqcommon.FieldType] != reqcommon.BlockTypeText || block["text"] != "2 + 2 = 4" {
		t.Errorf("block = %v", block)
	}

	// input_tokens excludes the cache read, unlike prompt_tokens.
	usage, _ := got[reqcommon.FieldUsage].(map[string]any)
	if usage[reqcommon.FieldInputTokens] != float64(10) || usage[reqcommon.FieldOutputTokens] != float64(9) ||
		usage[reqcommon.FieldCacheReadInputTokens] != float64(5) {
		t.Errorf("usage = %v", usage)
	}
	if _, ok := usage[reqcommon.FieldTotalTokens]; ok {
		t.Error("total_tokens reached the client")
	}
}

func TestWrapMessages_StopReasons(t *testing.T) {
	for finish, want := range map[string]any{
		"stop":   reqcommon.StopReasonEndTurn,
		"length": reqcommon.StopReasonMaxTokens,
	} {
		got := wrapMessagesMap(t, `{"output_mode":"text","choices":[{"finish_reason":"`+finish+`","text":"x"}]}`)
		if got[reqcommon.FieldStopReason] != want {
			t.Errorf("finish_reason %q: stop_reason = %v, want %v", finish, got[reqcommon.FieldStopReason], want)
		}
	}
}

func TestWrapMessages_MissingUsageAndFallbackFields(t *testing.T) {
	got := wrapMessagesMap(t, `{"output_mode":"text","choices":[{"finish_reason":"stop","text":""}]}`)
	if got[reqcommon.FieldID] != "msg_fallback-id" {
		t.Errorf("id = %v, want the coordinator's request id with msg_ prefix", got[reqcommon.FieldID])
	}
	if got[reqcommon.FieldModel] != "fallback-model" {
		t.Errorf("model = %v, want fallback-model", got[reqcommon.FieldModel])
	}
	if _, ok := got[reqcommon.FieldUsage]; ok {
		t.Error("usage present although the generate response had none")
	}
	if content, ok := got[reqcommon.FieldContent].([]any); !ok || len(content) != 1 {
		t.Errorf("content = %v, want one text block", got[reqcommon.FieldContent])
	}
}

func TestWrapMessages_IDPrefixing(t *testing.T) {
	// A request id that already has msg_ is preserved without doubling.
	got := wrapMessagesMap(t, `{"request_id":"msg_already_prefixed","output_mode":"text","choices":[{"finish_reason":"stop","text":""}]}`)
	if got[reqcommon.FieldID] != "msg_already_prefixed" {
		t.Errorf("id = %v, want msg_already_prefixed", got[reqcommon.FieldID])
	}
}

func TestWrapMessages_UsageMissingCompletionTokens(t *testing.T) {
	got := wrapMessagesMap(t, `{"output_mode":"text","usage":{"prompt_tokens":7},"choices":[{"finish_reason":"stop","text":""}]}`)
	usage, _ := got[reqcommon.FieldUsage].(map[string]any)
	if usage[reqcommon.FieldInputTokens] != float64(7) || usage[reqcommon.FieldOutputTokens] != float64(0) {
		t.Errorf("usage = %v, want input_tokens=7 and output_tokens=0", usage)
	}
}

func TestWrapMessages_UsageWithoutCache(t *testing.T) {
	got := wrapMessagesMap(t, `{"output_mode":"text","usage":{"prompt_tokens":7,"completion_tokens":3},"choices":[{"finish_reason":"stop","text":""}]}`)
	usage, _ := got[reqcommon.FieldUsage].(map[string]any)
	if usage[reqcommon.FieldInputTokens] != float64(7) || usage[reqcommon.FieldOutputTokens] != float64(3) {
		t.Errorf("usage = %v", usage)
	}
	if _, ok := usage[reqcommon.FieldCacheReadInputTokens]; ok {
		t.Error("cache_read_input_tokens present without a cached count")
	}
}

// TestWrapMessages_UsageWithCacheCreation verifies cache creation tokens are
// deducted alongside cached read tokens.
func TestWrapMessages_UsageWithCacheCreation(t *testing.T) {
	got := wrapMessagesMap(t, `{
		"output_mode":"text",
		"usage":{
			"prompt_tokens":20,
			"completion_tokens":5,
			"cache_creation_input_tokens":6,
			"prompt_tokens_details":{"cached_tokens":10}
		},
		"choices":[{"finish_reason":"stop","text":""}]
	}`)
	usage, _ := got[reqcommon.FieldUsage].(map[string]any)
	// 20 prompt tokens - 10 cached - 6 created = 4 input tokens.
	if usage[reqcommon.FieldInputTokens] != float64(4) || usage[reqcommon.FieldCacheReadInputTokens] != float64(10) ||
		usage[reqcommon.FieldCacheCreationInputTokens] != float64(6) {
		t.Errorf("usage = %v", usage)
	}
}

// TestWrapMessages_Errors verifies responses the wrap cannot represent are
// reported rather than forwarded.
func TestWrapMessages_Errors(t *testing.T) {
	for name, generate := range map[string]string{
		"empty choices":                        `{"output_mode":"text","choices":[]}`,
		"multiple choices":                     `{"output_mode":"text","choices":[{"finish_reason":"stop"},{"finish_reason":"stop"}]}`,
		"missing finish reason":                `{"output_mode":"text","choices":[{"text":"hi"}]}`,
		"unmappable finish":                    `{"output_mode":"text","choices":[{"finish_reason":"content_filter","text":""}]}`,
		"negative prompt tokens":               `{"output_mode":"text","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":-1}}`,
		"cached tokens exceed prompt tokens":   `{"output_mode":"text","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"prompt_tokens_details":{"cached_tokens":10}}}`,
		"creation tokens exceed prompt tokens": `{"output_mode":"text","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"cache_creation_input_tokens":6}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := wrapMessages([]byte(generate), "req", "test-model", reqcommon.OutputModeText); err == nil {
				t.Fatal("expected an error")
			}
		})
	}

	t.Run("missing model", func(t *testing.T) {
		generate := `{"output_mode":"text","choices":[{"finish_reason":"stop","text":"ok"}]}`
		if _, err := wrapMessages([]byte(generate), "req", "", reqcommon.OutputModeText); err == nil {
			t.Fatal("expected an error when model is omitted")
		}
	})
}
