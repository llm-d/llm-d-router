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

func wrapResponsesMap(t *testing.T, generate string) map[string]any {
	t.Helper()
	wrapped, err := wrapResponses([]byte(generate), "fallback-id", "fallback-model", reqcommon.OutputModeText)
	if err != nil {
		t.Fatalf("wrapResponses: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(wrapped, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return got
}

// TestWrapResponses_TextResponse verifies the envelope rewrite: shared fields
// carry over, usage is renamed, the choice becomes a message item with empty
// annotations, and generate-only fields are dropped.
func TestWrapResponses_TextResponse(t *testing.T) {
	got := wrapResponsesMap(t, `{
		"request_id": "req-1", "model": "test-model", "created": 1700000000,
		"output_mode": "text",
		"usage": {"prompt_tokens": 15, "completion_tokens": 9, "total_tokens": 24,
			"prompt_tokens_details": {"cached_tokens": 5}},
		"kv_transfer_params": {"remote_host": "10.0.0.1"},
		"prompt_token_ids": [1, 2, 3],
		"choices": [{"index": 0, "finish_reason": "stop", "text": "2 + 2 = 4", "token_ids": [17]}]
	}`)

	if got[reqcommon.FieldObject] != reqcommon.ObjectResponse || got[reqcommon.FieldStatus] != reqcommon.StatusCompleted {
		t.Errorf("object/status = %v/%v", got[reqcommon.FieldObject], got[reqcommon.FieldStatus])
	}
	if got[reqcommon.FieldID] != "req-1" || got["model"] != "test-model" || got[reqcommon.FieldCreatedAt] != float64(1700000000) {
		t.Errorf("shared fields not carried over: %v", got)
	}
	if _, ok := got["incomplete_details"]; ok {
		t.Error("incomplete_details present on a completed response")
	}
	for _, field := range []string{
		reqcommon.FieldRequestID, reqcommon.FieldOutputMode, reqcommon.FieldKVTransferParams,
		"prompt_token_ids", reqcommon.FieldChoices, reqcommon.FieldCreated,
	} {
		if _, ok := got[field]; ok {
			t.Errorf("field %q reached the client", field)
		}
	}

	usage, _ := got[reqcommon.FieldUsage].(map[string]any)
	if usage[reqcommon.FieldInputTokens] != float64(15) || usage[reqcommon.FieldOutputTokens] != float64(9) ||
		usage[reqcommon.FieldTotalTokens] != float64(24) {
		t.Errorf("usage = %v", usage)
	}
	details, _ := usage[reqcommon.FieldInputTokensDetails].(map[string]any)
	if details[reqcommon.FieldCachedTokens] != float64(5) {
		t.Errorf("input_tokens_details = %v", details)
	}
	if _, ok := usage[reqcommon.FieldPromptTokens]; ok {
		t.Error("prompt_tokens remained in responses usage")
	}

	output, _ := got[reqcommon.FieldOutput].([]any)
	if len(output) != 1 {
		t.Fatalf("output = %v, want one item", got[reqcommon.FieldOutput])
	}
	item, _ := output[0].(map[string]any)
	if item[reqcommon.FieldID] != "msg_req-1_0" || item[reqcommon.FieldType] != reqcommon.TypeMessage ||
		item[reqcommon.FieldRole] != reqcommon.RoleAssistant || item[reqcommon.FieldStatus] != reqcommon.StatusCompleted {
		t.Errorf("item = %v", item)
	}
	part, _ := item[reqcommon.FieldContent].([]any)[0].(map[string]any)
	if part[reqcommon.FieldType] != reqcommon.PartTypeOutputText || part["text"] != "2 + 2 = 4" {
		t.Errorf("part = %v", part)
	}
	annotations, ok := part["annotations"].([]any)
	if !ok || len(annotations) != 0 {
		t.Errorf("annotations = %v, want empty array", part["annotations"])
	}
}

// TestWrapResponses_Length verifies a token-limit stop is reported as an
// incomplete response and marks its output item incomplete.
func TestWrapResponses_Length(t *testing.T) {
	got := wrapResponsesMap(t, `{"request_id":"r","output_mode":"text","choices":[{"finish_reason":"length","text":"cut"}]}`)
	if got[reqcommon.FieldStatus] != reqcommon.StatusIncomplete {
		t.Errorf("status = %v, want incomplete", got[reqcommon.FieldStatus])
	}
	details, _ := got["incomplete_details"].(map[string]any)
	if details["reason"] != "max_output_tokens" {
		t.Errorf("incomplete_details = %v", details)
	}
	output, _ := got[reqcommon.FieldOutput].([]any)
	if len(output) != 1 {
		t.Fatalf("output = %v, want 1 item", output)
	}
	item := output[0].(map[string]any)
	if item[reqcommon.FieldStatus] != reqcommon.StatusIncomplete {
		t.Errorf("output item status = %v, want incomplete", item[reqcommon.FieldStatus])
	}
}

func TestWrapResponses_MissingUsageAndFallbackFields(t *testing.T) {
	got := wrapResponsesMap(t, `{"output_mode":"text","choices":[{"finish_reason":"stop","text":"ok"}]}`)
	if got[reqcommon.FieldID] != "fallback-id" {
		t.Errorf("id = %v, want the coordinator's request id", got[reqcommon.FieldID])
	}
	if got[reqcommon.FieldModel] != "fallback-model" {
		t.Errorf("model = %v, want fallback-model", got[reqcommon.FieldModel])
	}
	if _, ok := got[reqcommon.FieldUsage]; ok {
		t.Error("usage present although the generate response had none")
	}
}

// TestWrapResponses_Errors verifies responses the wrap cannot represent are
// reported rather than forwarded.
func TestWrapResponses_Errors(t *testing.T) {
	for name, generate := range map[string]string{
		"empty choices":          `{"output_mode":"text","choices":[]}`,
		"multiple choices":       `{"output_mode":"text","choices":[{"finish_reason":"stop"},{"finish_reason":"stop"}]}`,
		"missing finish reason":  `{"output_mode":"text","choices":[{"text":"hi"}]}`,
		"unmappable finish":      `{"output_mode":"text","choices":[{"finish_reason":"content_filter","text":""}]}`,
		"negative prompt tokens": `{"output_mode":"text","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":-1}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := wrapResponses([]byte(generate), "req", "test-model", reqcommon.OutputModeText); err == nil {
				t.Fatal("expected an error")
			}
		})
	}

	t.Run("missing model", func(t *testing.T) {
		generate := `{"output_mode":"text","choices":[{"finish_reason":"stop","text":"ok"}]}`
		if _, err := wrapResponses([]byte(generate), "req", "", reqcommon.OutputModeText); err == nil {
			t.Fatal("expected an error when model is omitted")
		}
	})
}
