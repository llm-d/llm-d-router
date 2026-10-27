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

// TestWrapChatCompletion_DerenderResponse verifies the envelope rewrite of an
// output_mode derender response: the shared fields carry over, the message is
// preserved whole, and the generate-only coordination fields are dropped.
func TestWrapChatCompletion_DerenderResponse(t *testing.T) {
	generate := []byte(`{
		"request_id": "req-1",
		"model": "test-model",
		"created": 1700000000,
		"output_mode": "derender",
		"usage": {"prompt_tokens": 15, "completion_tokens": 9, "total_tokens": 24},
		"kv_transfer_params": {"remote_host": "10.0.0.1"},
		"ec_transfer_params": null,
		"prompt_token_ids": [1, 2, 3],
		"metrics": {"arrival_time": 1.0},
		"choices": [{
			"index": 0,
			"finish_reason": "stop",
			"message": {"role": "assistant", "content": "2 + 2 = 4", "reasoning": "a chain", "tool_calls": [{"id": "c1"}]},
			"token_ids": [17, 488],
			"weight_versions": [{"start": 0, "end": 2}]
		}]
	}`)

	wrapped, err := wrapChatCompletion(generate, "fallback-id", reqcommon.OutputModeDerender)
	if err != nil {
		t.Fatalf("wrapChatCompletion: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(wrapped, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got[reqcommon.FieldObject] != reqcommon.ObjectChatCompletion {
		t.Errorf("object = %v, want %q", got[reqcommon.FieldObject], reqcommon.ObjectChatCompletion)
	}
	// The generate response named the request, so that id is kept.
	if got[reqcommon.FieldID] != "req-1" {
		t.Errorf("id = %v, want req-1", got[reqcommon.FieldID])
	}
	if got["model"] != "test-model" {
		t.Errorf("model = %v, want test-model", got["model"])
	}
	if got[reqcommon.FieldCreated] != float64(1700000000) {
		t.Errorf("created = %v, want 1700000000", got[reqcommon.FieldCreated])
	}
	// Usage is shared by both schemas and must survive.
	usage, ok := got[reqcommon.FieldUsage].(map[string]any)
	if !ok || usage[reqcommon.FieldPromptTokens] != float64(15) {
		t.Errorf("usage = %v, want the generate block carried over", got[reqcommon.FieldUsage])
	}

	for _, field := range []string{
		reqcommon.FieldRequestID, reqcommon.FieldOutputMode,
		reqcommon.FieldKVTransferParams, reqcommon.FieldECTransferParams,
		"prompt_token_ids", "metrics",
	} {
		if _, present := got[field]; present {
			t.Errorf("generate-only field %q reached the client", field)
		}
	}

	choices, _ := got[reqcommon.FieldChoices].([]any)
	if len(choices) != 1 {
		t.Fatalf("choices = %v, want one", got[reqcommon.FieldChoices])
	}
	choice, _ := choices[0].(map[string]any)
	if choice[reqcommon.FieldFinishReason] != "stop" {
		t.Errorf("finish_reason = %v, want stop", choice[reqcommon.FieldFinishReason])
	}
	for _, field := range []string{"token_ids", "weight_versions"} {
		if _, present := choice[field]; present {
			t.Errorf("generate-only choice field %q reached the client", field)
		}
	}
	// The message moves across whole, so derender's parser output survives.
	msg, ok := choice[reqcommon.FieldMessage].(map[string]any)
	if !ok {
		t.Fatalf("message = %v, want an object", choice[reqcommon.FieldMessage])
	}
	if msg[reqcommon.FieldContent] != "2 + 2 = 4" {
		t.Errorf("content = %v, want the generated text", msg[reqcommon.FieldContent])
	}
	if msg["reasoning"] != "a chain" {
		t.Errorf("reasoning = %v, want it preserved", msg["reasoning"])
	}
	if calls, _ := msg["tool_calls"].([]any); len(calls) != 1 {
		t.Errorf("tool_calls = %v, want them preserved", msg["tool_calls"])
	}
}

// TestWrapChatCompletion_TextResponse verifies that an output_mode text
// response, whose choices carry text rather than a message, is given the
// message a chat completions client reads.
func TestWrapChatCompletion_TextResponse(t *testing.T) {
	generate := []byte(`{"request_id":"req-2","output_mode":"text",
		"choices":[{"index":0,"finish_reason":"length","text":" Hello","token_ids":[9707]}]}`)

	wrapped, err := wrapChatCompletion(generate, "fallback-id", reqcommon.OutputModeText)
	if err != nil {
		t.Fatalf("wrapChatCompletion: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(wrapped, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	choice, _ := got[reqcommon.FieldChoices].([]any)[0].(map[string]any)
	msg, ok := choice[reqcommon.FieldMessage].(map[string]any)
	if !ok {
		t.Fatalf("message = %v, want one built from the text", choice[reqcommon.FieldMessage])
	}
	if msg[reqcommon.FieldRole] != reqcommon.RoleAssistant {
		t.Errorf("role = %v, want %q", msg[reqcommon.FieldRole], reqcommon.RoleAssistant)
	}
	if msg[reqcommon.FieldContent] != " Hello" {
		t.Errorf("content = %v, want the text verbatim", msg[reqcommon.FieldContent])
	}
	// The raw text field is not part of a chat completions choice.
	if _, present := choice["text"]; present {
		t.Error("text field reached the client alongside the message")
	}
}

// TestWrapChatCompletion_FallbackID verifies a response that omitted
// request_id still names the request the client asked about.
func TestWrapChatCompletion_FallbackID(t *testing.T) {
	wrapped, err := wrapChatCompletion([]byte(`{"output_mode":"derender","choices":[]}`), "fallback-id", reqcommon.OutputModeDerender)
	if err != nil {
		t.Fatalf("wrapChatCompletion: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(wrapped, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got[reqcommon.FieldID] != "fallback-id" {
		t.Errorf("id = %v, want the coordinator's request id", got[reqcommon.FieldID])
	}
	// choices is required by the schema even when the answer carried none.
	if _, present := got[reqcommon.FieldChoices]; !present {
		t.Error("choices key is absent")
	}
}

// TestWrapChatCompletion_InvalidJSON verifies a body that is not a JSON object
// is reported rather than forwarded.
func TestWrapChatCompletion_InvalidJSON(t *testing.T) {
	if _, err := wrapChatCompletion([]byte(`not json`), "req", reqcommon.OutputModeText); err == nil {
		t.Fatal("expected an error for a non-JSON response")
	}
}

// TestWrapChatCompletion_ModeNotEchoed verifies a worker that did not answer in
// the requested mode is reported rather than rewritten. A worker predating
// output_mode ignores it and returns token IDs only, which would otherwise
// reach the client as a message with no content.
func TestWrapChatCompletion_ModeNotEchoed(t *testing.T) {
	for _, tc := range []struct {
		name     string
		generate string
	}{
		{name: "field absent", generate: `{"request_id":"r1","choices":[{"index":0,"token_ids":[17]}]}`},
		{name: "answered tokens", generate: `{"request_id":"r1","output_mode":"tokens","choices":[{"index":0,"token_ids":[17]}]}`},
		{name: "answered another mode", generate: `{"request_id":"r1","output_mode":"text","choices":[{"index":0,"text":"hi"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := wrapChatCompletion([]byte(tc.generate), "req", reqcommon.OutputModeDerender); err == nil {
				t.Fatal("expected an error when the worker did not echo the requested mode")
			}
		})
	}
}
