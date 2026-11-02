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

func TestIdsForwardedIn(t *testing.T) {
	tests := []struct {
		name    string
		body    map[string]any
		apiType reqcommon.APIType
		want    bool
	}{
		{"generate route", map[string]any{"token_ids": []int{1}}, reqcommon.APITypeVLLMGenerate, true},
		{"completions with ids as prompt", map[string]any{"prompt": []int{1, 2}}, reqcommon.APITypeCompletions, true},
		{"completions with text prompt", map[string]any{"prompt": "hi"}, reqcommon.APITypeCompletions, false},
		{"chat with forwarded ids", map[string]any{reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []int{1, 2}}}, reqcommon.APITypeChatCompletions, true},
		{"chat with kv params but no ids", map[string]any{reqcommon.FieldKVTransferParams: map[string]any{"do_remote_prefill": true}}, reqcommon.APITypeChatCompletions, false},
		{"responses without kv params", map[string]any{"input": "hi"}, reqcommon.APITypeResponses, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := idsForwardedIn(tc.body, tc.apiType); got != tc.want {
				t.Fatalf("idsForwardedIn = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestStreamOptions(t *testing.T) {
	tests := []struct {
		name    string
		value   any
		absent  bool
		present bool
		ok      bool
	}{
		{name: "absent", absent: true, present: false, ok: true},
		{name: "decoded map", value: map[string]any{"include_usage": true}, present: true, ok: true},
		{name: "raw object", value: json.RawMessage(`{"include_usage": true}`), present: true, ok: true},
		{name: "raw null counts as absent", value: json.RawMessage(`null`), present: false, ok: true},
		{name: "raw string is invalid", value: json.RawMessage(`"x"`), present: true, ok: false},
		{name: "raw array is invalid", value: json.RawMessage(`[1]`), present: true, ok: false},
		{name: "unexpected go type is invalid", value: 7, present: true, ok: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := map[string]any{}
			if !tc.absent {
				body[reqcommon.FieldStreamOptions] = tc.value
			}
			_, present, ok := streamOptions(body)
			if present != tc.present || ok != tc.ok {
				t.Fatalf("streamOptions = (present %v, ok %v), want (%v, %v)", present, ok, tc.present, tc.ok)
			}
		})
	}
}

// A malformed stream_options is forwarded as the client sent it, and the check
// is skipped: the worker's answer to a body the coordinator would not have
// sent is not something to compare.
func TestInjectIncludeUsage_LeavesMalformedStreamOptionsAlone(t *testing.T) {
	body := map[string]any{reqcommon.FieldStreamOptions: json.RawMessage(`"x"`)}
	injected, invalid := injectIncludeUsage(body, true, reqcommon.APITypeChatCompletions)
	if injected || !invalid {
		t.Fatalf("injectIncludeUsage = (injected %v, invalid %v), want (false, true)", injected, invalid)
	}
	if raw, ok := body[reqcommon.FieldStreamOptions].(json.RawMessage); !ok || string(raw) != `"x"` {
		t.Fatalf("stream_options must be left as sent, got %v", body[reqcommon.FieldStreamOptions])
	}
}
