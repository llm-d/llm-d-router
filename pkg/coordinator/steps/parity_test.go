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
		{"completions with ids as prompt", map[string]any{"prompt": []uint32{1, 2}}, reqcommon.APITypeCompletions, true},
		{"completions with text prompt", map[string]any{"prompt": "hi"}, reqcommon.APITypeCompletions, false},
		{"chat with forwarded ids", map[string]any{reqcommon.FieldKVTransferParams: map[string]any{"prompt_token_ids": []uint32{1, 2}}}, reqcommon.APITypeChatCompletions, true},
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
