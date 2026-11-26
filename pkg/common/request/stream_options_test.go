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

package request

import (
	"encoding/json"
	"testing"
)

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
				body[FieldStreamOptions] = tc.value
			}
			_, present, ok := streamOptions(body)
			if present != tc.present || ok != tc.ok {
				t.Fatalf("streamOptions = (present %v, ok %v), want (%v, %v)", present, ok, tc.present, tc.ok)
			}
		})
	}
}

func TestInjectIncludeUsage(t *testing.T) {
	tests := []struct {
		name         string
		body         map[string]any
		stream       bool
		apiType      APIType
		wantInjected bool
		wantOptions  any
	}{
		{
			name:         "chat stream without stream_options",
			body:         map[string]any{},
			stream:       true,
			apiType:      APITypeChatCompletions,
			wantInjected: true,
			wantOptions:  map[string]any{FieldIncludeUsage: true},
		},
		{
			name:         "completions stream with include_usage false",
			body:         map[string]any{FieldStreamOptions: map[string]any{FieldIncludeUsage: false}},
			stream:       true,
			apiType:      APITypeCompletions,
			wantInjected: true,
			wantOptions:  map[string]any{FieldIncludeUsage: true},
		},
		{
			name:        "client already asked for usage",
			body:        map[string]any{FieldStreamOptions: map[string]any{FieldIncludeUsage: true}},
			stream:      true,
			apiType:     APITypeChatCompletions,
			wantOptions: map[string]any{FieldIncludeUsage: true},
		},
		{
			name:        "continuous usage alone is left alone",
			body:        map[string]any{FieldStreamOptions: map[string]any{FieldContinuousUsageStats: true}},
			stream:      true,
			apiType:     APITypeChatCompletions,
			wantOptions: map[string]any{FieldContinuousUsageStats: true},
		},
		{
			name:    "non-streaming request",
			body:    map[string]any{},
			apiType: APITypeChatCompletions,
		},
		{
			name:    "responses stream carries usage on its own",
			body:    map[string]any{},
			stream:  true,
			apiType: APITypeResponses,
		},
		{
			name:    "messages stream carries usage on its own",
			body:    map[string]any{},
			stream:  true,
			apiType: APITypeMessages,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			injected, invalid := InjectIncludeUsage(tc.body, tc.stream, tc.apiType)
			if injected != tc.wantInjected || invalid {
				t.Fatalf("InjectIncludeUsage = (injected %v, invalid %v), want (%v, false)", injected, invalid, tc.wantInjected)
			}
			got, present := tc.body[FieldStreamOptions]
			if tc.wantOptions == nil {
				if present {
					t.Fatalf("stream_options = %v, want absent", got)
				}
				return
			}
			gotJSON, _ := json.Marshal(got)
			wantJSON, _ := json.Marshal(tc.wantOptions)
			if string(gotJSON) != string(wantJSON) {
				t.Fatalf("stream_options = %s, want %s", gotJSON, wantJSON)
			}
		})
	}
}

// A malformed stream_options is forwarded as the client sent it, and nothing
// is injected.
func TestInjectIncludeUsage_LeavesMalformedStreamOptionsAlone(t *testing.T) {
	body := map[string]any{FieldStreamOptions: json.RawMessage(`"x"`)}
	injected, invalid := InjectIncludeUsage(body, true, APITypeChatCompletions)
	if injected || !invalid {
		t.Fatalf("InjectIncludeUsage = (injected %v, invalid %v), want (false, true)", injected, invalid)
	}
	if raw, ok := body[FieldStreamOptions].(json.RawMessage); !ok || string(raw) != `"x"` {
		t.Fatalf("stream_options must be left as sent, got %v", body[FieldStreamOptions])
	}
}

// stream_options can be the client's own decoded object, shared with another
// view of the request (the coordinator's conditional-decode shallow clone). The
// injection must build a copy, or a 412 would hand the decode step
// include_usage=true and the client a usage frame it never asked for.
func TestInjectIncludeUsage_CopiesClientStreamOptions(t *testing.T) {
	client := map[string]any{"include_obfuscation": false}
	body := map[string]any{FieldStreamOptions: client}

	injected, invalid := InjectIncludeUsage(body, true, APITypeChatCompletions)
	if !injected || invalid {
		t.Fatalf("InjectIncludeUsage = (injected %v, invalid %v), want (true, false)", injected, invalid)
	}
	if _, leaked := client[FieldIncludeUsage]; leaked {
		t.Fatal("injection wrote include_usage into the client's stream_options map")
	}
	merged, _ := body[FieldStreamOptions].(map[string]any)
	if merged[FieldIncludeUsage] != true || merged["include_obfuscation"] != false {
		t.Fatalf("merged stream_options = %v, want the client's fields plus include_usage", merged)
	}
}
