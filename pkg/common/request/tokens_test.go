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
	"maps"
	"reflect"
	"strings"
	"testing"
)

// Regression test for the sampling_params sharing that CapSingleToken documents.
func TestCapSingleToken_LeavesTheCallersNestedMapIntact(t *testing.T) {
	client := map[string]any{
		"token_ids":         []any{1, 2, 3},
		FieldSamplingParams: map[string]any{FieldMaxTokens: 200, FieldMinTokens: 10},
	}

	prefill := maps.Clone(client)
	CapSingleToken(prefill, APITypeVLLMGenerate)

	decodeLimits := client[FieldSamplingParams].(map[string]any)
	if got := decodeLimits[FieldMaxTokens]; got != 200 {
		t.Errorf("decode request max_tokens = %v, want the client's 200", got)
	}
	if got, ok := decodeLimits[FieldMinTokens]; !ok || got != 10 {
		t.Errorf("decode request min_tokens = %v (present %v), want the client's 10", got, ok)
	}

	prefillLimits := prefill[FieldSamplingParams].(map[string]any)
	if got := prefillLimits[FieldMaxTokens]; got != 1 {
		t.Errorf("prefill request max_tokens = %v, want 1", got)
	}
	if _, ok := prefillLimits[FieldMinTokens]; ok {
		t.Error("prefill request kept min_tokens")
	}
}

// Callers add transfer params to the returned map, so writes into it must
// reach the body that is sent.
func TestCapSingleToken_ReturnsTheCappedMap(t *testing.T) {
	tests := []struct {
		name    string
		apiType APIType
		body    map[string]any
		limits  func(body map[string]any) map[string]any
	}{
		{
			name:    "chat completions returns the body",
			apiType: APITypeChatCompletions,
			body:    map[string]any{"model": "m"},
			limits:  func(body map[string]any) map[string]any { return body },
		},
		{
			name:    "generate returns the body's sampling_params",
			apiType: APITypeVLLMGenerate,
			body:    map[string]any{"model": "m", FieldSamplingParams: map[string]any{"temperature": 0.5}},
			limits:  func(body map[string]any) map[string]any { return body[FieldSamplingParams].(map[string]any) },
		},
		{
			name:    "generate returns a synthesized sampling_params",
			apiType: APITypeVLLMGenerate,
			body:    map[string]any{"model": "m"},
			limits:  func(body map[string]any) map[string]any { return body[FieldSamplingParams].(map[string]any) },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CapSingleToken(tt.body, tt.apiType)
			got["marker"] = true

			if limits := tt.limits(tt.body); limits["marker"] != true {
				t.Fatalf("write into the returned map did not reach the body: %v", tt.body)
			}
		})
	}
}

func TestCapSingleToken(t *testing.T) {
	tests := []struct {
		name    string
		apiType APIType
		body    map[string]any
		want    map[string]any
	}{
		{
			name:    "chat completions caps output fields and forces non-streaming",
			apiType: APITypeChatCompletions,
			body: map[string]any{
				"model":                 "m",
				"max_tokens":            100,
				"min_tokens":            5,
				"max_completion_tokens": 100,
				"stream":                true,
				"stream_options":        map[string]any{"include_usage": true},
			},
			want: map[string]any{
				"model":                 "m",
				"max_tokens":            1,
				"max_completion_tokens": 1,
				"stream":                false,
			},
		},
		{
			name:    "max_completion_tokens is added even when the client omitted it",
			apiType: APITypeChatCompletions,
			body:    map[string]any{"model": "m"},
			want: map[string]any{
				"model":                 "m",
				"max_tokens":            1,
				"max_completion_tokens": 1,
				"stream":                false,
			},
		},
		{
			name:    "an unrecognized API is capped as chat completions",
			apiType: APIType(7),
			body: map[string]any{
				"model":           "m",
				"max_tokens":      100,
				"min_tokens":      5,
				"stream":          true,
				"stream_options":  map[string]any{"include_usage": true},
				"sampling_params": map[string]any{"max_tokens": 100},
			},
			want: map[string]any{
				"model":                 "m",
				"max_tokens":            1,
				"max_completion_tokens": 1,
				"stream":                false,
				"sampling_params":       map[string]any{"max_tokens": 100},
			},
		},
		{
			name:    "completions caps max_tokens, strips min_tokens, forces non-streaming",
			apiType: APITypeCompletions,
			body:    map[string]any{"model": "m", "max_tokens": 100, "min_tokens": 5},
			want:    map[string]any{"model": "m", "max_tokens": 1, "stream": false},
		},
		{
			name:    "messages caps only max_tokens, strips min_tokens, forces non-streaming",
			apiType: APITypeMessages,
			body:    map[string]any{"model": "m", "max_tokens": 50, "min_tokens": 5, "stream": true},
			want:    map[string]any{"model": "m", "max_tokens": 1, "stream": false},
		},
		{
			name:    "streaming is forced false and stream_options stripped",
			apiType: APITypeCompletions,
			body:    map[string]any{"stream": true, "stream_options": map[string]any{"include_usage": true}},
			want:    map[string]any{"stream": false, "max_tokens": 1},
		},
		{
			name:    "generate caps max_tokens and strips min_tokens inside sampling_params",
			apiType: APITypeVLLMGenerate,
			body: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 100, "min_tokens": 5},
			},
			want: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 1},
				"stream":          false,
			},
		},
		{
			name:    "generate synthesizes sampling_params when absent",
			apiType: APITypeVLLMGenerate,
			body:    map[string]any{"model": "m"},
			want: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 1},
				"stream":          false,
			},
		},
		{
			// The sidecar caps the request straight off the client body, with no
			// type guard for sampling_params ahead of it, so a malformed value
			// arrives here. The request still has to carry a cap, so the field
			// is replaced.
			name:    "generate replaces a non-object sampling_params",
			apiType: APITypeVLLMGenerate,
			body:    map[string]any{"model": "m", "sampling_params": "not-an-object"},
			want: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 1},
				"stream":          false,
			},
		},
		{
			name:    "generate replaces a null sampling_params",
			apiType: APITypeVLLMGenerate,
			body:    map[string]any{"model": "m", "sampling_params": nil},
			want: map[string]any{
				"model":           "m",
				"sampling_params": map[string]any{"max_tokens": 1},
				"stream":          false,
			},
		},
		{
			name:    "generate leaves the top-level fields alone",
			apiType: APITypeVLLMGenerate,
			body: map[string]any{
				"max_tokens":            100,
				"max_completion_tokens": 100,
				"sampling_params":       map[string]any{"max_tokens": 100},
			},
			want: map[string]any{
				"max_tokens":            100,
				"max_completion_tokens": 100,
				"sampling_params":       map[string]any{"max_tokens": 1},
				"stream":                false,
			},
		},
		{
			// store is pinned false so the synthetic leg leaves no stored
			// response object; vLLM defaults an absent store to true. The
			// client's own store on a Responses body is overwritten, since
			// only prefill and priming legs are capped.
			name:    "responses caps max_output_tokens and pins store",
			apiType: APITypeResponses,
			body:    map[string]any{"model": "m", "max_output_tokens": 800, "store": true},
			want:    map[string]any{"model": "m", "max_output_tokens": 1, "stream": false, "store": false},
		},
		{
			// The Responses API has no max_tokens field; vLLM's ResponsesRequest
			// ignores it, so max_output_tokens is the only field that caps output
			// length and must be set even when the client never sent it.
			name:    "responses caps max_output_tokens even when the client omitted it",
			apiType: APITypeResponses,
			body:    map[string]any{"model": "m"},
			want:    map[string]any{"model": "m", "max_output_tokens": 1, "stream": false, "store": false},
		},
		{
			// max_tokens and max_completion_tokens are not Responses fields, so
			// tokenLimitFields does not name them and they are left as sent.
			// min_tokens is stripped for every API; see CapSingleToken.
			name:    "responses leaves fields the API does not use",
			apiType: APITypeResponses,
			body:    map[string]any{"model": "m", "max_tokens": 100, "min_tokens": 5, "max_output_tokens": 800},
			want:    map[string]any{"model": "m", "max_tokens": 100, "max_output_tokens": 1, "stream": false, "store": false},
		},
		{
			name:    "generate preserves other sampling_params entries",
			apiType: APITypeVLLMGenerate,
			body: map[string]any{
				"sampling_params": map[string]any{
					"extra_args": map[string]any{"kv_transfer_params": "x"},
				},
			},
			want: map[string]any{
				"sampling_params": map[string]any{
					"max_tokens": 1,
					"extra_args": map[string]any{"kv_transfer_params": "x"},
				},
				"stream": false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			CapSingleToken(tt.body, tt.apiType)
			if !reflect.DeepEqual(tt.body, tt.want) {
				t.Fatalf("got %v, want %v", tt.body, tt.want)
			}
		})
	}
}

func TestRejectStatefulResponsesFields(t *testing.T) {
	tests := []struct {
		name      string
		body      map[string]any
		wantField string // empty means no error
	}{
		{
			name: "no stateful fields",
			body: map[string]any{"input": "hi"},
		},
		{
			name:      "previous_response_id present",
			body:      map[string]any{"input": "hi", FieldPreviousResponseID: "resp-123"},
			wantField: FieldPreviousResponseID,
		},
		{
			name:      "conversation present",
			body:      map[string]any{"input": "hi", FieldConversation: "conv-123"},
			wantField: FieldConversation,
		},
		// An SDK that serializes an unset optional as null sends the key with a
		// null value, which leaves the turn as stateless as omitting it.
		{
			name: "previous_response_id null as raw bytes is unset",
			body: map[string]any{"input": "hi", FieldPreviousResponseID: json.RawMessage(`null`)},
		},
		{
			name: "previous_response_id null decoded is unset",
			body: map[string]any{"input": "hi", FieldPreviousResponseID: nil},
		},
		{
			name: "conversation null as raw bytes is unset",
			body: map[string]any{"input": "hi", FieldConversation: json.RawMessage(`null`)},
		},
		{
			name:      "previous_response_id as raw bytes is rejected",
			body:      map[string]any{"input": "hi", FieldPreviousResponseID: json.RawMessage(`"resp-123"`)},
			wantField: FieldPreviousResponseID,
		},
		{
			name:      "undecodable previous_response_id bytes are rejected",
			body:      map[string]any{"input": "hi", FieldPreviousResponseID: json.RawMessage(`"resp`)},
			wantField: FieldPreviousResponseID,
		},
		{
			name: "background false is the default, not rejected",
			body: map[string]any{"input": "hi", FieldBackground: false},
		},
		{
			name:      "background true is rejected",
			body:      map[string]any{"input": "hi", FieldBackground: true},
			wantField: FieldBackground,
		},
		{
			name: "input is a plain string, no content to walk",
			body: map[string]any{"input": "hi"},
		},
		{
			name: "input_text part has no file_id",
			body: map[string]any{"input": []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "input_text", "text": "hi"},
				}},
			}},
		},
		{
			name: "input_image part references a file_id",
			body: map[string]any{"input": []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "input_image", FieldFileID: "file-123"},
				}},
			}},
			wantField: FieldFileID,
		},
		// The Responses input schema carries file_id outside a message's content
		// too: on a computer_call_output's output object, and inside a
		// function_call_output's output array. Both are input item types, so a
		// walk fixed to input[].content[] forwards the request the guard exists
		// to catch.
		{
			name: "function_call_output output array references a file_id",
			body: map[string]any{"input": []any{
				map[string]any{"type": "function_call_output", "call_id": "c1", "output": []any{
					map[string]any{"type": "input_image", FieldFileID: "file-123", "detail": "auto"},
				}},
			}},
			wantField: FieldFileID,
		},
		{
			name: "computer_call_output output object references a file_id",
			body: map[string]any{"input": []any{
				map[string]any{"type": "computer_call_output", "call_id": "c1", "output": map[string]any{
					"type": "computer_screenshot", FieldFileID: "file-123",
				}},
			}},
			wantField: FieldFileID,
		},
		// Replaying a prior response's output items is the only multi-turn
		// path left once previous_response_id and conversation are refused, so
		// a citation the completed turn reported has to survive the replay.
		// vLLM reads the text of such an item and nothing else.
		{
			name: "a replayed file citation annotation is served",
			body: map[string]any{"input": []any{
				map[string]any{"role": "assistant", "content": []any{
					map[string]any{"type": "output_text", "text": "see the chart", "annotations": []any{
						map[string]any{"type": "file_citation", FieldFileID: "file-123"},
					}},
				}},
			}},
		},
		{
			name: "an input_file part references a file_id",
			body: map[string]any{"input": []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "input_file", FieldFileID: "file-123"},
				}},
			}},
			wantField: FieldFileID,
		},
		// A nullable file_id the client left unset is serialized as null by an
		// SDK that emits every field, and by the server on a computer_screenshot
		// a client replays from a prior response.
		{
			name: "an input_image with a null file_id is served",
			body: map[string]any{"input": []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "input_image", "image_url": "https://example.com/a.png", FieldFileID: nil},
				}},
			}},
		},
		{
			name: "a replayed computer_screenshot with a null file_id is served",
			body: map[string]any{"input": []any{
				map[string]any{"type": "computer_call_output", "call_id": "c1", "output": map[string]any{
					"type": "computer_screenshot", "image_url": "https://example.com/s.png", FieldFileID: nil,
				}},
			}},
		},
		// A file_id on an object the input schema gives no content part type is
		// not a hydration request.
		{
			name: "a file_id on an untyped object is served",
			body: map[string]any{"input": []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "input_text", "text": "hi", "source": map[string]any{FieldFileID: "file-123"}},
				}},
			}},
		},
		{
			name: "a tool output carrying no file reference is served",
			body: map[string]any{"input": []any{
				map[string]any{"type": "function_call_output", "call_id": "c1", "output": "42"},
			}},
		},
		{
			name: "malformed input array does not panic",
			body: map[string]any{"input": []any{"not a map", 42, map[string]any{"content": "not an array"}}},
		},
		// A caller that decodes a body selectively, as the sidecar proxy does,
		// leaves the fields it does not read as raw bytes. Skipping those would
		// report the request as supported without having inspected it.
		{
			name:      "background true as raw bytes is rejected",
			body:      map[string]any{"input": "hi", FieldBackground: json.RawMessage(`true`)},
			wantField: FieldBackground,
		},
		{
			name: "background false as raw bytes is the default",
			body: map[string]any{"input": "hi", FieldBackground: json.RawMessage(`false`)},
		},
		{
			name: "background null as raw bytes is the default",
			body: map[string]any{"input": "hi", FieldBackground: json.RawMessage(`null`)},
		},
		{
			name: "background null decoded is the default",
			body: map[string]any{"input": "hi", FieldBackground: nil},
		},
		// vLLM's request model coerces these to true, so a value this package
		// cannot read as false asks for the unsupported behavior.
		{
			name:      "background 1 as raw bytes is rejected",
			body:      map[string]any{"input": "hi", FieldBackground: json.RawMessage(`1`)},
			wantField: FieldBackground,
		},
		{
			name:      "background 1.0 as raw bytes is rejected",
			body:      map[string]any{"input": "hi", FieldBackground: json.RawMessage(`1.0`)},
			wantField: FieldBackground,
		},
		{
			name:      `background "true" as raw bytes is rejected`,
			body:      map[string]any{"input": "hi", FieldBackground: json.RawMessage(`"true"`)},
			wantField: FieldBackground,
		},
		{
			name:      `background "yes" as raw bytes is rejected`,
			body:      map[string]any{"input": "hi", FieldBackground: json.RawMessage(`"yes"`)},
			wantField: FieldBackground,
		},
		{
			name:      "background decoded as a number is rejected",
			body:      map[string]any{"input": "hi", FieldBackground: float64(1)},
			wantField: FieldBackground,
		},
		{
			name:      "undecodable background bytes are rejected",
			body:      map[string]any{"input": "hi", FieldBackground: json.RawMessage(`{`)},
			wantField: FieldBackground,
		},
		{
			name:      "file_id in an input array of raw bytes is rejected",
			body:      map[string]any{FieldInput: json.RawMessage(`[{"role":"user","content":[{"type":"input_image","file_id":"file-123"}]}]`)},
			wantField: FieldFileID,
		},
		{
			name: "input array of raw bytes with no file_id",
			body: map[string]any{FieldInput: json.RawMessage(`[{"role":"user","content":[{"type":"input_text","text":"hi"}]}]`)},
		},
		{
			name: "input string as raw bytes has nothing to walk",
			body: map[string]any{FieldInput: json.RawMessage(`"hi"`)},
		},
		// An input the walk cannot decode is refused rather than reported as
		// carrying no file_id.
		{
			name:      "input bytes that decode as neither array nor string are refused",
			body:      map[string]any{FieldInput: json.RawMessage(`[{"role":`)},
			wantField: FieldInput,
		},
		// A number outside float64 range is not an uninspectable input: the
		// model server parses it, so the walk has to reach the file_id beside
		// it and name that field rather than the whole input.
		{
			name:      "file_id beside an out-of-range number is named as file_id",
			body:      map[string]any{FieldInput: json.RawMessage(`[{"role":"user","content":[{"type":"input_image","file_id":"file-123"}]},1e999]`)},
			wantField: FieldFileID,
		},
		{
			name:      "out-of-range number nested in the content part still finds file_id",
			body:      map[string]any{FieldInput: json.RawMessage(`[{"role":"user","content":[{"type":"input_image","file_id":"file-123","detail":1e999}]}]`)},
			wantField: FieldFileID,
		},
		{
			name: "an out-of-range number with no file_id is served",
			body: map[string]any{FieldInput: json.RawMessage(`[{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/a.jpg","detail":"auto"}],"pinned":1e999}]`)},
		},
		{
			name:      "input as a bare number is refused",
			body:      map[string]any{FieldInput: json.RawMessage(`42`)},
			wantField: FieldInput,
		},
		{
			name:      "input as an object is refused",
			body:      map[string]any{FieldInput: json.RawMessage(`{"role":"user"}`)},
			wantField: FieldInput,
		},
		// The same values already decoded. A caller passing a fully decoded
		// body gets the same verdict as one leaving the field raw, so the
		// walk is never skipped over an input it could not inspect.
		{
			name:      "a decoded input object is refused",
			body:      map[string]any{FieldInput: map[string]any{"role": "user"}},
			wantField: FieldInput,
		},
		{
			name:      "a decoded input number is refused",
			body:      map[string]any{FieldInput: float64(42)},
			wantField: FieldInput,
		},
		{
			name: "an absent input is accepted",
			body: map[string]any{FieldModel: "m"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := RejectStatefulResponsesFields(tt.body)
			if tt.wantField == "" {
				if err != nil {
					t.Fatalf("got error %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantField) {
				t.Fatalf("got error %v, want it to name field %q", err, tt.wantField)
			}
		})
	}
}
