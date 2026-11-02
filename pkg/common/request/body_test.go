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
	"reflect"
	"testing"
)

func TestDecodeRequestBody(t *testing.T) {
	const body = `{"model":"m","max_tokens":5,"tools":{"b":{},"a":{}}}`

	tests := []struct {
		name      string
		raw       string
		inspected map[string]struct{}
		want      map[string]any
		wantErr   bool
	}{
		{
			name:      "inspected fields decoded, non-inspected kept as exact bytes",
			raw:       body,
			inspected: map[string]struct{}{"model": {}, "max_tokens": {}},
			want: map[string]any{
				"model":      "m",
				"max_tokens": float64(5),
				"tools":      json.RawMessage(`{"b":{},"a":{}}`),
			},
		},
		{
			name:      "caller-supplied set decides which fields are decoded",
			raw:       body,
			inspected: map[string]struct{}{"tools": {}},
			want: map[string]any{
				"model":      json.RawMessage(`"m"`),
				"max_tokens": json.RawMessage(`5`),
				"tools":      map[string]any{"b": map[string]any{}, "a": map[string]any{}},
			},
		},
		{
			name:      "nil inspected set keeps everything raw",
			raw:       body,
			inspected: nil,
			want: map[string]any{
				"model":      json.RawMessage(`"m"`),
				"max_tokens": json.RawMessage(`5`),
				"tools":      json.RawMessage(`{"b":{},"a":{}}`),
			},
		},
		{
			name:      "empty inspected set keeps everything raw",
			raw:       body,
			inspected: map[string]struct{}{},
			want: map[string]any{
				"model":      json.RawMessage(`"m"`),
				"max_tokens": json.RawMessage(`5`),
				"tools":      json.RawMessage(`{"b":{},"a":{}}`),
			},
		},
		{name: "array body errors", raw: `[1,2]`, wantErr: true},
		{name: "null body errors", raw: `null`, wantErr: true},
		{name: "string body errors", raw: `"text"`, wantErr: true},
		{name: "invalid JSON errors", raw: `{`, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DecodeRequestBody([]byte(tc.raw), tc.inspected)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("DecodeRequestBody(%s) = %v, want error", tc.raw, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("DecodeRequestBody(%s) unexpected error: %v", tc.raw, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("DecodeRequestBody(%s) = %#v, want %#v", tc.raw, got, tc.want)
			}
		})
	}

	t.Run("non-inspected large integer round-trips byte-identical", func(t *testing.T) {
		raw := `{"seed":9007199254740993}`
		got, err := DecodeRequestBody([]byte(raw), map[string]struct{}{"model": {}})
		if err != nil {
			t.Fatalf("DecodeRequestBody(%s) unexpected error: %v", raw, err)
		}
		out, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("json.Marshal: %v", err)
		}
		if string(out) != raw {
			t.Errorf("round-trip = %s, want %s", out, raw)
		}
	})
}
