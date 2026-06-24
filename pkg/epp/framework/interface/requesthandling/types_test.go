/*
Copyright 2025 The Kubernetes Authors.

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

package requesthandling

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPrompt_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    Prompt
		wantErr bool
	}{
		{
			name:  "string prompt",
			input: `"hello world"`,
			want:  Prompt{Raw: "hello world"},
		},
		{
			name:  "array of strings prompt",
			input: `["hello","world"]`,
			want:  Prompt{Strings: []string{"hello", "world"}},
		},
		{
			name:  "single-element array prompt",
			input: `["hello world"]`,
			want:  Prompt{Strings: []string{"hello world"}},
		},
		{
			name:  "array of integers prompt",
			input: `[1,2,3]`,
			want:  Prompt{TokenIDs: []uint32{1, 2, 3}},
		},
		{
			name:    "array of floats prompt is rejected",
			input:   `[1.5,2.7]`,
			wantErr: true,
		},

		{
			name:    "array of arrays of integers prompt is rejected for now",
			input:   `[[1,2],[3,4]]`,
			wantErr: true,
		},

		{
			name:    "integer prompt is rejected",
			input:   `123`,
			wantErr: true,
		},
		{
			name:    "object prompt is rejected",
			input:   `{"key":"value"}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var p Prompt
			err := p.UnmarshalJSON([]byte(tt.input))
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, p)
			}
		})
	}
}

func TestEmbeddingsInput_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    EmbeddingsInput
		wantErr bool
	}{
		{
			name:  "string input",
			input: `"hello world"`,
			want:  EmbeddingsInput{Raw: "hello world"},
		},
		{
			name:  "array of strings input",
			input: `["hello","world"]`,
			want:  EmbeddingsInput{Strings: []string{"hello", "world"}},
		},
		{
			name:  "array of integers input",
			input: `[1,2,3]`,
			want:  EmbeddingsInput{TokenIDs: []uint32{1, 2, 3}},
		},
		{
			name:    "array of floats input is rejected",
			input:   `[1.5,2.7]`,
			wantErr: true,
		},

		{
			name:    "array of arrays of integers input is rejected for now",
			input:   `[[1,2],[3,4]]`,
			wantErr: true,
		},

		{
			name:    "integer input is rejected",
			input:   `123`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var e EmbeddingsInput
			err := e.UnmarshalJSON([]byte(tt.input))
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, e)
			}
		})
	}
}

func TestPrompt_PlainText(t *testing.T) {
	tests := []struct {
		name string
		p    Prompt
		want string
	}{
		{name: "raw string", p: Prompt{Raw: "hello"}, want: "hello"},
		{name: "strings joined", p: Prompt{Strings: []string{"a", "b", "c"}}, want: "a b c"},
		{name: "single string in array", p: Prompt{Strings: []string{"hello"}}, want: "hello"},
		{name: "zero value", p: Prompt{}, want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.p.PlainText())
		})
	}
}

func TestPrompt_IsEmpty(t *testing.T) {
	assert.True(t, Prompt{}.IsEmpty())
	assert.True(t, Prompt{Strings: []string{}}.IsEmpty())
	assert.False(t, Prompt{Raw: "x"}.IsEmpty())
	assert.False(t, Prompt{Strings: []string{"x"}}.IsEmpty())
}

func TestPrompt_MarshalJSON(t *testing.T) {
	raw, _ := Prompt{Raw: "hello"}.MarshalJSON()
	assert.Equal(t, `"hello"`, string(raw))

	arr, _ := Prompt{Strings: []string{"a", "b"}}.MarshalJSON()
	assert.Equal(t, `["a","b"]`, string(arr))

	empty, _ := Prompt{}.MarshalJSON()
	assert.Equal(t, `""`, string(empty))
}

func TestGenerateRequest_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		want        []uint32
		wantErr     bool
		errContains string
	}{
		{
			name:  "valid token ids",
			input: `{"token_ids":[1,2,3]}`,
			want:  []uint32{1, 2, 3},
		},
		{
			name:  "max uint32 boundary accepted",
			input: `{"token_ids":[4294967295]}`,
			want:  []uint32{4294967295},
		},
		{
			name:        "negative token id rejected",
			input:       `{"token_ids":[1,2,-1]}`,
			wantErr:     true,
			errContains: "token_ids[2]: invalid value",
		},
		{
			name:        "non-integer token id rejected",
			input:       `{"token_ids":[1,2.5,3]}`,
			wantErr:     true,
			errContains: "token_ids[1]: invalid value",
		},
		{
			name:        "value above MaxUint32 rejected",
			input:       `{"token_ids":[4294967296]}`,
			wantErr:     true,
			errContains: "token_ids[0]: invalid value",
		},
		{
			name:        "NaN token id rejected",
			input:       `{"token_ids":[1,NaN]}`,
			wantErr:     true,
			errContains: "invalid character",
		},
		{
			name:        "malformed json rejected",
			input:       `{"token_ids":[`,
			wantErr:     true,
			errContains: "unexpected end of JSON",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var g GenerateRequest
			err := g.UnmarshalJSON([]byte(tt.input))
			if tt.wantErr {
				require.Error(t, err)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, g.TokenIDs)
		})
	}
}

func TestMaxOutputTokensFromPayload(t *testing.T) {
	i64 := func(v int64) *int64 { return &v }

	tests := []struct {
		name string
		m    PayloadMap
		keys []string
		want *int64
	}{
		{name: "absent", m: PayloadMap{"other": float64(1)}, keys: []string{"max_tokens"}, want: nil},
		{name: "float64 value", m: PayloadMap{"max_tokens": float64(64)}, keys: []string{"max_tokens"}, want: i64(64)},
		{name: "json.Number value", m: PayloadMap{"max_tokens": json.Number("128")}, keys: []string{"max_tokens"}, want: i64(128)},
		{name: "explicit zero binds", m: PayloadMap{"max_tokens": float64(0)}, keys: []string{"max_tokens"}, want: i64(0)},
		{name: "negative ignored", m: PayloadMap{"max_tokens": float64(-1)}, keys: []string{"max_tokens"}, want: nil},
		{name: "non-integral ignored", m: PayloadMap{"max_tokens": float64(1.5)}, keys: []string{"max_tokens"}, want: nil},
		{name: "wrong type ignored", m: PayloadMap{"max_tokens": "64"}, keys: []string{"max_tokens"}, want: nil},
		{
			name: "precedence: first present key wins",
			m:    PayloadMap{"max_completion_tokens": float64(100), "max_tokens": float64(50)},
			keys: []string{"max_completion_tokens", "max_tokens"},
			want: i64(100),
		},
		{
			name: "precedence: fall back to second key",
			m:    PayloadMap{"max_tokens": float64(50)},
			keys: []string{"max_completion_tokens", "max_tokens"},
			want: i64(50),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, MaxOutputTokensFromPayload(tt.m, tt.keys...))
		})
	}
}
