/*
Copyright 2026 The Kubernetes Authors.

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
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseCanonicalTokenIDArrays(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		canonical bool
		want      [][]uint32
	}{
		{
			name:      "flat integers",
			input:     `[1,2,3]`,
			canonical: true,
			want:      [][]uint32{{1, 2, 3}},
		},
		{
			name:      "zero",
			input:     `[0]`,
			canonical: true,
			want:      [][]uint32{{0}},
		},
		{
			name:      "max uint32",
			input:     `[4294967295]`,
			canonical: true,
			want:      [][]uint32{{4294967295}},
		},
		{
			name:      "whitespace",
			input:     "[ 1,\n\t2 ]",
			canonical: true,
			want:      [][]uint32{{1, 2}},
		},
		{
			name:      "nested integers",
			input:     `[[1,2],[3]]`,
			canonical: true,
			want:      [][]uint32{{1, 2}, {3}},
		},
		{
			name:  "empty array falls through",
			input: `[]`,
		},
		{
			name:  "whole decimal falls through",
			input: `[1.0]`,
		},
		{
			name:  "exponent falls through",
			input: `[1e0]`,
		},
		{
			name:  "negative falls through",
			input: `[-1]`,
		},
		{
			name:  "above MaxUint32 falls through",
			input: `[4294967296]`,
		},
		{
			name:  "leading zeros fall through",
			input: `[01]`,
		},
		{
			name:  "trailing junk falls through",
			input: `[1,2]true`,
		},
		{
			name:  "numeric strings stay strings",
			input: `["1","2"]`,
		},
		{
			name:  "empty nested array falls through",
			input: `[[]]`,
		},
		{
			name:  "triple nesting falls through",
			input: `[[[1]]]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := []byte(tt.input)
			got, ok := parseCanonicalTokenIDArrays(data)
			assert.Equal(t, tt.canonical, ok)
			if ok {
				require.True(t, json.Valid(data), "fast path must only accept valid JSON")
				assert.Equal(t, tt.want, got)
			} else {
				assert.Nil(t, got)
			}
		})
	}
}

func TestParseCanonicalTokenIDArraysMatchesLegacyPath(t *testing.T) {
	flat := make([]uint32, 1024)
	for i := range flat {
		flat[i] = uint32(i * 7919)
	}
	flat[len(flat)-1] = ^uint32(0)

	nested := make([][]uint32, 32)
	for row := range nested {
		nested[row] = make([]uint32, row+1)
		for column := range nested[row] {
			nested[row][column] = uint32((row+1)*100000 + column)
		}
	}

	for name, input := range map[string]any{
		"flat":   flat,
		"nested": nested,
	} {
		t.Run(name, func(t *testing.T) {
			data, err := json.Marshal(input)
			require.NoError(t, err)

			got, ok := parseCanonicalTokenIDArrays(data)
			require.True(t, ok)

			var raw any
			require.NoError(t, json.Unmarshal(data, &raw))
			legacy, err := parseArrayInput(raw.([]any), "prompt")
			require.NoError(t, err)
			assert.Equal(t, legacy.TokenIDs, got)
		})
	}
}

func TestParseCanonicalTokenIDArraysUsesContiguousBackingStorage(t *testing.T) {
	got, ok := parseCanonicalTokenIDArrays([]byte(`[[1,2],[3,4,5],[6]]`))
	require.True(t, ok)
	require.Len(t, got, 3)

	const totalTokens = 6
	base := reflect.ValueOf(&got[0][0]).Pointer()
	elementSize := reflect.TypeOf(uint32(0)).Size()
	offset := 0
	for _, row := range got {
		require.NotEmpty(t, row)
		assert.Equal(t, len(row), cap(row))
		if gotAddress := reflect.ValueOf(&row[0]).Pointer(); gotAddress != base+uintptr(offset)*elementSize {
			t.Fatalf("row at offset %d does not share the contiguous backing storage", offset)
		}
		offset += len(row)
	}
	assert.Equal(t, totalTokens, offset)

	got[0] = append(got[0], 99)
	assert.Equal(t, []uint32{1, 2, 99}, got[0])
	assert.Equal(t, []uint32{3, 4, 5}, got[1])
}

// referencePromptUnmarshal is Prompt.UnmarshalJSON without the canonical token-ID
// fast path: it decodes through encoding/json only. The fast path is correct iff
// it never changes the observable result of this reference.
func referencePromptUnmarshal(data []byte) (Prompt, error) {
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return Prompt{}, err
	}
	switch v := raw.(type) {
	case string:
		return Prompt{Raw: v}, nil
	case []any:
		res, err := parseArrayInput(v, "prompt")
		if err != nil {
			return Prompt{}, err
		}
		return Prompt{Strings: res.Strings, TokenIDs: res.TokenIDs}, nil
	default:
		return Prompt{}, errors.New("prompt: must be a string or an array")
	}
}

// FuzzParseCanonicalTokenIDArrays cross-checks the hand-written parser against
// encoding/json for arbitrary bytes. It guards two properties:
//   - safety: the parser never panics and never reports success for input that
//     encoding/json rejects (accepting invalid JSON would be an injection vector);
//   - correctness: the fast path never changes what Prompt.UnmarshalJSON produces
//     versus decoding through encoding/json alone.
func FuzzParseCanonicalTokenIDArrays(f *testing.F) {
	seeds := []string{
		`[1,2,3]`, `[[1,2],[3]]`, `[0]`, `[4294967295]`, `[4294967296]`,
		`[1.0]`, `[1e0]`, `[1e10]`, `[-1]`, `[01]`, `[1,2,]`, `[1,,2]`,
		`["a","b"]`, `"hello"`, `"123"`, `123`, `true`, `null`, `{}`, `[]`,
		`[[]]`, `[[1],,[2]]`, `[1,2]/*x*/`, `foo([1,2])`, "[1,2]\x00",
		`[1,2] [3]`, "[ 1,\n\t2 ]", `[[1,2],"x"]`, `[[[1]]]`, `[{}]`,
	}
	for _, s := range seeds {
		f.Add([]byte(s))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		got, ok := parseCanonicalTokenIDArrays(data)

		if ok {
			if !json.Valid(data) {
				t.Fatalf("fast path accepted invalid JSON: %q", data)
			}
			var raw any
			if err := json.Unmarshal(data, &raw); err != nil {
				t.Fatalf("fast path accepted but json.Unmarshal failed: %q: %v", data, err)
			}
			arr, isArr := raw.([]any)
			if !isArr {
				t.Fatalf("fast path accepted non-array %q", data)
			}
			legacy, err := parseArrayInput(arr, "prompt")
			if err != nil {
				t.Fatalf("fast path accepted but legacy path rejected %q: %v", data, err)
			}
			if !reflect.DeepEqual(legacy.TokenIDs, got) {
				t.Fatalf("fast path %v != legacy path %v for %q", got, legacy.TokenIDs, data)
			}
		}

		var p Prompt
		gotErr := p.UnmarshalJSON(data)
		refP, refErr := referencePromptUnmarshal(data)
		if (gotErr == nil) != (refErr == nil) {
			t.Fatalf("error mismatch for %q: fast=%v ref=%v", data, gotErr, refErr)
		}
		if gotErr == nil && !reflect.DeepEqual(refP, p) {
			t.Fatalf("UnmarshalJSON diverges from reference for %q: fast=%#v ref=%#v", data, p, refP)
		}
	})
}

var benchmarkPrompt Prompt

func benchmarkTokenArray(count int, nonCanonicalLast bool) string {
	var body strings.Builder
	body.Grow(count*6 + 2)
	body.WriteByte('[')
	for i := 0; i < count; i++ {
		if i > 0 {
			body.WriteByte(',')
		}
		if nonCanonicalLast && i == count-1 {
			body.WriteString("12345.0")
		} else {
			body.WriteString("12345")
		}
	}
	body.WriteByte(']')
	return body.String()
}

func benchmarkNestedTokenArray(rows, columns int) string {
	var body strings.Builder
	body.Grow(rows*columns*6 + rows*2 + 2)
	body.WriteByte('[')
	for row := 0; row < rows; row++ {
		if row > 0 {
			body.WriteByte(',')
		}
		body.WriteByte('[')
		for column := 0; column < columns; column++ {
			if column > 0 {
				body.WriteByte(',')
			}
			body.WriteString("12345")
		}
		body.WriteByte(']')
	}
	body.WriteByte(']')
	return body.String()
}

func benchmarkPromptUnmarshalJSON(b *testing.B, tokens string) {
	data := []byte(tokens)
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for b.Loop() {
		var p Prompt
		if err := p.UnmarshalJSON(data); err != nil {
			b.Fatal(err)
		}
		benchmarkPrompt = p
	}
}

func BenchmarkPromptUnmarshalJSONTokenIDs(b *testing.B) {
	for _, count := range []int{4 * 1024, 32 * 1024, 256 * 1024, 1_000_000} {
		name := map[int]string{
			4 * 1024:   "4K",
			32 * 1024:  "32K",
			256 * 1024: "256K",
			1_000_000:  "1M",
		}[count]
		tokens := benchmarkTokenArray(count, false)
		b.Run("Flat/"+name, func(b *testing.B) {
			benchmarkPromptUnmarshalJSON(b, tokens)
		})
	}

	b.Run("Nested/1M", func(b *testing.B) {
		benchmarkPromptUnmarshalJSON(b, benchmarkNestedTokenArray(1000, 1000))
	})
	b.Run("Fallback/1M", func(b *testing.B) {
		benchmarkPromptUnmarshalJSON(b, benchmarkTokenArray(1_000_000, true))
	})
}
