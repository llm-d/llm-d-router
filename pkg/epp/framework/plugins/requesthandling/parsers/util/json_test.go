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

package parserutil

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestUnmarshalUsesNumber(t *testing.T) {
	var got any
	if err := Unmarshal([]byte(`{"integer":9007199254740993,"nested":[1.5]}  `), &got); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	want := map[string]any{
		"integer": json.Number("9007199254740993"),
		"nested":  []any{json.Number("1.5")},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Unmarshal() mismatch (-want +got):\n%s", diff)
	}
}

func TestUnmarshalEnvelope(t *testing.T) {
	got, err := UnmarshalEnvelope([]byte(` {
 "model": "m", "stream": true, "seed": 9007199254740993, "null": null, "prompt": "raw\ntext",
 "tools": [ {"parameters": {"z": 1e0, "a": 2}} ],
 "extra": { "z": "last", "a": "<first>" }
}`), "prompt")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"model": "m", "stream": true, "seed": json.Number("9007199254740993"), "null": nil, "prompt": json.RawMessage(`"raw\ntext"`),
		"tools": json.RawMessage(`[ {"parameters": {"z": 1e0, "a": 2}} ]`),
		"extra": json.RawMessage(`{ "z": "last", "a": "<first>" }`),
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("envelope mismatch (-want +got):\n%s", diff)
	}
	for _, input := range []string{"{", "[]", "{} {}", "{} trailing"} {
		if _, err := UnmarshalEnvelope([]byte(input), "prompt"); err == nil {
			t.Errorf("accepted invalid envelope %q", input)
		}
	}
}

func TestUnmarshalEnvelopeScalars(t *testing.T) {
	const input = `{"prompt":"line\n\"quoted\" \\ \uD83D\uDE00","stream":false,"max_tokens":8,"seed":-9007199254740993,"large":1e1000,"fraction":-1.50e-2,"empty":"","null":null}`
	var want map[string]any
	if err := Unmarshal([]byte(input), &want); err != nil {
		t.Fatal(err)
	}
	got, err := UnmarshalEnvelope([]byte(input), "system")
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("scalar mismatch (-want +got):\n%s", diff)
	}
}

func TestUnmarshalEnvelopeStringAllocations(t *testing.T) {
	allocations := func(size int) float64 {
		data := []byte(`{"model":"m","max_tokens":1,"prompt":"` + strings.Repeat("x", size) + `"}`)
		return testing.AllocsPerRun(100, func() {
			if _, err := UnmarshalEnvelope(data, "prompt"); err != nil {
				t.Fatal(err)
			}
		})
	}
	small, large := allocations(1024), allocations(240*1024)
	if large > small+1 {
		t.Fatalf("string decoding allocations grow with input size: 1 KiB = %.0f, 240 KiB = %.0f", small, large)
	}
}

func BenchmarkUnmarshalEnvelope(b *testing.B) {
	for _, tc := range []struct {
		name string
		size int
	}{{"1KiB", 1024}, {"240KiB", 240 * 1024}} {
		b.Run(tc.name, func(b *testing.B) {
			data := []byte(`{"model":"m","max_tokens":1,"prompt":"` + strings.Repeat("x", tc.size) + `"}`)
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := UnmarshalEnvelope(data, "prompt"); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestUnmarshalRejectsTrailingData(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError string
	}{
		{
			name:      "second JSON value",
			input:     `{} {}`,
			wantError: ErrTrailingData.Error(),
		},
		{
			name:      "malformed trailing data",
			input:     `{} trailing`,
			wantError: "unexpected trailing data after JSON value: invalid character 'a' in literal true (expecting 'u')",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got any
			err := Unmarshal([]byte(tt.input), &got)
			if !errors.Is(err, ErrTrailingData) {
				t.Fatalf("Unmarshal() error = %v, want unexpected trailing data", err)
			}
			if err.Error() != tt.wantError {
				t.Errorf("Unmarshal() error = %q, want %q", err, tt.wantError)
			}
		})
	}
}

func TestUnmarshalMapWithRawField(t *testing.T) {
	const (
		input    = `{"model":"test","token_ids":[[1.0, 2e0], [3,4]],"seed":9007199254740993,"nested":{"value":1.5}}`
		rawField = "token_ids"
		wantRaw  = `[[1.0, 2e0], [3,4]]`
	)

	var want map[string]any
	if err := Unmarshal([]byte(input), &want); err != nil {
		t.Fatalf("full decode error = %v", err)
	}

	got, err := UnmarshalMapWithRawField([]byte(input), rawField)
	if err != nil {
		t.Fatalf("UnmarshalMapWithRawField() error = %v", err)
	}

	raw, ok := got[rawField].(json.RawMessage)
	if !ok {
		t.Fatalf("%s type = %T, want json.RawMessage", rawField, got[rawField])
	}
	if string(raw) != wantRaw {
		t.Fatalf("%s = %s, want %s", rawField, raw, wantRaw)
	}

	var decoded any
	if err := Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("raw field decode error = %v", err)
	}
	got[rawField] = decoded
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("UnmarshalMapWithRawField() mismatch (-want +got):\n%s", diff)
	}
}

func TestUnmarshalMapWithRawFieldRejectsInvalidJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr error
	}{
		{
			name:  "malformed raw field",
			input: `{"token_ids":[1,2,]}`,
		},
		{
			name:    "trailing value",
			input:   `{"token_ids":[1,2]} true`,
			wantErr: ErrTrailingData,
		},
		{
			name:  "non-object",
			input: `[1,2]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := UnmarshalMapWithRawField([]byte(tt.input), "token_ids")
			if err == nil {
				t.Fatal("UnmarshalMapWithRawField() unexpectedly succeeded")
			}
			if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
				t.Fatalf("UnmarshalMapWithRawField() error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}
