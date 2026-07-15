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
	"errors"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestReadErrorBody_CapsOversizedBody(t *testing.T) {
	body := readErrorBody(strings.NewReader(strings.Repeat("a", maxErrorBodySize*4)))
	if len(body) != maxErrorBodySize {
		t.Fatalf("expected body capped to %d bytes, got %d", maxErrorBodySize, len(body))
	}
}

func TestReadErrorBody_ReturnsSmallBodyVerbatim(t *testing.T) {
	body := readErrorBody(strings.NewReader("overloaded"))
	if string(body) != "overloaded" {
		t.Fatalf("expected %q, got %q", "overloaded", string(body))
	}
}

func TestExtractTokenIDs(t *testing.T) {
	tests := []struct {
		name    string
		input   any
		want    []int
		wantErr bool
	}{
		{name: "valid", input: []any{float64(1), float64(2345), float64(6789)}, want: []int{1, 2345, 6789}},
		{name: "nil", input: nil, wantErr: true},
		{name: "not_array", input: "hello", wantErr: true},
		{name: "empty_array", input: []any{}, wantErr: true},
		{name: "negative_token", input: []any{float64(-1)}, wantErr: true},
		{name: "non_integer_token", input: []any{float64(1.5)}, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := extractTokenIDs(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("expected len %d, got len %d: %v", len(tc.want), len(got), got)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("index %d: expected %d, got %d", i, tc.want[i], got[i])
				}
			}
		})
	}
}

func TestExtractMultimodalEntries(t *testing.T) {
	t.Run("nil_features_returns_nil", func(t *testing.T) {
		entries, err := extractMultimodalEntries(nil)
		if err != nil {
			t.Fatal(err)
		}
		if entries != nil {
			t.Fatalf("expected nil, got %v", entries)
		}
	})

	t.Run("no_mm_hashes_returns_nil", func(t *testing.T) {
		entries, err := extractMultimodalEntries(map[string]any{})
		if err != nil {
			t.Fatal(err)
		}
		if entries != nil {
			t.Fatalf("expected nil, got %v", entries)
		}
	})

	t.Run("valid_single_image", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []any{"abc123"}},
			"mm_placeholders": map[string]any{"image": []any{
				map[string]any{"offset": float64(1), "length": float64(3)},
			}},
			"kwargs_data": map[string]any{"image": []any{"tensordata"}},
		}
		entries, err := extractMultimodalEntries(features)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 1 {
			t.Fatalf("expected 1 entry, got %d", len(entries))
		}
		e := entries[0]
		if e.Hash != "abc123" {
			t.Errorf("hash: expected abc123, got %v", e.Hash)
		}
		if e.Placeholder.Offset != 1 {
			t.Errorf("offset: expected 1, got %v", e.Placeholder.Offset)
		}
		if e.Placeholder.Length != 3 {
			t.Errorf("length: expected 3, got %v", e.Placeholder.Length)
		}
		if e.KwargsData != "tensordata" {
			t.Errorf("kwargs: expected tensordata, got %v", e.KwargsData)
		}
		if e.Index != 0 {
			t.Errorf("index: expected 0, got %v", e.Index)
		}
	})

	t.Run("length_mismatch_placeholders", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []any{"hash1", "hash2"}},
			"mm_placeholders": map[string]any{"image": []any{
				map[string]any{"offset": float64(1), "length": float64(3)},
			}},
			"kwargs_data": map[string]any{"image": []any{"d1", "d2"}},
		}
		_, err := extractMultimodalEntries(features)
		if err == nil {
			t.Fatal("expected error for mismatched placeholder count")
		}
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})

	t.Run("length_mismatch_kwargs", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []any{"hash1", "hash2"}},
			"mm_placeholders": map[string]any{"image": []any{
				map[string]any{"offset": float64(1), "length": float64(3)},
				map[string]any{"offset": float64(4), "length": float64(3)},
			}},
			"kwargs_data": map[string]any{"image": []any{"d1"}},
		}
		_, err := extractMultimodalEntries(features)
		if err == nil {
			t.Fatal("expected error for mismatched kwargs count")
		}
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})
}
