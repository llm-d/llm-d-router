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
	"errors"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const testHash = "abc123"

// testKwargs is a base64 tensor blob standing in for a real (non-cache-hit)
// kwargs_data entry.
const testKwargs = "dGVuc29y"

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
		{name: "overflow_float_token", input: []any{float64(1e19)}, wantErr: true},
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
			"mm_hashes": map[string]any{"image": []any{testHash}},
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
		if e.Hash != testHash {
			t.Errorf("hash: expected %s, got %v", testHash, e.Hash)
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

	t.Run("valid_two_images", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []any{"hash1", "hash2"}},
			"mm_placeholders": map[string]any{"image": []any{
				map[string]any{"offset": float64(1), "length": float64(3)},
				map[string]any{"offset": float64(5), "length": float64(2)},
			}},
			"kwargs_data": map[string]any{"image": []any{"d1", "d2"}},
		}
		entries, err := extractMultimodalEntries(features)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 2 {
			t.Fatalf("expected 2 entries, got %d", len(entries))
		}
		want := []pipeline.MultimodalEntry{
			{Index: 0, Hash: "hash1", KwargsData: "d1", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Index: 1, Hash: "hash2", KwargsData: "d2", Placeholder: pipeline.PlaceholderRange{Offset: 5, Length: 2}},
		}
		for i, w := range want {
			if entries[i] != w {
				t.Errorf("entry %d: expected %+v, got %+v", i, w, entries[i])
			}
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

	t.Run("absent_kwargs_resolves_from_cache", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []any{testHash}},
			"mm_placeholders": map[string]any{"image": []any{
				map[string]any{"offset": float64(1), "length": float64(3)},
			}},
		}
		entries, err := extractMultimodalEntries(features)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 1 {
			t.Fatalf("expected 1 entry, got %d", len(entries))
		}
		if entries[0].Hash != testHash {
			t.Errorf("hash: expected %s, got %v", testHash, entries[0].Hash)
		}
		if entries[0].KwargsData != "" {
			t.Errorf("kwargs: expected empty (resolve from cache), got %q", entries[0].KwargsData)
		}
	})

	t.Run("mixed_batch_null_kwargs_item", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []any{"hash1", "hash2"}},
			"mm_placeholders": map[string]any{"image": []any{
				map[string]any{"offset": float64(1), "length": float64(3)},
				map[string]any{"offset": float64(4), "length": float64(3)},
			}},
			"kwargs_data": map[string]any{"image": []any{testKwargs, nil}},
		}
		entries, err := extractMultimodalEntries(features)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 2 {
			t.Fatalf("expected 2 entries, got %d", len(entries))
		}
		if entries[0].KwargsData != testKwargs {
			t.Errorf("entry 0 kwargs: expected dGVuc29y, got %q", entries[0].KwargsData)
		}
		if entries[1].KwargsData != "" {
			t.Errorf("entry 1 kwargs: expected empty (cache hit), got %q", entries[1].KwargsData)
		}
	})

	t.Run("kwargs_wrong_type", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []any{"hash1"}},
			"mm_placeholders": map[string]any{"image": []any{
				map[string]any{"offset": float64(1), "length": float64(3)},
			}},
			"kwargs_data": map[string]any{"image": []any{float64(42)}},
		}
		_, err := extractMultimodalEntries(features)
		if err == nil {
			t.Fatal("expected error for non-string kwargs item")
		}
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})

	// A present but malformed field must fail loudly, not be silently coerced to
	// absent (which would process a multimodal request as text-only).
	t.Run("mm_hashes_not_object", func(t *testing.T) {
		_, err := extractMultimodalEntries(map[string]any{"mm_hashes": "garbage"})
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})

	t.Run("mm_hashes_image_not_array", func(t *testing.T) {
		features := map[string]any{"mm_hashes": map[string]any{"image": "garbage"}}
		_, err := extractMultimodalEntries(features)
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})

	t.Run("mm_hashes_no_image_modality_returns_nil", func(t *testing.T) {
		features := map[string]any{"mm_hashes": map[string]any{"audio": []any{testHash}}}
		entries, err := extractMultimodalEntries(features)
		if err != nil {
			t.Fatal(err)
		}
		if entries != nil {
			t.Fatalf("expected nil, got %v", entries)
		}
	})

	t.Run("mm_placeholders_not_object", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes":       map[string]any{"image": []any{testHash}},
			"mm_placeholders": "garbage",
		}
		_, err := extractMultimodalEntries(features)
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})

	// mm_placeholders is required once mm_hashes[image] is set; its absence must
	// fail loudly rather than being processed as a text-only request.
	t.Run("mm_placeholders_absent", func(t *testing.T) {
		features := map[string]any{"mm_hashes": map[string]any{"image": []any{testHash}}}
		_, err := extractMultimodalEntries(features)
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})

	t.Run("kwargs_data_not_object", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []any{testHash}},
			"mm_placeholders": map[string]any{"image": []any{
				map[string]any{"offset": float64(1), "length": float64(3)},
			}},
			"kwargs_data": "garbage",
		}
		_, err := extractMultimodalEntries(features)
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})
}

// mmImageKwargs extracts features["kwargs_data"].image as a []any, marshaling
// through JSON so the test sees exactly what the encoder/decoder receives on the
// wire (where the cache-hit sentinel must be null, never "").
func mmImageKwargs(t *testing.T, features map[string]any) []any {
	t.Helper()
	raw, err := json.Marshal(features["kwargs_data"])
	if err != nil {
		t.Fatalf("marshal kwargs_data: %v", err)
	}
	var decoded map[string][]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal kwargs_data: %v", err)
	}
	return decoded[ModalityImage]
}

func TestBuildMMFeatures_CacheHitSentinelSerializesAsNull(t *testing.T) {
	// The empty-string KwargsData is the "resolve from cache" sentinel. On the
	// wire it must be JSON null, not "": vLLM decodes "" as an inline tensor and
	// fails with "Input data was truncated", while null means a cache-hit item.
	entry := func(kwargs string) pipeline.MultimodalEntry {
		return pipeline.MultimodalEntry{Hash: testHash, KwargsData: kwargs}
	}

	t.Run("all cache-hit -> all null", func(t *testing.T) {
		features := buildMMFeatures([]pipeline.MultimodalEntry{entry(""), entry("")}, true)
		items := mmImageKwargs(t, features)
		if len(items) != 2 {
			t.Fatalf("expected 2 kwargs_data entries, got %d: %v", len(items), items)
		}
		for i, it := range items {
			if it != nil {
				t.Errorf("kwargs_data[%d] = %#v, want null", i, it)
			}
		}
		// Regression guard: the raw JSON must contain null, not "".
		raw, _ := json.Marshal(features["kwargs_data"])
		if strings.Contains(string(raw), `""`) {
			t.Errorf("kwargs_data emitted empty string instead of null: %s", raw)
		}
	})

	t.Run("mixed batch keeps inline, nulls cache hits", func(t *testing.T) {
		features := buildMMFeatures([]pipeline.MultimodalEntry{entry(testKwargs), entry("")}, true)
		items := mmImageKwargs(t, features)
		if len(items) != 2 || items[0] != testKwargs || items[1] != nil {
			t.Fatalf("expected [\"dGVuc29y\", null], got %#v", items)
		}
	})

	t.Run("includeKwargs=false omits the field", func(t *testing.T) {
		features := buildMMFeatures([]pipeline.MultimodalEntry{entry("")}, false)
		if _, ok := features["kwargs_data"]; ok {
			t.Errorf("expected kwargs_data absent when includeKwargs is false")
		}
	})
}
