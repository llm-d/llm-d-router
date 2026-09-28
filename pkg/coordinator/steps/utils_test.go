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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
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

func TestResolveFormat(t *testing.T) {
	tests := []struct {
		name            string
		useOpenAIFormat bool
		path            string
		want            reqcommon.APIType
	}{
		{name: "chat completions with openai format", useOpenAIFormat: true, path: reqcommon.PathChatCompletions, want: reqcommon.APITypeChatCompletions},
		{name: "chat completions without openai format collapses to generate", path: reqcommon.PathChatCompletions, want: reqcommon.APITypeVLLMGenerate},
		{name: "completions ignores openai format", path: reqcommon.PathCompletions, want: reqcommon.APITypeCompletions},
		{name: "generate", useOpenAIFormat: true, path: reqcommon.PathVLLMGenerate, want: reqcommon.APITypeVLLMGenerate},
		{name: "responses collapses to generate", useOpenAIFormat: true, path: reqcommon.PathResponses, want: reqcommon.APITypeVLLMGenerate},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveFormat(tt.useOpenAIFormat, tt.path); got != tt.want {
				t.Errorf("resolveFormat(%t, %q) = %v, want %v", tt.useOpenAIFormat, tt.path, got, tt.want)
			}
		})
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
		{name: "string_element", input: []any{"abc"}, wantErr: true},
		{name: "bool_element", input: []any{true}, wantErr: true},
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

	// A wrong-typed element inside an otherwise well-formed parallel array must
	// fail loudly, not be coerced. These exercise the per-element type asserts.
	t.Run("mm_hashes_element_not_string", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []any{float64(42)}},
			"mm_placeholders": map[string]any{"image": []any{
				map[string]any{"offset": float64(1), "length": float64(3)},
			}},
		}
		_, err := extractMultimodalEntries(features)
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})

	t.Run("mm_placeholders_element_not_object", func(t *testing.T) {
		features := map[string]any{
			"mm_hashes":       map[string]any{"image": []any{testHash}},
			"mm_placeholders": map[string]any{"image": []any{"not-an-object"}},
		}
		_, err := extractMultimodalEntries(features)
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Errorf("expected ErrBadRequest, got %v", err)
		}
	})

	// A negative offset or length must be rejected here: EncodeStep indexes
	// token_ids[offset] and allocates make([]int, 1+length), so a negative value
	// panics downstream. This guard is the only thing preventing that, so it is
	// exercised directly. Do not weaken anyToNonNegativeInt to a plain int parse.
	for _, tc := range []struct {
		name   string
		offset float64
		length float64
	}{
		{"negative_offset", -1, 3},
		{"negative_length", 1, -3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			features := map[string]any{
				"mm_hashes": map[string]any{"image": []any{testHash}},
				"mm_placeholders": map[string]any{"image": []any{
					map[string]any{"offset": tc.offset, "length": tc.length},
				}},
			}
			_, err := extractMultimodalEntries(features)
			if !errors.Is(err, pipeline.ErrBadRequest) {
				t.Errorf("expected ErrBadRequest, got %v", err)
			}
		})
	}
}

func TestValidatePlaceholderBounds(t *testing.T) {
	entry := func(offset, length int) pipeline.MultimodalEntry {
		return pipeline.MultimodalEntry{
			Placeholder: pipeline.PlaceholderRange{Offset: offset, Length: length},
		}
	}

	tests := []struct {
		name       string
		entries    []pipeline.MultimodalEntry
		tokenCount int
		wantErr    bool
	}{
		{name: "empty_entries", entries: nil, tokenCount: 4},
		{name: "span_fills_prompt", entries: []pipeline.MultimodalEntry{entry(0, 4)}, tokenCount: 4},
		{name: "boundary_offset_plus_length_equals_count", entries: []pipeline.MultimodalEntry{entry(1, 3)}, tokenCount: 4},
		{name: "zero_length_in_range", entries: []pipeline.MultimodalEntry{entry(3, 0)}, tokenCount: 4},
		{name: "offset_at_count", entries: []pipeline.MultimodalEntry{entry(4, 0)}, tokenCount: 4, wantErr: true},
		{name: "offset_beyond_count", entries: []pipeline.MultimodalEntry{entry(5, 1)}, tokenCount: 4, wantErr: true},
		{name: "span_one_past_end", entries: []pipeline.MultimodalEntry{entry(2, 3)}, tokenCount: 4, wantErr: true},
		{name: "length_overflows_prompt", entries: []pipeline.MultimodalEntry{entry(0, 9007199254740992)}, tokenCount: 4, wantErr: true},
		{name: "second_entry_out_of_range", entries: []pipeline.MultimodalEntry{entry(0, 2), entry(10, 1)}, tokenCount: 4, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validatePlaceholderBounds(tc.entries, tc.tokenCount)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !errors.Is(err, pipeline.ErrBadRequest) {
					t.Errorf("expected ErrBadRequest, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
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

func TestGatewayHeaders(t *testing.T) {
	reqCtx := &pipeline.RequestContext{
		RequestID: "req-1",
		OriginalHeaders: http.Header{
			"X-Custom":       {"v"},
			"X-Request-Id":   {"client-id"},
			"Content-Length": {"12"},
		},
	}

	headers := gatewayHeaders(reqCtx, gateway.PhaseEncode)

	want := map[string]string{
		"x-custom":                   "v",
		reqcommon.RequestIDHeaderKey: "req-1",
		gateway.EPPProfileHeader:     gateway.PhaseEncode,
	}
	if len(headers) != len(want) {
		t.Errorf("headers = %v, want %v", headers, want)
	}
	for k, v := range want {
		if headers[k] != v {
			t.Errorf("%s = %q, want %q", k, headers[k], v)
		}
	}

	headers["x-added"] = "1"
	if _, present := gatewayHeaders(reqCtx, gateway.PhaseEncode)["x-added"]; present {
		t.Error("a change to the returned map is visible in the next call")
	}
}

func TestGatewayHeaders_NoClientHeaders(t *testing.T) {
	headers := gatewayHeaders(&pipeline.RequestContext{RequestID: "req-1"}, gateway.PhasePrefill)
	if headers[reqcommon.RequestIDHeaderKey] != "req-1" || headers[gateway.EPPProfileHeader] != gateway.PhasePrefill {
		t.Errorf("headers = %v, want the request id and the prefill profile", headers)
	}
}

func TestCheckStatus(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		body     string
		wantBody string
		wantErr  bool
	}{
		{name: "ok", status: http.StatusOK, body: "answer"},
		{name: "client error", status: http.StatusBadRequest, body: "bad prompt", wantBody: "bad prompt", wantErr: true},
		{name: "server error", status: http.StatusServiceUnavailable, body: "overloaded", wantBody: "overloaded", wantErr: true},
		{name: "success status other than 200", status: http.StatusAccepted, wantErr: true},
		{name: "empty error body", status: http.StatusBadGateway, wantErr: true},
		{
			name:     "oversized error body",
			status:   http.StatusInternalServerError,
			body:     strings.Repeat("a", maxErrorBodySize*2),
			wantBody: strings.Repeat("a", maxErrorBodySize),
			wantErr:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &http.Response{StatusCode: tt.status, Body: io.NopCloser(strings.NewReader(tt.body))}

			err := checkStatus(RenderStepName, resp)
			if !tt.wantErr {
				if err != nil {
					t.Fatalf("checkStatus: %v", err)
				}
				// The body of a 200 response stays unread for the caller.
				if body, _ := io.ReadAll(resp.Body); string(body) != tt.body {
					t.Errorf("response body = %q, want %q", body, tt.body)
				}
				return
			}
			var upstream *pipeline.UpstreamError
			if !errors.As(err, &upstream) {
				t.Fatalf("error = %v, want a pipeline.UpstreamError", err)
			}
			if upstream.Step != RenderStepName || upstream.StatusCode != tt.status || upstream.Body != tt.wantBody {
				t.Errorf("error = {Step: %q, StatusCode: %d, len(Body): %d}, want {%q, %d, %d}",
					upstream.Step, upstream.StatusCode, len(upstream.Body), RenderStepName, tt.status, len(tt.wantBody))
			}
		})
	}
}

// testGatewayRequest is the base request of the postToGateway tests.
var testGatewayRequest = gatewayRequest{
	step:     PrefillStepName,
	upstream: coordmetrics.UpstreamPrefill,
	path:     reqcommon.PathCompletions,
	body:     []byte(`{}`),
}

// captureLogger returns a logger and a function that returns the lines that
// the logger recorded. The lock permits the concurrent encode sub-requests.
func captureLogger(verbosity int) (logr.Logger, func() []string) {
	var mu sync.Mutex
	var records []string
	logger := funcr.New(func(_, args string) {
		mu.Lock()
		defer mu.Unlock()
		records = append(records, args)
	}, funcr.Options{Verbosity: verbosity})
	return logger, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(records)
	}
}

func countRecords(records []string, substrs ...string) int {
	n := 0
	for _, record := range records {
		missing := func(substr string) bool { return !strings.Contains(record, substr) }
		if !slices.ContainsFunc(substrs, missing) {
			n++
		}
	}
	return n
}

func TestPostToGateway_ReturnsResponse(t *testing.T) {
	var gotPath, gotBody, gotHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotPath, gotBody, gotHeader = r.URL.Path, string(body), r.Header.Get("x-custom")
		_, _ = w.Write([]byte("answer"))
	}))
	defer server.Close()

	req := testGatewayRequest
	req.body = []byte(`{"model":"m"}`)
	req.headers = map[string]string{"x-custom": "v"}
	resp, err := postToGateway(context.Background(), logr.Discard(), gateway.New(config.GatewayConfig{Address: server.URL}), req)
	if err != nil {
		t.Fatalf("postToGateway: %v", err)
	}
	defer resp.Body.Close()

	if gotPath != reqcommon.PathCompletions || gotBody != `{"model":"m"}` || gotHeader != "v" {
		t.Errorf("gateway got path=%q body=%q x-custom=%q", gotPath, gotBody, gotHeader)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	if string(body) != "answer" {
		t.Errorf("response body = %q, want %q", body, "answer")
	}
}

func TestPostToGateway_DebugRecord(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	tests := []struct {
		name        string
		verbosity   int
		logMsg      string
		body        []byte
		headers     map[string]string
		wantMsg     string
		wantKeys    []string
		wantRecords int
	}{
		{
			name:        "message from the caller",
			verbosity:   logutil.DEBUG,
			logMsg:      "sub-request body",
			body:        []byte(`{"model":"m"}`),
			headers:     map[string]string{gateway.EPPProfileHeader: gateway.PhaseEncode},
			wantMsg:     "sub-request body",
			wantKeys:    []string{`"index"=2`, `"path"="` + reqcommon.PathCompletions + `"`, `"bodyLen"=13`},
			wantRecords: 1,
		},
		{
			name:        "default message with an empty body and no headers",
			verbosity:   logutil.DEBUG,
			wantMsg:     defaultGatewayLogMsg,
			wantKeys:    []string{`"bodyLen"=0`},
			wantRecords: 1,
		},
		{
			name:        "message from the caller that is equal to the default",
			verbosity:   logutil.DEBUG,
			logMsg:      defaultGatewayLogMsg,
			wantMsg:     defaultGatewayLogMsg,
			wantRecords: 1,
		},
		{name: "below debug", verbosity: logutil.VERBOSE, wantMsg: defaultGatewayLogMsg},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, records := captureLogger(tt.verbosity)

			req := testGatewayRequest
			req.logMsg, req.body, req.headers = tt.logMsg, tt.body, tt.headers
			resp, err := postToGateway(context.Background(), logger.WithValues("index", 2), gwClient, req)
			if err != nil {
				t.Fatalf("postToGateway: %v", err)
			}
			resp.Body.Close()

			want := append([]string{fmt.Sprintf(`"msg"=%q`, tt.wantMsg)}, tt.wantKeys...)
			if got := countRecords(records(), want...); got != tt.wantRecords {
				t.Errorf("%d records contain %v, want %d, records=%v", got, want, tt.wantRecords, records())
			}
		})
	}
}

func TestPostToGateway_StatusIsUpstreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	req := testGatewayRequest
	req.step = "encode[2]"
	resp, err := postToGateway(context.Background(), logr.Discard(), gateway.New(config.GatewayConfig{Address: server.URL}), req)
	if resp != nil {
		resp.Body.Close()
		t.Error("expected no response with an error")
	}
	var upstream *pipeline.UpstreamError
	if !errors.As(err, &upstream) {
		t.Fatalf("error = %v, want a pipeline.UpstreamError", err)
	}
	if upstream.Step != req.step || upstream.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("error = {Step: %q, StatusCode: %d}, want {%q, %d}",
			upstream.Step, upstream.StatusCode, req.step, http.StatusServiceUnavailable)
	}
}

func TestPostToGateway_TransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	server.Close()

	tests := []struct {
		name string
		ctx  func() context.Context
	}{
		{name: "gateway unreachable", ctx: context.Background},
		{name: "context cancelled", ctx: func() context.Context {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			return ctx
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := postToGateway(tt.ctx(), logr.Discard(), gwClient, testGatewayRequest)
			if resp != nil {
				resp.Body.Close()
				t.Error("expected no response with an error")
			}
			if err == nil || !strings.HasPrefix(err.Error(), "prefill: request: ") {
				t.Fatalf("error = %v, want the prefix %q", err, "prefill: request: ")
			}
			var upstream *pipeline.UpstreamError
			if errors.As(err, &upstream) {
				t.Errorf("a transport failure must not be a pipeline.UpstreamError: %v", err)
			}
		})
	}
}
