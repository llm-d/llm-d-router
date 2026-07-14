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

package vllmhttp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"k8s.io/utils/ptr"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

func TestNewVllmHTTPParser(t *testing.T) {
	parser := NewVllmHTTPParser()
	want := fwkplugin.TypedName{Type: VllmHTTPParserType, Name: VllmHTTPParserType}
	if parser.TypedName() != want {
		t.Errorf("TypedName() = %v, want %v", parser.TypedName(), want)
	}
}

func TestVllmHTTPParser_ParseRequest_Generate(t *testing.T) {
	parser := NewVllmHTTPParser()

	tests := []struct {
		name    string
		headers map[string]string
		body    map[string]any
		want    *fwkrh.InferenceRequestBody
		wantErr bool
	}{
		{
			name:    "generate request with token_ids",
			headers: map[string]string{":path": "/inference/v1/generate"},
			body: map[string]any{
				"token_ids": []any{1, 2, 3},
			},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{
					TokenIDs: []uint32{1, 2, 3},
				},
				Payload: fwkrh.PayloadMap{
					"token_ids": []any{float64(1), float64(2), float64(3)},
				},
			},
		},
		{
			name:    "generate request with token_ids and cache_salt",
			headers: map[string]string{":path": "/inference/v1/generate"},
			body: map[string]any{
				"token_ids":  []any{10, 20, 30},
				"cache_salt": "abc123",
			},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{
					TokenIDs:  []uint32{10, 20, 30},
					CacheSalt: "abc123",
				},
				Payload: fwkrh.PayloadMap{
					"token_ids":  []any{float64(10), float64(20), float64(30)},
					"cache_salt": "abc123",
				},
			},
		},
		{
			name:    "generate request with token_ids and sampling_params",
			headers: map[string]string{":path": "/inference/v1/generate"},
			body: map[string]any{
				"token_ids": []any{1, 2, 3},
				"sampling_params": map[string]any{
					"temperature": 0.8,
					"max_tokens":  128,
				},
				"stream": true,
			},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{
					TokenIDs: []uint32{1, 2, 3},
				},
				Payload: fwkrh.PayloadMap{
					"token_ids": []any{float64(1), float64(2), float64(3)},
					"sampling_params": map[string]any{
						"temperature": 0.8,
						"max_tokens":  float64(128),
					},
					"stream": true,
				},
				Stream:          true,
				MaxOutputTokens: ptr.To(int64(128)),
			},
		},
		{
			name:    "generate request missing token_ids",
			headers: map[string]string{":path": "/inference/v1/generate"},
			body: map[string]any{
				"sampling_params": map[string]any{"temperature": 0.8},
			},
			wantErr: true,
		},
		{
			name:    "generate request with empty token_ids",
			headers: map[string]string{":path": "/inference/v1/generate"},
			body: map[string]any{
				"token_ids": []any{},
			},
			wantErr: true,
		},
		{
			name:    "generate request via x-original-path header",
			headers: map[string]string{"x-original-path": "/inference/v1/generate"},
			body: map[string]any{
				"token_ids": []any{5, 6, 7},
			},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{
					TokenIDs: []uint32{5, 6, 7},
				},
				Payload: fwkrh.PayloadMap{
					"token_ids": []any{float64(5), float64(6), float64(7)},
				},
			},
		},
		{
			name:    "generate request with multimodal features",
			headers: map[string]string{":path": "/inference/v1/generate"},
			body: map[string]any{
				"token_ids": []any{151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198},
				"features": map[string]any{
					"mm_hashes": map[string]any{
						"image": []any{"abc123hash", "def456hash"},
					},
					"mm_placeholders": map[string]any{
						"image": []any{
							map[string]any{"offset": 1, "length": 3},
							map[string]any{"offset": 4, "length": 3},
						},
					},
				},
			},
			want: &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{
					TokenIDs: []uint32{151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198},
					Features: &tokenization.MultiModalFeatures{
						MMHashes: map[string][]string{
							"image": {"abc123hash", "def456hash"},
						},
						MMPlaceholders: map[string][]kvblock.PlaceholderRange{
							"image": {
								{Offset: 1, Length: 3},
								{Offset: 4, Length: 3},
							},
						},
					},
				},
				Payload: fwkrh.PayloadMap{
					"token_ids": []any{
						float64(151644), float64(872), float64(198), float64(3838), float64(374), float64(279),
						float64(6722), float64(315), float64(9625), float64(30), float64(151645), float64(198),
						float64(151644), float64(77091), float64(198),
					},
					"features": map[string]any{
						"mm_hashes": map[string]any{
							"image": []any{"abc123hash", "def456hash"},
						},
						"mm_placeholders": map[string]any{
							"image": []any{
								map[string]any{"offset": float64(1), "length": float64(3)},
								map[string]any{"offset": float64(4), "length": float64(3)},
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyBytes, err := json.Marshal(tt.body)
			if err != nil {
				t.Fatalf("Invalid tt.body %v: cannot convert to bytes", tt.body)
			}
			got, err := parser.ParseRequest(context.Background(), bodyBytes, tt.headers)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if got.SkipResponseProcessing {
				t.Errorf("ParseRequest() got.SkipResponseProcessing = true, want false")
			}
			if diff := cmp.Diff(tt.want, got.Body); diff != "" {
				t.Errorf("ParseRequest() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestVllmHTTPParser_ParseRequest_GenerateErrorPaths confirms that
// unmarshal errors and the empty-token_ids case surface distinct messages,
// so callers see the underlying validation problem (e.g. negative token IDs)
// instead of a generic "must have non-empty token_ids" message.
func TestVllmHTTPParser_ParseRequest_GenerateErrorPaths(t *testing.T) {
	parser := NewVllmHTTPParser()
	headers := map[string]string{":path": "/inference/v1/generate"}

	tests := []struct {
		name        string
		body        string
		errContains string
	}{
		{
			name:        "negative token id surfaces unmarshal error",
			body:        `{"token_ids":[1,2,-1]}`,
			errContains: "token_ids[2]: invalid value",
		},
		{
			name:        "empty token_ids surfaces empty-field error",
			body:        `{"token_ids":[]}`,
			errContains: "must have non-empty token_ids field",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parser.ParseRequest(context.Background(), []byte(tt.body), headers)
			if err == nil {
				t.Fatalf("ParseRequest() error = nil, want error containing %q", tt.errContains)
			}
			if !strings.Contains(err.Error(), tt.errContains) {
				t.Errorf("ParseRequest() error = %q, want substring %q", err.Error(), tt.errContains)
			}
		})
	}
}

// TestVllmHTTPParser_RejectsNonGeneratePaths confirms that non-generate paths
// are rejected with an error.
func TestVllmHTTPParser_RejectsNonGeneratePaths(t *testing.T) {
	parser := NewVllmHTTPParser()

	body, _ := json.Marshal(map[string]any{
		"prompt": "hello world",
	})
	_, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": "/v1/completions"})
	if err == nil {
		t.Fatal("ParseRequest() expected error for non-generate path, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported path") {
		t.Errorf("expected error to contain 'unsupported path', got: %v", err)
	}
}

func TestVllmHTTPParser_ParseRequest_MaxOutputTokens(t *testing.T) {
	parser := NewVllmHTTPParser()
	headers := map[string]string{":path": "/inference/v1/generate"}

	tests := []struct {
		name string
		body map[string]any
		want *int64
	}{
		{
			name: "sampling_params max_tokens present",
			body: map[string]any{
				"token_ids":       []any{1, 2, 3},
				"sampling_params": map[string]any{"max_tokens": float64(128)},
			},
			want: ptr.To(int64(128)),
		},
		{
			name: "no sampling_params",
			body: map[string]any{"token_ids": []any{1, 2, 3}},
			want: nil,
		},
		{
			name: "sampling_params without max_tokens",
			body: map[string]any{
				"token_ids":       []any{1, 2, 3},
				"sampling_params": map[string]any{"temperature": 0.8},
			},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyBytes, err := json.Marshal(tt.body)
			if err != nil {
				t.Fatalf("marshal body: %v", err)
			}
			got, err := parser.ParseRequest(context.Background(), bodyBytes, headers)
			if err != nil {
				t.Fatalf("ParseRequest() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got.Body.MaxOutputTokens); diff != "" {
				t.Errorf("MaxOutputTokens mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestVllmHTTPParser_Claims(t *testing.T) {
	parser := NewVllmHTTPParser()
	got := parser.Claims()
	want := fwkrh.Claims{
		Paths:     []string{generatePathSuffix},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Claims() mismatch (-want +got):\n%s", diff)
	}
}
