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

package request

import (
	"testing"
)

func TestGetRequestPath(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		want    string
	}{
		{
			name:    "primary path header",
			headers: map[string]string{":path": "/foo"},
			want:    "/foo",
		},
		{
			name:    "x-original-path header",
			headers: map[string]string{"x-original-path": "/bar"},
			want:    "/bar",
		},
		{
			name:    "x-forwarded-path header",
			headers: map[string]string{"x-forwarded-path": "/baz"},
			want:    "/baz",
		},
		{
			name:    "query parameters stripped from :path",
			headers: map[string]string{":path": "/v1/messages?beta=true"},
			want:    "/v1/messages",
		},
		{
			name:    "query parameters stripped from x-original-path",
			headers: map[string]string{"x-original-path": "/v1/messages?foo=bar&baz=1"},
			want:    "/v1/messages",
		},
		{
			name:    "query parameters stripped from x-forwarded-path",
			headers: map[string]string{"x-forwarded-path": "/v1/chat/completions?stream=true"},
			want:    "/v1/chat/completions",
		},
		{
			name:    "fallback to completions",
			headers: map[string]string{},
			want:    "/v1/completions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetRequestPath(tt.headers); got != tt.want {
				t.Errorf("GetRequestPath() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMatchPathSuffix(t *testing.T) {
	tests := []struct {
		name   string
		path   string
		suffix string
		want   bool
	}{
		{
			name:   "exact match",
			path:   "chat/completions",
			suffix: "chat/completions",
			want:   true,
		},
		{
			name:   "exact match with leading slash in path",
			path:   "/chat/completions",
			suffix: "chat/completions",
			want:   true,
		},
		{
			name:   "exact match with leading slash in suffix",
			path:   "chat/completions",
			suffix: "/chat/completions",
			want:   true,
		},
		{
			name:   "exact match with leading slash in both",
			path:   "/chat/completions",
			suffix: "/chat/completions",
			want:   true,
		},
		{
			name:   "suffix match with slash boundary",
			path:   "/v1/chat/completions",
			suffix: "chat/completions",
			want:   true,
		},
		{
			name:   "suffix match with dot boundary (gRPC)",
			path:   "/google.cloud.aiplatform.v1beta1.PredictionService/ChatCompletions",
			suffix: "PredictionService/ChatCompletions",
			want:   true,
		},
		{
			name:   "no match",
			path:   "/v1/chat/completions",
			suffix: "embeddings",
			want:   false,
		},
		{
			name:   "trailing slash in path",
			path:   "/v1/chat/completions/",
			suffix: "chat/completions",
			want:   true,
		},
		{
			name:   "trailing slash in suffix",
			path:   "/v1/chat/completions",
			suffix: "chat/completions/",
			want:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchPathSuffix(tt.path, tt.suffix); got != tt.want {
				t.Errorf("MatchPathSuffix(%q, %q) = %v, want %v", tt.path, tt.suffix, got, tt.want)
			}
		})
	}
}
