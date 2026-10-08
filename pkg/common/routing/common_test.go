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

// Package routing contains routing constants and utilities shared between
// the EPP/Inference-Scheduler and the Routing Sidecar.
//
//revive:disable:var-naming
package routing

import "testing"

func TestStripScheme(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "http scheme",
			input:    "http://localhost:4317",
			expected: "localhost:4317",
		},
		{
			name:     "https scheme",
			input:    "https://localhost:4317",
			expected: "localhost:4317",
		},
		{
			name:     "no scheme",
			input:    "localhost:4317",
			expected: "localhost:4317",
		},
		{
			name:     "host only",
			input:    "localhost",
			expected: "localhost",
		},
		{
			name:     "http with domain",
			input:    "http://otel-collector.monitoring.svc.cluster.local:4317",
			expected: "otel-collector.monitoring.svc.cluster.local:4317",
		},
		{
			name:     "https with domain",
			input:    "https://otel-collector.monitoring.svc.cluster.local:4317",
			expected: "otel-collector.monitoring.svc.cluster.local:4317",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "ip address with http",
			input:    "http://10.0.0.1:4317",
			expected: "10.0.0.1:4317",
		},
		{
			name:     "ip address with https",
			input:    "https://10.0.0.1:4317",
			expected: "10.0.0.1:4317",
		},
		{
			name:     "ip address without scheme",
			input:    "10.0.0.1:4317",
			expected: "10.0.0.1:4317",
		},
		{
			name:     "schemeless with double slash",
			input:    "//192.168.1.1:80",
			expected: "192.168.1.1:80",
		},
		{
			name:     "uppercase scheme",
			input:    "HTTP://localhost:4317",
			expected: "localhost:4317",
		},
		{
			name:     "port only",
			input:    ":9090",
			expected: ":9090",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := StripScheme(tt.input)
			if result != tt.expected {
				t.Errorf("StripScheme(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestIsConditionalDecode(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		want    bool
	}{
		{"nil headers", nil, false},
		{"empty headers", map[string]string{}, false},
		{"unrelated header", map[string]string{"x-other": "v"}, false},
		{"prefer return=minimal (not if-available)", map[string]string{PreferHeader: "return=minimal"}, false},
		{"prefer if-available", map[string]string{PreferHeader: PreferIfAvailable}, true},
		{"prefer If-Available case insensitive", map[string]string{PreferHeader: "If-Available"}, true},
		{"prefer with multiple tokens including if-available", map[string]string{PreferHeader: "return=minimal, if-available"}, true},
		{"prefer if-available with parameter", map[string]string{PreferHeader: "if-available;param=v"}, true},
		{"prefer if-available with a value", map[string]string{PreferHeader: "if-available=1"}, true},
		{"prefer if-available with leading whitespace", map[string]string{PreferHeader: "  if-available  "}, true},
		{"prefer with similar but distinct token", map[string]string{PreferHeader: "if-available-but-different"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsConditionalDecode(tt.headers); got != tt.want {
				t.Errorf("IsConditionalDecode(%v) = %v, want %v", tt.headers, got, tt.want)
			}
		})
	}
}

func TestHasPreference(t *testing.T) {
	// respond-async is a standard RFC 7240 token that the router does not use.
	const other = "respond-async"

	tests := []struct {
		name    string
		headers map[string]string
		token   string
		want    bool
	}{
		{"nil headers", nil, other, false},
		{"empty headers", map[string]string{}, other, false},
		{"empty Prefer value", map[string]string{PreferHeader: ""}, other, false},
		// The empty and whitespace-only token cases return at the empty want
		// guard before the Prefer value is split, so they test that guard only.
		{"empty token does not match an empty Prefer value", map[string]string{PreferHeader: ""}, "", false},
		{"empty token does not match a trailing comma", map[string]string{PreferHeader: "respond-async,"}, "", false},
		{"empty token does not match a leading comma", map[string]string{PreferHeader: ", respond-async"}, "", false},
		{"empty token does not match an entry with no token", map[string]string{PreferHeader: "=x"}, "", false},
		{"whitespace-only token does not match a trailing comma", map[string]string{PreferHeader: "respond-async, "}, " ", false},
		{"token", map[string]string{PreferHeader: other}, other, true},
		{"token with surrounding whitespace in want", map[string]string{PreferHeader: other}, "  respond-async  ", true},
		{"token case insensitive", map[string]string{PreferHeader: "Respond-Async"}, other, true},
		{"token among tokens with a parameter", map[string]string{PreferHeader: "if-available, respond-async;x=1"}, other, true},
		{"token with whitespace", map[string]string{PreferHeader: "  respond-async  "}, other, true},
		{"only a different token", map[string]string{PreferHeader: PreferIfAvailable}, other, false},
		{"a token does not match a different token", map[string]string{PreferHeader: other}, PreferIfAvailable, false},
		{"prefix of a longer token", map[string]string{PreferHeader: "respond-asynchronous"}, other, false},
		{"token with a value", map[string]string{PreferHeader: "return=minimal"}, "return", true},
		{"token with a value and whitespace around =", map[string]string{PreferHeader: "wait = 10, respond-async"}, "wait", true},
		{"token with a value and a parameter", map[string]string{PreferHeader: "return=minimal;x=1"}, "return", true},
		{"value is not matched as a token", map[string]string{PreferHeader: "return=minimal"}, "minimal", false},
		{"reserve-endpoint", map[string]string{PreferHeader: PreferReserveEndpoint}, PreferReserveEndpoint, true},
		{"reserve-endpoint case insensitive", map[string]string{PreferHeader: "Reserve-Endpoint"}, PreferReserveEndpoint, true},
		{"reserve-endpoint among tokens", map[string]string{PreferHeader: "if-available, reserve-endpoint;x=1"}, PreferReserveEndpoint, true},
		{"reserve-endpoint does not match if-available", map[string]string{PreferHeader: PreferReserveEndpoint}, PreferIfAvailable, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := HasPreference(tt.headers, tt.token); got != tt.want {
				t.Errorf("HasPreference(%v, %q) = %v, want %v", tt.headers, tt.token, got, tt.want)
			}
			if got := HasPreferenceToken(tt.headers[PreferHeader], tt.token); got != tt.want {
				t.Errorf("HasPreferenceToken(%q, %q) = %v, want %v", tt.headers[PreferHeader], tt.token, got, tt.want)
			}
		})
	}
}
