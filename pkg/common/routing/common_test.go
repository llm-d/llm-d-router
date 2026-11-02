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

import (
	"net/http"
	"testing"
)

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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := HasPreference(tt.headers, tt.token); got != tt.want {
				t.Errorf("HasPreference(%v, %q) = %v, want %v", tt.headers, tt.token, got, tt.want)
			}
		})
	}
}

func TestHeaderNames(t *testing.T) {
	tests := []struct {
		name string
		want []string
	}{
		{PrefillEndpointHeader, []string{"x-llm-d-prefiller-host-port", "x-prefiller-host-port"}},
		{EncoderEndpointsHeader, []string{"x-llm-d-encoder-hosts-ports", "x-encoder-hosts-ports"}},
		{KVCacheSourceHeader, []string{"x-llm-d-kv-cache-source-host-port", "x-kv-cache-source-host-port"}},
		// Deprecated without a rename, so it has no alias of its own.
		{DataParallelEndpointHeader, []string{"x-data-parallel-host-port"}},
		{"x-unrelated", []string{"x-unrelated"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := HeaderNames(tt.name)
			if len(got) != len(tt.want) {
				t.Fatalf("HeaderNames(%q) = %v, want %v", tt.name, got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("HeaderNames(%q)[%d] = %q, want %q", tt.name, i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestSetAndDeleteRoutingHeader(t *testing.T) {
	headers := map[string]string{}
	SetRoutingHeader(headers, PrefillEndpointHeader, "10.0.0.1:8000")

	// Both spellings are written so a sidecar on either side of the rename routes.
	for _, name := range HeaderNames(PrefillEndpointHeader) {
		if headers[name] != "10.0.0.1:8000" {
			t.Errorf("%s = %q, want %q", name, headers[name], "10.0.0.1:8000")
		}
	}

	DeleteRoutingHeader(headers, PrefillEndpointHeader)
	if len(headers) != 0 {
		t.Errorf("headers not empty after delete: %v", headers)
	}
}

func TestDeleteRoutingHeaderRemovesLegacyOnlyValue(t *testing.T) {
	headers := map[string]string{LegacyPrefillEndpointHeader: "attacker:9999"}
	DeleteRoutingHeader(headers, PrefillEndpointHeader)
	if _, ok := headers[LegacyPrefillEndpointHeader]; ok {
		t.Errorf("legacy spelling survived delete: %v", headers)
	}
}

func TestTakeRoutingHeaderValues(t *testing.T) {
	tests := []struct {
		name string
		set  map[string]string
		want []string
	}{
		// Only the legacy spelling is sanitized by every supported EPP, so it is
		// the only one trusted while the alias exists.
		{"legacy only", map[string]string{LegacyPrefillEndpointHeader: "b:2"}, []string{"b:2"}},
		{
			// An upgraded EPP writes both with one value, so this is the live path.
			"both present reads the legacy value",
			map[string]string{PrefillEndpointHeader: "b:2", LegacyPrefillEndpointHeader: "b:2"},
			[]string{"b:2"},
		},
		{
			// An older EPP forwards a client-supplied canonical name untouched; it
			// must not override the target that EPP itself set (#3087).
			"canonical does not override legacy",
			map[string]string{PrefillEndpointHeader: "attacker:9999", LegacyPrefillEndpointHeader: "b:2"},
			[]string{"b:2"},
		},
		{
			// Still honored: the alias comes out in a later release, after which
			// this is the only spelling. An EPP that sanitizes both names is what
			// keeps a client from reaching here (see InternalRoutingHeaders).
			"canonical only is still read",
			map[string]string{PrefillEndpointHeader: "a:1"},
			[]string{"a:1"},
		},
		{"neither present", map[string]string{}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := http.Header{}
			for k, v := range tt.set {
				h.Set(k, v)
			}

			got := TakeRoutingHeaderValues(h, PrefillEndpointHeader)
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("got[%d] = %q, want %q", i, got[i], tt.want[i])
				}
			}

			// Taking must strip both spellings, or a routing header reaches the worker.
			for _, name := range HeaderNames(PrefillEndpointHeader) {
				if h.Get(name) != "" {
					t.Errorf("%s survived the take: %q", name, h.Get(name))
				}
			}
		})
	}
}

func TestTakeRoutingHeaderValueMultiValued(t *testing.T) {
	h := http.Header{}
	h.Add(LegacyKVCacheSourceHeader, "first:1")
	h.Add(LegacyKVCacheSourceHeader, "second:2")

	if got := TakeRoutingHeaderValue(h, KVCacheSourceHeader); got != "first:1" {
		t.Errorf("TakeRoutingHeaderValue = %q, want %q", got, "first:1")
	}
	if got := TakeRoutingHeaderValue(h, KVCacheSourceHeader); got != "" {
		t.Errorf("second take = %q, want empty", got)
	}
}

func TestTakeRoutingHeaderValuesWithoutAlias(t *testing.T) {
	// DataParallelEndpointHeader has no alias, standing in for any header once its
	// alias is dropped from headerAliases: the name itself is read.
	h := http.Header{}
	h.Set(DataParallelEndpointHeader, "dp:1")

	if got := TakeRoutingHeaderValue(h, DataParallelEndpointHeader); got != "dp:1" {
		t.Errorf("TakeRoutingHeaderValue = %q, want %q", got, "dp:1")
	}
	if h.Get(DataParallelEndpointHeader) != "" {
		t.Error("header survived the take")
	}
}
