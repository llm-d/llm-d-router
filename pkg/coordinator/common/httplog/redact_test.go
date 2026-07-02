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

package httplog

import (
	"net/http"
	"strings"
	"testing"
)

func TestRedactedHeaders_LowercasesAndFlattensHTTPHeader(t *testing.T) {
	h := http.Header{
		"X-Request-Id": {"abc-123"},
		"Epp-Phase":    {"decode"},
	}

	out := RedactedHeaders(h)

	if got := out["x-request-id"]; got != "abc-123" {
		t.Errorf("x-request-id = %q, want %q", got, "abc-123")
	}
	if got := out["epp-phase"]; got != "decode" {
		t.Errorf("epp-phase = %q, want %q", got, "decode")
	}
	if _, ok := out["X-Request-Id"]; ok {
		t.Errorf("canonical key must not be present; keys are lowercased")
	}
}

func TestRedactedHeaders_LowercasesStringMap(t *testing.T) {
	out := RedactedHeaders(map[string]string{
		"x-request-id": "abc-123",
		"EPP-Phase":    "encode",
	})

	if got := out["x-request-id"]; got != "abc-123" {
		t.Errorf("x-request-id = %q, want %q", got, "abc-123")
	}
	if got := out["epp-phase"]; got != "encode" {
		t.Errorf("epp-phase = %q, want %q", got, "encode")
	}
}

func TestRedactedHeaders_RedactsSensitiveRegardlessOfInputCase(t *testing.T) {
	out := RedactedHeaders(http.Header{
		"Authorization": {"Bearer secret"},
		"X-Api-Key":     {"key"},
		"Accept":        {"*/*"},
	})

	if got := out["authorization"]; got != redactedValue {
		t.Errorf("authorization = %q, want %q", got, redactedValue)
	}
	if got := out["x-api-key"]; got != redactedValue {
		t.Errorf("x-api-key = %q, want %q", got, redactedValue)
	}
	if got := out["accept"]; got != "*/*" {
		t.Errorf("accept = %q, want %q", got, "*/*")
	}
}

// An empty value retains the key with an empty string, whether the input is a
// valueless slice or an empty string, so both input forms behave the same.
func TestRedactedHeaders_EmptyValueRetainsKey(t *testing.T) {
	fromSlice := RedactedHeaders(http.Header{"X-Empty": {}})
	if got, ok := fromSlice["x-empty"]; !ok || got != "" {
		t.Errorf("x-empty = %q (present=%v), want %q present", got, ok, "")
	}

	fromString := RedactedHeaders(map[string]string{"X-Empty": ""})
	if got, ok := fromString["x-empty"]; !ok || got != "" {
		t.Errorf("x-empty = %q (present=%v), want %q present", got, ok, "")
	}
}

func TestRedactedHeaders_TruncatesLongValue(t *testing.T) {
	long := strings.Repeat("a", maxValueLen+10)
	const phase = "decode"
	out := RedactedHeaders(http.Header{
		"X-Gateway-Peer-Metadata": {long},
		"Epp-Phase":               {phase},
	})

	want := long[:maxValueLen] + "...[truncated]"
	if got := out["x-gateway-peer-metadata"]; got != want {
		t.Errorf("long value not truncated:\n got %q\nwant %q", got, want)
	}
	if got := out["epp-phase"]; got != phase {
		t.Errorf("short value should be unchanged, got %q", got)
	}
}
