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

package pipeline

import (
	"net/http"
	"testing"
)

func TestForwardedHeaders_NormalizesKeysToLowercase(t *testing.T) {
	rc := &RequestContext{
		OriginalHeaders: http.Header{
			"X-Request-Id":    {"abc-123"},
			"X-Forwarded-For": {"10.0.0.1"},
		},
	}

	out := rc.ForwardedHeaders()

	if got := out["x-request-id"]; got != "abc-123" {
		t.Fatalf("x-request-id = %q, want %q", got, "abc-123")
	}
	if _, ok := out["X-Request-Id"]; ok {
		t.Errorf("canonical key X-Request-Id should not be present; keys must be lowercased")
	}
	if got := out["x-forwarded-for"]; got != "10.0.0.1" {
		t.Errorf("x-forwarded-for = %q, want %q", got, "10.0.0.1")
	}
}

// A forwarding step re-stamps x-request-id from the lowercase constant. The
// forwarded copy must use the same lowercase key so the two do not coexist as
// distinct map entries.
func TestForwardedHeaders_RequestIDDoesNotDuplicateOnRestamp(t *testing.T) {
	rc := &RequestContext{
		RequestID: "abc-123",
		OriginalHeaders: http.Header{
			"X-Request-Id": {"abc-123"},
		},
	}

	headers := rc.ForwardedHeaders()
	headers["x-request-id"] = rc.RequestID

	count := 0
	for k := range headers {
		if k == "x-request-id" || k == "X-Request-Id" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("request-id header present %d times, want 1: %v", count, headers)
	}
}

func TestForwardedHeaders_ExcludesHopByHopAndContentHeaders(t *testing.T) {
	rc := &RequestContext{
		OriginalHeaders: http.Header{
			"Connection":     {"keep-alive"},
			"Content-Length": {"42"},
			"Host":           {"example.com"},
			"Content-Type":   {"application/json"},
			"X-Request-Id":   {"abc-123"},
		},
	}

	out := rc.ForwardedHeaders()

	for _, excluded := range []string{"connection", "content-length", "host", "content-type"} {
		if _, ok := out[excluded]; ok {
			t.Errorf("header %q should be excluded from forwarded headers", excluded)
		}
	}
	if _, ok := out["x-request-id"]; !ok {
		t.Errorf("x-request-id should be forwarded")
	}
}

func TestForwardedHeaders_ExcludesInternalRoutingHeaders(t *testing.T) {
	rc := &RequestContext{
		OriginalHeaders: http.Header{
			"EPP-Phase":     {"decode"},
			"X-Request-Id":  {"abc-123"},
			"Authorization": {"Bearer token"},
		},
	}

	out := rc.ForwardedHeaders()

	if _, ok := out["epp-phase"]; ok {
		t.Fatalf("epp-phase should not be forwarded: %v", out)
	}
	if got := out["x-request-id"]; got != "abc-123" {
		t.Errorf("x-request-id = %q, want %q", got, "abc-123")
	}
	if got := out["authorization"]; got != "Bearer token" {
		t.Errorf("authorization = %q, want %q", got, "Bearer token")
	}
}

func TestForwardedHeaders_NilOriginalHeaders(t *testing.T) {
	rc := &RequestContext{}
	if out := rc.ForwardedHeaders(); len(out) != 0 {
		t.Fatalf("expected empty map for nil OriginalHeaders, got %v", out)
	}
}
