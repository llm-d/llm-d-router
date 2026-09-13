/*
Copyright 2025 The llm-d Authors.

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

package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/toolcalling"
)

func TestToolCallingSnapshotFromHTTPHeaders(t *testing.T) {
	t.Run("no_headers", func(t *testing.T) {
		h := http.Header{}
		if s := toolCallingSnapshotFromHTTPHeaders(h); s != nil {
			t.Fatalf("expected nil, got %+v", s)
		}
	})

	t.Run("partial_headers_without_hash", func(t *testing.T) {
		h := http.Header{}
		h.Set(toolcalling.HeaderToolPresent, "true")
		if s := toolCallingSnapshotFromHTTPHeaders(h); s != nil {
			t.Fatalf("expected nil without snapshot hash, got %+v", s)
		}
	})

	t.Run("full_headers", func(t *testing.T) {
		h := http.Header{}
		h.Set(toolcalling.HeaderSnapshotHash, "abc123")
		h.Set(toolcalling.HeaderToolPresent, "true")
		h.Set(toolcalling.HeaderToolChoiceKind, "auto")
		h.Set(toolcalling.HeaderToolDefsCount, "3")
		h.Set(toolcalling.HeaderParallelToolCalls, "true")

		s := toolCallingSnapshotFromHTTPHeaders(h)
		if s == nil {
			t.Fatal("expected snapshot, got nil")
		}
		if s.SnapshotHash != "abc123" {
			t.Errorf("SnapshotHash = %q, want %q", s.SnapshotHash, "abc123")
		}
		if !s.Present {
			t.Error("Present should be true")
		}
		if s.ToolChoiceKind != "auto" {
			t.Errorf("ToolChoiceKind = %q, want %q", s.ToolChoiceKind, "auto")
		}
		if s.ToolDefinitionsCount != 3 {
			t.Errorf("ToolDefinitionsCount = %d, want 3", s.ToolDefinitionsCount)
		}
		if s.ParallelToolCalls != "true" {
			t.Errorf("ParallelToolCalls = %q, want %q", s.ParallelToolCalls, "true")
		}
	})

	t.Run("canonical_casing", func(t *testing.T) {
		h := http.Header{}
		h.Set("X-Llm-D-Tool-Snapshot-Hash", "def456")
		h.Set("X-Llm-D-Tool-Present", "false")

		s := toolCallingSnapshotFromHTTPHeaders(h)
		if s == nil {
			t.Fatal("expected snapshot, got nil")
		}
		if s.SnapshotHash != "def456" {
			t.Errorf("SnapshotHash = %q, want %q", s.SnapshotHash, "def456")
		}
	})
}

func TestMetricsEndpoint(t *testing.T) {
	registerSidecarMetrics()
	handler := metricsHandler()

	recordToolCallingPreservation("true")

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	body := rec.Body.String()
	if !strings.Contains(body, "llm_d_sidecar_tool_calling_preserved_total") {
		t.Error("metrics response missing tool_calling_preserved_total")
	}
	if !strings.Contains(body, `preserved="true"`) {
		t.Error("metrics response missing preserved=true label")
	}
}
