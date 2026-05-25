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

package agentidentity

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// newDefaultPlugin builds a Plugin with no parameters — the built-in defaults.
// All PreAdmit tests below use this so they exercise the same code
// path as production-default configs.
func newDefaultPlugin(t *testing.T) *Plugin {
	t.Helper()
	pi, err := PluginFactory("test", nil, nil)
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	return pi.(*Plugin)
}

func TestPreAdmit(t *testing.T) {
	p := newDefaultPlugin(t)

	tests := []struct {
		name           string
		fairnessID     string
		headers        map[string]string
		body           *fwkrh.InferenceRequestBody
		wantFairnessID string
	}{
		{
			name:           "explicit fairness ID is preserved",
			fairnessID:     "my-explicit-id",
			headers:        map[string]string{ClaudeCodeSessionHeader: "session-abc"},
			wantFairnessID: "my-explicit-id",
		},
		{
			name:           "claude code session header used when fairness ID is empty",
			fairnessID:     "",
			headers:        map[string]string{ClaudeCodeSessionHeader: "session-abc"},
			wantFairnessID: "session-abc",
		},
		{
			name:           "opencode session header",
			fairnessID:     "",
			headers:        map[string]string{OpenCodeSessionHeader: "oc-session-1"},
			wantFairnessID: "oc-session-1",
		},
		{
			name:           "codex session header (hyphenated, >= 0.131.0)",
			fairnessID:     "",
			headers:        map[string]string{CodexSessionHeader: "codex-session-1"},
			wantFairnessID: "codex-session-1",
		},
		{
			name:           "codex legacy session header (underscored, <= 0.130.x)",
			fairnessID:     "",
			headers:        map[string]string{CodexSessionHeaderLegacy: "codex-legacy-1"},
			wantFairnessID: "codex-legacy-1",
		},
		{
			name:       "priority order: codex hyphenated wins over legacy underscored",
			fairnessID: "",
			headers: map[string]string{
				CodexSessionHeader:       "codex-new",
				CodexSessionHeaderLegacy: "codex-old",
			},
			wantFairnessID: "codex-new",
		},
		{
			name:       "priority order: claude code wins over opencode",
			fairnessID: "",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "session-abc",
				OpenCodeSessionHeader:   "oc-session-1",
			},
			wantFairnessID: "session-abc",
		},
		{
			name:       "priority order: opencode wins over codex",
			fairnessID: "",
			headers: map[string]string{
				OpenCodeSessionHeader: "oc-session-1",
				CodexSessionHeader:    "codex-session-1",
			},
			wantFairnessID: "oc-session-1",
		},
		{
			name:       "previous_response_id in body is ignored",
			fairnessID: "",
			headers:    map[string]string{},
			body: &fwkrh.InferenceRequestBody{
				Payload: fwkrh.PayloadMap{"previous_response_id": "resp-456"},
			},
			wantFairnessID: "",
		},
		{
			name:           "nil body does not panic",
			fairnessID:     "",
			headers:        map[string]string{},
			body:           nil,
			wantFairnessID: "",
		},
		{
			name:           "no matching headers leaves fairness ID empty",
			fairnessID:     "",
			headers:        map[string]string{"x-unrelated": "value"},
			wantFairnessID: "",
		},
		{
			name:           "empty headers leaves fairness ID empty",
			fairnessID:     "",
			headers:        map[string]string{},
			wantFairnessID: "",
		},
		{
			name:           "nil headers does not panic",
			fairnessID:     "",
			headers:        nil,
			wantFairnessID: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &scheduling.InferenceRequest{
				FairnessID: tt.fairnessID,
				Headers:    tt.headers,
				Body:       tt.body,
			}
			err := p.PreAdmit(context.Background(), req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if req.FairnessID != tt.wantFairnessID {
				t.Errorf("FairnessID = %q, want %q", req.FairnessID, tt.wantFairnessID)
			}
		})
	}
}

func TestPluginFactory_PriorityHeaders(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		want    []string
		wantErr bool
	}{
		{
			name: "nil parameters → defaults only",
			raw:  "",
			want: defaultPriorityHeaders,
		},
		{
			name: "empty additionalSessionHeaders → defaults only",
			raw:  `{"additionalSessionHeaders":[]}`,
			want: defaultPriorityHeaders,
		},
		{
			name: "extras prepended before defaults",
			raw:  `{"additionalSessionHeaders":["x-custom-1","x-custom-2"]}`,
			want: append([]string{"x-custom-1", "x-custom-2"}, defaultPriorityHeaders...),
		},
		{
			name: "mixed-case extras lowercased to match request header map",
			raw:  `{"additionalSessionHeaders":["X-Tenant-ID","X-User-Session"]}`,
			want: append([]string{"x-tenant-id", "x-user-session"}, defaultPriorityHeaders...),
		},
		{
			name:    "malformed json → factory error",
			raw:     `{"additionalSessionHeaders": not-json}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pi, err := PluginFactory("test", json.RawMessage(tt.raw), nil)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("PluginFactory: %v", err)
			}
			got := pi.(*Plugin).priorityHeaders
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("priorityHeaders mismatch:\n got=%v\nwant=%v", got, tt.want)
			}
		})
	}
}

// TestPreAdmit_CustomHeader proves end-to-end that a header added
// via additionalSessionHeaders is honored at request time.
func TestPreAdmit_CustomHeader(t *testing.T) {
	pi, err := PluginFactory("test",
		json.RawMessage(`{"additionalSessionHeaders":["x-tenant-id"]}`), nil)
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	p := pi.(*Plugin)

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{"x-tenant-id": "tenant-42"},
	}
	if err := p.PreAdmit(context.Background(), req); err != nil {
		t.Fatalf("PreAdmit: %v", err)
	}
	if req.FairnessID != "tenant-42" {
		t.Errorf("FairnessID = %q, want %q", req.FairnessID, "tenant-42")
	}

	// And it wins over a default-bucket header (because it is prepended).
	req2 := &scheduling.InferenceRequest{
		Headers: map[string]string{
			"x-tenant-id":           "tenant-42",
			ClaudeCodeSessionHeader: "claude-session",
		},
	}
	if err := p.PreAdmit(context.Background(), req2); err != nil {
		t.Fatalf("PreAdmit: %v", err)
	}
	if req2.FairnessID != "tenant-42" {
		t.Errorf("FairnessID = %q, want %q (custom should win)", req2.FairnessID, "tenant-42")
	}
}
