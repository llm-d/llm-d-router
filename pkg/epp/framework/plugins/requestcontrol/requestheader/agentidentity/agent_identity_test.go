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

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// newDefaultPlugin builds a Plugin with no parameters — the built-in defaults.
// All RequestHeader tests below use this so they exercise the same code
// path as production-default configs.
func newDefaultPlugin(t *testing.T) *Plugin {
	t.Helper()
	pi, err := PluginFactory("test", nil, nil)
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	return pi.(*Plugin)
}

func TestRequestHeader(t *testing.T) {
	p := newDefaultPlugin(t)

	tests := []struct {
		name          string
		headers       map[string]string
		body          *fwkrh.InferenceRequestBody
		wantIdentity  string
		wantAttrFound bool
	}{
		{
			name:          "claude code session header",
			headers:       map[string]string{ClaudeCodeSessionHeader: "session-abc"},
			wantIdentity:  "session-abc",
			wantAttrFound: true,
		},
		{
			name:          "opencode session header",
			headers:       map[string]string{OpenCodeSessionHeader: "oc-session-1"},
			wantIdentity:  "oc-session-1",
			wantAttrFound: true,
		},
		{
			name:          "codex session header (hyphenated, >= 0.131.0)",
			headers:       map[string]string{CodexSessionHeader: "codex-session-1"},
			wantIdentity:  "codex-session-1",
			wantAttrFound: true,
		},
		{
			name:          "codex legacy session header (underscored, <= 0.130.x)",
			headers:       map[string]string{CodexSessionHeaderLegacy: "codex-legacy-1"},
			wantIdentity:  "codex-legacy-1",
			wantAttrFound: true,
		},
		{
			name: "priority order: codex hyphenated wins over legacy underscored",
			headers: map[string]string{
				CodexSessionHeader:       "codex-new",
				CodexSessionHeaderLegacy: "codex-old",
			},
			wantIdentity:  "codex-new",
			wantAttrFound: true,
		},
		{
			name: "priority order: claude code wins over opencode",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "session-abc",
				OpenCodeSessionHeader:   "oc-session-1",
			},
			wantIdentity:  "session-abc",
			wantAttrFound: true,
		},
		{
			name: "priority order: opencode wins over codex",
			headers: map[string]string{
				OpenCodeSessionHeader: "oc-session-1",
				CodexSessionHeader:    "codex-session-1",
			},
			wantIdentity:  "oc-session-1",
			wantAttrFound: true,
		},
		{
			name:          "previous_response_id in body is ignored",
			headers:       map[string]string{},
			body:          &fwkrh.InferenceRequestBody{Payload: fwkrh.PayloadMap{"previous_response_id": "resp-456"}},
			wantAttrFound: false,
		},
		{
			name:          "nil body does not panic",
			headers:       map[string]string{},
			body:          nil,
			wantAttrFound: false,
		},
		{
			name:          "no matching headers leaves no attribute",
			headers:       map[string]string{"x-unrelated": "value"},
			wantAttrFound: false,
		},
		{
			name:          "empty headers leaves no attribute",
			headers:       map[string]string{},
			wantAttrFound: false,
		},
		{
			name:          "nil headers does not panic",
			headers:       nil,
			wantAttrFound: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &scheduling.InferenceRequest{
				Headers: tt.headers,
				Body:    tt.body,
			}
			err := p.RequestHeader(context.Background(), req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			got, ok := scheduling.ReadRequestAttribute[string](req, AgentIdentityKey)
			if ok != tt.wantAttrFound {
				t.Errorf("attribute found = %v, want %v", ok, tt.wantAttrFound)
			}
			if ok && got != tt.wantIdentity {
				t.Errorf("agent identity = %q, want %q", got, tt.wantIdentity)
			}
			// FairnessID must never be set by the plugin.
			if req.FairnessID != "" {
				t.Errorf("FairnessID = %q, want empty (plugin must not set FairnessID)", req.FairnessID)
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
			pi, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(tt.raw)), nil)
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

// TestRequestHeader_CustomHeader proves end-to-end that a header added
// via additionalSessionHeaders is honored at request time.
func TestRequestHeader_CustomHeader(t *testing.T) {
	pi, err := PluginFactory("test",
		fwkplugin.StrictDecoder(json.RawMessage(`{"additionalSessionHeaders":["x-tenant-id"]}`)), nil)
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	p := pi.(*Plugin)

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{"x-tenant-id": "tenant-42"},
	}
	if err := p.RequestHeader(context.Background(), req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	got, ok := scheduling.ReadRequestAttribute[string](req, AgentIdentityKey)
	if !ok {
		t.Fatal("agent-identity attribute not found")
	}
	if got != "tenant-42" {
		t.Errorf("agent identity = %q, want %q", got, "tenant-42")
	}

	// And it wins over a default-bucket header (because it is prepended).
	req2 := &scheduling.InferenceRequest{
		Headers: map[string]string{
			"x-tenant-id":           "tenant-42",
			ClaudeCodeSessionHeader: "claude-session",
		},
	}
	if err := p.RequestHeader(context.Background(), req2); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	got2, ok2 := scheduling.ReadRequestAttribute[string](req2, AgentIdentityKey)
	if !ok2 {
		t.Fatal("agent-identity attribute not found")
	}
	if got2 != "tenant-42" {
		t.Errorf("agent identity = %q, want %q (custom should win)", got2, "tenant-42")
	}
}
