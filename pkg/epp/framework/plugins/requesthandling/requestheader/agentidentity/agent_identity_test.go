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

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
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
		body          *fwkrequest.InferenceRequestBody
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
			body:          &fwkrequest.InferenceRequestBody{Payload: fwkrequest.PayloadMap{"previous_response_id": "resp-456"}},
			wantAttrFound: false,
		},
		{
			name: "whitespace-only header falls through to next priority header",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "   ",
				OpenCodeSessionHeader:   "oc-session-1",
			},
			wantIdentity:  "oc-session-1",
			wantAttrFound: true,
		},
		{
			name: "whitespace-only header with no fallback leaves no attribute",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "   ",
			},
			wantAttrFound: false,
		},
		{
			name: "leading and trailing whitespace is trimmed",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "  session-abc  ",
			},
			wantIdentity:  "session-abc",
			wantAttrFound: true,
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

// TestAgentIdentityKeyMatchesDocumentedConfig pins the published key to the
// name session-affinity sources use in configuration ("attribute:
// agent-identity"). A drift makes the session-affinity fallback silently
// resolve nothing.
func TestAgentIdentityKeyMatchesDocumentedConfig(t *testing.T) {
	const documented = "agent-identity"
	if got := fwkplugin.NewDataKey(documented, ""); got != AgentIdentityKey {
		t.Errorf("config %q resolves to %s, want %s", documented, got, AgentIdentityKey)
	}
}

func TestThreadAndParentKeysMatchDocumentedConfig(t *testing.T) {
	if fwkplugin.NewDataKey("thread-id", "") != ThreadIDKey {
		t.Errorf("config \"thread-id\" does not resolve to ThreadIDKey")
	}
	if fwkplugin.NewDataKey("parent-thread-id", "") != ParentThreadIDKey {
		t.Errorf("config \"parent-thread-id\" does not resolve to ParentThreadIDKey")
	}
}

func TestRequestHeader_ThreadAndParent(t *testing.T) {
	p := newDefaultPlugin(t)

	tests := []struct {
		name         string
		headers      map[string]string
		wantIdentity string
		wantThread   string
		wantParent   string
	}{
		{
			name:       "claude code main thread: thread falls back to session, no parent",
			headers:    map[string]string{ClaudeCodeSessionHeader: "S"},
			wantThread: "S",
		},
		{
			name:       "claude code depth-1 child: parent inferred as session",
			headers:    map[string]string{ClaudeCodeSessionHeader: "S", ClaudeCodeAgentHeader: "A"},
			wantThread: "A",
			wantParent: "S",
		},
		{
			name:       "claude code depth-2 child: parent header wins",
			headers:    map[string]string{ClaudeCodeSessionHeader: "S", ClaudeCodeAgentHeader: "A2", ClaudeCodeParentAgentHeader: "A1"},
			wantThread: "A2",
			wantParent: "A1",
		},
		{
			name:       "codex root: thread from thread-id, no parent",
			headers:    map[string]string{CodexSessionHeader: "R", CodexThreadHeader: "R"},
			wantThread: "R",
		},
		{
			name:       "codex child: parent from x-codex-parent-thread-id",
			headers:    map[string]string{CodexSessionHeader: "R", CodexThreadHeader: "T1", CodexParentThreadHeader: "R"},
			wantThread: "T1",
			wantParent: "R",
		},
		{
			name:       "codex legacy thread_id form",
			headers:    map[string]string{CodexSessionHeader: "R", CodexThreadHeaderLegacy: "T1"},
			wantThread: "T1",
		},
		{
			name:         "opencode child: own session as thread, parent one hop up",
			headers:      map[string]string{OpenCodeSessionHeader: "R", OpenCodeOwnSessionHeader: "C", ParentSessionHeader: "R"},
			wantIdentity: "R",
			wantThread:   "C",
			wantParent:   "R",
		},
		{
			name:       "standard pair only: no identity needed",
			headers:    map[string]string{StandardThreadHeader: "T", StandardParentThreadHeader: "P"},
			wantThread: "T",
			wantParent: "P",
		},
		{
			name:       "codex thread-id wins over legacy thread_id",
			headers:    map[string]string{CodexThreadHeader: "T-new", CodexThreadHeaderLegacy: "T-old"},
			wantThread: "T-new",
		},
		{
			name:         "standard thread wins over harness thread and the depth-1 rule does not fire",
			headers:      map[string]string{ClaudeCodeSessionHeader: "S", ClaudeCodeAgentHeader: "A", StandardThreadHeader: "T"},
			wantIdentity: "S",
			wantThread:   "T",
		},
		{
			name:       "standard parent wins over harness parent",
			headers:    map[string]string{ClaudeCodeAgentHeader: "A2", ClaudeCodeParentAgentHeader: "A1", StandardParentThreadHeader: "P"},
			wantThread: "A2",
			wantParent: "P",
		},
		{
			name:       "independent lists: claude code session with opencode parent header",
			headers:    map[string]string{ClaudeCodeSessionHeader: "S", ParentSessionHeader: "R"},
			wantThread: "S",
			wantParent: "R",
		},
		{
			name:       "no thread header: thread falls back to session",
			headers:    map[string]string{CodexSessionHeader: "R"},
			wantThread: "R",
		},
		{
			name:       "self-parent is not published",
			headers:    map[string]string{OpenCodeOwnSessionHeader: "X", ParentSessionHeader: "X"},
			wantThread: "X",
		},
		{
			name:       "whitespace-only thread falls through to fallback",
			headers:    map[string]string{ClaudeCodeSessionHeader: "S", ClaudeCodeAgentHeader: "  "},
			wantThread: "S",
		},
		{
			name:       "whitespace-only claude code parent triggers depth-1 rule",
			headers:    map[string]string{ClaudeCodeSessionHeader: "S", ClaudeCodeAgentHeader: "A", ClaudeCodeParentAgentHeader: "   "},
			wantThread: "A",
			wantParent: "S",
		},
		{
			name:       "thread values are trimmed",
			headers:    map[string]string{StandardThreadHeader: "  T  ", StandardParentThreadHeader: "  P  "},
			wantThread: "T",
			wantParent: "P",
		},
		{
			name:    "no headers: nothing published",
			headers: map[string]string{},
		},
		{
			name:    "nil headers: nothing published",
			headers: nil,
		},
		{
			name:    "parent only, no thread and no session: nothing published",
			headers: map[string]string{ParentSessionHeader: "R"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &scheduling.InferenceRequest{Headers: tt.headers}
			if err := p.RequestHeader(context.Background(), req); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			gotThread, ok := scheduling.ReadRequestAttribute[string](req, ThreadIDKey)
			if tt.wantThread == "" {
				if ok {
					t.Errorf("thread-id published as %q, want absent", gotThread)
				}
			} else if !ok || gotThread != tt.wantThread {
				t.Errorf("thread-id = (%q, %v), want (%q, true)", gotThread, ok, tt.wantThread)
			}
			if tt.wantIdentity != "" {
				gotIdentity, ok := scheduling.ReadRequestAttribute[string](req, AgentIdentityKey)
				if !ok || gotIdentity != tt.wantIdentity {
					t.Errorf("agent-identity = (%q, %v), want (%q, true)", gotIdentity, ok, tt.wantIdentity)
				}
			}
			gotParent, ok := scheduling.ReadRequestAttribute[string](req, ParentThreadIDKey)
			if tt.wantParent == "" {
				if ok {
					t.Errorf("parent-thread-id published as %q, want absent", gotParent)
				}
			} else if !ok || gotParent != tt.wantParent {
				t.Errorf("parent-thread-id = (%q, %v), want (%q, true)", gotParent, ok, tt.wantParent)
			}
		})
	}
}

func TestPluginFactory_ThreadHeaders(t *testing.T) {
	tests := []struct {
		name        string
		raw         string
		want        []string
		wantParents []string
		wantErr     bool
	}{
		{
			name:        "nil parameters → defaults only",
			raw:         "",
			want:        defaultThreadHeaders,
			wantParents: defaultParentThreadHeaders,
		},
		{
			name:        "extras prepended before defaults",
			raw:         `{"additionalThreadHeaders":["x-custom-thread"]}`,
			want:        append([]string{"x-custom-thread"}, defaultThreadHeaders...),
			wantParents: defaultParentThreadHeaders,
		},
		{
			name:        "mixed-case extras lowercased",
			raw:         `{"additionalThreadHeaders":["X-Custom-Thread"]}`,
			want:        append([]string{"x-custom-thread"}, defaultThreadHeaders...),
			wantParents: defaultParentThreadHeaders,
		},
		{
			name:        "parent extras prepended and lowercased",
			raw:         `{"additionalParentThreadHeaders":["X-My-Parent"]}`,
			want:        defaultThreadHeaders,
			wantParents: append([]string{"x-my-parent"}, defaultParentThreadHeaders...),
		},
		{
			name:    "unknown field → factory error",
			raw:     `{"additionalThreadHeader":[]}`,
			wantErr: true,
		},
		{
			name:    "malformed json → factory error",
			raw:     `{"additionalThreadHeaders": not-json}`,
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
			got := pi.(*Plugin).threadHeaders
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("threadHeaders mismatch:\n got=%v\nwant=%v", got, tt.want)
			}
			gotParents := pi.(*Plugin).parentThreadHeaders
			if !reflect.DeepEqual(gotParents, tt.wantParents) {
				t.Errorf("parentThreadHeaders mismatch:\n got=%v\nwant=%v", gotParents, tt.wantParents)
			}
		})
	}
}

// TestRequestHeader_CustomThreadHeaders proves end-to-end that headers
// added via additionalThreadHeaders / additionalParentThreadHeaders are
// honored at request time.
func TestRequestHeader_CustomThreadHeaders(t *testing.T) {
	pi, err := PluginFactory("test",
		fwkplugin.StrictDecoder(json.RawMessage(`{"additionalSessionHeaders":["x-my-session"],"additionalThreadHeaders":["x-my-thread"],"additionalParentThreadHeaders":["x-my-parent"]}`)), nil)
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	p := pi.(*Plugin)

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{
			"x-my-thread":              "T",
			"x-my-parent":              "P",
			StandardThreadHeader:       "ignored",
			StandardParentThreadHeader: "ignored",
		},
	}
	if err := p.RequestHeader(context.Background(), req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	gotThread, _ := scheduling.ReadRequestAttribute[string](req, ThreadIDKey)
	gotParent, _ := scheduling.ReadRequestAttribute[string](req, ParentThreadIDKey)
	if gotThread != "T" || gotParent != "P" {
		t.Errorf("thread = %q, parent = %q; want T, P (custom should win)", gotThread, gotParent)
	}

	// An operator session extra feeds the fallback thread and the Claude
	// Code depth-1 parent: the parent equals the resolved session value.
	req2 := &scheduling.InferenceRequest{
		Headers: map[string]string{
			"x-my-session":        "OP",
			ClaudeCodeAgentHeader: "A",
		},
	}
	if err := p.RequestHeader(context.Background(), req2); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	gotThread2, _ := scheduling.ReadRequestAttribute[string](req2, ThreadIDKey)
	gotParent2, _ := scheduling.ReadRequestAttribute[string](req2, ParentThreadIDKey)
	if gotThread2 != "A" || gotParent2 != "OP" {
		t.Errorf("thread = %q, parent = %q; want A, OP (depth-1 parent is the operator session)", gotThread2, gotParent2)
	}
}

func TestProducesAgentIdentity(t *testing.T) {
	t.Parallel()

	plugin := &Plugin{}
	for _, key := range []fwkplugin.DataKey{AgentIdentityKey, ThreadIDKey, ParentThreadIDKey} {
		produced, ok := plugin.Produces()[key]
		if !ok {
			t.Fatalf("Produces() does not declare %s", key)
		}
		if _, ok := produced.(string); !ok {
			t.Fatalf("Produces()[%s] has type %T, want string", key, produced)
		}
	}
}
