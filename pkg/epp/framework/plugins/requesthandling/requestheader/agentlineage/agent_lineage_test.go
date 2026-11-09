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

package agentlineage

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
)

// newPlugin builds a Plugin from raw JSON parameters; "" means no parameters.
func newPlugin(t *testing.T, raw string) *Plugin {
	t.Helper()
	var (
		pi  fwkplugin.Plugin
		err error
	)
	if raw == "" {
		pi, err = PluginFactory("test", nil, nil)
	} else {
		pi, err = PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	}
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	return pi.(*Plugin)
}

// send runs one request through the plugin and returns the derived root.
func send(t *testing.T, p *Plugin, fairnessID string, headers map[string]string) (string, bool) {
	t.Helper()
	req := &scheduling.InferenceRequest{Headers: headers, FairnessID: fairnessID}
	if err := p.RequestHeader(context.Background(), req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	return scheduling.ReadRequestAttribute[string](req, DerivedRootKey)
}

// wantRoot asserts the derived root of one request.
func wantRoot(t *testing.T, p *Plugin, fairnessID string, headers map[string]string, want string) {
	t.Helper()
	got, ok := send(t, p, fairnessID, headers)
	if !ok {
		t.Fatalf("derived-root attribute not found; want %q", want)
	}
	if got != want {
		t.Fatalf("derived root = %q, want %q", got, want)
	}
}

// newClock returns a controllable now function and an advance helper.
func newClock() (func() time.Time, func(time.Duration)) {
	current := time.Unix(0, 0)
	return func() time.Time { return current },
		func(d time.Duration) { current = current.Add(d) }
}

func TestRequestHeader_RootDerivation(t *testing.T) {
	tests := []struct {
		name      string
		params    string
		agentID   string // pre-set agent-identity attribute, when non-empty
		headers   map[string]string
		wantRoot  string
		wantFound bool
	}{
		{
			name:      "claude code session header is the root",
			headers:   map[string]string{agentidentity.ClaudeCodeSessionHeader: "R"},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name:      "opencode own session header, no parent, self-rooted",
			headers:   map[string]string{OpenCodeSessionHeader: "S"},
			wantRoot:  "S",
			wantFound: true,
		},
		{
			name:      "codex session header (sub-agent shape: root session)",
			headers:   map[string]string{agentidentity.CodexSessionHeader: "R"},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name:      "codex legacy session header",
			headers:   map[string]string{agentidentity.CodexSessionHeaderLegacy: "R"},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name: "session priority: claude code wins over codex",
			headers: map[string]string{
				agentidentity.ClaudeCodeSessionHeader: "CC",
				agentidentity.CodexSessionHeader:      "COD",
			},
			wantRoot:  "CC",
			wantFound: true,
		},
		{
			// The own-ID header must win: x-session-affinity carries the
			// parent one hop up, not this request's session ID.
			name: "session priority: opencode own session wins over affinity",
			headers: map[string]string{
				OpenCodeSessionHeader:               "S1",
				agentidentity.OpenCodeSessionHeader: "R",
			},
			wantRoot:  "S1",
			wantFound: true,
		},
		{
			name:      "session fallback: affinity header without own-ID header",
			headers:   map[string]string{agentidentity.OpenCodeSessionHeader: "R"},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name: "codex turn metadata session wins over session-id",
			headers: map[string]string{
				CodexTurnMetadataHeader:          `{"session_id":"R","thread_id":"T1","request_kind":"turn"}`,
				agentidentity.CodexSessionHeader: "derived-cache-key",
			},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name: "malformed turn metadata falls back to session-id",
			headers: map[string]string{
				CodexTurnMetadataHeader:          "not-json",
				agentidentity.CodexSessionHeader: "R",
			},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name: "turn metadata without session_id falls back to session-id",
			headers: map[string]string{
				CodexTurnMetadataHeader:          `{"thread_id":"T1"}`,
				agentidentity.CodexSessionHeader: "R",
			},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name: "oversized turn metadata falls back to session-id",
			headers: map[string]string{
				CodexTurnMetadataHeader:          `{"session_id":"X","pad":"` + strings.Repeat("a", maxTurnMetadataHeaderBytes) + `"}`,
				agentidentity.CodexSessionHeader: "R",
			},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name:      "stray child roots at the named parent",
			headers:   map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "R"},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name:      "opencode vendor parent header is honored",
			headers:   map[string]string{OpenCodeSessionHeader: "S1", OpenCodeParentSessionHeader: "R"},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name:      "parent without session roots at the parent",
			headers:   map[string]string{ParentSessionHeader: "R"},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name:      "whitespace is trimmed on session and parent",
			headers:   map[string]string{OpenCodeSessionHeader: "  S1  ", ParentSessionHeader: "  R  "},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name:      "whitespace-only parent falls back to the session as root",
			headers:   map[string]string{OpenCodeSessionHeader: "S", ParentSessionHeader: "   "},
			wantRoot:  "S",
			wantFound: true,
		},
		{
			// The agent-identity attribute is not a lineage session source:
			// for opencode it carries the parent one hop up, which would
			// corrupt the parent map's child keys.
			name:      "agent-identity attribute alone is not a session source",
			agentID:   "ATTR",
			headers:   map[string]string{},
			wantFound: false,
		},
		{
			name:      "no lineage headers publishes nothing",
			headers:   map[string]string{"x-unrelated": "v"},
			wantFound: false,
		},
		{
			name:      "empty headers publishes nothing",
			headers:   map[string]string{},
			wantFound: false,
		},
		{
			name:      "nil headers publishes nothing",
			headers:   nil,
			wantFound: false,
		},
		{
			name:      "additional parent header is honored",
			params:    `{"additionalParentHeaders":["x-custom-parent"]}`,
			headers:   map[string]string{OpenCodeSessionHeader: "S1", "x-custom-parent": "R"},
			wantRoot:  "R",
			wantFound: true,
		},
		{
			name:      "additional session header is honored",
			params:    `{"additionalSessionHeaders":["x-custom-session"]}`,
			headers:   map[string]string{"x-custom-session": "CS"},
			wantRoot:  "CS",
			wantFound: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newPlugin(t, tt.params)
			req := &scheduling.InferenceRequest{Headers: tt.headers}
			if tt.agentID != "" {
				req.PutAttribute(agentidentity.AgentIdentityKey, tt.agentID)
			}
			if err := p.RequestHeader(context.Background(), req); err != nil {
				t.Fatalf("RequestHeader: %v", err)
			}
			got, ok := scheduling.ReadRequestAttribute[string](req, DerivedRootKey)
			if ok != tt.wantFound {
				t.Fatalf("attribute found = %v, want %v", ok, tt.wantFound)
			}
			if ok && got != tt.wantRoot {
				t.Errorf("derived root = %q, want %q", got, tt.wantRoot)
			}
		})
	}
}

// TestRequestHeader_OpenCodeChildShape replays the exact header set opencode
// sends on a parented request: own session ID, both parent headers, and the
// affinity headers carrying the parent.
func TestRequestHeader_OpenCodeChildShape(t *testing.T) {
	p := newPlugin(t, "")

	child := map[string]string{
		OpenCodeSessionHeader:               "S1",
		OpenCodeParentSessionHeader:         "R",
		ParentSessionHeader:                 "R",
		agentidentity.OpenCodeSessionHeader: "R",
		"x-session-id":                      "R",
	}
	wantRoot(t, p, "", child, "R")

	// A later parentless replay of the child walks the recorded link to R.
	wantRoot(t, p, "", map[string]string{
		OpenCodeSessionHeader:               "S1",
		agentidentity.OpenCodeSessionHeader: "R",
	}, "R")
}

// TestRequestHeader_CodexShape replays the exact header set Codex sends:
// root threads carry a derived cache key in session-id, sub-agent threads
// carry the root session, and both carry the turn metadata with the true
// root session.
func TestRequestHeader_CodexShape(t *testing.T) {
	p := newPlugin(t, "")

	rootTurn := map[string]string{
		CodexTurnMetadataHeader:          `{"session_id":"R","thread_id":"T0","request_kind":"turn"}`,
		agentidentity.CodexSessionHeader: "derived-cache-key",
		"thread-id":                      "T0",
	}
	wantRoot(t, p, "", rootTurn, "R")

	subAgent := map[string]string{
		CodexTurnMetadataHeader:          `{"session_id":"R","thread_id":"T1","request_kind":"turn"}`,
		agentidentity.CodexSessionHeader: "R",
		"thread-id":                      "T1",
		"x-codex-parent-thread-id":       "T0",
	}
	wantRoot(t, p, "", subAgent, "R")

	// Without turn metadata, the sub-agent's session-id header still roots
	// the tree at R.
	wantRoot(t, p, "", map[string]string{
		agentidentity.CodexSessionHeader: "R",
		"thread-id":                      "T1",
	}, "R")
}

// TestRequestHeader_LineageWalk drives one three-layer tree (R -> S1 -> SS1 ->
// SSS1) through a single plugin instance, then replays arrivals without
// parent headers: every member of the tree must resolve to R.
func TestRequestHeader_LineageWalk(t *testing.T) {
	p := newPlugin(t, "")

	wantRoot(t, p, "", map[string]string{agentidentity.ClaudeCodeSessionHeader: "R"}, "R")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "R"}, "R")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "SS1", ParentSessionHeader: "S1"}, "R")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "SSS1", ParentSessionHeader: "SS1"}, "R")

	// Parentless replays walk the recorded links back to R.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "SS1"}, "R")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1"}, "R")
	wantRoot(t, p, "", map[string]string{agentidentity.ClaudeCodeSessionHeader: "R"}, "R")

	// A stray child naming an unknown parent roots at that parent.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S9", ParentSessionHeader: "UNKNOWN"}, "UNKNOWN")
}

// TestRequestHeader_TenantScoping pins that parent links never cross tenant
// scopes: the scope is the fairness-ID header value, "" when absent.
func TestRequestHeader_TenantScoping(t *testing.T) {
	p := newPlugin(t, "")

	wantRoot(t, p, "tenant-a", map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "R"}, "R")
	// tenant-b never sees tenant-a's link.
	wantRoot(t, p, "tenant-b", map[string]string{OpenCodeSessionHeader: "S1"}, "S1")
	// tenant-b records its own link for the same session ID.
	wantRoot(t, p, "tenant-b", map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "Q"}, "Q")
	// tenant-a's link is intact.
	wantRoot(t, p, "tenant-a", map[string]string{OpenCodeSessionHeader: "S1"}, "R")
	// The default tenant (no fairness header) has its own scope.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "N"}, "N")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1"}, "N")
}

// TestRequestHeader_ParentEntryTTL pins idle expiry: a link unused for the
// TTL disappears, and a link read within the TTL is refreshed by that read.
func TestRequestHeader_ParentEntryTTL(t *testing.T) {
	link := map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "R"}
	replay := map[string]string{OpenCodeSessionHeader: "S1"}

	t.Run("within ttl the link resolves", func(t *testing.T) {
		p := newPlugin(t, `{"parentEntryTtlSeconds": 10}`)
		now, advance := newClock()
		p.now = now
		wantRoot(t, p, "", link, "R")
		advance(9 * time.Second)
		wantRoot(t, p, "", replay, "R")
	})

	t.Run("past ttl without use the link is gone", func(t *testing.T) {
		p := newPlugin(t, `{"parentEntryTtlSeconds": 10}`)
		now, advance := newClock()
		p.now = now
		wantRoot(t, p, "", link, "R")
		advance(11 * time.Second)
		wantRoot(t, p, "", replay, "S1")
	})

	t.Run("a read refreshes the link", func(t *testing.T) {
		p := newPlugin(t, `{"parentEntryTtlSeconds": 10}`)
		now, advance := newClock()
		p.now = now
		wantRoot(t, p, "", link, "R")
		advance(9 * time.Second)
		wantRoot(t, p, "", replay, "R")
		advance(2 * time.Second)
		wantRoot(t, p, "", replay, "R")
	})

	t.Run("a re-declared link works again", func(t *testing.T) {
		p := newPlugin(t, `{"parentEntryTtlSeconds": 10}`)
		now, advance := newClock()
		p.now = now
		wantRoot(t, p, "", link, "R")
		advance(11 * time.Second)
		wantRoot(t, p, "", replay, "S1")
		wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "R2"}, "R2")
		wantRoot(t, p, "", replay, "R2")
	})
}

// TestRequestHeader_MapBound pins that inserting past maxParentEntries evicts
// the least-recently-used link.
func TestRequestHeader_MapBound(t *testing.T) {
	p := newPlugin(t, `{"maxParentEntries": 2}`)
	now, advance := newClock()
	p.now = now

	declare := func(session, parent string) {
		wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: session, ParentSessionHeader: parent}, parent)
	}

	declare("S1", "R")
	advance(time.Second)
	declare("S2", "R")
	advance(time.Second)
	declare("S3", "R") // bound reached: S1 is the least recently used

	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1"}, "S1")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S2"}, "R")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S3"}, "R")
}

// TestRequestHeader_CycleTerminates pins that a malformed lineage cycle
// (A -> B -> A) ends the walk with a stable root instead of looping. The
// exact member of the cycle returned is an implementation choice; the
// contract is termination and stability.
func TestRequestHeader_CycleTerminates(t *testing.T) {
	p := newPlugin(t, "")

	// A names B; B is unknown, so the walk from A stops there.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "A", ParentSessionHeader: "B"}, "B")
	// B names A: the map now holds A -> B and B -> A.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "B", ParentSessionHeader: "A"}, "A")
	// A later child naming A gets the same stable root.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "C", ParentSessionHeader: "A"}, "B")
}

// TestRequestHeader_ReDeclarationDoesNotEvict pins the bound's behavior for
// harnesses that resend both headers on every turn: re-declaring a recorded
// session must not evict an unrelated live link, because overwriting an
// existing key does not grow the map.
func TestRequestHeader_ReDeclarationDoesNotEvict(t *testing.T) {
	p := newPlugin(t, `{"maxParentEntries": 2}`)
	now, advance := newClock()
	p.now = now

	declare := func(session, parent string) {
		wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: session, ParentSessionHeader: parent}, parent)
	}

	declare("S1", "R")
	advance(time.Second)
	declare("S2", "R")
	advance(time.Second)
	declare("S3", "R") // bound reached: S1 is the least recently used

	// A turn of the recorded S2 re-declares its parent; S3 must survive it.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S2", ParentSessionHeader: "R"}, "R")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S3"}, "R")
}

// TestRequestHeader_FirstLiveParentWins pins that a live link is
// authoritative: a session's parent is fixed when the child is spawned, so a
// declaration naming a different parent is ignored until the entry expires.
func TestRequestHeader_FirstLiveParentWins(t *testing.T) {
	p := newPlugin(t, `{"parentEntryTtlSeconds": 10}`)
	now, advance := newClock()
	p.now = now

	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "R"}, "R")

	// A conflicting parent declaration does not move the session's root.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "EVIL"}, "R")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1"}, "R")

	// Once the entry expires, a different parent is accepted.
	advance(11 * time.Second)
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "R2"}, "R2")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S1"}, "R2")
}

// TestRequestHeader_DoesNotTouchFairnessIDOrAgentIdentity pins the contract
// boundaries: the fairness ID is an input (tenant scope), never rewritten,
// and the agent-identity attribute belongs to its producer.
func TestRequestHeader_DoesNotTouchFairnessIDOrAgentIdentity(t *testing.T) {
	p := newPlugin(t, "")
	req := &scheduling.InferenceRequest{
		Headers:    map[string]string{OpenCodeSessionHeader: "S1", ParentSessionHeader: "R"},
		FairnessID: "tenant-x",
	}
	req.PutAttribute(agentidentity.AgentIdentityKey, "S1")

	if err := p.RequestHeader(context.Background(), req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	if req.FairnessID != "tenant-x" {
		t.Errorf("FairnessID = %q, want unchanged %q", req.FairnessID, "tenant-x")
	}
	if id, _ := scheduling.ReadRequestAttribute[string](req, agentidentity.AgentIdentityKey); id != "S1" {
		t.Errorf("agent-identity attribute = %q, want unchanged %q", id, "S1")
	}
	if root, _ := scheduling.ReadRequestAttribute[string](req, DerivedRootKey); root != "R" {
		t.Errorf("derived root = %q, want %q", root, "R")
	}
}

// TestRequestHeader_WalkHopCap pins that a chain longer than maxWalkHops
// stops at a deterministic ancestor instead of traversing every link: a
// client can chain links up to the map bound, and the walk must not hold the
// lock for the whole chain.
func TestRequestHeader_WalkHopCap(t *testing.T) {
	p := newPlugin(t, "")

	// Declare a chain C0 <- C1 <- ... <- C70, longer than maxWalkHops.
	for i := 1; i <= 70; i++ {
		headers := map[string]string{
			OpenCodeSessionHeader: fmt.Sprintf("C%d", i),
			ParentSessionHeader:   fmt.Sprintf("C%d", i-1),
		}
		if _, ok := send(t, p, "", headers); !ok {
			t.Fatalf("declaration %d published no root", i)
		}
	}

	// Within the cap the walk reaches the chain's root.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "C5"}, "C0")

	// Past the cap the walk stops after maxWalkHops links.
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "C70"}, fmt.Sprintf("C%d", 70-maxWalkHops))
}

// TestRequestHeader_SelfLinkNotRecorded pins that a parent header naming the
// request's own session is dropped rather than stored: it carries no lineage
// and only consumes a map entry.
func TestRequestHeader_SelfLinkNotRecorded(t *testing.T) {
	p := newPlugin(t, "")
	wantRoot(t, p, "", map[string]string{OpenCodeSessionHeader: "S", ParentSessionHeader: "S"}, "S")
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.parents) != 0 {
		t.Fatalf("parent map holds %d entries after a self-link, want 0", len(p.parents))
	}
}

func TestRequestHeader_ConcurrentAccess(t *testing.T) {
	p := newPlugin(t, `{"maxParentEntries": 16, "parentEntryTtlSeconds": 1}`)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				req := &scheduling.InferenceRequest{Headers: map[string]string{
					OpenCodeSessionHeader: fmt.Sprintf("S%d-%d", i, j%10),
					ParentSessionHeader:   fmt.Sprintf("P%d", j%5),
				}}
				if err := p.RequestHeader(context.Background(), req); err != nil {
					t.Errorf("RequestHeader: %v", err)
				}
			}
		}(i)
	}
	wg.Wait()
}

func TestPluginFactory_Defaults(t *testing.T) {
	p := newPlugin(t, "")
	if !reflect.DeepEqual(p.sessionHeaders, defaultSessionHeaders) {
		t.Errorf("sessionHeaders mismatch:\n got=%v\nwant=%v", p.sessionHeaders, defaultSessionHeaders)
	}
	if !reflect.DeepEqual(p.parentHeaders, defaultParentHeaders) {
		t.Errorf("parentHeaders mismatch:\n got=%v\nwant=%v", p.parentHeaders, defaultParentHeaders)
	}
	if p.maxParentEntries != defaultMaxParentEntries {
		t.Errorf("maxParentEntries = %d, want %d", p.maxParentEntries, defaultMaxParentEntries)
	}
	if p.parentEntryTTL != defaultParentEntryTTL {
		t.Errorf("parentEntryTTL = %v, want %v", p.parentEntryTTL, defaultParentEntryTTL)
	}
	if p.now == nil {
		t.Error("now is nil, want injectable clock")
	}
}

func TestPluginFactory_Parameters(t *testing.T) {
	p := newPlugin(t, `{"additionalSessionHeaders":["X-Custom-Session"],"additionalParentHeaders":["X-Parent"],"maxParentEntries":7,"parentEntryTtlSeconds":2.5}`)
	wantSession := append([]string{"x-custom-session"}, defaultSessionHeaders...)
	wantParent := append([]string{"x-parent"}, defaultParentHeaders...)
	if !reflect.DeepEqual(p.sessionHeaders, wantSession) {
		t.Errorf("sessionHeaders mismatch:\n got=%v\nwant=%v", p.sessionHeaders, wantSession)
	}
	if !reflect.DeepEqual(p.parentHeaders, wantParent) {
		t.Errorf("parentHeaders mismatch:\n got=%v\nwant=%v", p.parentHeaders, wantParent)
	}
	if p.maxParentEntries != 7 {
		t.Errorf("maxParentEntries = %d, want 7", p.maxParentEntries)
	}
	if p.parentEntryTTL != 2500*time.Millisecond {
		t.Errorf("parentEntryTTL = %v, want %v", p.parentEntryTTL, 2500*time.Millisecond)
	}
}

func TestPluginFactory_RejectsInvalidParameters(t *testing.T) {
	for _, raw := range []string{
		`{"maxParentEntries":-1}`,
		`{"parentEntryTtlSeconds":-1}`,
		// A positive value too small to represent as a duration must not
		// silently become the default.
		`{"parentEntryTtlSeconds": 1e-12}`,
		// A value beyond the duration range must not overflow negative and
		// disable the map.
		`{"parentEntryTtlSeconds": 1e18}`,
		`{"unknown-field": 1}`,
		`{"maxParentEntries": not-json}`,
	} {
		if _, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil); err == nil {
			t.Errorf("PluginFactory(%s) succeeded, want error", raw)
		}
	}
}

func TestPluginFactory_TrimsExtraHeaders(t *testing.T) {
	p := newPlugin(t, `{"additionalParentHeaders":["  X-Parent  ","","x-keep"]}`)
	wantParent := append([]string{"x-parent", "x-keep"}, defaultParentHeaders...)
	if !reflect.DeepEqual(p.parentHeaders, wantParent) {
		t.Errorf("parentHeaders mismatch:\n got=%v\nwant=%v", p.parentHeaders, wantParent)
	}
}

func TestProducesDerivedRoot(t *testing.T) {
	p := &Plugin{}
	produced, ok := p.Produces()[DerivedRootKey]
	if !ok {
		t.Fatalf("Produces() does not declare %s", DerivedRootKey)
	}
	if _, ok := produced.(string); !ok {
		t.Fatalf("Produces()[%s] has type %T, want string", DerivedRootKey, produced)
	}
}

// TestDerivedRootKeyMatchesDocumentedConfig pins the published key to the name
// the Director reads. A drift makes the composed fairness key silently
// resolve nothing.
func TestDerivedRootKeyMatchesDocumentedConfig(t *testing.T) {
	const documented = "derived-root"
	if got := fwkplugin.NewDataKey(documented, ""); got != DerivedRootKey {
		t.Errorf("config %q resolves to %s, want %s", documented, got, DerivedRootKey)
	}
}
