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
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const requestsMetric = "llm_d_epp_agent_identity_requests_total"

// newTestPlugin builds the plugin from JSON parameters with a handle whose
// metrics recorder is a private registry, and returns both.
func newTestPlugin(t *testing.T, params string) (*Plugin, *prometheus.Registry) {
	t.Helper()
	reg := prometheus.NewRegistry()
	handle := fwkplugin.NewEppHandle(t.Context(), nil, fwkplugin.WithMetricsRecorder(reg))
	pi, err := PluginFactory("test", json.NewDecoder(strings.NewReader(params)), handle)
	if err != nil {
		t.Fatalf("PluginFactory(%s): %v", params, err)
	}
	return pi.(*Plugin), reg
}

func run(t *testing.T, p *Plugin, headers map[string]string) *scheduling.InferenceRequest {
	t.Helper()
	req := &scheduling.InferenceRequest{Headers: headers}
	if err := p.RequestHeader(context.Background(), req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	return req
}

func count(p *Plugin, harness, header string) float64 {
	return testutil.ToFloat64(p.requests.WithLabelValues(harness, header))
}

func TestParametersDecode(t *testing.T) {
	for _, tc := range []struct {
		params      string
		wantSilent  bool
		wantMetrics bool
	}{
		{params: `{}`},
		{params: `{"publishIdentity": true}`},
		{params: `{"metrics": true}`, wantMetrics: true},
		{params: `{"publishIdentity": false, "metrics": true}`, wantSilent: true, wantMetrics: true},
	} {
		t.Run(tc.params, func(t *testing.T) {
			p, _ := newTestPlugin(t, tc.params)
			if p.silent != tc.wantSilent {
				t.Errorf("silent = %v, want %v", p.silent, tc.wantSilent)
			}
			if (p.requests != nil) != tc.wantMetrics {
				t.Errorf("metrics enabled = %v, want %v", p.requests != nil, tc.wantMetrics)
			}
		})
	}

	if _, err := PluginFactory("bad", json.NewDecoder(strings.NewReader(`{"publishIdentity": "no"}`)), nil); err == nil {
		t.Fatal("expected a decode error for a non-boolean publishIdentity")
	}
}

// TestDefaultsRegisterNothing: without metrics the plugin registers no
// collector and keeps publishing the attribute, as before this change.
func TestDefaultsRegisterNothing(t *testing.T) {
	p, reg := newTestPlugin(t, `{}`)
	req := run(t, p, map[string]string{ClaudeCodeSessionHeader: "s1"})

	if got, ok := scheduling.ReadRequestAttribute[string](req, AgentIdentityKey); !ok || got != "s1" {
		t.Fatalf("attribute = %q, %v; want s1, true", got, ok)
	}
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	if len(families) != 0 {
		t.Fatalf("defaults registered %d metric families, want 0", len(families))
	}
}

// TestPublishIdentityFalseWritesNoAttribute covers every default header and an
// additional one.
func TestPublishIdentityFalseWritesNoAttribute(t *testing.T) {
	p, _ := newTestPlugin(t, `{"publishIdentity": false, "metrics": true, "additionalSessionHeaders": ["x-custom"]}`)
	for _, h := range append([]string{"x-custom"}, defaultPriorityHeaders...) {
		req := run(t, p, map[string]string{h: "s1"})
		if v, ok := scheduling.ReadRequestAttribute[string](req, AgentIdentityKey); ok {
			t.Errorf("header %s: attribute written (%q) with publishIdentity false", h, v)
		}
		if got := count(p, harnessUnknown, h); got != 1 {
			t.Errorf("header %s: counted %v, want 1", h, got)
		}
	}

	// Thread and parent attributes are suppressed too.
	req := run(t, p, map[string]string{
		ClaudeCodeSessionHeader:     "s1",
		ClaudeCodeAgentHeader:       "agent-1",
		ClaudeCodeParentAgentHeader: "agent-0",
	})
	for _, key := range []fwkplugin.DataKey{AgentIdentityKey, ThreadIDKey, ParentThreadIDKey} {
		if v, ok := scheduling.ReadRequestAttribute[string](req, key); ok {
			t.Errorf("%s written (%q) with publishIdentity false", key, v)
		}
	}
}

func TestRequestCounting(t *testing.T) {
	p, _ := newTestPlugin(t, `{"metrics": true}`)
	for i := 0; i < 3; i++ {
		run(t, p, map[string]string{ClaudeCodeSessionHeader: fmt.Sprintf("s%d", i), "user-agent": "claude-cli/2.1.3 (external, cli)"})
	}
	run(t, p, map[string]string{CodexSessionHeader: "c1", "user-agent": "codex_cli_rs/0.131.0 (Mac OS)"})
	run(t, p, map[string]string{"user-agent": "python-requests/2.32"})
	run(t, p, map[string]string{ClaudeCodeSessionHeader: "   ", "user-agent": "python-requests/2.32"})
	run(t, p, map[string]string{})

	for _, tc := range []struct {
		harness, header string
		want            float64
	}{
		{"claude-cli", ClaudeCodeSessionHeader, 3},
		{"codex_cli_rs", CodexSessionHeader, 1},
		{"python-requests", sessionHeaderNone, 2}, // no header, and a blank one
		{harnessUnknown, sessionHeaderNone, 1},
	} {
		if got := count(p, tc.harness, tc.header); got != tc.want {
			t.Errorf("requests{harness=%q,session_header=%q} = %v, want %v", tc.harness, tc.header, got, tc.want)
		}
	}
}

// TestSessionHeaderLabelFollowsPriority: the label is the header the attribute
// would be taken from.
func TestSessionHeaderLabelFollowsPriority(t *testing.T) {
	p, _ := newTestPlugin(t, `{"metrics": true, "additionalSessionHeaders": ["X-Opencode-Session-Id"]}`)
	req := run(t, p, map[string]string{
		"x-opencode-session-id": "own",
		OpenCodeSessionHeader:   "parent",
		CodexSessionHeader:      "codex",
	})
	if got, _ := scheduling.ReadRequestAttribute[string](req, AgentIdentityKey); got != "own" {
		t.Fatalf("attribute = %q, want own", got)
	}
	if got := count(p, harnessUnknown, "x-opencode-session-id"); got != 1 {
		t.Fatalf("counted under x-opencode-session-id %v times, want 1", got)
	}
	run(t, p, map[string]string{OpenCodeSessionHeader: "parent", CodexSessionHeader: "codex"})
	if got := count(p, harnessUnknown, OpenCodeSessionHeader); got != 1 {
		t.Fatalf("counted under %s %v times, want 1", OpenCodeSessionHeader, got)
	}
}

func TestParseHarnessToken(t *testing.T) {
	for in, want := range map[string]string{
		"claude-cli/2.1.3 (external, cli)": "claude-cli",
		"opencode/1.0.2":                   "opencode",
		"codex_cli_rs/0.131.0 (Mac OS)":    "codex_cli_rs",
		"OpenAI/Python 1.99.0":             "openai",
		"curl/8.5.0":                       "curl",
		"x$%^/1":                           harnessOther,
		"/1.0":                             harnessOther,
		strings.Repeat("a", 65):            harnessOther,
	} {
		if got := parseHarnessToken(in); got != want {
			t.Errorf("parseHarnessToken(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestHarnessLabelIsBounded(t *testing.T) {
	p, _ := newTestPlugin(t, `{"metrics": true}`)
	p.harnessLabels = newHarnessLabels(2)
	for _, ua := range []string{"a/1", "b/1", "c/1", "d/1"} {
		run(t, p, map[string]string{"user-agent": ua})
	}
	if got := count(p, "a", sessionHeaderNone) + count(p, "b", sessionHeaderNone); got != 2 {
		t.Errorf("first two harnesses counted %v, want 2", got)
	}
	if got := count(p, harnessOther, sessionHeaderNone); got != 2 {
		t.Errorf("overflow counted %v, want 2", got)
	}
	run(t, p, map[string]string{"user-agent": "  "})
	if got := count(p, harnessUnknown, sessionHeaderNone); got != 1 {
		t.Errorf("blank user-agent counted %v under unknown, want 1", got)
	}
}

func TestSessionIDsNeverInMetrics(t *testing.T) {
	p, reg := newTestPlugin(t, `{"metrics": true}`)
	ids := []string{"secret-session-1", "secret-session-2"}
	for _, id := range ids {
		run(t, p, map[string]string{ClaudeCodeSessionHeader: id, "user-agent": "claude-cli/2"})
	}
	if n := testutil.CollectAndCount(reg, requestsMetric); n != 1 {
		t.Fatalf("%s has %d series, want 1", requestsMetric, n)
	}
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	text := fmt.Sprint(families)
	for _, id := range ids {
		if strings.Contains(text, id) {
			t.Fatalf("session id %q found in gathered metrics", id)
		}
	}
}

// TestSecondMetricsInstanceRejected: a second instance enabling metrics on the
// same registry fails instead of sharing collectors, and leaves the first
// instance's collectors registered.
func TestSecondMetricsInstanceRejected(t *testing.T) {
	reg := prometheus.NewRegistry()
	handle := fwkplugin.NewEppHandle(t.Context(), nil, fwkplugin.WithMetricsRecorder(reg))
	first, err := PluginFactory("a", json.NewDecoder(strings.NewReader(`{"metrics": true}`)), handle)
	if err != nil {
		t.Fatalf("first instance: %v", err)
	}
	_, err = PluginFactory("b", json.NewDecoder(strings.NewReader(`{"metrics": true, "additionalSessionHeaders": ["x-other"]}`)), handle)
	if !errors.Is(err, errMetricsAlreadyEnabled) {
		t.Fatalf("second instance error = %v, want %v", err, errMetricsAlreadyEnabled)
	}
	if _, err := PluginFactory("c", json.NewDecoder(strings.NewReader(`{}`)), handle); err != nil {
		t.Fatalf("a second instance without metrics must still load: %v", err)
	}
	run(t, first.(*Plugin), map[string]string{})
	if n := testutil.CollectAndCount(reg, requestsMetric); n != 1 {
		t.Fatalf("first instance's counter has %d series after the rejected registration, want 1", n)
	}
}

// TestSilentWithoutMetricsRejected: publishIdentity false without metrics would
// make the plugin do nothing.
func TestSilentWithoutMetricsRejected(t *testing.T) {
	for _, params := range []string{`{"publishIdentity": false}`, `{"publishIdentity": false, "metrics": false}`} {
		if _, err := PluginFactory("x", json.NewDecoder(strings.NewReader(params)), nil); err == nil {
			t.Errorf("%s: expected a configuration error", params)
		}
	}
}

func gauge(p *Plugin, harness string) float64 {
	return testutil.ToFloat64(p.activeGauge.WithLabelValues(harness))
}

// TestActiveSessionsCountsDistinctSessions: repeated requests of one session
// count once, and a session keeps the harness of its first request.
func TestActiveSessionsCountsDistinctSessions(t *testing.T) {
	p, reg := newTestPlugin(t, `{"metrics": true}`)
	for turn := 0; turn < 4; turn++ {
		for _, id := range []string{"c1", "c2", "c3"} {
			run(t, p, map[string]string{ClaudeCodeSessionHeader: id, "user-agent": "claude-cli/2"})
		}
	}
	run(t, p, map[string]string{CodexSessionHeader: "x1", "user-agent": "codex_cli_rs/0.131"})
	run(t, p, map[string]string{CodexSessionHeader: "x1", "user-agent": "other-ua/1"}) // same session, new UA
	run(t, p, map[string]string{"user-agent": "python-requests/2"})                    // no session

	p.refreshActiveSessions(time.Now())
	if got := gauge(p, "claude-cli"); got != 3 {
		t.Errorf("active claude-cli sessions = %v, want 3", got)
	}
	if got := gauge(p, "codex_cli_rs"); got != 1 {
		t.Errorf("active codex_cli_rs sessions = %v, want 1", got)
	}
	if n := testutil.CollectAndCount(reg, "llm_d_epp_agent_identity_active_sessions"); n != 2 {
		t.Errorf("active_sessions has %d series, want 2 (no series for other-ua or session-less traffic)", n)
	}
}

// TestActiveSessionsExpire: a session drops out once it has sent nothing for
// the window, and its harness series is removed rather than left at 0.
func TestActiveSessionsExpire(t *testing.T) {
	p, reg := newTestPlugin(t, `{"metrics": true}`)
	start := time.Now()
	p.active.observe("old", "claude-cli", start)
	p.active.observe("recent", "opencode", start.Add(activeSessionWindow))

	p.refreshActiveSessions(start.Add(activeSessionWindow))
	if got := gauge(p, "claude-cli"); got != 1 {
		t.Fatalf("session at exactly the window edge counted %v, want 1", got)
	}

	p.refreshActiveSessions(start.Add(activeSessionWindow + time.Second))
	if got := gauge(p, "opencode"); got != 1 {
		t.Errorf("recent opencode session counted %v, want 1", got)
	}
	if n := testutil.CollectAndCount(reg, "llm_d_epp_agent_identity_active_sessions"); n != 1 {
		t.Errorf("active_sessions has %d series after expiry, want 1", n)
	}
	if n := len(p.active.sessions); n != 1 {
		t.Errorf("%d sessions retained after expiry, want 1", n)
	}
}

// TestActiveSessionsBounded: past the cap new sessions are not tracked, while
// known sessions keep being refreshed.
func TestActiveSessionsBounded(t *testing.T) {
	p, _ := newTestPlugin(t, `{"metrics": true}`)
	p.active = newActiveSessions(2)
	now := time.Now()
	for _, id := range []string{"a", "b", "c"} {
		p.active.observe(id, "h", now)
	}
	later := now.Add(activeSessionWindow - time.Second)
	p.active.observe("a", "h", later)

	p.refreshActiveSessions(now.Add(activeSessionWindow + time.Millisecond))
	if got := gauge(p, "h"); got != 1 {
		t.Fatalf("active sessions = %v, want 1 (only the refreshed session within the cap)", got)
	}
}

// TestActiveSessionsStoresNoRawIDs: the set is keyed by hash only.
func TestActiveSessionsStoresNoRawIDs(t *testing.T) {
	a := newActiveSessions(10)
	a.observe("secret-session", "h", time.Now())
	if strings.Contains(fmt.Sprintf("%+v", a.sessions), "secret-session") {
		t.Fatal("raw session id retained in the active-session set")
	}
}

// TestActiveSessionsRefreshStopsWithContext: the refresh goroutine exits when
// the plugin handle's context is cancelled.
func TestActiveSessionsRefreshStopsWithContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p, _ := newTestPlugin(t, `{"metrics": true}`)
	done := make(chan struct{})
	go func() {
		p.runActiveSessions(ctx)
		close(done)
	}()
	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("runActiveSessions did not return after cancel")
	}
}
