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

// Package agentidentity provides a RequestHeaderProcessor plugin that resolves
// agent identity from provider-specific headers and stores it as a request
// attribute for use by other subsystems (flow control, scheduling, KV cache, etc.).
package agentidentity

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// AgentIdentityKey is the request-attribute key under which this plugin
// publishes the resolved agent identity. Downstream consumers such as the
// Director read it via scheduling.ReadRequestAttribute to derive the
// FairnessID.
// The key carries no producer name: configuration names this attribute bare
// ("attribute: agent-identity"), and any plugin able to resolve an agent
// identity is an acceptable source.
var AgentIdentityKey = plugin.NewDataKey("agent-identity", "")

// ThreadIDKey is the request-attribute key under which this plugin publishes
// the thread that sent the request: the main conversation or one sub-agent.
// It equals the agent-identity value when no thread header is present, so a
// client that sends none yields one thread per session.
var ThreadIDKey = plugin.NewDataKey("thread-id", "")

// ParentThreadIDKey is the request-attribute key under which this plugin
// publishes the thread that spawned the sending thread. It is not published
// on a root thread; an absent or empty read means root.
var ParentThreadIDKey = plugin.NewDataKey("parent-thread-id", "")

const (
	PluginType              = "agent-identity"
	ClaudeCodeSessionHeader = "x-claude-code-session-id"
	OpenCodeSessionHeader   = "x-session-affinity"
	// CodexSessionHeader is the current (Codex >= 0.131.0) hyphenated form.
	CodexSessionHeader = "session-id"
	// CodexSessionHeaderLegacy is the underscored form used by Codex 0.130.x;
	// kept as a fallback for the brief window before users upgrade.
	CodexSessionHeaderLegacy = "session_id"
	// ParentSessionHeader is the shared wire-convention header carrying the
	// parent session ID one hop up.
	ParentSessionHeader = "x-parent-session-id"
	// OpenCodeOwnSessionHeader carries the request's own opencode session ID,
	// one conversation per value. It is not interchangeable with
	// OpenCodeSessionHeader (x-session-affinity), which opencode uses for
	// cache affinity and which carries the parent one hop up on opencode 2.x.
	OpenCodeOwnSessionHeader = "x-opencode-session-id"
	// StandardThreadHeader and StandardParentThreadHeader are the canonical
	// llm-d headers: any client can send the thread and parent directly,
	// without harness-specific detection. They are checked before the
	// harness headers.
	StandardThreadHeader       = "x-llm-d-thread-id"
	StandardParentThreadHeader = "x-llm-d-parent-thread-id"
	// Claude Code: a sub-agent sends its agent id; a nested one also sends
	// the parent's. The main thread sends neither.
	ClaudeCodeAgentHeader       = "x-claude-code-agent-id"
	ClaudeCodeParentAgentHeader = "x-claude-code-parent-agent-id"
	// Codex: every thread sends thread-id, and a sub-agent adds the parent
	// thread's id. thread_id is the 0.130.x legacy form.
	CodexThreadHeader       = "thread-id"
	CodexThreadHeaderLegacy = "thread_id"
	CodexParentThreadHeader = "x-codex-parent-thread-id"
)

// defaultPriorityHeaders is the built-in ordered list, always checked. User-
// supplied headers from Parameters.AdditionalSessionHeaders are inserted
// before this list so operators can prepend higher-priority entries without
// losing the defaults.
var defaultPriorityHeaders = []string{
	ClaudeCodeSessionHeader,
	OpenCodeSessionHeader,
	CodexSessionHeader,
	CodexSessionHeaderLegacy,
}

// defaultThreadHeaders is the built-in ordered list of thread-ID headers:
// the standard pair first, then the harness forms. User-supplied headers
// from Parameters.AdditionalThreadHeaders are inserted before it.
var defaultThreadHeaders = []string{
	StandardThreadHeader,
	ClaudeCodeAgentHeader,
	CodexThreadHeader,
	CodexThreadHeaderLegacy,
	OpenCodeOwnSessionHeader,
}

// defaultParentThreadHeaders is the built-in ordered list of parent-ID
// headers: the standard pair first, then the harness forms. User-supplied
// headers from Parameters.AdditionalParentThreadHeaders are inserted before
// it.
var defaultParentThreadHeaders = []string{
	StandardParentThreadHeader,
	ClaudeCodeParentAgentHeader,
	CodexParentThreadHeader,
	ParentSessionHeader,
}

// Parameters is the user-facing plugin configuration block. See the package
// README for a configmap example showing how additionalSessionHeaders extends
// the built-in default list.
type Parameters struct {
	// AdditionalSessionHeaders is prepended to the built-in default list.
	// Order is preserved; the request-time loop short-circuits on first match.
	AdditionalSessionHeaders []string `json:"additionalSessionHeaders,omitempty"`
	// AdditionalThreadHeaders is prepended to the built-in thread list,
	// the same way AdditionalSessionHeaders works for the session list.
	AdditionalThreadHeaders []string `json:"additionalThreadHeaders,omitempty"`
	// AdditionalParentThreadHeaders is prepended to the built-in parent
	// list, the same way AdditionalSessionHeaders works for the session list.
	AdditionalParentThreadHeaders []string `json:"additionalParentThreadHeaders,omitempty"`
}

// PluginFactory is the factory function for the agent identity plugin.
func PluginFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var params Parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("agent-identity: failed to decode parameters: %w", err)
		}
	}

	return &Plugin{
		typedName:           plugin.TypedName{Type: PluginType, Name: name},
		priorityHeaders:     mergeHeaders(params.AdditionalSessionHeaders, defaultPriorityHeaders),
		threadHeaders:       mergeHeaders(params.AdditionalThreadHeaders, defaultThreadHeaders),
		parentThreadHeaders: mergeHeaders(params.AdditionalParentThreadHeaders, defaultParentThreadHeaders),
	}, nil
}

// mergeHeaders returns extras followed by defaults, lowercased. Request headers
// are stored with lowercased keys (see handlers.HandleRequestHeaders), so the
// priority list must match. Duplicates and empty strings are left in — the
// request-time loop short-circuits on first match.
func mergeHeaders(extras, defaults []string) []string {
	merged := make([]string, 0, len(extras)+len(defaults))
	for _, h := range extras {
		merged = append(merged, strings.ToLower(h))
	}
	for _, h := range defaults {
		merged = append(merged, strings.ToLower(h))
	}
	return merged
}

// compile-time interface assertions
var (
	_ fwkrh.RequestHeaderProcessor = &Plugin{}
	_ plugin.ProducerPlugin        = &Plugin{}
)

// Plugin resolves agent identity from provider-specific headers and stores it
// as a request attribute for use by other subsystems.
type Plugin struct {
	typedName           plugin.TypedName
	priorityHeaders     []string
	threadHeaders       []string
	parentThreadHeaders []string
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces declares the agent identity request attributes written by this
// request-header processor.
func (p *Plugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{
		AgentIdentityKey:  "",
		ThreadIDKey:       "",
		ParentThreadIDKey: "",
	}
}

func (p *Plugin) RequestHeader(_ context.Context, request *scheduling.InferenceRequest) error {
	session := ""
	for _, header := range p.priorityHeaders {
		if id := strings.TrimSpace(request.Headers[header]); id != "" {
			session = id
			request.PutAttribute(AgentIdentityKey, id)
			break
		}
	}

	thread, threadHeader := "", ""
	for _, header := range p.threadHeaders {
		if id := strings.TrimSpace(request.Headers[header]); id != "" {
			thread, threadHeader = id, header
			break
		}
	}
	if thread == "" {
		// No thread header: the session is the only thread.
		thread = session
	}
	if thread == "" {
		return nil
	}
	request.PutAttribute(ThreadIDKey, thread)

	parent := ""
	for _, header := range p.parentThreadHeaders {
		if id := strings.TrimSpace(request.Headers[header]); id != "" {
			parent = id
			break
		}
	}
	if parent == "" && threadHeader == ClaudeCodeAgentHeader {
		// A Claude Code sub-agent with no parent header is a depth-1 child
		// of the main thread, whose thread id is the session id.
		parent = session
	}
	if parent != "" && parent != thread {
		request.PutAttribute(ParentThreadIDKey, parent)
	}
	return nil
}
