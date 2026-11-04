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

// Package agentlineage provides a RequestHeaderProcessor plugin that derives
// the program root of a request from agent-lineage headers and stores it as a
// request attribute for the Director's fairness-key composition.
package agentlineage

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
)

// DerivedRootKey is the request-attribute key under which this plugin
// publishes the derived program root: the session ID of the root request of
// the tree the request belongs to. The Director reads it via
// scheduling.ReadRequestAttribute to compose the fairness key.
// The key carries no producer name: any plugin able to derive a program root
// is an acceptable source.
var DerivedRootKey = plugin.NewDataKey("derived-root", "")

const (
	PluginType = "agent-lineage"
	// ParentSessionHeader is the shared wire-convention header carrying the
	// parent session ID one hop up. OpenCodeParentSessionHeader is
	// opencode's vendor-specific form of the same link; opencode sends both.
	ParentSessionHeader         = "x-parent-session-id"
	OpenCodeParentSessionHeader = "x-opencode-parent-session-id"
	// OpenCodeSessionHeader carries the request's own session ID. It is not
	// interchangeable with agentidentity.OpenCodeSessionHeader
	// (x-session-affinity), which carries the parent one hop up.
	OpenCodeSessionHeader = "x-opencode-session-id"
	// CodexTurnMetadataHeader carries a JSON blob whose session_id field is
	// the root session identity shared by every thread in the tree. The flat
	// session-id header is a derived cache-affinity key on root-thread
	// requests, so the turn metadata is the authoritative Codex session
	// source.
	CodexTurnMetadataHeader = "x-codex-turn-metadata"
	// maxTurnMetadataHeaderBytes bounds the JSON parse of the turn-metadata
	// header; larger values are ignored and the session falls back to the
	// header list.
	maxTurnMetadataHeaderBytes = 8192
	// maxWalkHops bounds a single parent-map walk. Nesting reaches three to
	// four layers in the harnesses, and a client can chain links up to the
	// map bound, so the walk stops past this many links and roots at the
	// current node, keeping the lock hold bounded.
	maxWalkHops = 64
)

// defaultParentHeaders is the built-in ordered list of parent-ID headers.
// Codex's x-codex-parent-thread-id is deliberately absent: it names a thread
// ID while the parent map is keyed by session IDs, and Codex needs no parent
// map because the turn-metadata session_id carries the root on every request.
var defaultParentHeaders = []string{ParentSessionHeader, OpenCodeParentSessionHeader}

// defaultSessionHeaders resolves the request's own session ID, in priority
// order. opencode's own-ID header comes first because x-session-affinity
// carries the parent one hop up rather than the session's own ID; the
// affinity header stays last as a fallback for traffic without the own-ID
// header. Codex's turn-metadata header is handled before this list.
var defaultSessionHeaders = []string{
	OpenCodeSessionHeader,
	agentidentity.ClaudeCodeSessionHeader,
	agentidentity.CodexSessionHeader,
	agentidentity.CodexSessionHeaderLegacy,
	agentidentity.OpenCodeSessionHeader,
}

// Interim parent-map bounds. The per-tenant program cap replaces
// defaultMaxParentEntries once it lands with the Sharing pillar, and the
// program-end window replaces defaultParentEntryTTL.
const (
	defaultMaxParentEntries = 4096
	defaultParentEntryTTL   = 1800 * time.Second
)

// Parameters is the user-facing plugin configuration block.
type Parameters struct {
	// AdditionalSessionHeaders is prepended to the built-in session list.
	AdditionalSessionHeaders []string `json:"additionalSessionHeaders,omitempty"`
	// AdditionalParentHeaders is prepended to the built-in parent list.
	AdditionalParentHeaders []string `json:"additionalParentHeaders,omitempty"`
	// MaxParentEntries bounds the per-replica parent map; the
	// least-recently-used entry is evicted when the bound is reached.
	MaxParentEntries int `json:"maxParentEntries,omitempty"`
	// ParentEntryTTLSeconds is how long a parent link survives unused.
	ParentEntryTTLSeconds float64 `json:"parentEntryTtlSeconds,omitempty"`
}

// lineageKey scopes a parent link to the tenant the request arrived under:
// the fairness-ID header value at classification time, "" when absent.
type lineageKey struct {
	tenant  string
	session string
}

// parentLink records one child -> parent edge and when it was last used.
type parentLink struct {
	parent    string
	updatedAt time.Time
}

// PluginFactory is the factory function for the agent lineage plugin.
func PluginFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var params Parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("agent-lineage: failed to decode parameters: %w", err)
		}
	}
	if params.MaxParentEntries < 0 {
		return nil, fmt.Errorf("agent-lineage: maxParentEntries must be >= 0, got %d", params.MaxParentEntries)
	}
	if params.ParentEntryTTLSeconds < 0 {
		return nil, fmt.Errorf("agent-lineage: parentEntryTtlSeconds must be >= 0, got %v", params.ParentEntryTTLSeconds)
	}
	maxEntries := params.MaxParentEntries
	if maxEntries == 0 {
		maxEntries = defaultMaxParentEntries
	}
	ttl := time.Duration(params.ParentEntryTTLSeconds * float64(time.Second))
	if ttl == 0 {
		ttl = defaultParentEntryTTL
	}
	return &Plugin{
		typedName:        plugin.TypedName{Type: PluginType, Name: name},
		sessionHeaders:   mergeHeaders(params.AdditionalSessionHeaders, defaultSessionHeaders),
		parentHeaders:    mergeHeaders(params.AdditionalParentHeaders, defaultParentHeaders),
		maxParentEntries: maxEntries,
		parentEntryTTL:   ttl,
		now:              time.Now,
		parents:          map[lineageKey]parentLink{},
	}, nil
}

// mergeHeaders returns extras followed by defaults, lowercased. Request
// headers are stored with lowercased keys (see handlers.HandleRequestHeaders),
// so both lists must match. Duplicates and empty strings are left in — the
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

// Plugin derives the program root of a request from agent-lineage headers
// and stores it as a request attribute.
type Plugin struct {
	typedName plugin.TypedName

	sessionHeaders   []string
	parentHeaders    []string
	maxParentEntries int
	parentEntryTTL   time.Duration

	// now is injectable so tests control TTL expiry and eviction order.
	now func() time.Time

	mu      sync.Mutex
	parents map[lineageKey]parentLink
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces declares the derived-root request attribute written by this
// request-header processor.
func (p *Plugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{DerivedRootKey: ""}
}

// RequestHeader resolves the request's own session and parent IDs, records
// the child -> parent link when a parent is named, and publishes the root
// reached by walking recorded links. A request with neither a session nor a
// parent publishes nothing and leaves the fairness key to its other inputs.
func (p *Plugin) RequestHeader(_ context.Context, request *scheduling.InferenceRequest) error {
	session := codexTurnMetadataSession(request.Headers[CodexTurnMetadataHeader])
	if session == "" {
		session = firstHeaderValue(request.Headers, p.sessionHeaders)
	}
	parent := firstHeaderValue(request.Headers, p.parentHeaders)

	switch {
	case parent != "":
		if session != "" {
			p.recordLink(request.FairnessID, session, parent)
		}
		request.PutAttribute(DerivedRootKey, p.walkToRoot(request.FairnessID, parent))
	case session != "":
		request.PutAttribute(DerivedRootKey, p.walkToRoot(request.FairnessID, session))
	}
	return nil
}

// firstHeaderValue returns the first trimmed non-empty value among the
// ordered header names.
func firstHeaderValue(headers map[string]string, names []string) string {
	for _, name := range names {
		if value := strings.TrimSpace(headers[name]); value != "" {
			return value
		}
	}
	return ""
}

// codexTurnMetadataSession returns the session_id field of the Codex
// turn-metadata header: the root session identity shared by every thread in
// the tree. Unparsable, oversized, or session-less values return "" so the
// caller falls back to the session header list.
func codexTurnMetadataSession(value string) string {
	if len(value) == 0 || len(value) > maxTurnMetadataHeaderBytes {
		return ""
	}
	var payload struct {
		SessionID string `json:"session_id"`
	}
	if err := json.Unmarshal([]byte(value), &payload); err != nil {
		return ""
	}
	return strings.TrimSpace(payload.SessionID)
}

// recordLink stores child -> parent under the tenant scope. A link naming
// the session's own ID is dropped: it carries no lineage. At the bound the
// least-recently-used link is evicted; expired links are the oldest, so the
// bound also reclaims them. The eviction scan is O(len(parents)), kept small
// by the bound.
func (p *Plugin) recordLink(tenant, session, parent string) {
	if session == parent {
		return
	}
	now := p.now()
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.parents) >= p.maxParentEntries {
		var oldestKey lineageKey
		var oldest time.Time
		first := true
		for key, link := range p.parents {
			if first || link.updatedAt.Before(oldest) {
				oldestKey, oldest, first = key, link.updatedAt, false
			}
		}
		if !first {
			delete(p.parents, oldestKey)
		}
	}
	p.parents[lineageKey{tenant: tenant, session: session}] = parentLink{parent: parent, updatedAt: now}
}

// walkToRoot follows recorded parent links from start to the first ancestor
// with no live link, at most maxWalkHops links. Reading a link refreshes it,
// so an in-use program never expires; an expired link is deleted and treated
// as absent. A cycle ends the walk at the current node.
func (p *Plugin) walkToRoot(tenant, start string) string {
	now := p.now()
	p.mu.Lock()
	defer p.mu.Unlock()

	current := start
	// visited is allocated only when a link is followed: a request that
	// walks nothing must not pay a map allocation.
	var visited map[string]bool
	for hops := 0; ; hops++ {
		if hops >= maxWalkHops {
			return current
		}
		key := lineageKey{tenant: tenant, session: current}
		link, ok := p.parents[key]
		if !ok || now.Sub(link.updatedAt) > p.parentEntryTTL {
			if ok {
				delete(p.parents, key)
			}
			return current
		}
		if visited == nil {
			visited = map[string]bool{start: true}
		}
		if visited[link.parent] {
			return current
		}
		visited[link.parent] = true
		p.parents[key] = parentLink{parent: link.parent, updatedAt: now}
		current = link.parent
	}
}
