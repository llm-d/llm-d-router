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
	"math"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// DerivedRootKey is the request-attribute key under which this plugin
// publishes the derived program root: the session ID of the root request of
// the tree the request belongs to. The Director reads it via
// scheduling.ReadRequestAttribute to compose the fairness key.
// The key carries no producer name: any plugin able to derive a program root
// is an acceptable source.
var DerivedRootKey = plugin.NewDataKey("derived-root", "")

// ResolvedBandKey is the request-attribute key under which this plugin
// publishes the request's resolved priority band: the seeded objective
// priority, or the parent's band for a parented child without an explicit
// objective, or the minimum of the two when the request opts down
// explicitly. The Director applies it once, before admission — bands
// resolve once and queued requests never migrate.
var ResolvedBandKey = plugin.NewDataKey("resolved-band", "")

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

// lineageKey scopes a node record to the tenant the request arrived under:
// the fairness-ID header value at classification time, "" when absent.
type lineageKey struct {
	tenant  string
	session string
}

// nodeRecord is one program node: its parent link ("" for a root) and its
// resolved band, refreshed on arrival.
type nodeRecord struct {
	parent    string
	band      int
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
	ttl := defaultParentEntryTTL
	if params.ParentEntryTTLSeconds != 0 {
		if params.ParentEntryTTLSeconds > float64(math.MaxInt64)/float64(time.Second) {
			return nil, fmt.Errorf("agent-lineage: parentEntryTtlSeconds %v exceeds the representable duration", params.ParentEntryTTLSeconds)
		}
		ttl = time.Duration(params.ParentEntryTTLSeconds * float64(time.Second))
		if ttl <= 0 {
			return nil, fmt.Errorf("agent-lineage: parentEntryTtlSeconds %v is too small to represent as a duration", params.ParentEntryTTLSeconds)
		}
	}
	maxEntries := params.MaxParentEntries
	if maxEntries == 0 {
		maxEntries = defaultMaxParentEntries
	}
	return &Plugin{
		typedName:        plugin.TypedName{Type: PluginType, Name: name},
		sessionHeaders:   mergeHeaders(params.AdditionalSessionHeaders, defaultSessionHeaders),
		parentHeaders:    mergeHeaders(params.AdditionalParentHeaders, defaultParentHeaders),
		maxParentEntries: maxEntries,
		parentEntryTTL:   ttl,
		now:              time.Now,
		nodes:            map[lineageKey]nodeRecord{},
	}, nil
}

// mergeHeaders returns extras followed by defaults, lowercased. Empty and
// whitespace-only extras are dropped. Request headers are stored with
// lowercased keys (see handlers.HandleRequestHeaders), so both lists must
// match. Duplicates are left in — the request-time loop short-circuits on
// first match.
func mergeHeaders(extras, defaults []string) []string {
	merged := make([]string, 0, len(extras)+len(defaults))
	for _, h := range extras {
		if h = strings.ToLower(strings.TrimSpace(h)); h != "" {
			merged = append(merged, h)
		}
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

	mu    sync.Mutex
	nodes map[lineageKey]nodeRecord
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces declares the derived-root and resolved-band request attributes
// written by this request-header processor.
func (p *Plugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{DerivedRootKey: "", ResolvedBandKey: 0}
}

// RequestHeader resolves the request's own session and parent IDs, records
// the session's node — its parent link and its resolved band — and publishes
// the root reached by walking recorded links together with the resolved
// band. A request with neither a session nor a parent publishes nothing and
// leaves the fairness key and priority to their other inputs.
func (p *Plugin) RequestHeader(_ context.Context, request *scheduling.InferenceRequest) error {
	session := codexTurnMetadataSession(request.Headers[CodexTurnMetadataHeader])
	if session == "" {
		session = firstHeaderValue(request.Headers, p.sessionHeaders)
	}
	parent := firstHeaderValue(request.Headers, p.parentHeaders)

	if session == "" && parent == "" {
		return nil
	}
	seeded := request.Objectives.Priority
	explicit := strings.TrimSpace(request.Headers[metadata.ObjectiveKey]) != ""
	root, band := p.resolveRoot(request.FairnessID, session, parent, seeded, explicit)
	request.PutAttribute(DerivedRootKey, root)
	request.PutAttribute(ResolvedBandKey, band)
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

// resolveRoot records the session's node and returns the root reached by
// walking recorded links from the request's session, or from its parent when
// the request carries no session, together with the request's resolved band.
// One critical section covers resolution, record, and walk, so a concurrent
// request cannot evict or rewrite state between them.
//
// The band follows the design's resolution table: a parented child whose
// parent has a live node inherits the parent's band, an explicit objective
// opts to min(explicit, parent band) — never above the parent — and every
// other request keeps its seeded band. A declared workload kind is not a
// band input.
func (p *Plugin) resolveRoot(tenant, session, parent string, seeded int, explicit bool) (string, int) {
	now := p.now()
	p.mu.Lock()
	defer p.mu.Unlock()

	// While a live linked node exists it is authoritative for the session's
	// lineage: the recorded parent, not a conflicting re-declaration,
	// supplies both the band and the walk. A parentless node has no link to
	// conflict with, so a later parent declaration is accepted.
	if record, ok := p.nodes[lineageKey{tenant: tenant, session: session}]; ok && now.Sub(record.updatedAt) <= p.parentEntryTTL && record.parent != "" {
		parent = record.parent
	}
	band := seeded
	if parent != "" {
		if record, ok := p.nodes[lineageKey{tenant: tenant, session: parent}]; ok && now.Sub(record.updatedAt) <= p.parentEntryTTL {
			if explicit {
				band = min(seeded, record.band)
			} else {
				band = record.band
			}
		}
	}
	p.recordNodeLocked(tenant, session, parent, band, now)
	start := session
	if start == "" {
		start = parent
	}
	return p.walkToRootLocked(tenant, start, now), band
}

// recordNodeLocked stores the session's node under the tenant scope: its
// parent link and its resolved band. A node naming itself as parent is
// stored parentless: the self-link carries no lineage. A session's parent is
// fixed when the child is spawned, so a live link keeps its first parent —
// a declaration naming a different parent is ignored until the node expires,
// which keeps a caller sharing the default tenant from splicing a live
// session onto another program or cycling a root. The band always refreshes
// to the latest resolution. At the bound the least-recently-used node is
// evicted, but never for a key already in the map: harnesses resend both
// headers every turn, and overwriting an existing key does not grow the
// map. A parentless node is not recorded at the bound at all: a link
// carries lineage a bare band does not, so parentless arrivals must not
// evict live links. The eviction scan is O(len(nodes)), kept small by the
// bound.
func (p *Plugin) recordNodeLocked(tenant, session, parent string, band int, now time.Time) {
	if parent == session {
		parent = ""
	}
	key := lineageKey{tenant: tenant, session: session}
	if record, ok := p.nodes[key]; ok {
		if now.Sub(record.updatedAt) <= p.parentEntryTTL && record.parent != "" && record.parent != parent {
			p.nodes[key] = nodeRecord{parent: record.parent, band: band, updatedAt: now}
			return
		}
	} else if parent == "" && len(p.nodes) >= p.maxParentEntries {
		return
	} else if len(p.nodes) >= p.maxParentEntries {
		var oldestKey lineageKey
		var oldest time.Time
		first := true
		for k, record := range p.nodes {
			if first || record.updatedAt.Before(oldest) {
				oldestKey, oldest, first = k, record.updatedAt, false
			}
		}
		if !first {
			delete(p.nodes, oldestKey)
		}
	}
	p.nodes[key] = nodeRecord{parent: parent, band: band, updatedAt: now}
}

// walkToRootLocked follows recorded parent links from start to the first
// ancestor with no live parent link, at most maxWalkHops links. Reading a
// node refreshes it, so an in-use program never expires; an expired node is
// deleted and treated as absent. A cycle ends the walk at the current node.
func (p *Plugin) walkToRootLocked(tenant, start string, now time.Time) string {
	current := start
	// visited is allocated only when a link is followed: a request that
	// walks nothing must not pay a map allocation.
	var visited map[string]bool
	for hops := 0; ; hops++ {
		if hops >= maxWalkHops {
			return current
		}
		key := lineageKey{tenant: tenant, session: current}
		record, ok := p.nodes[key]
		if !ok {
			return current
		}
		if now.Sub(record.updatedAt) > p.parentEntryTTL {
			delete(p.nodes, key)
			return current
		}
		if record.parent == "" {
			return current
		}
		if visited == nil {
			visited = map[string]bool{start: true}
		}
		if visited[record.parent] {
			return current
		}
		visited[record.parent] = true
		p.nodes[key] = nodeRecord{parent: record.parent, band: record.band, updatedAt: now}
		current = record.parent
	}
}
