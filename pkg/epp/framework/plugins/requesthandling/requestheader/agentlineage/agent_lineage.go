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
// the program root and resolved priority band of a request from agent-lineage
// headers and stores both as request attributes for the Director.
package agentlineage

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/duration"
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

// ResolvedBandKey is the request-attribute key under which a plugin publishes
// the request's resolved priority band: the seeded objective priority, a
// parent's band for a parented child without an explicit objective, or the
// minimum of the two when the request opts down explicitly. The Director
// applies it once, before admission — bands resolve once and queued requests
// never migrate.
// The key carries no producer name: any plugin able to resolve a band is an
// acceptable source.
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
	// MaxParentEntries bounds the per-replica node map; at the bound an
	// expired node is reclaimed first, then the oldest standalone node is
	// evicted.
	MaxParentEntries int `json:"maxParentEntries,omitempty"`
	// ParentEntryTTLSeconds is how long a node survives unused.
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
		var err error
		if ttl, err = duration.FromSeconds("parentEntryTtlSeconds", params.ParentEntryTTLSeconds); err != nil {
			return nil, fmt.Errorf("agent-lineage: %w", err)
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
		childCounts:      map[lineageKey]int{},
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

// Plugin derives the program root and resolved priority band of a request from
// agent-lineage headers and stores them as request attributes.
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
	// childCounts counts the live child nodes recorded under each parent key,
	// including parents that are not currently in nodes. An entry is removed
	// when its last child is removed, so the map is bounded by nodes.
	childCounts map[lineageKey]int
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
	objective, _ := metadata.GetLowerCaseHeaderValue(request.Headers, metadata.ObjectiveKey)
	explicit := strings.TrimSpace(objective) != ""
	root, band := p.resolve(request.FairnessID, session, parent, seeded, explicit)
	request.PutAttribute(DerivedRootKey, root)
	if band != seeded {
		// Published only when lineage changed the band: the Director
		// overrides the priority for inheritance alone, and an echoed seed
		// could revert a priority another processor adjusted in between.
		request.PutAttribute(ResolvedBandKey, band)
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

// resolve records the session's node and returns the root reached by walking
// recorded links from the request's session, or from its parent when the
// request carries no session, together with the request's resolved band.
// One critical section covers resolution, record, and walk, so a concurrent
// request cannot evict or rewrite state between them.
//
// A live link is authoritative for a session's lineage: the recorded parent,
// not a conflicting re-declaration, supplies both the band and the walk. A
// root with a live child is authoritative too, so a later parent declaration
// cannot splice it into another program. A standalone node accepts a later
// parent. A self-link carries no lineage and is cleared before resolution.
//
// The band follows the design's resolution table: under a stamped tenant (a
// non-empty tenant scope) a parented child whose parent has a live node
// inherits the parent's band; an explicit objective opts to
// min(explicit, parent band) in every population; an unstamped child takes
// min(seeded, parent band) because the untrusted-priority ceiling is not
// enforced yet; every other request keeps its seeded band. A declared
// workload kind is not a band input.
func (p *Plugin) resolve(tenant, session, parent string, seeded int, explicit bool) (string, int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	now := p.now()
	if session != "" && parent == session {
		parent = ""
	}
	if session != "" {
		if _, record, ok := p.liveNodeLocked(tenant, session, now); ok && (record.parent != "" || p.hasChildLocked(tenant, session)) {
			parent = record.parent
		}
	}
	band := seeded
	if parent != "" {
		if key, record, ok := p.liveNodeLocked(tenant, parent, now); ok {
			record.updatedAt = now
			p.nodes[key] = record
			if explicit || tenant == "" {
				band = min(seeded, record.band)
			} else {
				band = record.band
			}
		}
	}
	if session != "" {
		p.recordNodeLocked(tenant, session, parent, band, now)
	}
	start := session
	if start == "" {
		start = parent
	}
	return p.walkToRootLocked(tenant, start, now), band
}

// recordNodeLocked stores the session's node under the tenant scope: its
// parent link and its resolved band. resolve has already cleared a self-link
// and applied an authoritative recorded parent, so this only refreshes the
// band, keeps the live-child counts consistent, and evicts at the bound. An
// expired node is removed before insertion. At the bound a node is evicted,
// but never for a key already in the map: harnesses resend both headers every
// turn, and overwriting an existing key does not grow the map.
func (p *Plugin) recordNodeLocked(tenant, session, parent string, band int, now time.Time) {
	key := lineageKey{tenant: tenant, session: session}
	if record, ok := p.nodes[key]; ok {
		if now.Sub(record.updatedAt) <= p.parentEntryTTL {
			if record.parent != parent {
				p.removeChildLocked(tenant, record.parent)
				p.addChildLocked(tenant, parent)
			}
			record.parent = parent
			record.band = band
			record.updatedAt = now
			p.nodes[key] = record
			return
		}
		p.deleteNodeLocked(key, record)
	}
	if len(p.nodes) >= p.maxParentEntries {
		if !p.evictLocked(tenant, parent, now) {
			return
		}
	}
	p.nodes[key] = nodeRecord{parent: parent, band: band, updatedAt: now}
	p.addChildLocked(tenant, parent)
}

// evictLocked removes one node to make room for an arrival whose parent is
// parent. The scan deletes every expired node it passes. If no slot is free it
// prefers the oldest standalone node (no parent and no live child), then the
// oldest leaf link (a linked node with no live child), and finally the oldest
// node with a live child, whether a root or an interior node. A parentless
// arrival is rejected when every slot is live and linked or has a child. The
// arriving parent is never evicted. The scan is O(len(nodes)), kept small by
// the bound.
func (p *Plugin) evictLocked(tenant, parent string, now time.Time) bool {
	protected := lineageKey{tenant: tenant, session: parent}
	var standaloneKey, leafKey, hasChildKey lineageKey
	var standaloneAt, leafAt, hasChildAt time.Time
	haveStandalone, haveLeaf, haveHasChild := false, false, false
	for key, record := range p.nodes {
		if parent != "" && key == protected {
			continue
		}
		if now.Sub(record.updatedAt) > p.parentEntryTTL {
			p.deleteNodeLocked(key, record)
			continue
		}
		if record.parent == "" && !p.hasChildLocked(key.tenant, key.session) {
			if !haveStandalone || record.updatedAt.Before(standaloneAt) {
				standaloneKey, standaloneAt, haveStandalone = key, record.updatedAt, true
			}
			continue
		}
		if record.parent != "" && !p.hasChildLocked(key.tenant, key.session) {
			if !haveLeaf || record.updatedAt.Before(leafAt) {
				leafKey, leafAt, haveLeaf = key, record.updatedAt, true
			}
			continue
		}
		if !haveHasChild || record.updatedAt.Before(hasChildAt) {
			hasChildKey, hasChildAt, haveHasChild = key, record.updatedAt, true
		}
	}
	if len(p.nodes) < p.maxParentEntries {
		return true
	}
	switch {
	case haveStandalone:
		p.deleteNodeLocked(standaloneKey, p.nodes[standaloneKey])
	case parent == "":
		return false
	case haveLeaf:
		p.deleteNodeLocked(leafKey, p.nodes[leafKey])
	case haveHasChild:
		p.deleteNodeLocked(hasChildKey, p.nodes[hasChildKey])
	default:
		return false
	}
	return true
}

// hasChildLocked reports whether a live child node is recorded under the
// parent's key.
func (p *Plugin) hasChildLocked(tenant, session string) bool {
	return p.childCounts[lineageKey{tenant: tenant, session: session}] > 0
}

func (p *Plugin) addChildLocked(tenant, parent string) {
	if parent == "" {
		return
	}
	if p.childCounts == nil {
		p.childCounts = map[lineageKey]int{}
	}
	p.childCounts[lineageKey{tenant: tenant, session: parent}]++
}

func (p *Plugin) removeChildLocked(tenant, parent string) {
	if parent == "" {
		return
	}
	key := lineageKey{tenant: tenant, session: parent}
	if count := p.childCounts[key]; count > 1 {
		p.childCounts[key] = count - 1
	} else {
		delete(p.childCounts, key)
	}
}

// deleteNodeLocked removes a node and detaches it from its parent's live-child
// count. The node's own child count is left in place while its children remain.
func (p *Plugin) deleteNodeLocked(key lineageKey, record nodeRecord) {
	delete(p.nodes, key)
	p.removeChildLocked(key.tenant, record.parent)
}

// liveNodeLocked returns the node for session when it exists and has not
// expired, deleting an expired node. It does not refresh the node.
func (p *Plugin) liveNodeLocked(tenant, session string, now time.Time) (lineageKey, nodeRecord, bool) {
	key := lineageKey{tenant: tenant, session: session}
	record, ok := p.nodes[key]
	if !ok {
		return key, nodeRecord{}, false
	}
	if now.Sub(record.updatedAt) > p.parentEntryTTL {
		p.deleteNodeLocked(key, record)
		return key, nodeRecord{}, false
	}
	return key, record, true
}

// walkToRootLocked follows recorded parent links from start to the first
// ancestor with no live parent link, at most maxWalkHops links. Reading a
// node refreshes it — including the terminal root, so an idle root with
// active children stays alive — so an in-use program never expires; an
// expired node is deleted and treated as absent. A cycle ends the walk at
// the current node.
func (p *Plugin) walkToRootLocked(tenant, start string, now time.Time) string {
	current := start
	// visited is allocated only when a link is followed: a request that
	// walks nothing must not pay a map allocation.
	var visited map[string]bool
	for hops := 0; ; hops++ {
		if hops >= maxWalkHops {
			return current
		}
		key, record, ok := p.liveNodeLocked(tenant, current, now)
		if !ok {
			return current
		}
		if record.parent == "" {
			record.updatedAt = now
			p.nodes[key] = record
			return current
		}
		if visited == nil {
			visited = map[string]bool{start: true}
		}
		if visited[record.parent] {
			return current
		}
		visited[record.parent] = true
		record.updatedAt = now
		p.nodes[key] = record
		current = record.parent
	}
}
